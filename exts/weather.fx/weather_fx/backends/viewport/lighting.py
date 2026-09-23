"""Scales scene light intensities through the session layer (the user's files stay untouched)."""
from __future__ import annotations

import logging

from ..base import Effect

log = logging.getLogger("weather_fx")


class LightingEffect(Effect):
    name = "lighting"

    def __init__(self):
        # attr path -> (base_value, session_value_before_or_None)
        self._touched = {}
        self._stage = None

    def apply_state(self, state, changed):
        params = state.lighting
        stage = self.context.stage()
        if stage is None:
            return
        if self._stage is not None and stage != self._stage:
            self._touched.clear()  # old stage is gone; nothing to restore
        self._stage = stage
        if not (state.general.enabled and params.enabled):
            self._restore()
            return
        if "lighting" not in changed and "general" not in changed and self._touched:
            return
        from pxr import Usd, UsdLux

        session = stage.GetSessionLayer()
        with Usd.EditContext(stage, session):
            for prim in stage.Traverse():
                if not prim.HasAPI(UsdLux.LightAPI):
                    continue
                if not params.include_dome and prim.IsA(UsdLux.DomeLight):
                    self._restore_attr(stage, UsdLux.LightAPI(prim).GetIntensityAttr())
                    continue
                attr = UsdLux.LightAPI(prim).GetIntensityAttr()
                key = str(attr.GetPath())
                if key not in self._touched:
                    spec = session.GetAttributeAtPath(attr.GetPath())
                    before = spec.default if (spec and spec.HasDefaultValue()) else None
                    base = attr.Get()
                    if base is None:
                        continue
                    self._touched[key] = (float(base), before)
                base, _ = self._touched[key]
                attr.Set(base * params.light_scale)

    def _restore_attr(self, stage, attr):
        from pxr import Usd

        key = str(attr.GetPath())
        if key not in self._touched:
            return
        _, before = self._touched.pop(key)
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            if before is None:
                attr.Clear()
            else:
                attr.Set(before)

    def _restore(self):
        stage = self._stage
        if stage is None or not self._touched:
            self._touched.clear()
            return
        from pxr import Sdf, Usd

        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for key, (_, before) in list(self._touched.items()):
                attr = stage.GetAttributeAtPath(Sdf.Path(key))
                if not attr:
                    continue
                if before is None:
                    attr.Clear()
                else:
                    attr.Set(before)
        self._touched.clear()

    def detach(self):
        self._restore()

    def stats(self):
        return {"lights_scaled": len(self._touched)}
