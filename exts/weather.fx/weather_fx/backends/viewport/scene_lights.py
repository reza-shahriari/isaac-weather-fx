"""Hide the stage's own lights while weather-fx authors the sky.

A new Isaac Sim stage comes with a ``defaultLight`` (a ``DistantLight``) and often a dome of its
own. Left on, that light is a second sun in a direction nothing computed, and a second sky
brightening every shadow. So while the sky effect is active, every light outside ``/WeatherFX`` is
made invisible with a **session-layer** opinion, and ``restore`` removes exactly those opinions.
The user's layers are never written, and a light the user had already hidden stays hidden.

The viewport's "camera light" mode is a render setting rather than a prim; it is switched off the
same way the fog effect switches its settings: remember the value, write ours, put it back.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger("weather_fx")

#: Render settings that make the viewport light the scene from the camera instead of from the
#: stage's lights. Missing paths are skipped: the name has moved between Kit releases.
VIEW_LIGHTING_SETTINGS = ("/rtx/useViewLightingMode",)


def is_own_prim(path: str, root: str = "/WeatherFX") -> bool:
    """True for ``root`` itself and anything under it."""
    return path == root or path.startswith(root + "/")


class SceneLightSuppressor:
    """Makes foreign lights invisible in the session layer, and undoes exactly that."""

    def __init__(self, root: str = "/WeatherFX"):
        self._root = root
        # visibility attribute path -> session-layer value before we wrote (None: no opinion)
        self._hidden: Dict[str, Optional[str]] = {}
        self._settings: Dict[str, Any] = {}
        self._stage = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def hidden_count(self) -> int:
        return len(self._hidden)

    def suppress(self, stage: Any, rescan: bool = False) -> None:
        """Hide every light not under the root.

        One stage traversal on activation, not one per state change: the sky's clock advancing
        changes the state every tick, and walking a large stage that often is not free. Pass
        ``rescan`` to pick up lights added since.
        """
        from pxr import Usd, UsdGeom, UsdLux

        if self._stage is not None and stage != self._stage:
            self._hidden.clear()  # the old stage is gone; nothing to restore there
            self._active = False
        if self._active and not rescan:
            return
        self._stage = stage
        session = stage.GetSessionLayer()
        with Usd.EditContext(stage, session):
            for prim in stage.Traverse():
                if not prim.HasAPI(UsdLux.LightAPI) or is_own_prim(str(prim.GetPath()), self._root):
                    continue
                imageable = UsdGeom.Imageable(prim)
                attr = imageable.GetVisibilityAttr()
                key = str(attr.GetPath())
                if key in self._hidden:
                    continue
                spec = session.GetAttributeAtPath(attr.GetPath())
                before = spec.default if (spec and spec.HasDefaultValue()) else None
                if before is None and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                    continue  # already hidden by the user; not ours to touch or restore
                self._hidden[key] = before
                imageable.MakeInvisible()
        self._suppress_view_lighting()
        self._active = True

    def restore(self) -> None:
        stage = self._stage
        if stage is not None and self._hidden:
            from pxr import Sdf, Usd

            with Usd.EditContext(stage, stage.GetSessionLayer()):
                for key, before in self._hidden.items():
                    attr = stage.GetAttributeAtPath(Sdf.Path(key))
                    if not attr:
                        continue
                    if before is None:
                        attr.Clear()
                    else:
                        attr.Set(before)
        self._hidden.clear()
        self._active = False
        self._restore_view_lighting()

    # --- render settings -------------------------------------------------------------

    def _suppress_view_lighting(self) -> None:
        try:
            import carb.settings
        except ImportError:
            return
        settings = carb.settings.get_settings()
        for path in VIEW_LIGHTING_SETTINGS:
            current = settings.get(path)
            if current is None:
                continue
            self._settings.setdefault(path, current)
            settings.set(path, False)

    def _restore_view_lighting(self) -> None:
        if not self._settings:
            return
        try:
            import carb.settings
        except ImportError:
            self._settings.clear()
            return
        settings = carb.settings.get_settings()
        for path, value in self._settings.items():
            settings.set(path, value)
        self._settings.clear()
