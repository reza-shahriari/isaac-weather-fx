"""Viewport backend: everything an RTX camera can see."""
from __future__ import annotations

import logging

from ...runtime.context import SESSION_ROOT
from ..base import SensorBackend
from .fog import FogEffect
from .lighting import LightingEffect
from .precipitation import PrecipitationEffect

log = logging.getLogger("weather_fx")


class ViewportBackend(SensorBackend):
    name = "viewport"

    def __init__(self, effects=None):
        self.effects = effects or [FogEffect(), PrecipitationEffect("rain"),
                                   PrecipitationEffect("snow"), LightingEffect()]

    def add_effect(self, effect) -> None:
        """Extension point: plug in your own Effect (e.g. lens droplets)."""
        effect.attach(self.context)
        self.effects.append(effect)

    def attach(self, context):
        super().attach(context)
        for effect in self.effects:
            effect.attach(context)

    def apply_state(self, state, changed):
        for effect in self.effects:
            try:
                effect.apply_state(state, changed)
            except Exception:
                log.exception("weather_fx: effect %s failed", effect.name)

    def update(self, dt, t):
        for effect in self.effects:
            try:
                effect.update(dt, t)
            except Exception:
                log.exception("weather_fx: effect %s update failed", effect.name)

    def detach(self):
        for effect in reversed(self.effects):
            try:
                effect.detach()
            except Exception:
                log.exception("weather_fx: effect %s detach failed", effect.name)
        try:
            from pxr import Usd

            stage = self.context.stage()
            if stage is not None and stage.GetPrimAtPath(SESSION_ROOT):
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    stage.RemovePrim(SESSION_ROOT)
        except Exception:
            pass

    def stats(self):
        return {effect.name: effect.stats() for effect in self.effects}
