"""Fog through RTX Simple Fog render settings (affects viewport and RTX cameras)."""
from __future__ import annotations

import logging

from ...core.physics import extinction_from_visibility
from ..base import Effect
from .rtx_settings import FOG_SETTINGS

log = logging.getLogger("weather_fx")


class FogEffect(Effect):
    name = "fog"

    def __init__(self):
        self._originals = {}   # path -> value before we touched it
        self._missing = set()
        self._active = False
        self._beta = 0.0

    def _settings(self):
        import carb.settings

        return carb.settings.get_settings()

    def _write(self, key, value):
        path = FOG_SETTINGS[key]
        settings = self._settings()
        current = settings.get(path)
        if current is None:
            if path not in self._missing:
                self._missing.add(path)
                log.warning("weather_fx: RTX setting %s not found, skipping (see rtx_settings.py)", path)
            return
        if path not in self._originals:
            self._originals[path] = current
        if isinstance(value, (list, tuple)):
            settings.set_float_array(path, [float(v) for v in value])
        else:
            settings.set(path, value)

    def apply_state(self, state, changed):
        fog = state.fog
        if not (state.general.enabled and fog.enabled):
            self._restore()
            return
        mpu = self.context.meters_per_unit()
        self._beta = extinction_from_visibility(fog.visibility_m)
        density = self._beta * mpu * fog.density_calibration  # per stage unit

        self._write("enabled", True)
        self._write("color", fog.color)
        self._write("color_intensity", float(fog.color_intensity))
        self._write("z_up", self.context.up_axis() == 2)
        self._write("start_distance", fog.start_distance_m / mpu)
        self._write("end_distance", fog.end_distance_m / mpu)
        self._write("distance_density", float(density))
        self._write("start_height", fog.base_height_m / mpu)
        self._write("height_density", float(fog.height_density if fog.height_fog else 0.0))
        self._write("height_falloff", float(fog.height_falloff))
        self._active = True

    def _restore(self):
        if not self._originals:
            self._active = False
            return
        settings = self._settings()
        for path, value in self._originals.items():
            try:
                if isinstance(value, (list, tuple)):
                    settings.set_float_array(path, list(value))
                else:
                    settings.set(path, value)
            except Exception:
                log.exception("weather_fx: could not restore %s", path)
        self._originals.clear()
        self._active = False

    def detach(self):
        self._restore()

    def stats(self):
        if not self._active:
            return {"active": False}
        return {"active": True, "extinction_per_m": round(self._beta, 6),
                "missing_settings": sorted(self._missing)}
