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
        self._mode = None  # "fog", "haze" or None
        self._published = None  # the sky colour the settings were last written with

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
        if not state.general.enabled:
            self._restore()
            return
        if fog.enabled:
            self._apply_fog(fog)
        elif state.sky.enabled and state.sky.aerial_perspective:
            self._apply_haze(state)
        else:
            self._restore()

    def _sky_luminance(self):
        """The sky's horizon luminance in renderer units, or None when no sky is authored."""
        rgb = self.context.sky_horizon_rgb
        if rgb is None:
            return None
        return max(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2], 1e-9)

    def _apply_fog(self, fog):
        mpu = self.context.meters_per_unit()
        self._beta = extinction_from_visibility(fog.visibility_m)
        density = self._beta * mpu * fog.density_calibration  # per stage unit

        # RTX fog colours are in the renderer's units. Under the authored sky the dome is drawn at
        # about 150 of those, so a colour of 0.7 at intensity 1 is a black fog: scale it by the
        # sky's own horizon level and keep the user's colour as the tint. With no sky authored,
        # the colour means what it always meant.
        level = self._sky_luminance()
        intensity = float(fog.color_intensity) * (level if level is not None else 1.0)
        self._published = self.context.sky_horizon_rgb

        self._write("enabled", True)
        self._write("color", fog.color)
        self._write("color_intensity", intensity)
        self._write("z_up", self.context.up_axis() == 2)
        self._write("start_distance", fog.start_distance_m / mpu)
        self._write("end_distance", fog.end_distance_m / mpu)
        self._write("distance_density", float(density))
        self._write("start_height", fog.base_height_m / mpu)
        self._write("height_density", float(fog.height_density if fog.height_fog else 0.0))
        self._write("height_falloff", float(fog.height_falloff))
        self._active = True
        self._mode = "fog"

    def _apply_haze(self, state):
        """Aerial perspective: the sky's own haze on the stage's geometry (real-time only).

        The extinction is the visibility the turbidity implies (the number an infrared model
        scales its aerosol from) and the colour is the sky's horizon, in renderer units, as the
        sky effect measured it on the dome it baked. Until a dome exists there is nothing to
        match, so the fog stays off rather than guessing a colour.
        """
        from ...core.atmosphere import haze_visibility_m

        rgb = self.context.sky_horizon_rgb
        self._published = rgb
        if rgb is None:
            self._restore()
            self._mode = "haze"  # keep watching for the sky's colour
            return
        peak = max(max(rgb), 1e-9)
        mpu = self.context.meters_per_unit()
        self._beta = extinction_from_visibility(haze_visibility_m(state.sky.turbidity))
        self._write("enabled", True)
        self._write("color", tuple(float(c) / peak for c in rgb))
        self._write("color_intensity", float(peak))
        self._write("z_up", self.context.up_axis() == 2)
        self._write("start_distance", 0.0)
        self._write("end_distance", 200_000.0 / mpu)
        self._write("distance_density", float(self._beta * mpu * state.fog.density_calibration))
        self._write("height_density", 0.0)
        self._active = True
        self._mode = "haze"

    def update(self, dt, t):
        """Follow the sky: each new bake publishes a new horizon colour (and level)."""
        if self._mode is None or self.context.sky_horizon_rgb == self._published:
            return
        self.apply_state(self.context.state, {"sky"})

    def _restore(self):
        if not self._originals:
            self._active = False
            self._mode = None
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
        self._mode = None

    def detach(self):
        self._restore()

    def stats(self):
        if not self._active:
            return {"active": False}
        return {"active": True, "mode": self._mode, "extinction_per_m": round(self._beta, 6),
                "missing_settings": sorted(self._missing)}
