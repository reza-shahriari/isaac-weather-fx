"""Fog through RTX Simple Fog render settings (affects viewport and RTX cameras)."""
from __future__ import annotations

import logging

from ...core.jobs import LatestJob
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
        self._haze_colours = {}
        self._haze_job = LatestJob("weather_fx-haze")

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
            if self._mode != "haze" or {"sky", "fog", "general"} & set(changed):
                self._apply_haze(state)
        else:
            self._restore()

    def _apply_fog(self, fog):
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
        self._mode = "fog"

    def _apply_haze(self, state):
        """Aerial perspective: the sky's own haze on the stage's geometry.

        With no fog asked for, distant objects should still fade into the horizon the way they do
        in the sky above them -- otherwise a building two kilometres away is as crisp as one at
        twenty metres and meets the dome's hazy horizon at a hard edge. The extinction is the
        visibility the turbidity implies (the same number an infrared model scales its aerosol
        from) and the colour is the sky's own horizon.
        """
        from ...core.atmosphere import atmosphere_for, haze_visibility_m, sky_view
        from ...core.sky import conditions_from_state

        sky = state.sky
        conditions = conditions_from_state(state, build_cloud=False)
        # Half-degree steps: the colour changes slowly, and a table per step is cached.
        elevation = round(max(conditions.sun.elevation_deg, -18.0) * 2.0) / 2.0
        key = (round(float(sky.turbidity), 3), tuple(sky.ground_albedo), elevation)
        colour = self._haze_colours.get(key)
        if colour is None and key != self._haze_job.running_key:
            # A new sun position or turbidity costs a sky table (0.2 s, or 1.3 s for a new
            # turbidity): computed in the background so dragging the site or the clock never
            # stalls the UI. The previous colour stays until it lands.
            turbidity, albedo = float(sky.turbidity), tuple(sky.ground_albedo)

            def compute():
                return sky_view(atmosphere_for(turbidity, albedo), elevation).horizon_colour()

            if state.general.time_source == "manual":
                self._haze_job.cancel()
                colour = compute()
                self._remember_colour(key, colour)
            else:
                self._haze_job.submit(key, compute)

        mpu = self.context.meters_per_unit()
        self._beta = extinction_from_visibility(haze_visibility_m(sky.turbidity))
        self._write("enabled", True)
        if colour is not None:
            peak = float(max(colour.max(), 1e-12))
            self._write("color", tuple(float(c) / peak for c in colour))
        self._write("color_intensity", 1.0)
        self._write("z_up", self.context.up_axis() == 2)
        self._write("start_distance", 0.0)
        self._write("end_distance", 200_000.0 / mpu)
        self._write("distance_density", float(self._beta * mpu * state.fog.density_calibration))
        self._write("height_density", 0.0)
        self._active = True
        self._mode = "haze"

    def update(self, dt, t):
        try:
            done = self._haze_job.poll()
        except Exception:
            log.exception("weather_fx: computing the haze colour failed")
            return
        if done is None:
            return
        key, colour = done
        self._remember_colour(key, colour)
        if self._mode == "haze":
            self._apply_haze(self.context.state)

    def _remember_colour(self, key, colour):
        if len(self._haze_colours) > 256:
            self._haze_colours.clear()
        self._haze_colours[key] = colour

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
