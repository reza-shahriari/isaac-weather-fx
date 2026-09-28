"""Auto exposure for RTX Real-Time.

On Kit 110 RTX Real-Time (``RealTimePathTracing``) exposes with a fixed camera exposure that is
orders of magnitude too dark for daylight. A plain stage with a 3000 lux sun and a 1000 nit dome
renders black there while the path tracer shows it fine (``tools/probe_realtime_renderer.py``).
Turning on the histogram auto exposure brings the frame back, so this effect turns it on whenever
the viewport runs anything other than the path tracer, and puts the setting back when the path
tracer runs, when ``general.realtime_auto_exposure`` is turned off, or when the weather detaches.
"""
from __future__ import annotations

import logging

from ..base import Effect
from .render_mode import PATH_TRACED_MODES, current_render_mode

log = logging.getLogger("weather_fx")

HISTOGRAM_SETTING = "/rtx/post/histogram/enabled"


class RealTimeExposureEffect(Effect):
    name = "realtime_exposure"

    def __init__(self):
        self._wanted = False
        self._original = None  # the setting's value before we changed it; None = untouched
        self._mode = None

    def _settings(self):
        import carb.settings

        return carb.settings.get_settings()

    def apply_state(self, state, changed):
        self._wanted = bool(state.general.enabled and state.general.realtime_auto_exposure)
        self._sync(force=True)

    def update(self, dt, t):
        self._sync()

    def _sync(self, force=False):
        mode = current_render_mode()
        if mode == self._mode and not force:
            return
        self._mode = mode
        if self._wanted and mode is not None and mode not in PATH_TRACED_MODES:
            self._enable(mode)
        else:
            self._restore()

    def _enable(self, mode):
        settings = self._settings()
        current = settings.get(HISTOGRAM_SETTING)
        if current is None:
            return
        if self._original is None:
            self._original = bool(current)
        if not current:
            settings.set(HISTOGRAM_SETTING, True)
            log.info("weather_fx: auto exposure on for real-time mode %r", mode)

    def _restore(self):
        if self._original is None:
            return
        try:
            self._settings().set(HISTOGRAM_SETTING, self._original)
        except Exception:
            log.exception("weather_fx: could not restore %s", HISTOGRAM_SETTING)
        self._original = None

    def detach(self):
        self._restore()

    def stats(self):
        return {"active": self._original is not None, "render_mode": self._mode}
