"""Auto exposure, the same in RTX Real-Time and in the path tracer.

On Kit 110 RTX Real-Time (``RealTimePathTracing``) exposes with a fixed camera exposure that is
orders of magnitude too dark for daylight. A plain stage with a 3000 lux sun and a 1000 nit dome
renders black there while the path tracer shows it fine (``tools/probe_realtime_renderer.py``).
The histogram auto exposure brings the frame back. It is turned on in the path tracer as well,
because one sky exposed two ways is two pictures: with the sun in frame Real-Time came out dark
and the path tracer washed out (measured again with both on: the frames' means agree within 3
of 255). Its white point is lowered from RTX's 10 to :data:`WHITE_SCALE`, which is what puts a
sunlit cloud near white and the sky at a daylight blue, as a camera metering a sky does.

Both settings are put back when ``general.realtime_auto_exposure`` is turned off or the weather
detaches.
"""
from __future__ import annotations

import logging

from ..base import Effect
from .render_mode import PATH_TRACED_MODES, current_render_mode

log = logging.getLogger("weather_fx")

HISTOGRAM_SETTING = "/rtx/post/histogram/enabled"
WHITE_SCALE_SETTING = "/rtx/post/histogram/whiteScale"
#: RTX's default is 10. At 7 a sunlit cloud keeps its shading below white and the sky is a deep daylight blue.
WHITE_SCALE = 7.0


class RealTimeExposureEffect(Effect):
    name = "realtime_exposure"

    def __init__(self):
        self._wanted = False
        self._original = None  # the setting's value before we changed it; None = untouched
        self._original_white = None
        self._mode = None

    def _settings(self):
        import carb.settings

        return carb.settings.get_settings()

    def apply_state(self, state, changed):
        self._wanted = bool(state.general.enabled and state.general.realtime_auto_exposure)
        self._sync(force=True)

    def update(self, dt, t):
        self._sync()
        # Kit puts the white point back to its default when the render mode or the stage
        # changes, so it is held, not set once.
        if self._original is not None:
            settings = self._settings()
            white = settings.get(WHITE_SCALE_SETTING)
            if white is None or abs(float(white) - WHITE_SCALE) > 1e-6:
                if self._original_white is None:
                    self._original_white = 10.0 if white is None else float(white)
                settings.set(WHITE_SCALE_SETTING, WHITE_SCALE)

    def _sync(self, force=False):
        mode = current_render_mode()
        if mode == self._mode and not force:
            return
        self._mode = mode
        if self._wanted and mode is not None:
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
            log.info("weather_fx: auto exposure on for render mode %r", mode)
        white = settings.get(WHITE_SCALE_SETTING)
        if white is not None and self._original_white is None:
            self._original_white = float(white)
            settings.set(WHITE_SCALE_SETTING, WHITE_SCALE)

    def _restore(self):
        if self._original is None:
            return
        try:
            self._settings().set(HISTOGRAM_SETTING, self._original)
        except Exception:
            log.exception("weather_fx: could not restore %s", HISTOGRAM_SETTING)
        self._original = None
        if self._original_white is not None:
            try:
                self._settings().set(WHITE_SCALE_SETTING, self._original_white)
            except Exception:
                log.exception("weather_fx: could not restore %s", WHITE_SCALE_SETTING)
            self._original_white = None

    def detach(self):
        self._restore()

    def stats(self):
        return {"active": self._original is not None, "render_mode": self._mode}
