"""Calls a function on every Kit app update, across Kit versions."""
from __future__ import annotations

import logging
import time

log = logging.getLogger("weather_fx")


class UpdateTicker:
    MAX_DT = 0.1  # avoid huge jumps after a stall

    def __init__(self, callback):
        self._callback = callback
        self._sub = None
        self._last = None

    @property
    def running(self) -> bool:
        return self._sub is not None

    def start(self) -> None:
        if self._sub is not None:
            return
        self._last = time.perf_counter()
        try:  # Kit 107+
            import carb.eventdispatcher
            import omni.kit.app

            self._sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
                observer_name="weather_fx.update",
                event_name=omni.kit.app.GLOBAL_EVENT_UPDATE,
                on_event=self._on_update,
            )
        except (ImportError, AttributeError):  # older Kit
            import omni.kit.app

            self._sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
                self._on_update, name="weather_fx.update"
            )

    def stop(self) -> None:
        sub, self._sub = self._sub, None
        if sub is not None:
            for name in ("reset", "unsubscribe"):
                fn = getattr(sub, name, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
                    break

    def _on_update(self, *_):
        now = time.perf_counter()
        dt = min(now - self._last, self.MAX_DT)
        self._last = now
        try:
            self._callback(dt)
        except Exception:
            log.exception("weather_fx: update failed")
