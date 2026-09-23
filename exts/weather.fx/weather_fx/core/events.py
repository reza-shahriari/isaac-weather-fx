"""Tiny observer used to keep the UI, scripts and backends in sync."""
import logging

log = logging.getLogger("weather_fx")


class Signal:
    def __init__(self):
        self._subscribers = {}
        self._next_token = 0

    def connect(self, fn):
        token = self._next_token
        self._next_token += 1
        self._subscribers[token] = fn
        return token

    def disconnect(self, token):
        self._subscribers.pop(token, None)

    def emit(self, *args, **kwargs):
        for fn in list(self._subscribers.values()):
            try:
                fn(*args, **kwargs)
            except Exception:  # a broken listener must not break the others
                log.exception("weather_fx: listener failed")
