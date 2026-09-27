"""Owns the weather state and fans changes out to sensor backends."""
from __future__ import annotations

import logging
from typing import Dict

from ..core.clouds import drift_velocity_m_s
from ..core.events import Signal
from ..core.presets import build_preset
from ..core.state import SECTION_TYPES, WeatherState
from .context import WeatherContext
from .ticker import UpdateTicker

log = logging.getLogger("weather_fx")


class WeatherManager:
    def __init__(self):
        self._state = WeatherState()
        self._backends: Dict[str, object] = {}
        self.time = 0.0
        self.context = WeatherContext(self)
        self.state_changed = Signal()  # (state_copy, changed_sections)
        self._ticker = UpdateTicker(self._on_app_update)

    # ------------------------------------------------------------ state
    @property
    def state_ref(self) -> WeatherState:
        """Live state for backends. Do not mutate."""
        return self._state

    def get_state(self) -> WeatherState:
        return self._state.copy()

    def set_state(self, new_state: WeatherState, force: bool = False) -> None:
        changed = set(SECTION_TYPES) if force else self._state.diff(new_state)
        if not changed:
            return
        self._state = new_state.copy()
        for backend in self._backends.values():
            try:
                backend.apply_state(self._state, changed)
            except Exception:
                log.exception("weather_fx: backend %s failed to apply state", backend.name)
        self.state_changed.emit(self._state.copy(), changed)

    def update_sections(self, **sections) -> None:
        state = self._state
        for name, values in sections.items():
            state = state.with_updates(name, **values)
        self.set_state(state)

    def apply_preset(self, name: str) -> None:
        state = build_preset(name)
        state.general = self._state.general  # keep follow prim, time source, seed
        self.set_state(state)

    # ------------------------------------------------------------ backends
    def register_backend(self, backend) -> None:
        if backend.name in self._backends:
            raise ValueError(f"Backend {backend.name!r} already registered")
        backend.attach(self.context)
        self._backends[backend.name] = backend
        backend.apply_state(self._state, set(SECTION_TYPES))

    def unregister_backend(self, name: str) -> None:
        backend = self._backends.pop(name, None)
        if backend is not None:
            backend.detach()

    def get_backend(self, name: str):
        return self._backends.get(name)

    @property
    def backends(self):
        return dict(self._backends)

    # ------------------------------------------------------------ time
    def start_auto_update(self) -> None:
        self._ticker.start()

    def stop_auto_update(self) -> None:
        self._ticker.stop()

    def _on_app_update(self, dt: float) -> None:
        if self._state.general.time_source == "wall":
            self.step(dt)

    def step(self, dt: float) -> None:
        """Advance every backend by ``dt`` seconds (scaled by general.time_scale)."""
        if not self._state.general.enabled:
            return
        dt = dt * self._state.general.time_scale
        self.time += dt
        self._advance_cloud_drift(dt)
        for backend in self._backends.values():
            try:
                backend.update(dt, self.time)
            except Exception:
                log.exception("weather_fx: backend %s failed to update", backend.name)

    def _advance_cloud_drift(self, dt: float) -> None:
        clouds, wind = self._state.clouds, self._state.wind
        if not clouds.enabled or clouds.wind_factor <= 0.0 or wind.speed_mps <= 0.0:
            return
        try:
            up_axis = self.context.up_axis()
        except Exception:
            up_axis = 2
        self.context.cloud_drift_m = self.context.cloud_drift_m + drift_velocity_m_s(
            wind.speed_mps, wind.direction_deg, clouds.wind_factor, up_axis) * dt

    # ------------------------------------------------------------ misc
    def stats(self) -> dict:
        out = {}
        for name, backend in self._backends.items():
            try:
                out[name] = backend.stats()
            except Exception:
                out[name] = {"error": "stats failed"}
        return out

    def shutdown(self) -> None:
        self._ticker.stop()
        for name in list(self._backends):
            self.unregister_backend(name)
