"""Public Python API. The UI panel uses exactly the same controller.

Example (Script Editor or standalone script)::

    from weather_fx import api as weather

    wx = weather.get_controller()
    wx.apply_preset("heavy_rain")
    wx.set_fog(visibility_m=150)
    wx.set_wind(speed_mps=6, direction_deg=45)
"""
from __future__ import annotations

import json
from typing import Callable, Optional

from .backends.viewport import ViewportBackend
from .core.presets import list_presets as _list_presets
from .core.presets import register_preset as _register_preset
from .core.state import SECTION_TYPES, WeatherState
from .runtime.manager import WeatherManager

_controller: Optional["WeatherController"] = None


def get_controller(auto_update: bool = True) -> "WeatherController":
    """Return the shared controller, creating it on first use."""
    global _controller
    if _controller is None:
        _controller = WeatherController(auto_update=auto_update)
    return _controller


def shutdown_controller() -> None:
    global _controller
    if _controller is not None:
        _controller.shutdown()
        _controller = None


class WeatherController:
    SECTIONS = tuple(SECTION_TYPES)

    def __init__(self, auto_update: bool = True):
        self._manager = WeatherManager()
        self._manager.register_backend(ViewportBackend())
        if auto_update:
            self._manager.start_auto_update()

    # ------------------------------------------------------------ state
    @property
    def state(self) -> WeatherState:
        """A copy of the current state."""
        return self._manager.get_state()

    def configure(self, **sections) -> None:
        """Change several sections in one update: ``configure(fog={...}, rain={...})``."""
        self._manager.update_sections(**sections)

    def set_state(self, state: WeatherState) -> None:
        self._manager.set_state(state)

    def set_general(self, **values) -> None:
        self.configure(general=values)

    def set_fog(self, **values) -> None:
        self.configure(fog=values)

    def set_wind(self, **values) -> None:
        self.configure(wind=values)

    def set_rain(self, **values) -> None:
        self.configure(rain=values)

    def set_snow(self, **values) -> None:
        self.configure(snow=values)

    def set_lighting(self, **values) -> None:
        self.configure(lighting=values)

    def enable(self, enabled: bool = True) -> None:
        self.set_general(enabled=enabled)

    def disable(self) -> None:
        self.enable(False)

    # ------------------------------------------------------------ presets and files
    def apply_preset(self, name: str) -> None:
        self._manager.apply_preset(name)

    def clear(self) -> None:
        self.apply_preset("clear")

    @staticmethod
    def list_presets():
        return _list_presets()

    @staticmethod
    def register_preset(name: str, overrides: dict) -> None:
        _register_preset(name, overrides)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._manager.set_state(WeatherState.from_dict(data))

    # ------------------------------------------------------------ time and anchoring
    def step(self, dt: float) -> None:
        """Advance the simulation manually (use with general.time_source='manual')."""
        self._manager.step(dt)

    def follow(self, prim_path: str) -> None:
        """Make the particle volume follow a prim, e.g. a robot camera. '' = viewport camera."""
        self.set_general(follow_prim=prim_path)

    def set_anchor_provider(self, fn: Optional[Callable[[], object]]) -> None:
        """Custom anchor, e.g. ``lambda: camera.get_world_pose()[0]``. Overrides follow_prim."""
        self._manager.context.anchor_provider = fn

    # ------------------------------------------------------------ events, backends, info
    def on_change(self, fn: Callable) -> int:
        """``fn(state, changed_sections)`` is called after every change. Returns a token."""
        return self._manager.state_changed.connect(fn)

    def remove_on_change(self, token: int) -> None:
        self._manager.state_changed.disconnect(token)

    def register_backend(self, backend) -> None:
        self._manager.register_backend(backend)

    def unregister_backend(self, name: str) -> None:
        self._manager.unregister_backend(name)

    def stats(self) -> dict:
        return self._manager.stats()

    @property
    def manager(self) -> WeatherManager:
        return self._manager

    def shutdown(self) -> None:
        self._manager.shutdown()
