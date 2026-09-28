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

    def set_sky(self, **values) -> None:
        """Place, date, hour, turbidity, exposure: ``set_sky(hour_utc=17.5, turbidity=4)``."""
        self.configure(sky=values)

    def set_clouds(self, **values) -> None:
        """``set_clouds(enabled=True, cover=0.5, genus="cumulus")``."""
        self.configure(clouds=values)

    def set_time(self, hour_utc: float = None, date_utc: str = None) -> None:
        """Scrub the clock. The sun, the moon, the sky colour and the shadows move together."""
        values = {}
        if hour_utc is not None:
            values["hour_utc"] = float(hour_utc)
        if date_utc is not None:
            values["date_utc"] = str(date_utc)
        if values:
            self.configure(sky=values)

    def set_site(self, latitude_deg: float, longitude_deg: float) -> None:
        """Where on earth the scene is. Longitude is positive east."""
        self.configure(sky={"latitude_deg": latitude_deg, "longitude_deg": longitude_deg})

    # ------------------------------------------------------------ randomisation
    def randomize(self, seed: int = None, **kwargs) -> str:
        """Draw a coherent random weather and apply it. Returns the regime it chose.

        Safe to call at runtime, from a UI button or from a dataset loop: the ``general`` section
        is untouched, so a deterministic run stays on its manual clock and keeps its follow prim.

        ``kwargs`` are passed to :func:`weather_fx.core.random_weather.random_overrides` --
        ``regime=``, ``latitude_deg=``, ``night_fraction=`` and so on.
        """
        from .core.random_weather import random_overrides

        overrides = random_overrides(seed, **kwargs)
        self.configure(**{k: v for k, v in overrides.items() if v})
        return kwargs.get("regime", "drawn")

    def sky_conditions(self, build_cloud: bool = False):
        """The resolved sun, moon and cloud for the current state -- what the sky actually is.

        Every number in it is derived from the place and the clock, so this is the first thing to
        print when a scene is lit from the wrong quarter.
        """
        from .core.sky import conditions_from_state

        return conditions_from_state(self.state, build_cloud=build_cloud)

    def surface_weather(self, hours: float = 48.0, step_s: float = 1800.0, **kwargs):
        """The diurnal surface-meteorology series implied by the current state.

        Air temperature, humidity, wind, cloud, irradiance, visibility and precipitation on a
        regular grid, in SI units -- what a thermal model integrates, and what this extension
        hands to a sensor simulator so that one weather drives both the picture and the physics.
        The state's own clock is the anchor; see
        :func:`weather_fx.core.meteorology.diurnal_series`.
        """
        from .core.meteorology import diurnal_series

        return diurnal_series(self.state, hours=hours, step_s=step_s, **kwargs)

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
    def diagnose(self, show: bool = True) -> dict:
        """Everything that decides what the viewport shows, as a dict; printed unless ``show=False``.

        The renderer the extension detected and how the clouds are drawn, every prim under
        /WeatherFX with its visibility and light intensity, whether the dome's texture exists,
        which stage lights are hidden, and the live RTX fog settings. Paste it when a frame looks
        wrong.
        """
        from .runtime.diagnostics import diagnose, format_report

        report = diagnose(self._manager)
        if show:
            print(format_report(report))
        return report

    def cloud_drift_m(self, frame: str = "field"):
        """How far the wind has carried the clouds, metres.

        ``frame="field"`` gives it in the cloud field's own Y-up frame, which is what a sensor
        model marching ``CloudField`` should subtract from its sample positions so its cloud sits
        where the rendered one does. ``frame="stage"`` gives stage axes.
        """
        from .core.clouds import stage_to_field

        drift = self._manager.context.cloud_drift_m
        if frame == "stage":
            return drift.copy()
        return stage_to_field(drift, self._manager.context.up_axis())

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
