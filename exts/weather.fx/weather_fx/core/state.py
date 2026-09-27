"""Weather state as plain dataclasses.

Every field carries UI metadata (label, range, unit, tooltip). The UI panel,
JSON presets and the Python API are all generated from these classes, so
adding a parameter here is enough to expose it everywhere.
"""
from __future__ import annotations

import copy
import logging
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Tuple

log = logging.getLogger("weather_fx")

SCHEMA_VERSION = 2
Vec3 = Tuple[float, float, float]


def param(default, label, *, min=None, max=None, unit="", tooltip="", widget=None,
          log_scale=False, choices=None, advanced=False):
    """Declare a parameter with the metadata the UI needs."""
    meta = dict(label=label, min=min, max=max, unit=unit, tooltip=tooltip, widget=widget,
                log_scale=log_scale, choices=choices, advanced=advanced)
    return field(default=default, metadata=meta)


# --------------------------------------------------------------------------- sections

@dataclass
class GeneralParams:
    enabled: bool = param(True, "Weather enabled", tooltip="Master switch for every effect.")
    follow_prim: str = param("", "Follow prim",
                             tooltip="Prim the particle volume follows (e.g. a robot camera). "
                                     "Empty = active viewport camera.")
    time_source: str = param("wall", "Time source", choices=("wall", "manual"),
                             tooltip="wall: animate on every app update. "
                                     "manual: only when step(dt) is called (deterministic data generation).")
    time_scale: float = param(1.0, "Time scale", min=0.0, max=4.0)
    seed: int = param(0, "Random seed", min=0, max=1_000_000, advanced=True)


@dataclass
class FogParams:
    enabled: bool = param(False, "Enabled")
    visibility_m: float = param(500.0, "Visibility", min=5.0, max=20000.0, unit="m", log_scale=True,
                                tooltip="Meteorological optical range: distance where contrast falls to 5%.")
    color: Vec3 = param((0.72, 0.74, 0.78), "Color", min=0.0, max=1.0, widget="color")
    color_intensity: float = param(1.0, "Color intensity", min=0.0, max=5.0)
    start_distance_m: float = param(0.0, "Start distance", min=0.0, max=2000.0, unit="m")
    end_distance_m: float = param(20000.0, "End distance", min=1.0, max=100000.0, unit="m", log_scale=True,
                                  advanced=True)
    height_fog: bool = param(False, "Height fog")
    base_height_m: float = param(0.0, "Base height", min=-100.0, max=500.0, unit="m")
    height_density: float = param(1.0, "Height density", min=0.0, max=10.0)
    height_falloff: float = param(0.5, "Height falloff", min=0.0, max=10.0)
    density_calibration: float = param(1.0, "Density calibration", min=0.001, max=1000.0, log_scale=True,
                                       advanced=True,
                                       tooltip="Maps physical extinction (1/m) to the renderer's fog density. "
                                               "Calibrate once per RTX mode (see README).")


@dataclass
class WindParams:
    speed_mps: float = param(0.0, "Speed", min=0.0, max=40.0, unit="m/s")
    direction_deg: float = param(0.0, "Direction", min=-180.0, max=180.0, unit="deg",
                                 tooltip="Direction the wind blows toward, in the horizontal plane, "
                                         "measured from +X (toward +Y on Z-up stages, +Z on Y-up stages).")
    vertical_mps: float = param(0.0, "Vertical", min=-5.0, max=5.0, unit="m/s", advanced=True)
    gust_strength: float = param(0.0, "Gust strength", min=0.0, max=1.0,
                                 tooltip="Fraction of the wind speed added as slow gusts.")
    gust_period_s: float = param(6.0, "Gust period", min=0.5, max=60.0, unit="s")


@dataclass
class PrecipitationParams:
    enabled: bool = param(False, "Enabled")
    volume_radius_m: float = param(12.0, "Volume half-width", min=2.0, max=100.0, unit="m",
                                   tooltip="Half size of the particle box that follows the camera.")
    volume_height_m: float = param(10.0, "Volume height", min=2.0, max=100.0, unit="m")
    max_particles: int = param(30000, "Max particles", min=0, max=200000, advanced=True)
    color: Vec3 = param((0.80, 0.85, 0.90), "Color", min=0.0, max=1.0, widget="color")
    opacity: float = param(0.35, "Opacity", min=0.0, max=1.0,
                           tooltip="Ignored by RTX Real-Time unless partial opacity is enabled in the renderer.")
    self_illumination: float = param(0.5, "Self-illumination", min=0.0, max=4.0,
                                     tooltip="Emission as a fraction of the color. Stands in for the sky light "
                                             "drops and flakes scatter toward the camera.")
    near_clearance_m: float = param(1.0, "Near clearance", min=0.0, max=5.0, unit="m",
                                    tooltip="Hide particles closer than this to the followed camera. "
                                            "A real lens defocuses them; rendered sharp they fill the frame.")
    material: str = param("preview", "Material", choices=("preview", "glass"), advanced=True,
                          tooltip="preview: UsdPreviewSurface (fast). glass: OmniGlass MDL.")
    cast_shadows: bool = param(False, "Cast shadows", advanced=True)


@dataclass
class RainParams(PrecipitationParams):
    volume_radius_m: float = param(8.0, "Volume half-width", min=2.0, max=100.0, unit="m",
                                   tooltip="Half size of the particle box that follows the camera. "
                                           "Drops beyond ~10 m are thinner than a pixel, so a small box "
                                           "puts the particle budget where it shows.")
    rate_mm_h: float = param(10.0, "Rain rate", min=0.1, max=150.0, unit="mm/h", log_scale=True)
    density_scale: float = param(4e-3, "Density scale", min=1e-5, max=1.0, log_scale=True,
                                 tooltip="Physical drop counts are millions per scene; this thins them "
                                         "to a renderable number. Visual control, not physics.")
    drop_min_diameter_mm: float = param(0.5, "Min drop diameter", min=0.1, max=3.0, unit="mm", advanced=True)
    drop_max_diameter_mm: float = param(6.0, "Max drop diameter", min=1.0, max=8.0, unit="mm", advanced=True)
    streak_exposure_s: float = param(1.0 / 60.0, "Streak exposure", min=0.0005, max=0.1, unit="s",
                                     log_scale=True,
                                     tooltip="Camera exposure time. Streak length = drop speed x exposure.")
    streak_width_scale: float = param(2.0, "Streak width scale", min=0.1, max=10.0)


@dataclass
class SnowParams(PrecipitationParams):
    color: Vec3 = param((0.95, 0.96, 1.0), "Color", min=0.0, max=1.0, widget="color")
    opacity: float = param(0.9, "Opacity", min=0.0, max=1.0)
    self_illumination: float = param(0.3, "Self-illumination", min=0.0, max=4.0)
    number_density_m3: float = param(2.0, "Flake density", min=0.01, max=200.0, unit="1/m^3", log_scale=True)
    flake_min_diameter_mm: float = param(1.0, "Min flake diameter", min=0.2, max=10.0, unit="mm", advanced=True)
    flake_max_diameter_mm: float = param(6.0, "Max flake diameter", min=1.0, max=30.0, unit="mm", advanced=True)
    fall_speed_mps: float = param(1.0, "Fall speed", min=0.1, max=4.0, unit="m/s")
    fall_speed_jitter: float = param(0.3, "Fall speed jitter", min=0.0, max=1.0)
    sway_amplitude_m: float = param(0.3, "Sway amplitude", min=0.0, max=2.0, unit="m")
    sway_frequency_hz: float = param(0.5, "Sway frequency", min=0.0, max=3.0, unit="Hz")


@dataclass
class SkyParams:
    """Where on earth, when, and what the air is like. The sun and moon follow from these."""

    enabled: bool = param(True, "Sky enabled",
                          tooltip="Author the sky dome, the sun and the moon. Off leaves whatever "
                                  "lighting the stage already has.")
    latitude_deg: float = param(48.1, "Latitude", min=-90.0, max=90.0, unit="deg")
    longitude_deg: float = param(11.6, "Longitude", min=-180.0, max=180.0, unit="deg",
                                 tooltip="Positive east.")
    date_utc: str = param("2024-06-21", "Date (UTC)",
                          tooltip="YYYY-MM-DD. With the hour below it fixes the sun and the moon; "
                                  "nothing else in the sky is authored by hand.")
    hour_utc: float = param(10.0, "Hour (UTC)", min=0.0, max=24.0, unit="h",
                            tooltip="Time of day, in UTC. Scrub this and the sun, the moon, the "
                                    "sky colour and the shadows all move together.")
    advance_clock: bool = param(False, "Advance the clock",
                                tooltip="Let time pass as the app ticks, scaled by general.time_scale. "
                                        "Off holds the sky at the hour above.")
    turbidity: float = param(2.8, "Turbidity", min=1.8, max=10.0,
                             tooltip="Linke turbidity: 2 is a very clear day, 6 is hazy, 10 is "
                                     "industrial murk. Drives the sky's colour and its brightness.")
    ground_albedo: Vec3 = param((0.16, 0.17, 0.12), "Ground albedo", min=0.0, max=1.0,
                                widget="color", advanced=True)
    exposure_scale: float = param(1.0, "Exposure", min=0.05, max=20.0, log_scale=True,
                                  tooltip="Multiplies the dome and both lights together, so the "
                                          "relative brightness of sky, sun and moon is preserved.")
    hide_scene_lights: bool = param(True, "Hide scene lights",
                                    tooltip="While the sky is on, hide every light outside "
                                            "/WeatherFX (the stage's default light and dome), so "
                                            "there is one sun. Session layer only; restored when "
                                            "the sky is turned off.")
    sun_enabled: bool = param(True, "Sun light")
    moon_enabled: bool = param(True, "Moon light",
                               tooltip="The only thing that lights an outdoor night scene.")
    star_intensity: float = param(1.0, "Starlight", min=0.0, max=10.0, advanced=True,
                                  tooltip="Scales the moonless night floor. Real starlight is "
                                          "about 0.002 lux, a hundredth of a full moon.")
    dome_resolution: int = param(1024, "Dome resolution", min=128, max=4096, advanced=True,
                                 tooltip="Rows in the latitude-longitude environment map. A "
                                         "marched cloud crosses from clear to opaque inside one "
                                         "texel, so too few rows draw a staircase along every edge.")


@dataclass
class CloudParams:
    """The cloud field. One parameterisation, read by every band that looks up."""

    enabled: bool = param(False, "Enabled")
    cover: float = param(0.35, "Sky cover", min=0.0, max=1.0,
                         tooltip="Fraction of the sky the cloud hides. This is solved for, not "
                                 "approximated: ask for 0.45 and the field measures 0.45.")
    genus: str = param("cumulus", "Cloud type",
                       choices=("cumulus", "congestus", "stratocumulus", "stratus", "cirrus"),
                       tooltip="Sets the vertical shape, the thickness and the optical depth. "
                               "A stratus is a sheet; a cumulus has a flat base and a "
                               "cauliflower top; cirrus is thin and streaked along the wind.")
    base_m: float = param(0.0, "Base height", min=0.0, max=12000.0, unit="m",
                          tooltip="0 computes it from the temperature and dew point below, which "
                                  "is why a whole field of cumulus has its bases on one level.")
    temperature_c: float = param(22.0, "Surface temperature", min=-40.0, max=50.0, unit="degC",
                                 advanced=True)
    dewpoint_c: float = param(12.0, "Surface dew point", min=-50.0, max=40.0, unit="degC",
                              advanced=True,
                              tooltip="The spread against the temperature sets the cloud base: "
                                      "125 m per kelvin.")
    thickness_m: float = param(0.0, "Thickness", min=0.0, max=14000.0, unit="m", advanced=True,
                               tooltip="0 uses the genus default.")
    optical_depth: float = param(0.0, "Optical depth", min=0.0, max=120.0, advanced=True,
                                 tooltip="Visible optical depth of the median cloudy column. "
                                         "0 uses the genus default.")
    feature_m: float = param(400.0, "Feature size", min=80.0, max=4000.0, unit="m", advanced=True,
                             tooltip="Diameter of the smallest structure the field carries. 400 m "
                                     "is the low end of the observed fair-weather cumulus mode.")
    erosion_scale: float = param(1.0, "Erosion scale", min=0.0, max=5.0, advanced=True,
                                 tooltip="Scales the high-frequency erosion. 0 is smooth blobs, >1 is wispier.")
    beta_scale: float = param(1.0, "Smoothness (beta scale)", min=0.5, max=2.0, advanced=True,
                              tooltip="Scales the fractal noise spectrum. Higher is smoother, lower adds overall high-frequency detail.")
    cells: int = param(256, "Grid cells", min=32, max=512, advanced=True)
    levels: int = param(48, "Grid levels", min=8, max=128, advanced=True)
    cell_m: float = param(60.0, "Cell size", min=10.0, max=400.0, unit="m", advanced=True,
                          tooltip="Horizontal grid spacing. cells x cell_m is the tile width, and "
                                  "the field tiles exactly, so there is no edge to reach.")
    seed: int = param(0, "Cloud seed", min=0, max=1_000_000, advanced=True)
    march_steps: int = param(
        64, "March steps", min=16, max=512, advanced=True,
        tooltip="Samples per ray when the dome is baked. 64 is right for a viewport; a still "
                "someone will look at closely wants more, because the step pattern is visible "
                "on a backlit cloud edge before anything else is.",
    )


@dataclass
class LightingParams:
    enabled: bool = param(False, "Enabled", tooltip="Scale scene light intensities for an overcast look.")
    light_scale: float = param(1.0, "Light scale", min=0.0, max=2.0)
    include_dome: bool = param(True, "Include dome lights")


SECTION_TYPES: Dict[str, type] = {
    "general": GeneralParams,
    "sky": SkyParams,
    "clouds": CloudParams,
    "fog": FogParams,
    "wind": WindParams,
    "rain": RainParams,
    "snow": SnowParams,
    "lighting": LightingParams,
}


# --------------------------------------------------------------------------- coercion

def _clamp(value, meta):
    lo, hi = meta.get("min"), meta.get("max")
    if lo is not None:
        value = type(value)(max(lo, value))
    if hi is not None:
        value = type(value)(min(hi, value))
    return value


def coerce_value(f, value):
    """Convert ``value`` to the type of field ``f`` and clamp it to the declared range."""
    meta, default = f.metadata, f.default
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        return _clamp(int(round(float(value))), meta)
    if isinstance(default, float):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{f.name} must be finite")
        return _clamp(value, meta)
    if isinstance(default, tuple):
        values = tuple(float(v) for v in value)
        if len(values) != len(default):
            raise ValueError(f"{f.name} needs {len(default)} values")
        return tuple(_clamp(v, meta) for v in values)
    if isinstance(default, str):
        value = str(value)
        choices = meta.get("choices")
        if choices and value not in choices:
            raise ValueError(f"{f.name} must be one of {choices}, got {value!r}")
        return value
    return value


# --------------------------------------------------------------------------- state

@dataclass
class WeatherState:
    general: GeneralParams = field(default_factory=GeneralParams)
    sky: SkyParams = field(default_factory=SkyParams)
    clouds: CloudParams = field(default_factory=CloudParams)
    fog: FogParams = field(default_factory=FogParams)
    wind: WindParams = field(default_factory=WindParams)
    rain: RainParams = field(default_factory=RainParams)
    snow: SnowParams = field(default_factory=SnowParams)
    lighting: LightingParams = field(default_factory=LightingParams)

    def copy(self) -> "WeatherState":
        return copy.deepcopy(self)

    def section(self, name: str):
        if name not in SECTION_TYPES:
            raise KeyError(f"Unknown weather section {name!r}. Valid: {list(SECTION_TYPES)}")
        return getattr(self, name)

    def with_updates(self, section: str, strict: bool = True, **values: Any) -> "WeatherState":
        """Return a new state with ``values`` applied to ``section`` (typed and clamped)."""
        new = self.copy()
        target = new.section(section)
        known = {f.name: f for f in fields(target)}
        for key, value in values.items():
            if key not in known:
                if strict:
                    raise KeyError(f"Unknown parameter {section}.{key}. Valid: {sorted(known)}")
                log.warning("weather_fx: ignoring unknown parameter %s.%s", section, key)
                continue
            setattr(target, key, coerce_value(known[key], value))
        _cross_validate(new)
        return new

    def to_dict(self) -> Dict[str, Any]:
        data = {name: asdict(getattr(self, name)) for name in SECTION_TYPES}
        data["schema_version"] = SCHEMA_VERSION
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base: "WeatherState" = None, strict: bool = False) -> "WeatherState":
        state = (base or cls()).copy()
        for name, values in data.items():
            if name == "schema_version":
                continue
            if name not in SECTION_TYPES:
                if strict:
                    raise KeyError(f"Unknown weather section {name!r}")
                log.warning("weather_fx: ignoring unknown section %r", name)
                continue
            state = state.with_updates(name, strict=strict, **values)
        return state

    def diff(self, other: "WeatherState"):
        """Names of sections whose values differ between the two states."""
        return {name for name in SECTION_TYPES
                if asdict(getattr(self, name)) != asdict(getattr(other, name))}


def _cross_validate(state: WeatherState) -> None:
    rain, snow = state.rain, state.snow
    if rain.drop_min_diameter_mm > rain.drop_max_diameter_mm:
        rain.drop_max_diameter_mm = rain.drop_min_diameter_mm
    if snow.flake_min_diameter_mm > snow.flake_max_diameter_mm:
        snow.flake_max_diameter_mm = snow.flake_min_diameter_mm
    if state.fog.end_distance_m <= state.fog.start_distance_m:
        state.fog.end_distance_m = state.fog.start_distance_m + 1.0
    # A cloud base above its own top is not a cloud. Both are free parameters and a UI lets you
    # drag them past each other, so the pair is fixed here rather than refused.
    clouds = state.clouds
    if clouds.thickness_m > 0.0 and clouds.base_m > 0.0:
        ceiling = 20000.0 - clouds.thickness_m
        if clouds.base_m > ceiling:
            clouds.base_m = max(ceiling, 0.0)
    # The hour is a wall-clock hour and 24:00 is the next day's 00:00, not a 25th hour.
    if state.sky.hour_utc >= 24.0:
        state.sky.hour_utc = state.sky.hour_utc % 24.0
