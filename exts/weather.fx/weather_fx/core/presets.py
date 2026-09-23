"""Named weather presets. These are visual starting points, not calibrated weather."""
from __future__ import annotations

import copy
from typing import Any, Dict, List

from .state import WeatherState

_PRESETS: Dict[str, Dict[str, Any]] = {
    "clear": {},
    "haze": {"fog": {"enabled": True, "visibility_m": 5000.0}},
    "light_fog": {"fog": {"enabled": True, "visibility_m": 1000.0}},
    "dense_fog": {
        "fog": {"enabled": True, "visibility_m": 80.0},
        "lighting": {"enabled": True, "light_scale": 0.6},
    },
    "drizzle": {
        "rain": {"enabled": True, "rate_mm_h": 1.0},
        "fog": {"enabled": True, "visibility_m": 3000.0},
    },
    "moderate_rain": {
        "rain": {"enabled": True, "rate_mm_h": 5.0},
        "fog": {"enabled": True, "visibility_m": 2000.0},
        "lighting": {"enabled": True, "light_scale": 0.7},
    },
    "heavy_rain": {
        "rain": {"enabled": True, "rate_mm_h": 30.0},
        "wind": {"speed_mps": 2.0},
        "fog": {"enabled": True, "visibility_m": 600.0},
        "lighting": {"enabled": True, "light_scale": 0.5},
    },
    "storm": {
        "rain": {"enabled": True, "rate_mm_h": 80.0},
        "wind": {"speed_mps": 10.0, "direction_deg": 20.0, "gust_strength": 0.5, "gust_period_s": 4.0},
        "fog": {"enabled": True, "visibility_m": 300.0},
        "lighting": {"enabled": True, "light_scale": 0.35},
    },
    "light_snow": {
        "snow": {"enabled": True, "number_density_m3": 1.0},
        "fog": {"enabled": True, "visibility_m": 2000.0},
    },
    "blizzard": {
        "snow": {"enabled": True, "number_density_m3": 20.0, "fall_speed_mps": 1.5, "sway_amplitude_m": 0.8},
        "wind": {"speed_mps": 12.0, "direction_deg": 15.0, "gust_strength": 0.6, "gust_period_s": 3.0},
        "fog": {"enabled": True, "visibility_m": 150.0, "color": (0.85, 0.87, 0.90)},
        "lighting": {"enabled": True, "light_scale": 0.6},
    },
}


def list_presets() -> List[str]:
    return list(_PRESETS)


def register_preset(name: str, overrides: Dict[str, Any]) -> None:
    """Add or replace a preset. ``overrides`` uses the same layout as ``WeatherState.to_dict()``."""
    WeatherState.from_dict(overrides, strict=True)  # validate early
    _PRESETS[name] = copy.deepcopy(overrides)


def build_preset(name: str) -> WeatherState:
    if name not in _PRESETS:
        raise KeyError(f"Unknown preset {name!r}. Available: {list_presets()}")
    return WeatherState.from_dict(_PRESETS[name], strict=True)
