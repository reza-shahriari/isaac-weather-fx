import json

import pytest

from weather_fx.core import WeatherState, build_preset, list_presets, register_preset


def test_roundtrip_json():
    state = WeatherState().with_updates("fog", enabled=True, visibility_m=123.0, color=[0.1, 0.2, 0.3])
    data = json.loads(json.dumps(state.to_dict()))
    again = WeatherState.from_dict(data)
    assert again == state
    assert again.fog.color == (0.1, 0.2, 0.3)


def test_clamping_and_types():
    state = WeatherState().with_updates("rain", rate_mm_h=10_000, max_particles="12.6", enabled=1)
    assert state.rain.rate_mm_h == 150.0
    assert state.rain.max_particles == 13
    assert state.rain.enabled is True


def test_strict_unknown_key_raises():
    with pytest.raises(KeyError):
        WeatherState().with_updates("fog", visiblity_m=10)
    with pytest.raises(KeyError):
        WeatherState().with_updates("hail", enabled=True)


def test_lenient_from_dict_ignores_unknown():
    state = WeatherState.from_dict({"fog": {"nope": 1}, "future_section": {}})
    assert state == WeatherState()


def test_bad_choice_raises():
    with pytest.raises(ValueError):
        WeatherState().with_updates("general", time_source="sometimes")


def test_cross_validation():
    state = WeatherState().with_updates("rain", drop_min_diameter_mm=3.0, drop_max_diameter_mm=1.0)
    assert state.rain.drop_max_diameter_mm >= state.rain.drop_min_diameter_mm


def test_diff():
    a = WeatherState()
    b = a.with_updates("snow", enabled=True)
    assert a.diff(b) == {"snow"}


def test_all_presets_build():
    for name in list_presets():
        build_preset(name)


def test_register_preset_validates():
    register_preset("test_mist", {"fog": {"enabled": True, "visibility_m": 900}})
    assert build_preset("test_mist").fog.visibility_m == 900
    with pytest.raises(KeyError):
        register_preset("broken", {"fog": {"wrong": 1}})


def test_inherited_defaults_differ_per_section():
    s = WeatherState()
    assert s.rain.opacity != s.snow.opacity
