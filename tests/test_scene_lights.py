from weather_fx.backends.viewport.scene_lights import is_own_prim
from weather_fx.core.state import WeatherState


def test_own_prims_are_the_root_and_its_children_only():
    assert is_own_prim("/WeatherFX")
    assert is_own_prim("/WeatherFX/Sky/Sun")
    # A sibling whose name merely starts with the root is somebody else's light.
    assert not is_own_prim("/WeatherFXLights/Key")
    assert not is_own_prim("/Environment/defaultLight")


def test_scene_lights_are_hidden_by_default_and_can_be_kept():
    state = WeatherState()
    assert state.sky.hide_scene_lights is True
    kept = state.with_updates("sky", hide_scene_lights=False)
    assert kept.sky.hide_scene_lights is False


def test_distance_haze_is_opt_in():
    """RTX fog is real-time only and fogs the dome at infinity; it stays off until asked for."""
    assert WeatherState().sky.aerial_perspective is False


def test_realtime_auto_exposure_is_on_by_default():
    """RTX Real-Time on Kit 110 is black at its fixed exposure; the extension turns auto exposure on."""
    assert WeatherState().general.realtime_auto_exposure is True
