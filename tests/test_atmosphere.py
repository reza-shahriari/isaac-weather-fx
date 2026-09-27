import math

import numpy as np
import pytest

from weather_fx.core import atmosphere as A
from weather_fx.core.sky import conditions_from_state, environment_map, latlong_directions
from weather_fx.core.state import WeatherState

LUMA = np.array([0.2126, 0.7152, 0.0722])


def _radiance(view, elevation_deg, azimuth_deg):
    value = view.radiance(np.array([math.radians(elevation_deg)]),
                          np.array([math.radians(azimuth_deg)]))[0]
    return value * A.SOLAR_ILLUMINANCE_LUX


def test_a_clear_noon_zenith_is_blue_and_in_the_measured_range():
    view = A.sky_view(A.atmosphere_for(2.8), 60.0)
    zenith = _radiance(view, 90.0, 0.0)
    # Measured clear-sky zenith luminance at a high sun: a few thousand cd/m2.
    assert 2_000.0 < float(zenith @ LUMA) < 9_000.0
    assert zenith[2] > zenith[1] > zenith[0]
    # Diffuse illuminance on the ground under a clear high sun: 10-25 klux.
    assert 10_000.0 < float(view.irradiance @ LUMA) * A.SOLAR_ILLUMINANCE_LUX < 25_000.0


def test_a_low_sun_reddens_the_horizon_beneath_it():
    view = A.sky_view(A.atmosphere_for(3.0), 3.0)
    toward = _radiance(view, 2.0, 0.0)
    assert toward[0] > 1.5 * toward[2]


def test_the_upper_air_is_still_lit_after_sunset_at_the_ground():
    """The planet is round: at a -3 degree sun the zenith is dim, not black."""
    view = A.sky_view(A.atmosphere_for(2.8), -3.0)
    zenith = float(_radiance(view, 90.0, 0.0) @ LUMA)
    assert 1.0 < zenith < 200.0


def test_turbidity_whitens_the_horizon_and_shortens_the_visibility():
    clear = A.sky_view(A.atmosphere_for(2.0), 40.0).horizon_colour()
    hazy = A.sky_view(A.atmosphere_for(6.0), 40.0).horizon_colour()
    assert hazy[2] / hazy[0] < clear[2] / clear[0]
    visibilities = [A.haze_visibility_m(t) for t in (2.0, 3.0, 6.0, 10.0)]
    assert visibilities == sorted(visibilities, reverse=True)
    assert 3_000.0 < visibilities[-1] and visibilities[0] < 80_000.0


def test_the_ground_melts_into_the_horizon_instead_of_meeting_it():
    """The complaint that started this: a hard line where the dome's ground meets its sky."""
    state = WeatherState().with_updates("sky", hour_utc=10.0)
    image = environment_map(conditions_from_state(state), height=256)
    directions = latlong_directions(256)
    elevation = np.degrees(np.arcsin(directions[..., 1]))
    luminance = image @ LUMA
    band = [float(np.median(luminance[(elevation > e - 0.4) & (elevation <= e + 0.4)]))
            for e in np.arange(-6.0, 6.0, 0.7)]
    steps = np.abs(np.diff(np.log(band)))
    assert steps.max() < 0.25  # no jump above ~28 % between neighbouring bands

    bare = conditions_from_state(state.with_updates("sky", horizon_blend_deg=0.0))
    image = environment_map(bare, height=256)
    luminance = image @ LUMA
    band = [float(np.median(luminance[(elevation > e - 0.4) & (elevation <= e + 0.4)]))
            for e in np.arange(-6.0, 6.0, 0.7)]
    assert np.abs(np.diff(np.log(band))).max() > steps.max()


def test_the_sky_is_finite_and_non_negative_through_the_whole_day():
    for hour in np.arange(0.0, 24.0, 1.5):
        state = WeatherState().with_updates("sky", hour_utc=float(hour))
        image = environment_map(conditions_from_state(state), height=64)
        assert np.all(np.isfinite(image)) and image.min() >= 0.0


@pytest.mark.parametrize("model", ["atmosphere", "preetham"])
def test_both_models_remain_selectable(model):
    state = WeatherState().with_updates("sky", model=model)
    assert conditions_from_state(state).model == model


# --- camera: white balance, sun colour, exposure --------------------------------------------------

from weather_fx.core.sky import (  # noqa: E402
    LUMA as SKY_LUMA,
    WHITE_POINT_SKY_WEIGHT,
    dome_exposure,
    scene_illuminant,
    sun_colour,
    white_balance_gains,
)


def _scene(hour, date="2024-06-21"):
    state = WeatherState().with_updates("sky", hour_utc=hour, date_utc=date,
                                        latitude_deg=48.14, longitude_deg=11.58)
    conditions = conditions_from_state(state, build_cloud=False)
    return conditions, environment_map(conditions, height=96)


def test_the_sun_warms_toward_the_horizon_and_is_white_overhead():
    blue_to_red = [float(sun_colour(el)[2] / sun_colour(el)[0]) for el in (2, 5, 10, 20, 60)]
    assert blue_to_red == sorted(blue_to_red)
    high = sun_colour(60.0)
    assert high.max() / high.min() < 1.05
    for el in (1.0, 10.0, 60.0):
        assert float(sun_colour(el) @ SKY_LUMA) == pytest.approx(1.0, rel=1e-6)


def test_white_balance_changes_colour_and_not_level():
    gains = white_balance_gains((1.0, 0.7, 0.4), 1.0)
    balanced = gains * np.array([1.0, 0.7, 0.4])
    assert balanced.max() / balanced.min() == pytest.approx(1.0, rel=1e-6)
    assert float(balanced @ SKY_LUMA) == pytest.approx(float(np.array([1.0, 0.7, 0.4]) @ SKY_LUMA))
    assert np.allclose(white_balance_gains((1.0, 0.7, 0.4), 0.0), 1.0)


def test_a_morning_cloud_is_near_white_but_the_last_minutes_before_sunset_stay_warm():
    """The complaint: clouds went fully orange early in the day, which no one ever sees."""
    def sunlit_cloud(hour):
        conditions, image = _scene(hour)
        gains = white_balance_gains(
            scene_illuminant(image, conditions, sky_weight=WHITE_POINT_SKY_WEIGHT), 0.9)
        colour = sun_colour(conditions.sun.elevation_deg) * gains
        return colour / colour.max(), conditions.sun.elevation_deg

    morning, el = sunlit_cloud(5.3)
    assert 14.0 < el < 22.0 and morning[2] > 0.85
    sunset, el = sunlit_cloud(19.0)
    assert 0.0 < el < 4.0 and sunset[2] < 0.3


def test_a_moonlit_night_is_dark_but_readable_and_darker_than_the_day():
    """The other complaint: night went black. Exposure follows the light on the ground, so a
    full-moon landscape sits a few stops under noon, not seven."""
    def rendered_ground(hour, date="2024-06-21"):
        conditions, image = _scene(hour, date)
        return float(scene_illuminant(image, conditions) @ SKY_LUMA) * dome_exposure(image, conditions)

    noon = rendered_ground(12.97)
    moonlit = rendered_ground(22.0)
    moonless = rendered_ground(1.0, "2024-06-06")
    for night in (moonlit, moonless):
        stops = math.log2(noon / night)
        assert 2.5 < stops < 5.5, stops
    assert moonlit > moonless
