"""The sky model and the randomiser.

The sky's job is to be continuous and to have the right *orders of magnitude*: a night sky is six
decades below a noon one, a moonlit sky is a hundred times brighter than a moonless one, and none
of it may step. A model that gets the shape right and the scale wrong looks fine in isolation and
ruins any camera that has an exposure.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from weather_fx.core.random_weather import REGIMES, random_overrides, random_sequence, random_state
from weather_fx.core.sky import (
    STARLIGHT_LUX,
    conditions_from_state,
    environment_map,
    latlong_directions,
    sky_radiance_rgb,
)
from weather_fx.core.state import WeatherState


def state(**sky) -> WeatherState:
    return WeatherState().with_updates("sky", **sky)


def hemisphere(n: int = 400, seed: int = 0) -> np.ndarray:
    """Random directions in the upper hemisphere."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    v[:, 1] = np.abs(v[:, 1])
    return v


# --- the map's layout is the renderer's -----------------------------------------------------


def test_the_latlong_map_covers_the_sphere_exactly_once() -> None:
    d = latlong_directions(32)
    assert d.shape == (32, 64, 3)
    assert np.allclose(np.linalg.norm(d, axis=-1), 1.0, atol=1e-12)
    # Half the texels above the horizon and half below, since the horizon falls on a row boundary.
    assert np.mean(d[..., 1] > 0.0) == pytest.approx(0.5, abs=0.02)


def test_an_odd_or_tiny_map_is_refused() -> None:
    with pytest.raises(ValueError, match="even number of rows"):
        latlong_directions(33)
    with pytest.raises(ValueError, match="at least 8"):
        latlong_directions(4)


def test_the_map_is_float32_and_never_half() -> None:
    """A half-float sky is the habit that ends up in a temperature path."""
    assert environment_map(conditions_from_state(state()), height=16).dtype == np.float32


# --- orders of magnitude ----------------------------------------------------------------------


def test_a_noon_sky_is_thousands_of_candela_and_a_night_sky_is_thousandths() -> None:
    """Six decades between them. This is the assertion that catches a model that is the right
    shape on an arbitrary scale -- which is most of them."""
    # December, not June: at 48 deg north midsummer has no astronomical night at all, and the
    # model is right to report a bright sky at 01:00 UTC on the solstice. The test premise was
    # wrong before this line, not the sky.
    noon = sky_radiance_rgb(hemisphere(), conditions_from_state(state(hour_utc=11.0)))
    midnight = state(date_utc="2024-12-21", hour_utc=1.0)
    assert conditions_from_state(midnight).sun.elevation_deg < -18.0
    night = sky_radiance_rgb(hemisphere(), conditions_from_state(midnight))
    noon_mean = float(noon.mean())
    night_mean = float(night.mean())
    assert 1e3 < noon_mean < 5e4, f"noon sky at {noon_mean:.3g} cd/m2"
    assert 1e-4 < night_mean < 1e-1, f"night sky at {night_mean:.3g} cd/m2"
    assert noon_mean / night_mean > 1e4


def test_a_moonlit_night_is_far_brighter_than_a_moonless_one() -> None:
    """The moon is the only thing lighting an outdoor night scene, and the difference between
    having it and not is the difference between a usable frame and a black one."""
    # December 2024: full moon on the 15th, new moon on the 1st. Same hour, same place, and a
    # date where 48 deg north actually gets dark -- midsummer there does not.
    full = conditions_from_state(state(date_utc="2024-12-15", hour_utc=23.5))
    new = conditions_from_state(state(date_utc="2024-12-01", hour_utc=23.5))
    assert full.moon.illuminated_fraction > 0.97
    assert new.moon.illuminated_fraction < 0.05
    lit = float(sky_radiance_rgb(hemisphere(), full).mean())
    dark = float(sky_radiance_rgb(hemisphere(), new).mean())
    assert lit > 5.0 * dark


def test_a_moonless_night_still_has_starlight_rather_than_being_black() -> None:
    """A black sky is not dark, it is empty: a real sensor sees airglow and stars, and a
    background of exactly zero is a number no camera ever records."""
    dark = conditions_from_state(state(date_utc="2024-12-01", hour_utc=23.5))
    values = sky_radiance_rgb(hemisphere(), dark)
    assert float(values.min()) > 0.0
    assert float(values.mean()) == pytest.approx(STARLIGHT_LUX / math.pi, rel=0.8)


def test_turbidity_makes_the_sky_brighter_and_less_blue() -> None:
    """Which is what haze is. If turbidity only scaled brightness, a hazy day would render as a
    dimmer clear one and the colour would give the game away."""
    clear = sky_radiance_rgb(hemisphere(), conditions_from_state(state(turbidity=2.0)))
    hazy = sky_radiance_rgb(hemisphere(), conditions_from_state(state(turbidity=8.0)))
    assert hazy.mean() > clear.mean()

    def blueness(rgb):
        return float(rgb[..., 2].mean() / max(rgb[..., 0].mean(), 1e-9))

    assert blueness(hazy) < blueness(clear)


# --- continuity, which is what a time-lapse needs -------------------------------------------


def test_the_sky_does_not_step_anywhere_through_sunset() -> None:
    """Sunset is where every sky model that switches between a day branch and a night branch
    shows its seam, and it is exactly where a time-lapse is most likely to be filmed."""
    directions = hemisphere(120)
    hours = np.linspace(15.0, 22.0, 120)
    means = np.array(
        [float(sky_radiance_rgb(directions, conditions_from_state(state(hour_utc=h))).mean())
         for h in hours]
    )
    assert means[0] > means[-1], "it must actually get darker"
    # No adjacent pair may differ by more than a factor of four: the sky falls fast at sunset but
    # it falls smoothly, and a switch would show up here as a single enormous ratio.
    ratios = means[:-1] / np.maximum(means[1:], 1e-12)
    assert float(ratios.max()) < 4.0, f"discontinuity at sunset, worst ratio {ratios.max():.1f}"


def test_the_sky_is_never_negative_and_never_nan() -> None:
    """The Perez distribution diverges at the horizon and the xyY conversion can go negative in
    the gamut corners. Either one bakes a ring of NaN into an environment map."""
    for hour in (0.0, 4.0, 5.9, 12.0, 18.1, 23.9):
        values = environment_map(conditions_from_state(state(hour_utc=hour)), height=32)
        assert np.all(np.isfinite(values)), f"non-finite sky at {hour} h"
        assert float(values.min()) >= 0.0


def test_the_ground_half_of_the_map_is_lit_rather_than_black() -> None:
    """The dome lights the scene with the whole map. A black lower hemisphere is a scene lit from
    above only, with no bounce, which reads as a studio rather than as outdoors."""
    values = environment_map(conditions_from_state(state(hour_utc=11.0)), height=64)
    d = latlong_directions(64)
    below = d[..., 1] < -0.2
    assert float(values[below].mean()) > 0.0
    # And dimmer than the sky, because it is a 16 % albedo lit by that sky.
    above = d[..., 1] > 0.2
    assert float(values[below].mean()) < float(values[above].mean())


# --- the cloud is the same object in both bands -----------------------------------------------


def test_cloud_darkens_the_sky_it_covers_and_the_gaps_stay_clear() -> None:
    clear = conditions_from_state(state(hour_utc=11.0))
    cloudy_state = state(hour_utc=11.0).with_updates(
        "clouds", enabled=True, cover=0.9, cells=64, levels=16, cell_m=120.0, seed=4
    )
    cloudy = conditions_from_state(cloudy_state)
    assert cloudy.cloud is not None
    directions = hemisphere(600, seed=3)
    before = sky_radiance_rgb(directions, clear)
    after = sky_radiance_rgb(directions, cloudy)
    # Somewhere is dimmer (cloud in front of sky) and somewhere is brighter (a sunlit cloud top).
    assert np.any(after.mean(axis=-1) < before.mean(axis=-1) * 0.9)
    assert np.any(after.mean(axis=-1) > before.mean(axis=-1) * 1.05)


def test_the_conditions_object_reports_what_it_resolved() -> None:
    conditions = conditions_from_state(state(hour_utc=10.0, date_utc="2024-06-21"))
    line = conditions.describe()
    assert "2024-06-21 10:00 UTC" in line
    assert "sun" in line and "moon" in line


def test_a_malformed_date_is_refused_with_the_format_in_the_message() -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        conditions_from_state(state(date_utc="21/06/2024"))


# --- the randomiser ------------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_weather_and_different_seeds_do_not() -> None:
    assert random_overrides(1234) == random_overrides(1234)
    assert random_overrides(1) != random_overrides(2)


def test_a_sequence_uses_child_streams_rather_than_consecutive_seeds() -> None:
    """Nearby integer seeds give correlated first draws in most generators, which is how a
    'random' dataset ends up with the same kind of day in its first frame every time."""
    batch = random_sequence(24, seed=5)
    regimes = {(o["rain"]["enabled"], o["snow"]["enabled"], o["clouds"].get("enabled")) for o in batch}
    assert len(batch) == 24
    assert len(regimes) > 2


@pytest.mark.parametrize("regime", sorted(REGIMES))
def test_every_regime_produces_a_state_that_loads(regime: str) -> None:
    result = random_state(3, regime=regime)
    assert isinstance(result, WeatherState)
    # And it round-trips through JSON, which is what a preset and a dataset manifest are.
    assert WeatherState.from_dict(result.to_dict()).to_dict() == result.to_dict()


def test_fog_comes_with_a_saturated_surface_layer_and_no_wind() -> None:
    """The coupling that makes it fog rather than a low-visibility setting: fog *is* a tiny
    dew-point spread, and it does not survive a gale."""
    for seed in range(12):
        o = random_overrides(seed, regime="fog")
        assert o["fog"]["enabled"] and o["fog"]["visibility_m"] < 1000.0
        assert o["wind"]["speed_mps"] < 2.5
        assert not o["clouds"]["enabled"], "convective cloud above fog has nothing driving it"


def test_rain_comes_with_deep_low_cloud_and_a_washed_out_aerosol() -> None:
    """Rain scavenges aerosol, so the air between the drops is clearer than a hazy dry day even
    though the visibility *through* the rain is worse. Both are true at once."""
    for seed in range(12):
        o = random_overrides(seed, regime="rain")
        assert o["rain"]["enabled"]
        assert o["clouds"]["cover"] > 0.85
        assert o["clouds"]["temperature_c"] - o["clouds"]["dewpoint_c"] < 3.5
        assert o["sky"]["turbidity"] < 3.5
        assert o["fog"]["visibility_m"] < 9001.0


def test_snow_implies_air_below_freezing() -> None:
    for seed in range(12):
        o = random_overrides(seed, regime="snow")
        assert o["snow"]["enabled"]
        assert o["clouds"]["temperature_c"] <= 0.5


def test_the_cloud_base_is_derived_from_the_spread_rather_than_drawn() -> None:
    """125 m per kelvin. Drawing the base independently is how a randomiser produces a 1500 m
    base under a two-degree spread, which cannot happen."""
    for seed in range(20):
        o = random_overrides(seed)
        if o["clouds"].get("enabled") and "temperature_c" in o["clouds"]:
            assert o["clouds"]["base_m"] == 0.0, "0 means 'compute it', which is the point"


def test_the_clock_is_drawn_in_local_time_so_night_means_night_where_the_scene_is() -> None:
    """A randomiser that draws the hour in UTC puts the sun overhead in half the scenes it
    labelled as dark. Checked against the resolved sun, not against the hour."""
    elevations = [
        conditions_from_state(random_state(seed), build_cloud=False).sun.elevation_deg
        for seed in range(160)
    ]
    dark = float(np.mean(np.asarray(elevations) < 0.0))
    assert 0.2 < dark < 0.55, f"{dark:.2f} of draws at night; asked for about 0.35"


def test_randomising_leaves_the_general_section_alone() -> None:
    """Changing the weather must not take a deterministic data run off its manual clock."""
    base = WeatherState().with_updates(
        "general", time_source="manual", seed=99, follow_prim="/World/Camera"
    )
    out = random_state(7, base=base)
    assert out.general.time_source == "manual"
    assert out.general.seed == 99
    assert out.general.follow_prim == "/World/Camera"


def test_a_named_regime_that_does_not_exist_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown regime"):
        random_overrides(0, regime="hurricane")


def test_a_fixed_site_and_year_are_honoured() -> None:
    o = random_overrides(11, latitude_deg=51.5, longitude_deg=-0.13, year=2023)
    assert o["sky"]["latitude_deg"] == pytest.approx(51.5)
    assert o["sky"]["longitude_deg"] == pytest.approx(-0.13)
    assert date.fromisoformat(o["sky"]["date_utc"]).year == 2023


# --- exposure ------------------------------------------------------------------------------------


def _sky_at(hour, *, cover=0.0, turbidity=2.6, date="2024-06-21", height=128, seed=17):
    from weather_fx.core.sky import conditions_from_state, environment_map
    from weather_fx.core.state import WeatherState

    state = WeatherState().with_updates(
        "sky", latitude_deg=48.14, longitude_deg=11.58, date_utc=date,
        hour_utc=hour, turbidity=turbidity,
    )
    state = state.with_updates(
        "clouds", enabled=cover > 0.0, cover=cover, genus="cumulus",
        temperature_c=22.0, dewpoint_c=12.0, seed=seed, cells=64, cell_m=140.0, levels=24,
    )
    conditions = conditions_from_state(state)
    image = environment_map(conditions, height=height).view(_Sky)
    image.conditions = conditions  # the meter needs to know where the sun and moon are
    return image


class _Sky(np.ndarray):
    """An environment map that remembers the conditions it was baked from."""


def test_a_clear_midday_sky_is_exposed_where_the_measurement_put_it():
    """The anchor: a clear midday is exposed at the 0.025 dome intensity the sweep on a demo stage
    found, whichever meter produced it, and its sky's median lands in a legible band."""
    from weather_fx.core.sky import dome_exposure, latlong_directions

    image = _sky_at(12.97, turbidity=2.1)
    exposure = dome_exposure(image, image.conditions)
    luminance = image[latlong_directions(image.shape[0])[..., 1] > 0.0].mean(axis=-1)
    assert exposure == pytest.approx(0.025, rel=0.05)
    assert 100.0 < float(np.median(luminance)) * exposure < 250.0


def test_a_sunset_does_not_clip_the_way_a_median_meter_made_it():
    """The defect this rule replaced. Metering on the median put a 4 degree sun's highlights 2.4x
    above where noon puts them, so the entire solar half of the frame rendered white."""
    from weather_fx.core.sky import dome_exposure, latlong_directions

    noon, sunset = _sky_at(12.97, cover=0.35), _sky_at(18.73, cover=0.35, turbidity=3.6)
    peaks = []
    for image in (noon, sunset):
        luminance = image[latlong_directions(image.shape[0])[..., 1] > 0.0].mean(axis=-1)
        peaks.append(float(np.percentile(luminance, 99.0)) * dome_exposure(image, image.conditions))
    assert peaks[1] < 1.15 * peaks[0]


def test_civil_twilight_does_not_run_the_exposure_to_its_clamp():
    """The other direction, and the uglier one: the median of the upper hemisphere at a -9 degree
    sun is zero to float precision, so a median meter divided by nothing and returned a white
    frame at the darkest moment of the day."""
    from weather_fx.core.sky import dome_exposure, latlong_directions

    image = _sky_at(20.5, cover=0.3, turbidity=3.0)
    luminance = image[latlong_directions(image.shape[0])[..., 1] > 0.0].mean(axis=-1)
    peak = float(np.percentile(luminance, 99.0))
    # The trap is real: the median is orders of magnitude below the highlights.
    assert float(np.median(luminance)) < 1e-2 * peak
    rendered = peak * dome_exposure(image, image.conditions)
    assert 50.0 < rendered < 450.0


def test_the_day_stays_brighter_than_the_night_by_a_legible_margin():
    """An adaptation, not a normalisation. Seven decades of physical luminance are compressed,
    but the order and a readable gap have to survive or the gallery is a set of grey squares."""
    from weather_fx.core.sky import dome_exposure, latlong_directions

    rendered = []
    for hour, cover, date in ((12.97, 0.35, "2024-06-21"), (18.73, 0.35, "2024-06-21"),
                              (20.5, 0.3, "2024-06-21"), (0.83, 0.25, "2024-07-23")):
        image = _sky_at(hour, cover=cover, date=date)
        luminance = image[latlong_directions(image.shape[0])[..., 1] > 0.0].mean(axis=-1)
        rendered.append(float(np.percentile(luminance, 99.0)) * dome_exposure(image, image.conditions))
    assert all(a > b for a, b in zip(rendered, rendered[1:])), rendered
    assert rendered[0] / rendered[-1] > 4.0     # night is clearly night
    assert rendered[0] / rendered[-1] < 100.0   # and not simply black


def test_an_all_black_sky_does_not_divide_by_it():
    from weather_fx.core.sky import dome_exposure

    exposure = dome_exposure(np.zeros((32, 64, 3), dtype=np.float32))
    assert np.isfinite(exposure) and exposure > 0.0
