"""Surface meteorology: the series a thermal model integrates.

These tests are written to fail if the *physics* is wrong, not merely if the code throws. The
recurring trap in this module is a quantity that is plausible in isolation and wrong in relation
to another: a diffuse irradiance that falls when cloud arrives, a relative humidity that does not
rise at night, a December that is warm because a phase was anchored on the new year.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from weather_fx.core.meteorology import (
    COLUMNS,
    aerosol_visibility_m,
    airmass,
    clear_sky_irradiance,
    diurnal_series,
    precipitation_rate_mm_h,
    relative_humidity,
    saturation_vapour_pressure_hpa,
    solar_irradiance,
    surface_visibility_m,
)
from weather_fx.core.random_weather import _seasonal_temperature, random_state
from weather_fx.core.state import WeatherState


def _state(**sections):
    state = WeatherState()
    for name, values in sections.items():
        state = state.with_updates(name, **values)
    return state


# --- humidity --------------------------------------------------------------------------------


def test_saturation_vapour_pressure_matches_the_reference_values():
    # Smithsonian tables: 6.11 hPa at 0 degC, 23.37 at 20, 12.27 at 10.
    assert saturation_vapour_pressure_hpa(0.0) == pytest.approx(6.11, abs=0.02)
    assert saturation_vapour_pressure_hpa(10.0) == pytest.approx(12.27, abs=0.05)
    assert saturation_vapour_pressure_hpa(20.0) == pytest.approx(23.37, abs=0.1)


def test_relative_humidity_is_one_at_the_dew_point_and_falls_as_the_air_warms():
    assert relative_humidity(11.0, 11.0) == pytest.approx(1.0)
    rh = relative_humidity(np.array([11.0, 16.0, 21.0, 26.0]), 11.0)
    assert np.all(np.diff(rh) < 0.0)
    assert rh[-1] == pytest.approx(0.39, abs=0.02)


def test_relative_humidity_never_exceeds_one_when_the_air_is_below_its_dew_point():
    assert relative_humidity(5.0, 11.0) == pytest.approx(1.0)


# --- solar geometry --------------------------------------------------------------------------


def test_airmass_is_one_at_the_zenith_and_finite_at_the_horizon():
    assert airmass(90.0) == pytest.approx(1.0, rel=1e-3)
    assert airmass(30.0) == pytest.approx(2.0, rel=0.02)
    horizon = float(airmass(0.0))
    assert 35.0 < horizon < 40.0  # the 1/sin form would be infinite here


def test_airmass_decreases_with_elevation():
    el = np.array([1.0, 5.0, 15.0, 45.0, 90.0])
    assert np.all(np.diff(airmass(el)) < 0.0)


# --- clear-sky irradiance --------------------------------------------------------------------


def test_no_sun_below_the_horizon():
    dni, dhi = clear_sky_irradiance(np.array([-30.0, -5.0, -0.1]), 2.8)
    assert np.all(dni == 0.0)
    assert np.all(dhi == 0.0)


def test_clear_sky_beam_and_diffuse_are_the_magnitudes_everyone_knows():
    dni, dhi = clear_sky_irradiance(60.0, 2.8)
    assert 850.0 < float(dni) < 1000.0
    assert 80.0 < float(dhi) < 160.0


def test_turbidity_dims_the_beam_and_brightens_the_diffuse():
    """The signature of haze. A model that dims both is describing a smaller sun, not an aerosol:
    the energy the beam loses to scattering is where the diffuse sky comes from."""
    clean_dni, clean_dhi = clear_sky_irradiance(50.0, 2.0)
    hazy_dni, hazy_dhi = clear_sky_irradiance(50.0, 7.0)
    assert hazy_dni < clean_dni
    assert hazy_dhi > clean_dhi


def test_a_days_clear_sky_energy_is_in_the_right_range():
    """Midsummer at 48 degN: a clear day delivers 6-9 kWh/m2 on the horizontal."""
    when = datetime(2024, 6, 21, tzinfo=timezone.utc)
    state = _state(sky={"latitude_deg": 48.1, "longitude_deg": 11.6,
                        "date_utc": "2024-06-21", "hour_utc": 12.0, "turbidity": 2.8})
    series = diurnal_series(state, start_utc=when, hours=24.0, step_s=600.0)
    from weather_fx.core.celestial import sun_position

    elevation = np.array([
        sun_position(48.1, 11.6, when + timedelta(seconds=float(t))).elevation_deg
        for t in series.time_s
    ])
    ghi = series.dni_w_m2 * np.maximum(np.sin(np.radians(elevation)), 0.0) + series.dhi_w_m2
    kwh = float(np.trapezoid(ghi, series.time_s)) / 3.6e6
    assert 6.5 < kwh < 9.0


# --- cloud -------------------------------------------------------------------------------------


def test_overcast_removes_the_beam_and_leaves_a_quarter_of_the_global():
    """Kasten-Czeplak. The quarter is the number an overcast pyranometer actually reports, and it
    is far more than ``1 - cover`` would give -- a fully covered sky is not a dark one."""
    el = 55.0
    clear_dni, clear_dhi = clear_sky_irradiance(el, 2.8)
    sin_el = np.sin(np.radians(el))
    clear_ghi = float(clear_dni) * sin_el + float(clear_dhi)

    dni, dhi = solar_irradiance(el, 2.8, cloud_cover=1.0)
    assert float(dni) == 0.0
    assert float(dhi) == pytest.approx(0.25 * clear_ghi, rel=1e-6)


def test_broken_cloud_gives_more_diffuse_than_a_clear_sky():
    """The bright side of a cumulus is a light source. Measured diffuse peaks near half cover,
    and a model that only subtracts cloud runs a scene cold at midday."""
    _, clear_dhi = clear_sky_irradiance(55.0, 2.8)
    _, broken_dhi = solar_irradiance(55.0, 2.8, cloud_cover=0.5)
    assert float(broken_dhi) > float(clear_dhi)


def test_global_irradiance_falls_monotonically_with_cover():
    el = 55.0
    sin_el = np.sin(np.radians(el))
    ghi = []
    for cover in np.linspace(0.0, 1.0, 11):
        dni, dhi = solar_irradiance(el, 2.8, cloud_cover=float(cover))
        ghi.append(float(dni) * sin_el + float(dhi))
    assert np.all(np.diff(ghi) < 0.0)


def test_cloud_cannot_make_the_sun_rise():
    dni, dhi = solar_irradiance(-10.0, 2.8, cloud_cover=0.5)
    assert float(dni) == 0.0
    assert float(dhi) == 0.0


# --- visibility and precipitation ---------------------------------------------------------------


def test_aerosol_visibility_tracks_turbidity_over_the_familiar_range():
    assert 30_000.0 < aerosol_visibility_m(2.0) < 50_000.0
    assert 18_000.0 < aerosol_visibility_m(3.0) < 32_000.0
    assert 2_000.0 < aerosol_visibility_m(8.0) < 5_000.0
    turbidity = np.array([2.0, 3.0, 5.0, 8.0, 10.0])
    values = [aerosol_visibility_m(float(t)) for t in turbidity]
    assert np.all(np.diff(values) < 0.0)


def test_fog_shortens_the_visibility_below_its_own_figure():
    """Extinction adds, visibility does not. Fog at 200 m in air that was already hazy must give
    *less* than 200 m, and a model that simply takes the minimum will never say so."""
    state = _state(sky={"turbidity": 6.0}, fog={"enabled": True, "visibility_m": 200.0})
    assert surface_visibility_m(state) < 200.0
    assert surface_visibility_m(state) > 150.0


def test_without_fog_the_visibility_is_the_aerosols():
    state = _state(sky={"turbidity": 3.4}, fog={"enabled": False, "visibility_m": 50.0})
    assert surface_visibility_m(state) == pytest.approx(aerosol_visibility_m(3.4))


def test_rain_rate_passes_through_and_nothing_falls_when_it_is_off():
    assert precipitation_rate_mm_h(_state(rain={"enabled": True, "rate_mm_h": 7.5})) == 7.5
    assert precipitation_rate_mm_h(_state(rain={"enabled": False, "rate_mm_h": 7.5})) == 0.0


def test_snow_water_equivalent_is_derived_from_the_flake_population():
    """No second parameter to keep in step with the first: the flakes already say how much water
    is in the air and how fast it is arriving."""
    light = _state(snow={"enabled": True, "number_density_m3": 2.0, "fall_speed_mps": 1.0})
    heavy = _state(snow={"enabled": True, "number_density_m3": 40.0, "fall_speed_mps": 1.5})
    light_rate = precipitation_rate_mm_h(light)
    heavy_rate = precipitation_rate_mm_h(heavy)
    # A snowfall is a fraction of a millimetre to a few millimetres of water an hour.
    assert 0.005 < light_rate < 0.5
    assert 0.2 < heavy_rate < 5.0
    # Twice the flakes, twice the water.
    doubled = _state(snow={"enabled": True, "number_density_m3": 4.0, "fall_speed_mps": 1.0})
    assert precipitation_rate_mm_h(doubled) == pytest.approx(2.0 * light_rate)


# --- the series ----------------------------------------------------------------------------------


def _clear_noon_state(**sky):
    params = {"latitude_deg": 48.1, "longitude_deg": 11.6, "date_utc": "2024-06-21",
              "hour_utc": 10.0, "turbidity": 2.8}
    params.update(sky)
    return _state(sky=params, clouds={"enabled": True, "cover": 0.3,
                                      "temperature_c": 22.0, "dewpoint_c": 11.0})


def test_the_series_carries_exactly_the_documented_columns():
    series = diurnal_series(_clear_noon_state())
    assert set(series.columns()) == set(COLUMNS)
    assert all(series.columns()[name].shape == series.time_s.shape for name in COLUMNS)
    assert series.epoch_utc.tzinfo is not None


def test_the_air_temperature_passes_through_the_states_own_value_at_the_states_own_hour():
    """The anchor property, and the reason the series can be trusted next to a rendered frame:
    the frame is lit for one instant, and the thermal history must arrive at that instant."""
    state = _clear_noon_state()
    series = diurnal_series(state, hours=48.0, step_s=1800.0)
    anchor_index = int(round(24.0 * 3600.0 / 1800.0))  # the default start is 24 h earlier
    assert float(series.t_air_k[anchor_index]) == pytest.approx(22.0 + 273.15, abs=1e-6)


def test_the_air_never_cools_below_its_dew_point():
    """A humid night saturates and stalls; a bare sinusoid would drive it several kelvin past."""
    state = _state(
        sky={"latitude_deg": 48.1, "date_utc": "2024-06-21", "hour_utc": 12.0},
        clouds={"enabled": True, "cover": 0.1, "temperature_c": 20.0, "dewpoint_c": 18.5},
    )
    series = diurnal_series(state)
    assert float(series.t_air_k.min()) >= 18.5 + 273.15 - 1e-9
    assert float(series.rh_fraction.max()) == pytest.approx(1.0)


def test_humidity_rises_at_night_because_the_air_cooled_not_because_the_water_changed():
    series = diurnal_series(_clear_noon_state())
    correlation = float(np.corrcoef(series.t_air_k, series.rh_fraction)[0, 1])
    assert correlation < -0.95


def test_cloud_damps_the_daily_temperature_range():
    """The cloudy-night signature: it does not get cold."""
    clear = diurnal_series(_clear_noon_state().with_updates("clouds", enabled=False, cover=0.0))
    overcast = diurnal_series(_clear_noon_state().with_updates("clouds", cover=1.0))
    clear_range = float(clear.t_air_k.max() - clear.t_air_k.min())
    overcast_range = float(overcast.t_air_k.max() - overcast.t_air_k.min())
    assert overcast_range < 0.5 * clear_range


def test_the_sun_rises_and_sets_in_the_series():
    series = diurnal_series(_clear_noon_state(), hours=24.0, step_s=900.0)
    assert float(series.dni_w_m2.min()) == 0.0
    assert float(series.dni_w_m2.max()) > 500.0
    # Midsummer at 48 degN: roughly sixteen hours of sun, so a third of the samples are dark.
    dark = float(np.mean(series.dni_w_m2 == 0.0))
    assert 0.2 < dark < 0.45


def test_a_polar_winter_series_stays_dark_rather_than_failing():
    state = _clear_noon_state(latitude_deg=78.0, longitude_deg=15.0, date_utc="2024-01-15")
    series = diurnal_series(state, hours=24.0, step_s=1800.0)
    assert float(series.dni_w_m2.max()) == 0.0
    assert float(series.dhi_w_m2.max()) == 0.0


def test_the_series_refuses_a_naive_start_and_a_degenerate_grid():
    state = _clear_noon_state()
    with pytest.raises(ValueError):
        diurnal_series(state, start_utc=datetime(2024, 6, 21, 0, 0))
    with pytest.raises(ValueError):
        diurnal_series(state, hours=0.0)
    with pytest.raises(ValueError):
        diurnal_series(state, step_s=-60.0)


def test_a_bad_date_says_which_field_and_what_shape():
    state = _clear_noon_state().with_updates("sky", date_utc="21/06/2024")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        diurnal_series(state)


@pytest.mark.parametrize("seed", [1, 3, 7, 11, 23])
def test_every_random_regime_produces_a_physically_admissible_series(seed):
    """The columns downstream are range-checked by the consumer; failing there is a stack trace
    in a render, and failing here is a line of output."""
    series = diurnal_series(random_state(seed))
    assert np.all(np.isfinite(series.t_air_k))
    assert np.all((series.t_air_k > 180.0) & (series.t_air_k < 340.0))
    assert np.all((series.rh_fraction >= 0.0) & (series.rh_fraction <= 1.0))
    assert np.all(series.wind_speed_m_s >= 0.0)
    assert np.all((series.cloud_fraction >= 0.0) & (series.cloud_fraction <= 1.0))
    assert np.all((series.dni_w_m2 >= 0.0) & (series.dni_w_m2 <= 1361.0))
    assert np.all((series.dhi_w_m2 >= 0.0) & (series.dhi_w_m2 <= 1361.0))
    assert np.all(series.visibility_m > 0.0)
    assert np.all(series.precip_mm_h >= 0.0)


def test_a_rain_regime_is_wet_dim_and_hard_to_see_through():
    for seed in range(200):
        state = random_state(seed, regime="rain")
        series = diurnal_series(state)
        assert float(series.precip_mm_h[0]) > 0.0
        assert float(series.cloud_fraction[0]) > 0.85
        assert float(series.visibility_m[0]) < 10_000.0
        break


# --- the seasonal anchor -------------------------------------------------------------------------


def test_the_northern_hemisphere_is_warm_in_july_and_cold_in_january():
    """Anchored on the warmest day rather than the new year. With the phase a half-year out, a
    log line reading 17 degC in December at 59 degN looks entirely ordinary."""
    rng = np.random.default_rng
    july = np.mean([_seasonal_temperature(rng(i), 196, 59.0) for i in range(300)])
    january = np.mean([_seasonal_temperature(rng(i), 15, 59.0) for i in range(300)])
    assert july > january + 12.0
    assert january < 5.0


def test_the_southern_hemisphere_is_the_other_way_round():
    rng = np.random.default_rng
    july = np.mean([_seasonal_temperature(rng(i), 196, -35.0) for i in range(300)])
    january = np.mean([_seasonal_temperature(rng(i), 15, -35.0) for i in range(300)])
    assert january > july + 5.0


def test_the_tropics_barely_have_a_season():
    rng = np.random.default_rng
    july = np.mean([_seasonal_temperature(rng(i), 196, 0.0) for i in range(300)])
    january = np.mean([_seasonal_temperature(rng(i), 15, 0.0) for i in range(300)])
    assert abs(july - january) < 6.0
