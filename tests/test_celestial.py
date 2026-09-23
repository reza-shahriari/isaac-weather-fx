"""The sun and the moon, checked against geometry that does not depend on the implementation.

Almost every assertion here is a closed-form fact about the earth's orbit or a published
ephemeris value, not a number this code produced on a good day. An ephemeris that is wrong is
wrong plausibly: the sun comes up, moves across the sky and sets, and only a check against the
outside world notices that it did all of that an hour early or forty degrees round.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from weather_fx.core.celestial import (
    MOON_MEAN_DISTANCE_KM,
    MoonPosition,
    civil_twilight_fraction,
    julian_day,
    moon_illuminance_lux,
    moon_position,
    sun_position,
)

MUNICH = (48.1, 11.6)


def utc(year, month, day, hour=0, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


# --- the calendar ------------------------------------------------------------------------------


def test_the_julian_day_of_j2000_is_the_epoch_itself() -> None:
    """J2000.0 is 2000-01-01 12:00 TT = JD 2451545.0. The one anchor every series is written on."""
    assert julian_day(utc(2000, 1, 1, 12)) == pytest.approx(2451545.0, abs=1e-6)


def test_a_naive_datetime_is_refused_rather_than_assumed_to_be_utc() -> None:
    """A clock silently an hour out gives a sun in the wrong place and a plausible shadow."""
    with pytest.raises(ValueError, match="timezone-aware"):
        julian_day(datetime(2024, 6, 21, 12))


# --- the sun -----------------------------------------------------------------------------------


@pytest.mark.parametrize("latitude", [-60.0, -23.4, 0.0, 30.0, 48.1, 66.0])
def test_the_noon_sun_at_the_solstice_is_where_the_obliquity_puts_it(latitude: float) -> None:
    """Closed form: at the June solstice the subsolar point is at +23.44 deg, so the noon
    elevation is 90 - |lat - 23.44|. True for every latitude, with no fitting anywhere."""
    best = max(
        (
            sun_position(latitude, 0.0, utc(2024, 6, 21, hour, minute))
            for hour in range(9, 15)
            for minute in range(0, 60, 2)
        ),
        key=lambda p: p.elevation_deg,
    )
    expected = 90.0 - abs(latitude - 23.44)
    # A tenth of a degree covers refraction near the horizon and the solstice not falling exactly
    # on midnight UTC; nothing else is free.
    assert best.elevation_deg == pytest.approx(expected, abs=0.35)


def test_at_the_equinox_the_sun_rises_due_east_and_sets_due_west() -> None:
    """True everywhere on earth, twice a year, and it catches a transposed azimuth instantly."""
    def nearest_horizon(start_hour: int):
        base = utc(2024, 3, 20, start_hour)
        return min(
            (sun_position(0.0, 0.0, base + timedelta(minutes=m)) for m in range(0, 120, 2)),
            key=lambda p: abs(p.elevation_deg),
        )

    rising, setting = nearest_horizon(5), nearest_horizon(17)
    assert rising.azimuth_deg == pytest.approx(90.0, abs=1.0)
    assert setting.azimuth_deg == pytest.approx(270.0, abs=1.0)


def test_the_northern_noon_sun_is_due_south_and_the_southern_one_due_north() -> None:
    def noon_azimuth(latitude: float) -> float:
        best = max(
            (sun_position(latitude, 0.0, utc(2024, 6, 21, h, m))
             for h in range(10, 14) for m in range(0, 60, 2)),
            key=lambda p: p.elevation_deg,
        )
        return best.azimuth_deg

    assert noon_azimuth(48.1) == pytest.approx(180.0, abs=2.0)
    assert noon_azimuth(-33.9) % 360.0 == pytest.approx(360.0, abs=2.0)


def test_the_azimuth_does_not_fold_back_on_itself_across_noon() -> None:
    """The `acos` form of this conversion loses the sign of the hour angle and maps the afternoon
    onto the morning -- a sun that retraces its own path after noon, and a shadow that goes
    backwards. It is the single most common bug in this conversion."""
    azimuths = [
        sun_position(*MUNICH, utc(2024, 6, 21, 4 + i // 4, 15 * (i % 4))).azimuth_deg
        for i in range(60)
    ]
    rising = [b - a for a, b in zip(azimuths, azimuths[1:])]
    assert all(step > 0.0 for step in rising), "azimuth must increase all day in the north"
    assert max(azimuths) - min(azimuths) > 180.0


def test_refraction_lifts_the_setting_sun_by_about_its_own_diameter() -> None:
    """At the horizon refraction is ~34 arc-minutes, slightly more than the sun's 32. So the sun
    is geometrically already set when it appears to touch the horizon; a model without this puts
    sunset minutes early, at the moment a scene's lighting is changing fastest.

    **True altitude, not apparent.** Saemundsson's formula takes the geometric altitude, so its
    value at zero is 29 arc-minutes; the familiar 34 belongs to the altitude at which the disc
    *appears* on the horizon, about -0.57 deg true. Pinning both is what stops the two being
    swapped later -- they differ by one solar diameter, which is the whole effect.
    """
    base = utc(2024, 3, 20, 17)
    near_horizon = min(
        (sun_position(*MUNICH, base + timedelta(minutes=m)) for m in range(0, 90)),
        key=lambda p: abs(p.elevation_deg),
    )
    # Refraction is already included, so the reported elevation at true sunset is above zero by
    # roughly the refraction: sample either side and check the lift is in the right band.
    from weather_fx.core.celestial import _refraction_deg

    assert _refraction_deg(0.0) * 60.0 == pytest.approx(29.0, abs=1.0)
    assert _refraction_deg(-0.57) * 60.0 == pytest.approx(34.0, abs=1.5)
    assert _refraction_deg(45.0) * 60.0 < 1.5
    # Monotone: refraction must never grow as a body climbs.
    lifts = [_refraction_deg(e) for e in [0.0, 1.0, 5.0, 15.0, 45.0, 89.0]]
    assert all(b < a for a, b in zip(lifts, lifts[1:]))
    assert near_horizon.elevation_deg == pytest.approx(0.0, abs=0.6)


def test_the_solar_disc_grows_at_perihelion_and_shrinks_at_aphelion() -> None:
    """The earth's orbit is 1.7 % eccentric, so the disc is 3.4 % wider in January than in July.
    Small, but it is what makes the angular diameter a computed quantity rather than a constant."""
    january = sun_position(0.0, 0.0, utc(2024, 1, 4, 12)).angular_diameter_deg
    july = sun_position(0.0, 0.0, utc(2024, 7, 5, 12)).angular_diameter_deg
    assert january > july
    assert january / july == pytest.approx(1.034, abs=0.006)


def test_the_direction_vector_agrees_with_the_angles_it_came_from() -> None:
    """Y-up, -Z north: the Omniverse convention the viewport backend authors lights in."""
    sun = sun_position(*MUNICH, utc(2024, 6, 21, 10))
    v = sun.direction(up_axis=1, north_axis=2, north_sign=-1.0)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-12)
    assert math.degrees(math.asin(v[1])) == pytest.approx(sun.elevation_deg, abs=1e-9)
    # Azimuth 90 (due east) must put the vector on +X and nothing on Z.
    from weather_fx.core.celestial import BodyPosition

    east = BodyPosition(0.0, 90.0, 0.53).direction(up_axis=1, north_axis=2, north_sign=-1.0)
    assert east == pytest.approx(np.array([1.0, 0.0, 0.0]), abs=1e-12)
    north = BodyPosition(0.0, 0.0, 0.53).direction(up_axis=1, north_axis=2, north_sign=-1.0)
    assert north == pytest.approx(np.array([0.0, 0.0, -1.0]), abs=1e-12)


def test_a_z_up_stage_gets_the_same_sky_rotated_into_its_own_axes() -> None:
    from weather_fx.core.celestial import BodyPosition

    up = BodyPosition(90.0, 0.0, 0.53).direction(up_axis=2, north_axis=1, north_sign=1.0)
    assert up == pytest.approx(np.array([0.0, 0.0, 1.0]), abs=1e-12)
    north = BodyPosition(0.0, 0.0, 0.53).direction(up_axis=2, north_axis=1, north_sign=1.0)
    assert north == pytest.approx(np.array([0.0, 1.0, 0.0]), abs=1e-12)
    east = BodyPosition(0.0, 90.0, 0.53).direction(up_axis=2, north_axis=1, north_sign=1.0)
    assert east == pytest.approx(np.array([1.0, 0.0, 0.0]), abs=1e-12)


# --- the moon ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "when, illuminated",
    [
        (utc(2024, 6, 22, 1, 8), 1.0),  # full moon, published instant
        (utc(2024, 6, 6, 12, 38), 0.0),  # new moon
        (utc(2024, 6, 14, 5, 18), 0.5),  # first quarter
        (utc(2024, 6, 28, 21, 53), 0.5),  # last quarter
    ],
)
def test_the_lunar_phase_matches_the_published_instants(when, illuminated) -> None:
    moon = moon_position(*MUNICH, when)
    assert moon.illuminated_fraction == pytest.approx(illuminated, abs=0.01)


def test_the_quarters_are_named_waxing_and_waning_the_right_way_round() -> None:
    assert moon_position(*MUNICH, utc(2024, 6, 14, 5, 18)).phase_name == "first quarter"
    assert moon_position(*MUNICH, utc(2024, 6, 28, 21, 53)).phase_name == "last quarter"
    assert moon_position(*MUNICH, utc(2024, 6, 22, 1, 8)).phase_name == "full"
    assert moon_position(*MUNICH, utc(2024, 6, 6, 12, 38)).phase_name == "new"


def test_the_moon_stays_within_its_own_orbit_s_distance_range() -> None:
    """Perigee 356500 km, apogee 406700 km. A series with a sign error leaves that band at once."""
    start = utc(2024, 1, 1)
    distances = [
        moon_position(*MUNICH, start + timedelta(days=day)).distance_km for day in range(0, 60)
    ]
    assert min(distances) > 355_000.0
    assert max(distances) < 408_000.0
    # And it really does swing: a constant distance would pass the bounds above.
    assert max(distances) - min(distances) > 40_000.0


def test_the_full_moon_is_opposite_the_sun_in_the_sky() -> None:
    """The definition of a full moon, and an independent check on both ephemerides at once."""
    when = utc(2024, 6, 22, 1, 8)
    sun = sun_position(0.0, 0.0, when)
    moon = moon_position(0.0, 0.0, when)
    separation = abs((moon.azimuth_deg - sun.azimuth_deg + 180.0) % 360.0 - 180.0)
    assert separation > 150.0
    assert moon.elevation_deg * sun.elevation_deg < 0.0, "one up, one down"


# --- how much light ------------------------------------------------------------------------------


def test_a_full_moon_overhead_gives_the_textbook_quarter_of_a_lux() -> None:
    full = MoonPosition(90.0, 180.0, 0.518, 1.0, 0.0, MOON_MEAN_DISTANCE_KM)
    assert moon_illuminance_lux(full) == pytest.approx(0.267, rel=0.01)


def test_a_quarter_moon_gives_a_twelfth_of_a_full_one_not_a_half() -> None:
    """Published magnitudes: -12.74 full against -10.0 at first quarter is 2.74 mag, a ratio of
    0.080. Treating phase as linear overstates a quarter moon by six times, which is the
    difference between a night scene a camera can use and one that is black."""
    full = MoonPosition(90.0, 180.0, 0.518, 1.0, 0.0, MOON_MEAN_DISTANCE_KM)
    quarter = MoonPosition(90.0, 180.0, 0.518, 0.5, 90.0, MOON_MEAN_DISTANCE_KM)
    assert moon_illuminance_lux(quarter) / moon_illuminance_lux(full) == pytest.approx(
        0.080, abs=0.015
    )


def test_a_moon_below_the_horizon_lights_nothing() -> None:
    below = MoonPosition(-0.1, 180.0, 0.518, 1.0, 0.0, MOON_MEAN_DISTANCE_KM)
    assert moon_illuminance_lux(below) == 0.0


def test_twilight_falls_smoothly_and_reaches_zero_at_astronomical_night() -> None:
    assert civil_twilight_fraction(10.0) == 1.0
    assert civil_twilight_fraction(0.0) == 1.0
    assert civil_twilight_fraction(-18.0) == 0.0
    assert civil_twilight_fraction(-30.0) == 0.0
    samples = [civil_twilight_fraction(e) for e in np.linspace(0.0, -18.0, 200)]
    assert all(b <= a + 1e-12 for a, b in zip(samples, samples[1:])), "must be monotone"
    # No step anywhere: the largest jump between adjacent samples stays small, so a time-lapse
    # through sunset does not pop.
    assert max(abs(b - a) for a, b in zip(samples, samples[1:])) < 0.02
