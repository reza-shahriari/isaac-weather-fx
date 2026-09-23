"""Where the sun and the moon are, and how bright they are.

Pure Python plus numpy, like the rest of ``core``: an extension that can only tell you where the
sun is once Kit has booted is no use for generating a dataset, and the same numbers have to be
available to a renderer, to a physically-based sensor model and to a unit test.

Two bodies, two different jobs:

* **The sun** sets the whole daylight picture -- the sky's colour and brightness, the direction
  shadows fall, and the short-wave load a thermal model puts on every surface. Its position comes
  from the NOAA Solar Position Algorithm, which is accurate to about 0.01 degrees over the years
  this will ever be used for and is short enough to read.
* **The moon** matters for one reason: it is the only thing that lights an outdoor night scene.
  Its position uses the truncated lunar theory from Meeus, *Astronomical Algorithms* ch. 47,
  which is good to roughly 10 arc-minutes -- far finer than the half-degree disc itself, and far
  finer than anything a renderer does with it.

Angles are degrees, times are timezone-aware UTC ``datetime``, and azimuth is measured **from
north, clockwise through east**, which is the meteorological convention and the one the sky model
and the viewport backend both read.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

__all__ = [
    "MOON_ALBEDO",
    "MOON_MEAN_DISTANCE_KM",
    "SOLAR_CONSTANT_W_M2",
    "SUN_ANGULAR_DIAMETER_DEG",
    "MOON_ANGULAR_DIAMETER_DEG",
    "BodyPosition",
    "MoonPosition",
    "julian_day",
    "julian_centuries",
    "sun_position",
    "moon_position",
    "moon_illuminance_lux",
    "civil_twilight_fraction",
]

#: Solar irradiance at the top of the atmosphere, 1 AU (WMO/CODATA, 2015 value).
SOLAR_CONSTANT_W_M2 = 1361.0
#: Mean apparent angular diameter of the solar disc, degrees. Used for shadow softness.
SUN_ANGULAR_DIAMETER_DEG = 0.533
#: Mean apparent angular diameter of the lunar disc, degrees. Very nearly the sun's, which is why
#: total eclipses are possible; kept separate because the moon's varies by 12 % over a month.
MOON_ANGULAR_DIAMETER_DEG = 0.518
#: Geometric albedo of the moon (Bond albedo is 0.12; this is the one that sets brightness).
MOON_ALBEDO = 0.136
MOON_MEAN_DISTANCE_KM = 384400.0

#: Full-moon illuminance on a horizontal surface at the zenith, lux. The textbook figure; the
#: phase and altitude laws below are applied on top of it.
_FULL_MOON_ZENITH_LUX = 0.267


def julian_day(when: datetime) -> float:
    """Julian Day Number, including the fraction of a day, for an **aware** UTC datetime.

    Naive datetimes are refused rather than assumed to be UTC. A scene whose clock is silently an
    hour out gets a sun in the wrong place and a shadow that looks entirely plausible, which is
    the kind of error that survives review.
    """
    if when.tzinfo is None:
        raise ValueError("celestial times must be timezone-aware; got a naive datetime")
    t = when.astimezone(timezone.utc)
    year, month = t.year, t.month
    day = (
        t.day
        + (t.hour + (t.minute + (t.second + t.microsecond * 1e-6) / 60.0) / 60.0) / 24.0
    )
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    return (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + day
        + b
        - 1524.5
    )


def julian_centuries(jd: float) -> float:
    """Julian centuries since J2000.0, the time argument every series below is written in."""
    return (jd - 2451545.0) / 36525.0


@dataclass(frozen=True)
class BodyPosition:
    """Where a body is in the observer's sky."""

    elevation_deg: float
    azimuth_deg: float
    #: Apparent angular diameter, degrees -- the softness of the shadow the body casts.
    angular_diameter_deg: float

    @property
    def zenith_deg(self) -> float:
        return 90.0 - self.elevation_deg

    @property
    def is_up(self) -> bool:
        """Above the true horizon. The *disc* is still partly visible slightly below this."""
        return self.elevation_deg > 0.0

    def direction(self, up_axis: int = 1, north_axis: int = 2, north_sign: float = -1.0) -> Any:
        """Unit vector **toward** the body, in stage axes.

        The defaults are the Omniverse convention -- +Y up and -Z north -- which is what the
        viewport backend needs. A Z-up stage passes ``up_axis=2, north_axis=1, north_sign=1``.
        Azimuth runs from north clockwise through east, so east is the remaining axis and its
        sign follows from the right-handed triple.
        """
        el = math.radians(self.elevation_deg)
        az = math.radians(self.azimuth_deg)
        east_axis = 3 - up_axis - north_axis
        out = np.zeros(3, dtype=np.float64)
        out[up_axis] = math.sin(el)
        out[north_axis] = north_sign * math.cos(el) * math.cos(az)
        # Right-handed: east = north x up, whose sign depends on the axis ordering.
        north = np.zeros(3)
        north[north_axis] = north_sign
        up = np.zeros(3)
        up[up_axis] = 1.0
        east = np.cross(north, up)
        out[east_axis] = float(east[east_axis]) * math.cos(el) * math.sin(az)
        return out


@dataclass(frozen=True)
class MoonPosition(BodyPosition):
    """The moon, plus the two numbers that decide how much light it gives."""

    #: 0 at new moon, 1 at full. The *illuminated fraction of the disc*, not the phase angle.
    illuminated_fraction: float
    #: Phase angle sun-moon-observer, degrees: 0 at full moon, 180 at new.
    phase_angle_deg: float
    distance_km: float
    #: True while the lit fraction is growing. The phase *angle* cannot say: it is 90 degrees at
    #: both quarters. Only the sign of the moon's elongation east of the sun distinguishes them,
    #: so it is carried here rather than guessed from the angle.
    waxing: bool = True

    @property
    def phase_name(self) -> str:
        """A human label, for a UI that has to say something short."""
        f = self.illuminated_fraction
        if f < 0.02:
            return "new"
        if f > 0.98:
            return "full"
        if f < 0.45:
            return "waxing crescent" if self.waxing else "waning crescent"
        if f < 0.55:
            return "first quarter" if self.waxing else "last quarter"
        return "waxing gibbous" if self.waxing else "waning gibbous"


def _equatorial_to_horizontal(
    right_ascension_deg: float, declination_deg: float, lat_deg: float, local_sidereal_deg: float
) -> tuple[float, float]:
    """Hour angle, then the standard rotation into altitude and azimuth (from north, via east)."""
    hour_angle = math.radians((local_sidereal_deg - right_ascension_deg + 180.0) % 360.0 - 180.0)
    dec = math.radians(declination_deg)
    lat = math.radians(lat_deg)
    sin_el = math.sin(dec) * math.sin(lat) + math.cos(dec) * math.cos(lat) * math.cos(hour_angle)
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))
    # atan2 form rather than acos: it is continuous through due south, where the acos form loses
    # the sign of the hour angle and folds the afternoon onto the morning.
    y = -math.sin(hour_angle)
    x = math.tan(dec) * math.cos(lat) - math.sin(lat) * math.cos(hour_angle)
    azimuth = math.degrees(math.atan2(y, x)) % 360.0
    return elevation, azimuth


def _greenwich_sidereal_deg(jd: float) -> float:
    """Greenwich apparent sidereal time, degrees (Meeus eq. 12.4, mean; good to ~1 arcsec here)."""
    t = julian_centuries(jd)
    theta = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t * t
        - t * t * t / 38710000.0
    )
    return theta % 360.0


def _refraction_deg(elevation_deg: float) -> float:
    """Atmospheric refraction lifting a body near the horizon (Saemundsson, Meeus eq. 16.4).

    The argument is the **true** (geometric) altitude, which is what Saemundsson's form takes and
    what this module has before the correction is applied. At true altitude zero it is 29
    arc-minutes; the familiar 34 belongs to the altitude at which the disc *appears* on the
    horizon, about -0.57 deg true. The two differ by a whole solar diameter, so which one a
    formula wants is worth stating rather than discovering.

    Kept at all because a sky model that ignores refraction puts sunset a couple of minutes early
    -- at exactly the moment a scene's lighting is changing fastest.
    """
    if elevation_deg < -2.0:
        return 0.0
    h = elevation_deg
    return 1.02 / math.tan(math.radians(h + 10.3 / (h + 5.11))) / 60.0


def sun_position(lat_deg: float, lon_deg: float, when: datetime) -> BodyPosition:
    """The sun's apparent position, by the NOAA Solar Position Algorithm.

    ``lon_deg`` is positive **east**. The result includes atmospheric refraction, so it is where
    the sun appears rather than where it geometrically is.
    """
    jd = julian_day(when)
    t = julian_centuries(jd)

    geom_mean_long = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    geom_mean_anom = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    eccentricity = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    m = math.radians(geom_mean_anom)
    centre = (
        math.sin(m) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + math.sin(2 * m) * (0.019993 - 0.000101 * t)
        + math.sin(3 * m) * 0.000289
    )
    true_long = geom_mean_long + centre
    omega = 125.04 - 1934.136 * t
    apparent_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))

    mean_obliquity = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    obliquity = mean_obliquity + 0.00256 * math.cos(math.radians(omega))

    lam = math.radians(apparent_long)
    eps = math.radians(obliquity)
    right_ascension = math.degrees(math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam)))
    declination = math.degrees(math.asin(math.sin(eps) * math.sin(lam)))

    local_sidereal = (_greenwich_sidereal_deg(jd) + lon_deg) % 360.0
    elevation, azimuth = _equatorial_to_horizontal(
        right_ascension, declination, lat_deg, local_sidereal
    )
    # The earth-sun distance moves 3.3 % over a year, and the disc with it.
    radius_au = (1.000001018 * (1 - eccentricity * eccentricity)) / (
        1 + eccentricity * math.cos(math.radians(geom_mean_anom + centre))
    )
    return BodyPosition(
        elevation_deg=elevation + _refraction_deg(elevation),
        azimuth_deg=azimuth,
        angular_diameter_deg=SUN_ANGULAR_DIAMETER_DEG / radius_au,
    )


def moon_position(lat_deg: float, lon_deg: float, when: datetime) -> MoonPosition:
    """The moon's apparent position, phase and distance (Meeus ch. 47, principal terms).

    The truncated series keeps the terms above roughly 0.05 degrees, which lands the moon inside
    about 10 arc-minutes -- a fifth of its own diameter. That is far better than a renderer needs
    for a light source, and it is the difference between a night scene lit from the right quarter
    and one lit from anywhere.
    """
    jd = julian_day(when)
    t = julian_centuries(jd)

    # Mean elements, degrees.
    lp = (218.3164477 + 481267.88123421 * t - 0.0015786 * t * t) % 360.0  # mean longitude
    d = (297.8501921 + 445267.1114034 * t - 0.0018819 * t * t) % 360.0  # mean elongation
    m = (357.5291092 + 35999.0502909 * t) % 360.0  # sun's mean anomaly
    mp = (134.9633964 + 477198.8675055 * t + 0.0087414 * t * t) % 360.0  # moon's mean anomaly
    f = (93.2720950 + 483202.0175233 * t - 0.0036539 * t * t) % 360.0  # argument of latitude

    dr, mr, mpr, fr = (math.radians(x) for x in (d, m, mp, f))

    longitude = lp + (
        6.288774 * math.sin(mpr)
        + 1.274027 * math.sin(2 * dr - mpr)
        + 0.658314 * math.sin(2 * dr)
        + 0.213618 * math.sin(2 * mpr)
        - 0.185116 * math.sin(mr)
        - 0.114332 * math.sin(2 * fr)
        + 0.058793 * math.sin(2 * dr - 2 * mpr)
        + 0.057066 * math.sin(2 * dr - mr - mpr)
        + 0.053322 * math.sin(2 * dr + mpr)
        + 0.045758 * math.sin(2 * dr - mr)
        - 0.040923 * math.sin(mr - mpr)
        - 0.034720 * math.sin(dr)
        - 0.030383 * math.sin(mr + mpr)
    )
    latitude = (
        5.128122 * math.sin(fr)
        + 0.280602 * math.sin(mpr + fr)
        + 0.277693 * math.sin(mpr - fr)
        + 0.173237 * math.sin(2 * dr - fr)
        + 0.055413 * math.sin(2 * dr - mpr + fr)
        + 0.046271 * math.sin(2 * dr - mpr - fr)
        + 0.032573 * math.sin(2 * dr + fr)
    )
    distance_km = (
        385000.56
        - 20905.355 * math.cos(mpr)
        - 3699.111 * math.cos(2 * dr - mpr)
        - 2955.968 * math.cos(2 * dr)
        - 569.925 * math.cos(2 * mpr)
    )

    eps = math.radians(23.4392911 - 0.0130042 * t)
    lam, beta = math.radians(longitude % 360.0), math.radians(latitude)
    right_ascension = math.degrees(
        math.atan2(
            math.sin(lam) * math.cos(eps) - math.tan(beta) * math.sin(eps),
            math.cos(lam),
        )
    )
    declination = math.degrees(
        math.asin(
            math.sin(beta) * math.cos(eps) + math.cos(beta) * math.sin(eps) * math.sin(lam)
        )
    )
    local_sidereal = (_greenwich_sidereal_deg(jd) + lon_deg) % 360.0
    elevation, azimuth = _equatorial_to_horizontal(
        right_ascension, declination, lat_deg, local_sidereal
    )

    # Phase from the elongation of the moon from the sun. The separation alone is symmetric
    # about new and full, so the *signed* difference in ecliptic longitude is kept too: positive
    # means the moon is east of the sun, which is the half of the month it is waxing.
    signed_elongation = (longitude - _sun_apparent_long(t) + 180.0) % 360.0 - 180.0
    elongation = math.degrees(
        math.acos(
            max(-1.0, min(1.0, math.cos(beta) * math.cos(math.radians(signed_elongation))))
        )
    )
    phase_angle = 180.0 - elongation - 0.1468 * math.sin(math.radians(elongation))
    phase_angle = max(0.0, min(180.0, phase_angle))
    illuminated = 0.5 * (1.0 + math.cos(math.radians(phase_angle)))

    return MoonPosition(
        elevation_deg=elevation + _refraction_deg(elevation),
        azimuth_deg=azimuth,
        angular_diameter_deg=MOON_ANGULAR_DIAMETER_DEG * MOON_MEAN_DISTANCE_KM / distance_km,
        illuminated_fraction=illuminated,
        phase_angle_deg=phase_angle,
        distance_km=distance_km,
        waxing=signed_elongation > 0.0,
    )


def _sun_apparent_long(t: float) -> float:
    """The sun's apparent ecliptic longitude, degrees -- the reference the lunar phase is off."""
    geom_mean_long = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    m = math.radians(357.52911 + t * (35999.05029 - 0.0001537 * t))
    centre = (
        math.sin(m) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + math.sin(2 * m) * (0.019993 - 0.000101 * t)
        + math.sin(3 * m) * 0.000289
    )
    return (geom_mean_long + centre) % 360.0


def moon_illuminance_lux(moon: MoonPosition) -> float:
    """Illuminance on a horizontal surface from the moon alone, lux.

    Three factors, each of which matters by an order of magnitude or more:

    * **phase**, which is strongly non-linear -- a quarter moon gives about **8 %** of a full one,
      not 50 %, because at full moon the surface is lit straight on and the regolith backscatters
      hard (the opposition surge). Anchored on the published magnitudes: -12.74 at full and -10.0
      at first quarter is 2.74 magnitudes, a ratio of 0.080, which ``exp(-2.52 a / 90)`` gives;
    * **altitude**, as ``sin`` of it, plus the atmospheric extinction a low moon suffers;
    * **distance**, inverse square over the 12 % the orbit varies by.

    Returns 0 with the moon below the horizon. The absolute scale is the textbook 0.267 lux for a
    full moon overhead, so this is a defensible order of magnitude rather than a measurement --
    and the extinction term is normalised **to that geometry**, so the constant means what it says
    rather than being quietly darkened by one airmass of atmosphere it already includes.
    """
    if moon.elevation_deg <= 0.0:
        return 0.0
    phase = math.exp(-2.52 * moon.phase_angle_deg / 90.0)
    sin_el = math.sin(math.radians(moon.elevation_deg))
    # Rayleigh + aerosol extinction along the airmass, as a crude but monotone 0.28 mag/airmass,
    # relative to the zenith where the 0.267 lux figure was measured.
    airmass = 1.0 / max(sin_el, 0.02)
    extinction = 10.0 ** (-0.4 * 0.28 * (min(airmass, 10.0) - 1.0))
    distance = (MOON_MEAN_DISTANCE_KM / moon.distance_km) ** 2
    return _FULL_MOON_ZENITH_LUX * phase * sin_el * extinction * distance


def civil_twilight_fraction(sun_elevation_deg: float) -> float:
    """How much of full daylight is left, as the sun goes down. 1 by day, 0 deep in the night.

    A smooth ramp across the twilight band rather than a switch at the horizon, because the sky's
    brightness falls by four orders of magnitude between sunset and astronomical night and any
    hard cutoff there produces a visible pop in a time-lapse. Crosses 0.5 at about -2 degrees,
    which is roughly where a camera has to change exposure.
    """
    # -0.833 is sunset proper (refraction plus the sun's own semi-diameter); -18 is astronomical
    # night. The exponent shapes the fall; it is chosen to look right, not derived.
    if sun_elevation_deg >= 0.0:
        return 1.0
    if sun_elevation_deg <= -18.0:
        return 0.0
    u = (sun_elevation_deg + 18.0) / 18.0
    return float(u**3.2)
