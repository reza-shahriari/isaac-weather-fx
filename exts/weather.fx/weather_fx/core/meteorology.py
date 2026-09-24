"""Surface meteorology: the diurnal time series a thermal model needs, from one weather state.

A :class:`~weather_fx.core.state.WeatherState` describes **an instant** -- this air temperature,
this dew point, this cloud cover, at this hour on this date. A renderer needs nothing more. A
*thermal* model needs the hours before the instant as well, because a surface's temperature is an
integral of its history: asphalt at noon is hot because of the morning, and the same asphalt under
the same noon sun is cooler on the day after rain.

So this module turns the instant into a series. What is genuinely diurnal is given a diurnal shape
and anchored on the state's own value at the state's own hour; what the state has no basis to vary
is held constant and said so:

===================  =========================================================================
``t_air_k``          sinusoid about a solved mean, peaking at ``T_MAX_SOLAR_HOUR`` solar time,
                     with the swing damped by cloud and by fog; floored at the dew point
``rh_fraction``      derived, from a **constant** dew point and the varying air temperature
``dni_w_m2``         computed per step from the real sun elevation, the turbidity and the cover
``dhi_w_m2``         likewise
``cloud_fraction``   constant (the state carries one cover)
``wind_speed_m_s``   constant (the state carries one wind)
``visibility_m``     constant
``precip_mm_h``      constant, derived from the rain rate and the snow flake population
===================  =========================================================================

**Why the dew point is the constant one and the temperature is not.** Dew point is a property of
the air mass; it moves when the weather changes, not when the sun does. Relative humidity, by
contrast, swings from 40 % at mid-afternoon to saturation before dawn purely because the air
cooled. Holding RH constant and letting the dew point float -- the easy way round -- produces a
night that never reaches saturation and therefore never forms dew or fog, and a thermal model
that consequently never sees the latent heat that stops a surface cooling.

**Why the air temperature is floored at the dew point.** Nocturnal cooling stalls when the surface
layer saturates: further cooling condenses water and releases its latent heat rather than dropping
the temperature. A sinusoid does not know this, and left alone it will drive a humid night 4 K
below the dew point, which is not a state the atmosphere has.

Everything here is pure Python and numpy. A dataset generator calls it in a loop with no renderer
attached, and the infrared simulator that reads this package consumes the columns directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .celestial import SOLAR_CONSTANT_W_M2, sun_position
from .physics import extinction_from_visibility, visibility_from_extinction

__all__ = [
    "COLUMNS",
    "beam_tint",
    "DiurnalSeries",
    "aerosol_visibility_m",
    "airmass",
    "clear_sky_irradiance",
    "diurnal_series",
    "precipitation_rate_mm_h",
    "relative_humidity",
    "saturation_vapour_pressure_hpa",
    "solar_irradiance",
    "surface_visibility_m",
]

#: The columns a surface-meteorology series carries, in a fixed order. These are the standard
#: names, and the infrared simulator downstream uses exactly this set.
COLUMNS: Tuple[str, ...] = (
    "t_air_k",
    "rh_fraction",
    "wind_speed_m_s",
    "cloud_fraction",
    "dni_w_m2",
    "dhi_w_m2",
    "visibility_m",
    "precip_mm_h",
)

#: Broadband atmospheric transmittance at the zenith on a clean, clear day. The airmass law is
#: applied on top of it; this is the anchor. It puts a clear-day zenith sun near 970 W/m2 and a
#: midsummer mid-latitude clear day near 8.5 kWh/m2, which is where pyranometers put them.
CLEAR_SKY_TRANSMITTANCE = 0.72
#: Of the direct beam that the clear atmosphere removes, the fraction that reaches the ground
#: again as diffuse sky. Slightly under a half: scattering is close to symmetric, and absorption
#: by water vapour and ozone takes the rest. Puts clear-sky DHI near 120 W/m2 under a high sun,
#: which is where measurements put it.
DIFFUSE_DOWNWARD_FRACTION = 0.35

#: Kasten-Czeplak: GHI_cloudy / GHI_clear = 1 - A * cover^B. Fitted to Hamburg pyranometer data
#: and used almost universally since; the cube-and-a-bit exponent is why a half-covered sky is
#: barely dimmer than a clear one and a fully covered one is a quarter as bright.
KASTEN_CZEPLAK_A = 0.75
KASTEN_CZEPLAK_B = 3.4

#: Aerosol extinction at the surface as a function of turbidity, 1/m: beta = a * exp(b * T).
#: Anchored on the pairs everyone recognises -- turbidity 2 is a 38 km day, 3 a 25 km day, 5 an
#: 11 km haze, 8 a 3 km one -- with visibility read at the WMO 5 % contrast threshold, which is
#: what :func:`~weather_fx.core.physics.visibility_from_extinction` uses. Read at the 2 %
#: threshold instead, the same aerosol would be quoted 30 % further and every anchor here would
#: be wrong by that much. ESTIMATED: a smooth curve through familiar pairs, not a fit.
AEROSOL_BETA_CLEAN_1_M = 3.36e-5
AEROSOL_BETA_PER_TURBIDITY = 0.424

#: Bulk density of a snowflake aggregate, kg/m3 -- an order of magnitude below solid ice, because
#: a flake is mostly air. Sets the water-equivalent rate the flake population carries.
SNOW_FLAKE_DENSITY_KG_M3 = 100.0

#: Peak-to-trough swing of the air temperature on a clear, calm day at a mid-latitude land site.
#: ESTIMATED.
DIURNAL_SWING_CLEAR_K = 11.0
#: How much of that swing cloud cover removes. Overcast keeps about a third of a clear day's
#: range, which is the classic signature of a cloudy night: it does not get cold.
DIURNAL_CLOUD_DAMPING = 0.65
#: Solar hour of the daily maximum. The sun peaks at 12; the ground and then the air lag it by a
#: couple of hours, because the surface is still gaining heat while the sun is already falling.
T_MAX_SOLAR_HOUR = 14.5


# --- humidity ------------------------------------------------------------------------------


def saturation_vapour_pressure_hpa(temperature_c: Any) -> Any:
    """Magnus-Tetens saturation vapour pressure over liquid water, hPa.

    Valid to about 0.1 % between -40 and +50 degC, which covers every state this package can be
    put in. Over water, not over ice, on both sides of zero: a supercooled surface layer is the
    normal case in a snow regime, and the ice curve would put the dew point in the wrong place.
    """
    t = np.asarray(temperature_c, dtype=np.float64)
    return 6.112 * np.exp(17.62 * t / (243.12 + t))


def relative_humidity(temperature_c: Any, dewpoint_c: Any) -> Any:
    """RH as a fraction, from the temperature and the dew point: e_s(T_d) / e_s(T)."""
    ratio = saturation_vapour_pressure_hpa(dewpoint_c) / saturation_vapour_pressure_hpa(
        temperature_c
    )
    return np.clip(ratio, 0.0, 1.0)


# --- solar ---------------------------------------------------------------------------------


def airmass(elevation_deg: Any) -> Any:
    """Kasten-Young relative optical airmass. 1 at the zenith, about 38 at the horizon.

    The schoolbook ``1 / sin h`` diverges at the horizon, which would make a setting sun
    infinitely attenuated rather than merely red, and would put a hard zero in every irradiance
    series exactly where the interesting part of the day is.
    """
    el = np.maximum(np.asarray(elevation_deg, dtype=np.float64), 0.0)
    return 1.0 / (np.sin(np.radians(el)) + 0.50572 * (el + 6.07995) ** -1.6364)


def clear_sky_irradiance(elevation_deg: Any, turbidity: float) -> Tuple[Any, Any]:
    """Cloudless direct-normal and diffuse-horizontal irradiance, W/m2.

    Beer-Lambert on the airmass for the beam, and a fixed downward share of what the beam lost
    for the diffuse. Deliberately a two-parameter model rather than a spectral one: the thermal
    model downstream integrates it over a day, where the total energy matters and the spectrum
    does not.
    """
    el = np.asarray(elevation_deg, dtype=np.float64)
    optical_depth = -math.log(CLEAR_SKY_TRANSMITTANCE) * (0.7 + 0.12 * float(turbidity))
    beam = SOLAR_CONSTANT_W_M2 * np.exp(-optical_depth * airmass(el))
    dni = np.where(el > 0.0, beam, 0.0)
    sin_el = np.maximum(np.sin(np.radians(el)), 0.0)
    dhi = DIFFUSE_DOWNWARD_FRACTION * np.maximum(SOLAR_CONSTANT_W_M2 * sin_el - dni * sin_el, 0.0)
    return dni, dhi


def solar_irradiance(
    elevation_deg: Any, turbidity: float, cloud_cover: float = 0.0
) -> Tuple[Any, Any]:
    """Direct-normal and diffuse-horizontal irradiance under a partly covered sky, W/m2.

    Two things happen at once under broken cloud and both are needed. The **beam** is simply
    interrupted -- over an hour the sun is out for the uncovered fraction of the time, so
    ``DNI x (1 - cover)`` is the hour's mean. The **global** total falls far more slowly than that,
    because the cloud that blocked the beam is itself a bright scatterer; Kasten-Czeplak pins the
    global, and the diffuse takes up whatever the beam no longer carries.

    That is why a half-covered sky is *brighter* diffusely than a clear one, and why a thermal
    model given only ``GHI x (1 - cover)`` runs a scene several kelvin too cold at midday.
    """
    el = np.asarray(elevation_deg, dtype=np.float64)
    cover = float(np.clip(cloud_cover, 0.0, 1.0))
    dni_clear, dhi_clear = clear_sky_irradiance(el, turbidity)
    sin_el = np.maximum(np.sin(np.radians(el)), 0.0)

    global_clear = dni_clear * sin_el + dhi_clear
    global_cloudy = global_clear * (1.0 - KASTEN_CZEPLAK_A * cover**KASTEN_CZEPLAK_B)
    dni = dni_clear * (1.0 - cover)
    dhi = np.maximum(global_cloudy - dni * sin_el, 0.0)
    return dni, np.minimum(dhi, SOLAR_CONSTANT_W_M2)


#: Representative wavelengths of the linear sRGB primaries, micrometres. Three numbers standing in
#: for three response curves: enough to get the *direction* of the colour shift right, which is
#: what a sunset is, and far cheaper than a spectral render.
RGB_WAVELENGTHS_UM = (0.610, 0.550, 0.465)
#: Luminance weights of those primaries (Rec. 709). The tint is divided by the luminance it
#: carries, so reddening changes the colour of the beam without changing how bright it is -- the
#: brightness is already in :func:`clear_sky_irradiance`, and applying it twice dims every sunset.
RGB_LUMINANCE = (0.2126, 0.7152, 0.0722)
#: Angstrom exponent of the aerosol extinction. 1.3 is the continental value; maritime aerosol is
#: nearer 0.5 (bigger particles, flatter spectrum) and smoke nearer 2.
ANGSTROM_EXPONENT = 1.3
#: Aerosol optical depth at 550 nm, as a function of turbidity: ``clean + per_turbidity * (T - 1)``.
#: **Only the aerosol is spectrally selective here.** The rest of what
#: :data:`CLEAR_SKY_TRANSMITTANCE` accounts for -- water vapour, ozone, the mixed gases -- absorbs
#: broadly across the visible and does not redden anything, so attributing all of it to a
#: wavelength^-1.3 aerosol makes a midday cloud beige. Measured against the rendered gallery: it
#: did. These give AOD 0.075 for very clean air and 0.4 for a thick haze, which is the range
#: sun photometers report.
AEROSOL_OPTICAL_DEPTH_CLEAN = 0.03
AEROSOL_OPTICAL_DEPTH_PER_TURBIDITY = 0.045
#: The illuminant the tint is balanced against: a 45 degree sun through clean air. **Every
#: photograph anyone has seen of a white cloud was white-balanced**, by a camera or by the eye,
#: and the raw beam at midday is already a good deal warmer than the daylight white point -- so a
#: render that skips this step produces clouds that are correctly lit and visibly khaki. Balancing
#: against a fixed daylight reference rather than against the scene's own illuminant is what keeps
#: the sunset: auto white balance would neutralise that too, which is why cameras have a daylight
#: preset and photographers use it at golden hour.
WHITE_BALANCE_ELEVATION_DEG = 45.0
WHITE_BALANCE_TURBIDITY = 2.5


def _rayleigh_optical_depth(wavelength_um: float) -> float:
    """Sea-level Rayleigh optical depth at the zenith (Hansen & Travis 1974)."""
    inverse = wavelength_um**-2
    return 0.008569 * inverse**2 * (1.0 + 0.0113 * inverse + 0.00013 * inverse**2)


def _beam_transmittance(elevation_deg: float, turbidity: float) -> Tuple[float, float, float]:
    """Per-channel transmittance of the direct beam, Rayleigh plus aerosol on the airmass."""
    airmass_value = float(airmass(max(float(elevation_deg), 0.0)))
    aerosol_550 = max(
        AEROSOL_OPTICAL_DEPTH_CLEAN
        + AEROSOL_OPTICAL_DEPTH_PER_TURBIDITY * (float(turbidity) - 1.0),
        0.0,
    )
    out = []
    for wavelength in RGB_WAVELENGTHS_UM:
        aerosol = aerosol_550 * (wavelength / 0.55) ** -ANGSTROM_EXPONENT
        depth = _rayleigh_optical_depth(wavelength) + aerosol
        out.append(math.exp(-depth * airmass_value))
    return tuple(out)


def beam_tint(
    elevation_deg: float, turbidity: float, *, white_balance: bool = True
) -> Tuple[float, float, float]:
    """The colour of the direct beam at this sun or moon elevation, luminance-preserving.

    **This is what makes a sunset orange**, and leaving it out is why a low sun can light a scene
    that is merely dim rather than warm. The beam reaching the ground at four degrees has crossed
    twelve airmasses; Rayleigh scattering alone removes 58 % of the blue against 26 % of the red
    over one, so over twelve the blue is gone and what is left is the colour everyone photographs.
    Clouds do not make themselves orange at sunset -- they are white, and they are lit by that.

    Returned as a multiplier about unit luminance, so it changes the beam's *colour* and not its
    brightness: the brightness is :func:`clear_sky_irradiance`'s answer and applying the extinction
    twice would leave every golden hour underexposed.
    """
    airmass_value = float(airmass(max(float(elevation_deg), 0.0)))
    aerosol_550 = max(
        AEROSOL_OPTICAL_DEPTH_CLEAN
        + AEROSOL_OPTICAL_DEPTH_PER_TURBIDITY * (float(turbidity) - 1.0),
        0.0,
    )

    channels = list(_beam_transmittance(elevation_deg, turbidity))
    if white_balance:
        reference = _beam_transmittance(WHITE_BALANCE_ELEVATION_DEG, WHITE_BALANCE_TURBIDITY)
        channels = [c / r for c, r in zip(channels, reference)]
    luminance = sum(w * c for w, c in zip(RGB_LUMINANCE, channels))
    if luminance <= 0.0:
        # Below any usable elevation the beam is gone; return white rather than a divide.
        return (1.0, 1.0, 1.0)
    return tuple(c / luminance for c in channels)


# --- visibility and precipitation ------------------------------------------------------------


def aerosol_visibility_m(turbidity: float) -> float:
    """Meteorological optical range implied by the sky's turbidity, metres.

    Turbidity and visibility are two views of the same aerosol, and a scene that sets one without
    the other looks internally inconsistent in a way that is hard to name and easy to see: a sky
    the colour of an industrial afternoon with a horizon sharp to fifty kilometres.
    """
    beta = AEROSOL_BETA_CLEAN_1_M * math.exp(AEROSOL_BETA_PER_TURBIDITY * float(turbidity))
    return visibility_from_extinction(beta)


def surface_visibility_m(state: Any) -> float:
    """The state's near-surface meteorological optical range, metres.

    Two contributions, added where extinction adds rather than where visibility does: the aerosol
    that the turbidity implies, and the fog volume if one is enabled.

    **Precipitation is deliberately not a third term.** The ``fog`` section is this package's
    single authority for near-surface extinction, and the presets and the randomiser already lower
    it when they turn rain or snow on -- the rain regime sets a visibility from its own rate. A
    separate precipitation term here would count the same drops twice.
    """
    beta = extinction_from_visibility(aerosol_visibility_m(state.sky.turbidity))
    if getattr(state.fog, "enabled", False):
        beta += extinction_from_visibility(float(state.fog.visibility_m))
    return visibility_from_extinction(beta)


def precipitation_rate_mm_h(state: Any) -> float:
    """Total precipitation rate as **water equivalent**, mm/h.

    Rain carries its rate directly. Snow does not, and does not need to: the flake population is
    fully specified -- a number density, a diameter range and a fall speed -- so the water flux is
    ``n x v x rho_flake x <V>``, which is a derivation rather than another parameter to keep
    consistent with the first one. A flake is mostly air, hence
    :data:`SNOW_FLAKE_DENSITY_KG_M3`; using solid ice here would overstate a snowfall ninefold.
    """
    total = 0.0
    if getattr(state.rain, "enabled", False):
        total += max(float(state.rain.rate_mm_h), 0.0)
    if getattr(state.snow, "enabled", False):
        snow = state.snow
        d_min = max(float(snow.flake_min_diameter_mm), 0.0) * 1e-3
        d_max = max(float(snow.flake_max_diameter_mm), d_min) * 1e-3
        if d_max > d_min:
            # <D^3> over a uniform diameter distribution.
            mean_d3 = (d_max**4 - d_min**4) / (4.0 * (d_max - d_min))
        else:
            mean_d3 = d_min**3
        mean_volume_m3 = math.pi / 6.0 * mean_d3
        flux_kg_m2_s = (
            max(float(snow.number_density_m3), 0.0)
            * max(float(snow.fall_speed_mps), 0.0)
            * SNOW_FLAKE_DENSITY_KG_M3
            * mean_volume_m3
        )
        total += flux_kg_m2_s * 3600.0  # 1 kg/m2 of water is 1 mm.
    return total


# --- the series ------------------------------------------------------------------------------


@dataclass(frozen=True)
class DiurnalSeries:
    """Surface meteorology on a regular grid, in SI units, ready for a thermal solver."""

    epoch_utc: datetime
    time_s: np.ndarray
    t_air_k: np.ndarray
    rh_fraction: np.ndarray
    wind_speed_m_s: np.ndarray
    cloud_fraction: np.ndarray
    dni_w_m2: np.ndarray
    dhi_w_m2: np.ndarray
    visibility_m: np.ndarray
    precip_mm_h: np.ndarray

    @property
    def n(self) -> int:
        return int(self.time_s.size)

    def columns(self) -> Dict[str, np.ndarray]:
        """The eight columns by name -- what a consumer wraps in its own weather type."""
        return {name: getattr(self, name) for name in COLUMNS}

    def describe(self) -> str:
        """One line for a log or a dataset manifest."""
        hours = float(self.time_s[-1] - self.time_s[0]) / 3600.0
        return (
            f"{self.epoch_utc:%Y-%m-%d %H:%M} UTC + {hours:.0f} h, {self.n} samples  "
            f"air {self.t_air_k.min() - 273.15:.1f} .. {self.t_air_k.max() - 273.15:.1f} degC  "
            f"RH {self.rh_fraction.min() * 100:.0f} .. {self.rh_fraction.max() * 100:.0f} %  "
            f"peak DNI {self.dni_w_m2.max():.0f} W/m2  "
            f"cloud {self.cloud_fraction[0] * 100:.0f} %  "
            f"visibility {self.visibility_m[0] / 1000.0:.1f} km"
        )


def _solar_hour(when: datetime, longitude_deg: float) -> float:
    """Mean solar time at a longitude, in hours. Not the equation of time -- 15 minutes of
    correction is well inside the error of a sinusoidal temperature model."""
    utc = when.astimezone(timezone.utc)
    hour = utc.hour + utc.minute / 60.0 + utc.second / 3600.0
    return (hour + float(longitude_deg) / 15.0) % 24.0


def _diurnal_shape(solar_hour: Any) -> Any:
    """-1 at the daily minimum, +1 at the maximum, one cycle a day."""
    return np.sin(2.0 * np.pi * (np.asarray(solar_hour, dtype=np.float64) - T_MAX_SOLAR_HOUR + 6.0) / 24.0)


def diurnal_series(
    state: Any,
    *,
    start_utc: Optional[datetime] = None,
    hours: float = 48.0,
    step_s: float = 1800.0,
    swing_k: Optional[float] = None,
) -> DiurnalSeries:
    """Build a surface-meteorology series around the instant a ``WeatherState`` describes.

    The state's own clock is the **anchor**: the returned air temperature passes through
    ``state.clouds.temperature_c`` at ``state.sky.hour_utc``, so a scene rendered at that instant
    sees exactly the air the state asked for, and the hours on either side are what would have led
    to it. By default the series starts a day earlier, which is enough spin-up for any surface
    thinner than a wall.

    :param start_utc: first sample. Defaults to 24 h before the state's own instant.
    :param hours: total span. :param step_s: sample spacing.
    :param swing_k: override the peak-to-trough temperature range instead of deriving it from
        the cloud and fog.
    """
    if hours <= 0.0 or step_s <= 0.0:
        raise ValueError("hours and step_s must be positive")

    sky, clouds = state.sky, state.clouds
    try:
        day = datetime.strptime(sky.date_utc, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"sky.date_utc must be YYYY-MM-DD, got {sky.date_utc!r}") from exc
    anchor = day + timedelta(hours=float(sky.hour_utc))
    start = (anchor - timedelta(hours=24.0)) if start_utc is None else start_utc
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("start_utc must be timezone-aware")
    start = start.astimezone(timezone.utc)

    n = int(round(hours * 3600.0 / step_s)) + 1
    if n < 2:
        raise ValueError("hours and step_s must give at least two samples")
    time_s = np.arange(n, dtype=np.float64) * float(step_s)
    times = [start + timedelta(seconds=float(t)) for t in time_s]

    cover = float(np.clip(clouds.cover, 0.0, 1.0)) if getattr(clouds, "enabled", False) else 0.0

    # --- air temperature: a sinusoid solved so it passes through the state's own value --------
    if swing_k is None:
        swing = DIURNAL_SWING_CLEAR_K * (1.0 - DIURNAL_CLOUD_DAMPING * cover)
        if getattr(state.fog, "enabled", False):
            # Fog is both a symptom of a small swing and a cause of one: it reflects the sun by
            # day and blocks the radiative loss at night.
            swing *= float(np.clip(float(state.fog.visibility_m) / 2000.0, 0.15, 1.0))
    else:
        swing = float(swing_k)
    if swing < 0.0:
        raise ValueError("swing_k must be non-negative")

    dewpoint_c = float(clouds.dewpoint_c)
    anchor_shape = _diurnal_shape(_solar_hour(anchor, sky.longitude_deg))
    mean_c = float(clouds.temperature_c) - 0.5 * swing * float(anchor_shape)
    solar_hours = np.array([_solar_hour(w, sky.longitude_deg) for w in times], dtype=np.float64)
    t_air_c = mean_c + 0.5 * swing * _diurnal_shape(solar_hours)
    # Nocturnal cooling stalls at saturation; see the module docstring.
    t_air_c = np.maximum(t_air_c, dewpoint_c)

    # --- irradiance: the real sun, step by step ----------------------------------------------
    elevation = np.array(
        [
            sun_position(sky.latitude_deg, sky.longitude_deg, w).elevation_deg
            for w in times
        ],
        dtype=np.float64,
    )
    dni, dhi = solar_irradiance(elevation, float(sky.turbidity), cover)

    ones = np.ones(n, dtype=np.float64)
    return DiurnalSeries(
        epoch_utc=start,
        time_s=time_s,
        t_air_k=t_air_c + 273.15,
        rh_fraction=np.asarray(relative_humidity(t_air_c, dewpoint_c), dtype=np.float64),
        wind_speed_m_s=ones * max(float(state.wind.speed_mps), 0.0),
        cloud_fraction=ones * cover,
        dni_w_m2=np.asarray(dni, dtype=np.float64),
        dhi_w_m2=np.asarray(dhi, dtype=np.float64),
        visibility_m=ones * surface_visibility_m(state),
        precip_mm_h=ones * precipitation_rate_mm_h(state),
    )
