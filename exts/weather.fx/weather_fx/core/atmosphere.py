"""A physically based atmosphere: the sky over a round planet, and the haze in front of the ground.

This is the model Unreal Engine's *Sky Atmosphere* is built on -- Hillaire, "A Scalable and
Production Ready Sky and Atmosphere Rendering Technique" (EGSR 2020) -- reduced to what an
environment map needs and written in numpy:

1. a **transmittance** table ``T(h, mu)`` from any height, along any zenith cosine, to space;
2. a **multiple-scattering** table ``Psi(h, mu_s)`` (Hillaire section 5.5): every order of
   scattering past the first, from the observation that light scattered twice or more is close to
   isotropic, so the infinite series closes as ``L2 / (1 - f_ms)``;
3. a **sky-view** table: single scattering marched along each view ray plus the ``Psi`` term, over
   view elevation and azimuth *relative to the sun* (the sky is mirror-symmetric about the solar
   vertical, so half the azimuths suffice), with rows packed toward the horizon where the sky
   changes fastest.

What this gives that the analytic Preetham fit cannot: the planet is a sphere, so the sun lights
the upper air after it has set at the ground and twilight comes out of the geometry rather than a
blend; the horizon brightens because a grazing ray crosses a thousand kilometres of air, not
because a fitted exponential says so; ozone makes the twilight zenith blue; and the ground below
the horizon is seen *through* the same air, so it fades into the sky instead of meeting it.

Units. Everything is computed for a unit solar illuminance and scaled by the caller, so the result
is in the photometric units of the illuminance it is scaled by: lux in, cd/m2 out. Channels are
linear sRGB primaries, with the Rayleigh and ozone coefficients quoted at 680, 550 and 440 nm.

Engine-free, like the rest of ``core``. The infrared bands do **not** read this: their extinction
and path radiance are band physics and live in the sensor model. What they share with it is the
state -- turbidity, and the visibility :func:`haze_visibility_m` derives from it -- so a hazier sky
here is a hazier path there.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Tuple

import numpy as np

from weather_fx.core.physics import visibility_from_extinction

__all__ = [
    "AtmosphereParams",
    "SOLAR_ILLUMINANCE_LUX",
    "aerosol_optical_depth",
    "haze_visibility_m",
    "atmosphere_for",
    "transmittance_to_space",
    "SkyView",
    "sky_view",
]

#: Solar illuminance at the top of the atmosphere, lux (the photometric solar constant).
SOLAR_ILLUMINANCE_LUX = 128_000.0

PLANET_RADIUS_M = 6_360_000.0
ATMOSPHERE_TOP_M = 100_000.0

#: Rayleigh scattering at sea level, per metre, at 680/550/440 nm (Bruneton & Neyret 2008).
RAYLEIGH_SCATTERING = np.array([5.802e-6, 13.558e-6, 33.100e-6])
RAYLEIGH_SCALE_HEIGHT_M = 8_000.0
#: Ozone absorption at the layer's peak, per metre (Hillaire 2020, table 1).
OZONE_ABSORPTION = np.array([0.650e-6, 1.881e-6, 0.085e-6])
OZONE_CENTRE_M = 25_000.0
OZONE_HALF_WIDTH_M = 15_000.0
MIE_SCALE_HEIGHT_M = 1_200.0
#: Aerosol single-scattering albedo. Continental aerosol absorbs about a tenth of what it removes.
MIE_SINGLE_SCATTERING_ALBEDO = 0.9
MIE_ASYMMETRY = 0.8

_G = 1  # index of the 550 nm channel


def aerosol_optical_depth(turbidity: float) -> float:
    """Vertical aerosol optical depth at 550 nm from Linke turbidity.

    **ESTIMATED.** Linke turbidity is a broadband ratio of total to clean-dry-air extinction and
    has no single conversion; this linear fit puts a very clear day (T = 2) at 0.05, a typical
    continental day (T = 3) at 0.15 and a hazy one (T = 6) at 0.45, which is where AERONET
    climatologies put those conditions. It is monotonic, which is the property the state relies on.
    """
    return max(0.1 * (float(turbidity) - 1.5), 0.01)


def haze_visibility_m(turbidity: float) -> float:
    """Horizontal meteorological visibility at the ground, metres, implied by the turbidity.

    The one number every band can share: the viewport's aerial-perspective fog is set from it,
    and an infrared sensor model scales its aerosol extinction from it.
    """
    beta = aerosol_optical_depth(turbidity) / MIE_SCALE_HEIGHT_M + RAYLEIGH_SCATTERING[_G]
    # Meteorological optical range: the project's one visibility convention (5 % contrast, WMO).
    return visibility_from_extinction(beta)


@dataclass(frozen=True)
class AtmosphereParams:
    """The medium. Frozen and hashable, so the precomputed tables can be cached on it."""

    mie_scattering: float  # per metre at sea level, grey
    mie_extinction: float
    ground_albedo: Tuple[float, float, float] = (0.16, 0.17, 0.12)
    mie_asymmetry: float = MIE_ASYMMETRY


def atmosphere_for(turbidity: float, ground_albedo: Any = (0.16, 0.17, 0.12)) -> AtmosphereParams:
    """The medium for a turbidity: aerosol loading from :func:`aerosol_optical_depth`."""
    extinction = aerosol_optical_depth(turbidity) / MIE_SCALE_HEIGHT_M
    albedo = tuple(round(float(a), 6) for a in ground_albedo)
    return AtmosphereParams(
        mie_scattering=extinction * MIE_SINGLE_SCATTERING_ALBEDO,
        mie_extinction=extinction,
        ground_albedo=albedo,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- the medium

def _densities(height_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.maximum(height_m, 0.0)
    rayleigh = np.exp(-h / RAYLEIGH_SCALE_HEIGHT_M)
    mie = np.exp(-h / MIE_SCALE_HEIGHT_M)
    ozone = np.maximum(1.0 - np.abs(h - OZONE_CENTRE_M) / OZONE_HALF_WIDTH_M, 0.0)
    return rayleigh, mie, ozone


def _medium(height_m: np.ndarray, atm: AtmosphereParams):
    """Rayleigh scattering, Mie scattering (both ``(..., 3)``) and total extinction ``(..., 3)``."""
    rayleigh, mie, ozone = _densities(height_m)
    s_r = rayleigh[..., None] * RAYLEIGH_SCATTERING
    s_m = (mie * atm.mie_scattering)[..., None] * np.ones(3)
    ext = s_r + (mie * atm.mie_extinction)[..., None] + ozone[..., None] * OZONE_ABSORPTION
    return s_r, s_m, ext


def _rayleigh_phase(cos_theta: np.ndarray) -> np.ndarray:
    return 3.0 / (16.0 * math.pi) * (1.0 + cos_theta * cos_theta)


def _mie_phase(cos_theta: np.ndarray, g: float) -> np.ndarray:
    """Cornette-Shanks: Henyey-Greenstein with the Rayleigh-like correction, normalised."""
    g2 = g * g
    k = 3.0 / (8.0 * math.pi) * (1.0 - g2) / (2.0 + g2)
    return k * (1.0 + cos_theta ** 2) / np.power(1.0 + g2 - 2.0 * g * cos_theta, 1.5)


# --------------------------------------------------------------------------- geometry

def _ray_sphere_far(r: np.ndarray, mu: np.ndarray, radius: float) -> np.ndarray:
    """Distance along a ray from radius ``r`` at zenith cosine ``mu`` to where it leaves a sphere."""
    disc = r * r * (mu * mu - 1.0) + radius * radius
    return -r * mu + np.sqrt(np.maximum(disc, 0.0))


def _hits_ground(r: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return (mu < 0.0) & (r * r * (mu * mu - 1.0) + PLANET_RADIUS_M ** 2 >= 0.0)


def _ray_ground(r: np.ndarray, mu: np.ndarray) -> np.ndarray:
    disc = r * r * (mu * mu - 1.0) + PLANET_RADIUS_M ** 2
    return -r * mu - np.sqrt(np.maximum(disc, 0.0))


# --------------------------------------------------------------------------- transmittance

_T_HEIGHTS = 64
_T_MUS = 256
_T_STEPS = 48


def _mu_to_u(mu: np.ndarray) -> np.ndarray:
    """Pack table columns toward the horizon, where transmittance changes by decades per degree."""
    return 0.5 + 0.5 * np.sign(mu) * np.sqrt(np.abs(mu))


def _u_to_mu(u: np.ndarray) -> np.ndarray:
    v = 2.0 * u - 1.0
    return np.sign(v) * v * v


def _h_to_u(h: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(h / ATMOSPHERE_TOP_M, 0.0, 1.0))


def transmittance_to_space(height_m: Any, mu: Any, atm: AtmosphereParams, steps: int = _T_STEPS) -> np.ndarray:
    """Transmittance ``(..., 3)`` from a height, along a zenith cosine, to the top of the air.

    Integrated directly; the planet is ignored (the ground shadow is a separate test), which is
    what makes one table serve both the sun path and the view path.
    """
    h = np.asarray(height_m, dtype=np.float64)
    m = np.asarray(mu, dtype=np.float64)
    h, m = np.broadcast_arrays(h, m)
    r = PLANET_RADIUS_M + h
    length = _ray_sphere_far(r, m, PLANET_RADIUS_M + ATMOSPHERE_TOP_M)
    s = (np.arange(steps) + 0.5) / steps
    t = length[..., None] * s  # (..., steps)
    rr = np.sqrt(r[..., None] ** 2 + t * t + 2.0 * r[..., None] * m[..., None] * t)
    _, _, ext = _medium(rr - PLANET_RADIUS_M, atm)
    depth = ext.sum(axis=-2) * (length / steps)[..., None]
    return np.exp(-depth)


@lru_cache(maxsize=8)
def _transmittance_table(atm: AtmosphereParams) -> np.ndarray:
    u_h = (np.arange(_T_HEIGHTS) + 0.5) / _T_HEIGHTS
    u_m = (np.arange(_T_MUS) + 0.5) / _T_MUS
    heights = (u_h ** 2) * ATMOSPHERE_TOP_M
    mus = _u_to_mu(u_m)
    hh, mm = np.meshgrid(heights, mus, indexing="ij")
    table = transmittance_to_space(hh, mm, atm)
    table.setflags(write=False)
    return table


def _bilinear(table: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample ``table[(rows, cols, 3)]`` at texel-centre coordinates ``u`` (rows), ``v`` (cols) in [0, 1]."""
    rows, cols = table.shape[:2]
    x = np.clip(u * rows - 0.5, 0.0, rows - 1.0)
    y = np.clip(v * cols - 0.5, 0.0, cols - 1.0)
    x0 = np.minimum(np.floor(x).astype(np.int64), rows - 2) if rows > 1 else np.zeros_like(x, dtype=np.int64)
    y0 = np.minimum(np.floor(y).astype(np.int64), cols - 2) if cols > 1 else np.zeros_like(y, dtype=np.int64)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    a = table[x0, y0]
    b = table[x0 + 1, y0]
    c = table[x0, y0 + 1]
    d = table[x0 + 1, y0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _lookup_transmittance(table: np.ndarray, h: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return _bilinear(table, _h_to_u(h), _mu_to_u(np.clip(mu, -1.0, 1.0)))


def _sun_visibility(r: np.ndarray, mu_s: np.ndarray) -> np.ndarray:
    """0 where the planet is between a point and the sun, 1 elsewhere, softened over the sun's disc."""
    # The cosine of the horizon's zenith angle seen from radius r; the sun is up where mu_s exceeds it.
    horizon = -np.sqrt(np.maximum(1.0 - (PLANET_RADIUS_M / r) ** 2, 0.0))
    half_disc = math.radians(0.27)
    return np.clip((mu_s - horizon) / half_disc * 0.5 + 0.5, 0.0, 1.0)


# --------------------------------------------------------------------------- multiple scattering

_MS_SIZE = 24
_MS_DIRECTIONS = 64
_MS_STEPS = 20


def _sphere_directions(n: int) -> np.ndarray:
    """``n`` near-uniform unit vectors (a Fibonacci sphere), y up."""
    i = np.arange(n) + 0.5
    y = 1.0 - 2.0 * i / n
    radius = np.sqrt(1.0 - y * y)
    phi = math.pi * (3.0 - math.sqrt(5.0)) * i
    return np.stack([radius * np.cos(phi), y, radius * np.sin(phi)], axis=-1)


@lru_cache(maxsize=8)
def _multiscattering_table(atm: AtmosphereParams) -> np.ndarray:
    """``Psi(h, mu_s)``: the luminance of all scattering orders past the first, for unit sun.

    Hillaire 2020 section 5.5. At each table point, march a sphere of directions: ``L2`` is the
    second-order luminance reaching the point (single scattering with the isotropic phase, plus
    the lit ground), ``f_ms`` the fraction of isotropic light the surroundings send back. Each
    further order is the previous one times ``f_ms``, so the whole series is ``L2 / (1 - f_ms)``.
    """
    trans = _transmittance_table(atm)
    u = (np.arange(_MS_SIZE) + 0.5) / _MS_SIZE
    heights = (u ** 2) * ATMOSPHERE_TOP_M          # rows
    mu_s = 2.0 * u - 1.0                            # columns
    dirs = _sphere_directions(_MS_DIRECTIONS)       # (D, 3)
    albedo = np.asarray(atm.ground_albedo)
    iso = 1.0 / (4.0 * math.pi)

    hh, ms = np.meshgrid(heights, mu_s, indexing="ij")          # (H, S)
    r0 = PLANET_RADIUS_M + hh[..., None]                        # (H, S, 1)
    # Sun in the x-y plane; view directions from the sphere.
    sun = np.stack([np.sqrt(np.maximum(1.0 - ms ** 2, 0.0)), ms, np.zeros_like(ms)], axis=-1)
    mu_v = dirs[:, 1][None, None, :] * np.ones_like(r0)         # (H, S, D)
    ground = _hits_ground(r0, mu_v)
    top = _ray_sphere_far(r0, mu_v, PLANET_RADIUS_M + ATMOSPHERE_TOP_M)
    length = np.where(ground, _ray_ground(r0, mu_v), top)

    luminance = np.zeros(mu_v.shape + (3,))
    f_ms = np.zeros(mu_v.shape + (3,))
    throughput = np.ones(mu_v.shape + (3,))
    dt = length / _MS_STEPS
    cos_sv = np.einsum("hsk,dk->hsd", sun, dirs)  # (H, S, D)
    for i in range(_MS_STEPS):
        t = (i + 0.5) * dt
        r = np.sqrt(r0 ** 2 + t * t + 2.0 * r0 * mu_v * t)
        h = r - PLANET_RADIUS_M
        # The zenith at the sample has turned; the sun's cosine there follows from the geometry.
        mu_sun = (r0 * sun[..., 1:2] + t * cos_sv) / r
        s_r, s_m, ext = _medium(h, atm)
        scat = s_r + s_m
        step_t = np.exp(-ext * dt[..., None])
        t_sun = _lookup_transmittance(trans, h, mu_sun) * _sun_visibility(r, mu_sun)[..., None]
        s = t_sun * scat * iso
        safe = np.maximum(ext, 1e-12)
        luminance += throughput * (s - s * step_t) / safe
        f_ms += throughput * (scat - scat * step_t) / safe
        throughput *= step_t

    # The lit ground at the end of the rays that reach it.
    r_g = np.full_like(r0, PLANET_RADIUS_M)
    mu_sun_g = (r0 * sun[..., 1:2] + length * cos_sv) / PLANET_RADIUS_M
    t_sun_g = _lookup_transmittance(trans, np.zeros_like(mu_sun_g), mu_sun_g)
    lit = np.clip(mu_sun_g, 0.0, 1.0)[..., None] * t_sun_g * albedo / math.pi
    luminance += np.where(ground[..., None], throughput * lit, 0.0)
    del r_g

    weight = 4.0 * math.pi / _MS_DIRECTIONS * iso
    l2 = luminance.sum(axis=2) * weight
    fms = f_ms.sum(axis=2) * weight
    table = l2 / np.maximum(1.0 - fms, 1e-3)
    table.setflags(write=False)
    return table


def _lookup_multiscattering(table: np.ndarray, h: np.ndarray, mu_s: np.ndarray) -> np.ndarray:
    return _bilinear(table, _h_to_u(h), 0.5 * (np.clip(mu_s, -1.0, 1.0) + 1.0))


# --------------------------------------------------------------------------- the sky view

@dataclass
class SkyView:
    """Luminance of the sky and the ground, for unit illuminance from one body.

    Sample it with :meth:`radiance`. The table covers view elevation (rows, packed toward the
    horizon) by azimuth from the body (columns, 0 to 180 degrees).
    """

    table: np.ndarray               # (rows, cols, 3)
    horizon_elevation: float        # radians; slightly negative, the dip at the observer's height
    irradiance: np.ndarray          # (3,) diffuse illuminance on a horizontal plane, unit sun

    def radiance(self, elevation: np.ndarray, relative_azimuth: np.ndarray) -> np.ndarray:
        return _bilinear(self.table, _elevation_to_u(elevation, self.horizon_elevation),
                         np.abs(np.angle(np.exp(1j * relative_azimuth))) / math.pi)

    def horizon_colour(self) -> np.ndarray:
        """Mean luminance of the band just above the horizon, all round: the haze colour."""
        az = np.linspace(0.0, math.pi, 33)
        el = np.full_like(az, self.horizon_elevation + math.radians(1.0))
        return self.radiance(el, az).mean(axis=0)


def _elevation_to_u(elevation: np.ndarray, horizon: float) -> np.ndarray:
    """Hillaire's non-linear latitude: rows packed quadratically toward the horizon."""
    e = np.asarray(elevation, dtype=np.float64) - horizon
    above = np.clip(e / (0.5 * math.pi - horizon), 0.0, 1.0)
    below = np.clip(-e / (0.5 * math.pi + horizon), 0.0, 1.0)
    return np.where(e >= 0.0, 0.5 + 0.5 * np.sqrt(above), 0.5 - 0.5 * np.sqrt(below))


def _u_to_elevation(u: np.ndarray, horizon: float) -> np.ndarray:
    above = u >= 0.5
    a = (2.0 * u - 1.0) ** 2 * (0.5 * math.pi - horizon)
    b = (1.0 - 2.0 * u) ** 2 * (0.5 * math.pi + horizon)
    return horizon + np.where(above, a, -b)


_SV_ROWS = 128
_SV_COLS = 64
_SV_STEPS = 40


@lru_cache(maxsize=16)
def _sky_view_cached(atm: AtmosphereParams, body_elevation_q: float, observer_m_q: float) -> SkyView:
    return _compute_sky_view(atm, body_elevation_q, observer_m_q)


def sky_view(atm: AtmosphereParams, body_elevation_deg: float, observer_height_m: float = 2.0) -> SkyView:
    """The sky for a body at an elevation, seen from a height. Cached on hundredths of a degree."""
    return _sky_view_cached(atm, round(float(body_elevation_deg), 2),
                            round(max(float(observer_height_m), 0.5), 1))


def _compute_sky_view(atm: AtmosphereParams, body_elevation_deg: float, observer_m: float) -> SkyView:
    trans = _transmittance_table(atm)
    ms_table = _multiscattering_table(atm)
    r0 = PLANET_RADIUS_M + observer_m
    horizon = -math.acos(min(PLANET_RADIUS_M / r0, 1.0))

    u = (np.arange(_SV_ROWS) + 0.5) / _SV_ROWS
    v = (np.arange(_SV_COLS) + 0.5) / _SV_COLS
    elevation = _u_to_elevation(u, horizon)
    azimuth = v * math.pi
    el, az = np.meshgrid(elevation, azimuth, indexing="ij")
    mu_v = np.sin(el)
    view = np.stack([np.cos(el) * np.cos(az), mu_v, np.cos(el) * np.sin(az)], axis=-1)
    sun_el = math.radians(body_elevation_deg)
    sun = np.array([math.cos(sun_el), math.sin(sun_el), 0.0])
    cos_sv = view @ sun

    r0a = np.full(mu_v.shape, r0)
    ground = _hits_ground(r0a, mu_v)
    top = _ray_sphere_far(r0a, mu_v, PLANET_RADIUS_M + ATMOSPHERE_TOP_M)
    length = np.where(ground, _ray_ground(r0a, mu_v), top)

    phase_r = _rayleigh_phase(cos_sv)[..., None]
    phase_m = _mie_phase(cos_sv, atm.mie_asymmetry)[..., None]
    luminance = np.zeros(mu_v.shape + (3,))
    throughput = np.ones(mu_v.shape + (3,))
    # Quadratic step distribution: short steps in the dense air near the observer, long ones out
    # where a grazing ray spends hundreds of kilometres in almost nothing.
    edges = (np.arange(_SV_STEPS + 1) / _SV_STEPS) ** 2
    for i in range(_SV_STEPS):
        t0 = edges[i] * length
        t1 = edges[i + 1] * length
        dt = t1 - t0
        t = 0.5 * (t0 + t1)
        r = np.sqrt(r0 ** 2 + t * t + 2.0 * r0 * mu_v * t)
        h = r - PLANET_RADIUS_M
        mu_sun = (r0 * sun[1] + t * cos_sv) / r
        s_r, s_m, ext = _medium(h, atm)
        step_t = np.exp(-ext * dt[..., None])
        t_sun = _lookup_transmittance(trans, h, mu_sun) * _sun_visibility(r, mu_sun)[..., None]
        psi = _lookup_multiscattering(ms_table, h, mu_sun)
        s = t_sun * (s_r * phase_r + s_m * phase_m) + psi * (s_r + s_m)
        luminance += throughput * (s - s * step_t) / np.maximum(ext, 1e-12)
        throughput *= step_t

    # Diffuse illuminance on the ground from the sky above it, from the rows above the horizon.
    above = elevation > horizon
    d_el = np.gradient(elevation)
    weight = (np.cos(elevation) * np.sin(np.clip(elevation, 0.0, None)) * d_el)[:, None] * (math.pi / _SV_COLS)
    weight = np.where(above[:, None], weight, 0.0) * 2.0  # two mirror halves of the azimuth
    irradiance = (luminance * weight[..., None]).sum(axis=(0, 1))

    # The ground at the end of the rays that reach it: sunlit and skylit Lambertian terrain, seen
    # through the air in between -- which is what fades it into the horizon.
    mu_sun_g = (r0 * sun[1] + length * cos_sv) / PLANET_RADIUS_M
    t_sun_g = _lookup_transmittance(trans, np.zeros_like(mu_sun_g), mu_sun_g)
    t_sun_g = t_sun_g * _sun_visibility(np.full_like(mu_sun_g, PLANET_RADIUS_M), mu_sun_g)[..., None]
    albedo = np.asarray(atm.ground_albedo)
    lit = (np.clip(mu_sun_g, 0.0, 1.0)[..., None] * t_sun_g + irradiance) * albedo / math.pi
    luminance += np.where(ground[..., None], throughput * lit, 0.0)

    luminance.setflags(write=False)
    return SkyView(table=luminance, horizon_elevation=horizon, irradiance=irradiance)
