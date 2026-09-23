"""Physical relations used by the effects (all SI units unless noted)."""
from __future__ import annotations

import math

import numpy as np

# Meteorological optical range uses a 5% contrast threshold (WMO).
MOR_CONTRAST_THRESHOLD = 0.05

# Marshall-Palmer drop size distribution: N(D) = N0 * exp(-Lambda * D)
MP_N0 = 8000.0  # m^-3 mm^-1


def extinction_from_visibility(visibility_m: float, contrast: float = MOR_CONTRAST_THRESHOLD) -> float:
    """Koschmieder: beta = -ln(contrast) / V  [1/m]."""
    return -math.log(contrast) / max(visibility_m, 1e-6)


def visibility_from_extinction(beta: float, contrast: float = MOR_CONTRAST_THRESHOLD) -> float:
    return -math.log(contrast) / max(beta, 1e-12)


def transmittance(beta: float, distance_m):
    """Beer-Lambert transmittance along a path."""
    return np.exp(-beta * np.asarray(distance_m, dtype=float))


def marshall_palmer_lambda(rate_mm_h: float) -> float:
    """Slope of the Marshall-Palmer distribution [1/mm]."""
    return 4.1 * max(rate_mm_h, 1e-6) ** -0.21


def rain_number_density(rate_mm_h: float, d_min_mm: float, d_max_mm: float) -> float:
    """Drops per m^3 with diameter in [d_min, d_max]."""
    lam = marshall_palmer_lambda(rate_mm_h)
    return MP_N0 / lam * (math.exp(-lam * d_min_mm) - math.exp(-lam * d_max_mm))


def sample_drop_diameters(rng: np.random.Generator, n: int, rate_mm_h: float,
                          d_min_mm: float, d_max_mm: float) -> np.ndarray:
    """Inverse-CDF sampling of the truncated Marshall-Palmer distribution [mm]."""
    lam = marshall_palmer_lambda(rate_mm_h)
    a, b = math.exp(-lam * d_min_mm), math.exp(-lam * d_max_mm)
    u = rng.random(n)
    return -np.log(a - u * (a - b)) / lam


def raindrop_terminal_velocity(diameter_mm) -> np.ndarray:
    """Atlas et al. (1973) fit to Gunn-Kinzer data [m/s]."""
    d = np.asarray(diameter_mm, dtype=float)
    return np.maximum(9.65 - 10.3 * np.exp(-0.6 * d), 0.0)


def horizontal_axes(up_axis: int):
    return (0, 1) if up_axis == 2 else (0, 2)


def wind_vector(speed_mps: float, direction_deg: float, vertical_mps: float, up_axis: int) -> np.ndarray:
    """Wind velocity in stage axes."""
    a = math.radians(direction_deg)
    h0, h1 = horizontal_axes(up_axis)
    v = np.zeros(3)
    v[h0] = speed_mps * math.cos(a)
    v[h1] = speed_mps * math.sin(a)
    v[up_axis] = vertical_mps
    return v


def gust_factor(t: float, strength: float, period_s: float) -> float:
    """Smooth, deterministic gust multiplier (>= 0)."""
    if strength <= 0.0:
        return 1.0
    w = 2.0 * math.pi / max(period_s, 1e-3)
    wave = 0.6 * math.sin(w * t) + 0.4 * math.sin(w * t / 0.37 + 1.3)
    return max(0.0, 1.0 + strength * wave)
