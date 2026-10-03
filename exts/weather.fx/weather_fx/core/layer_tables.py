"""What the per-pixel cloud march needs from the sky model, as small tables a GPU can sample.

The march (:mod:`weather_fx.gpu.cloud_march`) runs per camera pixel per frame; the sky model is
numpy and takes a second per frame at that size. Neither the clear sky nor the air in front of a
cloud changes until the sun moves, so both are tabulated here once per sun position and sampled
on the GPU:

* :func:`sky_table` -- the clear sky (and the ground below the horizon) by elevation and azimuth;
* :func:`air_tables` -- the aerial perspective, in-scatter and transmittance, by elevation,
  azimuth from the light, and distance;
* :func:`lighting_for` -- the direct beam and the hemisphere means that light the cloud.

Rows are packed toward the horizon, where the sky changes fastest: a row's elevation is
``sign(s) s^2 * 90 deg`` for ``s`` in -1..1 (:func:`elevation_of_row`, and its inverse
:func:`row_of_elevation`, which the kernel repeats). Units are the sky model's, cd/m2 before the
dome's exposure.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Tuple

import numpy as np

from . import sky as _sky

__all__ = ["elevation_of_row", "row_of_elevation", "sky_table", "air_tables", "lighting_for",
           "AIR_NEAR_M", "AIR_FAR_M", "lit_body"]

#: The distance axis of the air tables: logarithmic between these, metres.
AIR_NEAR_M = 500.0
AIR_FAR_M = 100_000.0


def elevation_of_row(v: Any) -> np.ndarray:
    """Elevation (radians) at normalised row coordinate ``v`` in 0..1."""
    s = 2.0 * np.asarray(v, dtype=np.float64) - 1.0
    return np.sign(s) * s * s * (math.pi / 2.0)


def row_of_elevation(elevation_rad: Any) -> np.ndarray:
    e = np.asarray(elevation_rad, dtype=np.float64) / (math.pi / 2.0)
    return 0.5 + 0.5 * np.sign(e) * np.sqrt(np.abs(e))


def _direction(elevation: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    """Field frame: +Y up, azimuth 0 toward -Z, 90 toward +X."""
    return np.stack([np.cos(elevation) * np.sin(azimuth), np.sin(elevation),
                     -np.cos(elevation) * np.cos(azimuth)], axis=-1)


def lit_body(conditions: Any):
    """The sun while it is up, else the moon."""
    return conditions.sun if conditions.sun.elevation_deg > 0.0 else conditions.moon


def sky_table(conditions: Any, rows: int = 256) -> np.ndarray:
    """``(rows, 2 rows, 4)`` float32: the clear sky's radiance by elevation row and azimuth column
    (column centre ``(i + 0.5) / (2 rows)`` of a full turn), alpha 1."""
    clear = replace(conditions, cloud=None)
    v = (np.arange(rows) + 0.5) / rows
    u = (np.arange(2 * rows) + 0.5) / (2 * rows)
    el, az = np.meshgrid(elevation_of_row(v), u * 2.0 * math.pi, indexing="ij")
    rgb = _sky.sky_radiance_rgb(_direction(el, az), clear) / max(float(conditions.exposure_scale), 1e-12)
    out = np.ones((rows, 2 * rows, 4), dtype=np.float32)
    out[..., :3] = rgb
    return out


def air_tables(conditions: Any, observer_height_m: float = 2.0, *, elevations: int = 32,
               azimuths: int = 24, distances: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """``(inscatter, transmittance)``, each ``(distances, elevations, azimuths, 4)`` float32.

    Azimuth is measured from the lit body, 0..180 degrees (the air light is symmetric about it);
    distance is logarithmic from :data:`AIR_NEAR_M` to :data:`AIR_FAR_M`.
    """
    body = lit_body(conditions)
    v = (np.arange(elevations) + 0.5) / elevations
    u = (np.arange(azimuths) + 0.5) / azimuths
    w = (np.arange(distances) + 0.5) / distances
    dist = AIR_NEAR_M * (AIR_FAR_M / AIR_NEAR_M) ** w
    d3, el3, az3 = np.meshgrid(dist, elevation_of_row(v),
                               math.radians(body.azimuth_deg) + u * math.pi, indexing="ij")
    inscatter, transmittance = _sky.aerial_perspective_rgb(
        _direction(el3, az3), d3, conditions, observer_height_m=float(observer_height_m))
    a = np.ones(d3.shape + (4,), dtype=np.float32)
    b = np.ones(d3.shape + (4,), dtype=np.float32)
    a[..., :3] = inscatter
    b[..., :3] = transmittance
    return a, b


def lighting_for(conditions: Any) -> dict:
    """The lights of the cloud: ``sun_direction``, ``sun_rgb`` (the beam's illuminance normal to
    itself, the scale the dome's own cloud uses), ``above_rgb`` and ``below_rgb``."""
    body = lit_body(conditions)
    tint = _sky.sun_colour(body.elevation_deg, conditions.turbidity)
    if body is conditions.sun:
        source = 1.6e5 * conditions.daylight * tint
    else:
        mu = max(math.sin(math.radians(max(body.elevation_deg, 1.0))), 0.02)
        source = 1.6 * conditions.moon_lux / mu * np.asarray(_sky.MOONLIGHT_TINT) * tint
    above, below = _sky.cloud_ambient_rgb(conditions)
    return dict(sun_direction=tuple(float(x) for x in body.direction()),
                sun_rgb=tuple(float(x) for x in source),
                above_rgb=tuple(float(x) for x in above), below_rgb=tuple(float(x) for x in below),
                azimuth_rad=math.radians(body.azimuth_deg))
