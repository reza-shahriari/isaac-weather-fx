"""What a cloud field looks like from above, measured the way satellite studies measure real clouds.

The shape of a cloud is a requirement on the field, not a matter of taste: real cumulus have an
outline dimension, a size distribution and an edge, all measured, and a field that misses them looks
wrong in every band that reads it. This module measures a :class:`~weather_fx.core.clouds.CloudField`
the way those studies measure satellite imagery: a plan-view mask of the columns a camera could not
see through, cut into separate clouds, each with an area and a perimeter.

* :func:`area_perimeter_dimension` -- ``P ~ A^(D/2)`` across clouds. 1 for smooth discs; 1.35 for
  real clouds from 1 to 10^6 km^2 (Lovejoy 1982), 1.28 for trade cumulus at 15 m (Zhao & Di
  Girolamo 2007).
* :func:`size_exponent` -- ``n(l) ~ l^-b`` with ``l = sqrt(A)``, fitted by maximum likelihood
  between two sizes. Real shallow cumulus give 1.7-2.0 below a break at 0.5-1 km (Neggers et al.
  2003; Benner & Curry 1998).
* :func:`stripe_ratio` -- the spectral power of the mask along one axis over its mean at the same
  wavelengths. 1 for an isotropic field; a field streaked along that axis puts its power there.
* :func:`fringe_width_m` -- the mean width of the band in which a column goes from optical depth
  0.1 to 1, the part of a cloud a camera sees as an edge.

Everything is plain numpy and periodic: the field wraps, so a cloud that crosses the tile's edge is
one cloud, not two cut halves.
"""
from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np

__all__ = [
    "MASK_OPTICAL_DEPTH",
    "plan_optical_depth",
    "label_periodic",
    "areas_and_perimeters",
    "area_perimeter_dimension",
    "size_exponent",
    "stripe_ratio",
    "fringe_width_m",
]

#: A column is cloud in the mask when its visible optical depth reaches 1: the sky behind it is
#: dimmed to a third, which is roughly where a satellite cloud mask calls a pixel cloudy.
MASK_OPTICAL_DEPTH = 1.0


def plan_optical_depth(field: Any, pitch_m: float) -> np.ndarray:
    """Vertical visible optical depth of every column of one tile, on a square grid ``pitch_m`` apart.

    Rows run along the field's z axis and columns along x, starting at the tile's corner. Each
    column is summed at the field's own finest vertical pitch, through ``density``, so the edge
    detail a camera sees is in it.
    """
    tile = float(field.cells * field.cell_m)
    n = int(round(tile / pitch_m))
    axis = -0.5 * tile + (np.arange(n) + 0.5) * (tile / n)
    dv = float(field.finest_pitch_m)
    heights = field.base_m + (np.arange(int(math.ceil(field.thickness_m / dv))) + 0.5) * dv
    tau = np.zeros((n, n))
    for start in range(0, n, 128):
        z, x = np.meshgrid(axis[start:start + 128], axis, indexing="ij")
        for y in heights:
            tau[start:start + 128] += field.density(x, y, z)
    return tau * dv * float(field.extinction_per_m)


def label_periodic(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Four-connected components of a mask that wraps on both axes.

    Returns labels 1..n (0 outside the mask) and n. Each pixel starts as its own label and takes
    the smallest of its neighbours', with pointer jumping so a long cloud converges in a few dozen
    passes rather than one per pixel of its length.
    """
    mask = np.asarray(mask, dtype=bool)
    size = mask.size
    labels = np.where(mask, np.arange(size, dtype=np.int64).reshape(mask.shape), size)
    while True:
        smallest = labels
        for axis in (0, 1):
            for shift in (1, -1):
                smallest = np.minimum(smallest, np.roll(labels, shift, axis=axis))
        flat = np.where(mask, smallest, size).reshape(-1)
        for _ in range(4):
            inside = flat < size
            followed = flat.copy()
            followed[inside] = flat[flat[inside]]
            flat = np.minimum(flat, followed)
        updated = flat.reshape(mask.shape)
        if np.array_equal(updated, labels):
            break
        labels = updated
    roots, index = np.unique(labels[mask], return_inverse=True)
    out = np.zeros(mask.shape, dtype=np.int64)
    out[mask] = index + 1
    return out, int(roots.size)


def areas_and_perimeters(labels: np.ndarray, count: int, pitch_m: float):
    """Area (m^2) and perimeter (m) of each labelled cloud; the perimeter counts pixel edges."""
    area = np.bincount(labels.reshape(-1), minlength=count + 1)[1:] * pitch_m**2
    edges = np.zeros(count + 1)
    for axis in (0, 1):
        neighbour = np.roll(labels, 1, axis=axis)
        differ = labels != neighbour
        edges += np.bincount(labels[differ], minlength=count + 1)
        edges += np.bincount(neighbour[differ], minlength=count + 1)
    return area.astype(np.float64), edges[1:] * pitch_m


def area_perimeter_dimension(area: np.ndarray, perimeter: np.ndarray, pitch_m: float,
                             min_pixels: int = 10) -> float:
    """``D`` in ``P ~ A^(D/2)``, by least squares in log-log over clouds of at least ``min_pixels``.

    Pixel-edge perimeters overstate a smooth outline by up to 4/pi, but by the same factor at every
    size, so the slope -- which is all ``D`` is -- is unaffected.
    """
    keep = area >= min_pixels * pitch_m**2
    if np.count_nonzero(keep) < 3:
        return float("nan")
    return 2.0 * float(np.polyfit(np.log(area[keep]), np.log(perimeter[keep]), 1)[0])


def size_exponent(sizes_m: np.ndarray, smallest_m: float, largest_m: float) -> Tuple[float, int]:
    """Maximum-likelihood ``b`` for ``n(l) ~ l^-b`` truncated to ``[smallest_m, largest_m]``.

    Returns ``(b, clouds used)``. A fit to a binned histogram leans on its sparsest bins; the
    likelihood weighs every cloud once. The truncated power law's mean log size is monotone in
    ``b``, so ``b`` is found by bisection on it.
    """
    l = np.asarray(sizes_m, dtype=np.float64)
    l = l[(l >= smallest_m) & (l <= largest_m)]
    if l.size < 2:
        return float("nan"), int(l.size)
    observed = float(np.mean(np.log(l)))
    lo_log, hi_log = math.log(smallest_m), math.log(largest_m)

    def mean_log(b: float) -> float:
        a = 1.0 - b
        if abs(a) < 1e-9:
            return 0.5 * (lo_log + hi_log)
        lo_a, hi_a = smallest_m**a, largest_m**a
        return (hi_a * hi_log - lo_a * lo_log) / (hi_a - lo_a) - 1.0 / a

    low, high = -2.0, 8.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if mean_log(mid) > observed:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high), int(l.size)


def stripe_ratio(mask: np.ndarray, pitch_m: float, axis: int = 0, shortest_m: float = 500.0,
                 longest_m: float = 5000.0, half_width_deg: float = 10.0) -> float:
    """How much of the mask's structure is streaked along ``axis``.

    The mean spectral power of the mask's wavevectors within ``half_width_deg`` of perpendicular to
    ``axis`` -- the ones that carry structure uniform along it -- over the mean power of all
    wavevectors with the same range of wavelengths. An isotropic field gives 1 within its own
    sampling noise (0.84-1.23 over four seeds of 35 % cumulus); a field of streaks along ``axis``
    gives several.
    """
    m = np.asarray(mask, dtype=np.float64)
    m = m - m.mean()
    power = np.abs(np.fft.fft2(m)) ** 2
    k0 = np.fft.fftfreq(m.shape[0], pitch_m)[:, None] * np.ones((1, m.shape[1]))
    k1 = np.fft.fftfreq(m.shape[1], pitch_m)[None, :] * np.ones((m.shape[0], 1))
    k = np.hypot(k0, k1)
    ring = (k >= 1.0 / longest_m) & (k <= 1.0 / shortest_m)
    along = np.abs(k0 if axis == 0 else k1)
    angle = np.degrees(np.arcsin(np.clip(along / np.maximum(k, 1e-30), 0.0, 1.0)))
    line = ring & (angle <= half_width_deg)
    return float(power[line].mean() / power[ring].mean())


def fringe_width_m(optical_depth: np.ndarray, pitch_m: float, labels: np.ndarray,
                   count: int) -> float:
    """Mean width of a cloud's fringe: columns between optical depth 0.1 and 1, per metre of outline.

    The area of the fringe divided by the length of the clouds' outlines. A hard stencil gives
    zero; a cloud whose edge is a ramp gives the ramp's width.
    """
    fringe = (optical_depth >= 0.1) & (optical_depth < MASK_OPTICAL_DEPTH)
    _, perimeter = areas_and_perimeters(labels, count, pitch_m)
    total = float(perimeter.sum())
    if total <= 0.0:
        return float("nan")
    return float(np.count_nonzero(fringe)) * pitch_m**2 / total
