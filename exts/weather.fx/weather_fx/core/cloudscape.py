"""A cloud layer defined the way real-time volumetric clouds are: a function, not a baked grid.

The layer is ``density(x, y, z)`` built from three small tiling textures and a few lines of
arithmetic, after Schneider's *Nubis* (Horizon Zero Dawn, SIGGRAPH 2015/2017) and the volumetric
clouds of Unreal Engine (Hillaire 2016):

* a **weather map**, two-dimensional and tens of kilometres across: where the clouds are
  (coverage) and how tall each one grows (type);
* a **shape** noise, three-dimensional and a few kilometres across: Perlin-Worley, the rounded
  lobes of a convective cloud;
* a **detail** noise, three-dimensional and a few hundred metres across: Worley, which erodes the
  shape's edge into wisps at the bottom and billows at the top.

Because it is a function, anything can evaluate it at any point and at any resolution: a GPU ray
march per camera pixel (:mod:`weather_fx.gpu.cloud_march`), a voxel export for the path tracer, or
an infrared march that reads the same density as extinction and the height as temperature. That
is what a picture of a sky cannot give. This module is the definition and its numpy reference;
the GPU kernel evaluates the same textures with the same arithmetic, and a test holds the two
together.

Frame: metres, +Y up, the observer's column at x = z = 0, as :class:`~weather_fx.core.clouds.CloudField`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

from . import clouds as _clouds

__all__ = ["CloudscapeProfile", "CLOUDSCAPE_TYPES", "Cloudscape", "sample_wrapped"]


@dataclass(frozen=True)
class CloudscapeProfile:
    """One genus as the function's constants."""

    name: str
    #: Depth of the layer the clouds may occupy, metres.
    thickness_m: float
    #: Visible extinction of a dense core, per metre. Cumulus measure 0.05-0.12.
    extinction_per_m: float
    #: Fraction of the layer the lowest and the tallest clouds reach (the weather map's type
    #: channel runs between them).
    top_min: float
    top_max: float
    #: How much the cloud narrows from its base to its top, 0..1. Cumulus taper; a sheet does not.
    taper: float
    #: Spacing of the clouds in the weather map, metres, and the smallest feature it carries.
    spacing_m: float
    smallest_m: float
    #: How strongly the detail noise eats the edge, in density units.
    erosion: float
    #: Tile of the shape noise, metres. Its first lobes are a quarter of this across.
    shape_tile_m: float
    #: Tile of the detail noise, metres.
    detail_tile_m: float = 520.0
    #: Gain from eroded shape to density: larger gives a harder, more solid cloud.
    gain: float = 3.0
    #: The shape noise's lobes and its finer cells, as divisors of the tile.
    lobe_div: float = 4.0
    fine_div: float = 8.0


CLOUDSCAPE_TYPES: Dict[str, CloudscapeProfile] = {
    "cumulus": CloudscapeProfile(
        name="cumulus", thickness_m=1600.0, extinction_per_m=0.06, top_min=0.5, top_max=1.0,
        taper=0.6, spacing_m=2600.0, smallest_m=500.0, erosion=0.5, shape_tile_m=4200.0,
        lobe_div=6.0, fine_div=12.0),
    "congestus": CloudscapeProfile(
        name="congestus", thickness_m=3800.0, extinction_per_m=0.07, top_min=0.3, top_max=1.0,
        taper=0.45, spacing_m=4200.0, smallest_m=800.0, erosion=0.36, shape_tile_m=6400.0),
    "stratocumulus": CloudscapeProfile(
        name="stratocumulus", thickness_m=550.0, extinction_per_m=0.045, top_min=0.75, top_max=1.0,
        taper=0.15, spacing_m=2200.0, smallest_m=500.0, erosion=0.30, shape_tile_m=3000.0),
    "stratus": CloudscapeProfile(
        name="stratus", thickness_m=400.0, extinction_per_m=0.03, top_min=0.9, top_max=1.0,
        taper=0.0, spacing_m=6000.0, smallest_m=1500.0, erosion=0.12, shape_tile_m=5000.0),
}


def sample_wrapped(texture: np.ndarray, *coords: Any) -> np.ndarray:
    """Linear interpolation of a tiling texture at normalised coordinates, texel centres at
    ``(i + 0.5) / n`` -- the convention of a GPU sampler in wrap mode.

    ``coords`` are ``(u, v)`` for a 2-D texture indexed ``[v, u]`` and ``(u, v, w)`` for a 3-D
    one indexed ``[w, v, u]``. A trailing channel axis on the texture is kept.
    """
    dims = len(coords)
    out = None
    shape = texture.shape[:dims]
    index, weight = [], []
    for axis, c in enumerate(coords):
        n = shape[dims - 1 - axis]
        f = np.asarray(c, dtype=np.float64) * n - 0.5
        i0 = np.floor(f).astype(np.int64)
        index.append((i0 % n, (i0 + 1) % n))
        weight.append(f - i0)
    for corner in range(2**dims):
        w = 1.0
        key = []
        for axis in range(dims):
            bit = (corner >> axis) & 1
            w = w * (weight[axis] if bit else 1.0 - weight[axis])
            key.append(index[axis][bit])
        value = texture[tuple(reversed(key))]
        w = w[..., None] if value.ndim > np.ndim(w) else w
        out = value * w if out is None else out + value * w
    return out


def _remap(v: Any, lo: Any, hi: Any) -> Any:
    """``(v - lo) / (hi - lo)`` clipped to 0..1, with a floor on the span."""
    return np.clip((v - lo) / np.maximum(hi - lo, 1e-4), 0.0, 1.0)


def _smoothstep(lo: float, hi: float, v: Any) -> Any:
    t = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class Cloudscape:
    """One cloud layer: three textures and the constants that combine them."""

    cover: float
    base_m: float
    profile: CloudscapeProfile
    seed: int = 0
    #: Weather-map texels per side and its tile, metres.
    weather_cells: int = 512
    weather_tile_m: float = 32_000.0
    shape_cells: int = 128
    detail_cells: int = 64
    #: (cells, cells, 2) float32: coverage and type, each 0..1.
    weather: np.ndarray = field(init=False, repr=False)
    #: (cells, cells, cells) float32, 0..1.
    shape: np.ndarray = field(init=False, repr=False)
    detail: np.ndarray = field(init=False, repr=False)
    #: The coverage bias the cover solve found (see :meth:`_solve_cover`).
    coverage_bias: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if not 0.0 <= self.cover <= 1.0:
            raise ValueError(f"cover is a fraction of sky, got {self.cover}")
        self._build()

    # --- construction ---------------------------------------------------------------------

    def _build(self) -> None:
        p = self.profile
        n = self.weather_cells
        cell = self.weather_tile_m / n
        flat = (1, n, n)
        size = (cell, cell, cell)
        # Where the clouds are: thermals of several sizes (cellular), modulated by a fractal so
        # some grow and some do not -- the recipe WX.6 measured against real fields' sizes.
        fractal = _clouds._periodic_fractal_noise(
            flat, 2.5, self.seed + 1543, cell_size_m=size, min_wavelength_m=p.smallest_m,
            max_wavelength_m=4.0 * p.spacing_m)
        lattice = max(1, int(round(self.weather_tile_m / p.spacing_m)))
        cells = _clouds._cellular_octaves(flat, lattice, self.seed + 2711)
        coverage = _clouds._rank_normalise(cells * (0.35 + fractal))[0]
        # How tall each one grows: big clouds are tall, with a slow variation across the field.
        slow = _clouds._periodic_fractal_noise(
            flat, 3.0, self.seed + 911, cell_size_m=size, min_wavelength_m=4.0 * p.spacing_m)[0]
        kind = np.clip(0.6 * coverage + 0.4 * slow, 0.0, 1.0)
        self.weather = np.stack([coverage, kind], axis=-1).astype(np.float32)

        m = self.shape_cells
        vox = p.shape_tile_m / m
        cube = (m, m, m)
        perlin = _clouds._periodic_fractal_noise(
            cube, 2.6, self.seed + 17, cell_size_m=(vox, vox, vox),
            min_wavelength_m=p.shape_tile_m / 12.0, max_wavelength_m=p.shape_tile_m / 2.0)
        lobes = _clouds._worley_fbm(cube, (vox, vox, vox), p.shape_tile_m / p.lobe_div, self.seed + 3571)
        fine = _clouds._worley_fbm(cube, (vox, vox, vox), p.shape_tile_m / p.fine_div, self.seed + 4409)
        perlin_worley = _clouds._perlin_worley(perlin, lobes, 1.0)
        self.shape = _remap(perlin_worley, fine - 1.0, 1.0).astype(np.float32)

        k = self.detail_cells
        dv = p.detail_tile_m / k
        self.detail = _clouds._worley_fbm(
            (k, k, k), (dv, dv, dv), p.detail_tile_m / 4.0, self.seed + 7907).astype(np.float32)
        self.coverage_bias = self._solve_cover()

    def _solve_cover(self) -> float:
        """The coverage bias at which the fraction of cloudy columns is the cover asked for.

        A column is cloudy when some level holds density above one half: at these extinctions a
        cloud a camera cannot see through. Measured on a coarse lattice of columns and solved by
        bisection, since more coverage can only cover more columns.
        """
        if self.cover <= 0.0:
            return -1.0
        n = 160
        axis = (np.arange(n) + 0.5) / n * self.weather_tile_m
        x, z = np.meshgrid(axis, axis, indexing="xy")
        heights = self.base_m + (np.arange(14) + 0.5) / 14.0 * self.profile.thickness_m
        lo, hi = -1.0, 1.0
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            cloudy = np.zeros(x.shape, dtype=bool)
            for y in heights:
                cloudy |= self._density(x, y, z, mid) > 0.5
            if float(cloudy.mean()) < self.cover:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # --- the function ---------------------------------------------------------------------

    @property
    def top_m(self) -> float:
        return self.base_m + self.profile.thickness_m

    @property
    def extinction_per_m(self) -> float:
        return float(self.profile.extinction_per_m)

    def density(self, x_m: Any, y_m: Any, z_m: Any) -> np.ndarray:
        """Normalised density 0..1 at a position; multiply by :attr:`extinction_per_m`."""
        if self.cover <= 0.0:
            return np.zeros(np.broadcast(x_m, y_m, z_m).shape)
        return self._density(x_m, y_m, z_m, self.coverage_bias)

    def _density(self, x_m: Any, y_m: Any, z_m: Any, bias: float) -> np.ndarray:
        """The arithmetic the GPU kernel repeats (``gpu/cloud_march.py::cloud_density``)."""
        p = self.profile
        x, y, z = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (x_m, y_m, z_m)))
        h = (y - self.base_m) / p.thickness_m
        inside = (h > 0.0) & (h < 1.0)
        weather = sample_wrapped(self.weather, x / self.weather_tile_m, z / self.weather_tile_m)
        coverage = np.clip(weather[..., 0] + bias, 0.0, 1.0)
        top = p.top_min + (p.top_max - p.top_min) * weather[..., 1]
        hr = h / top
        # A flat base, a rounded top: the height gradient of one cloud.
        gradient = _smoothstep(0.0, 0.07, hr) * (1.0 - _smoothstep(0.35, 1.0, hr))
        coverage = coverage * (1.0 - p.taper * np.clip(hr, 0.0, 1.0) ** 1.5)
        s = 1.0 / p.shape_tile_m
        shape = sample_wrapped(self.shape, x * s, y * s, z * s)
        base = _remap(shape * gradient, 1.0 - coverage, 1.0)
        d = 1.0 / p.detail_tile_m
        detail = sample_wrapped(self.detail, x * d, y * d, z * d)
        # Wisps under the cloud, billows on top of it: the detail inverts with height.
        blend = np.clip(hr * 5.0, 0.0, 1.0)
        modifier = detail * (1.0 - blend) + (1.0 - detail) * blend
        eroded = _remap(base, modifier * p.erosion, 1.0)
        return np.where(inside, np.clip(eroded * p.gain, 0.0, 1.0), 0.0)

    def kernel_constants(self) -> Dict[str, float]:
        """Every scalar the GPU kernel needs, by the names its struct uses."""
        p = self.profile
        return dict(
            base_m=float(self.base_m), thickness_m=float(p.thickness_m),
            weather_tile_m=float(self.weather_tile_m), shape_tile_m=float(p.shape_tile_m),
            detail_tile_m=float(p.detail_tile_m), coverage_bias=float(self.coverage_bias),
            top_min=float(p.top_min), top_max=float(p.top_max), taper=float(p.taper),
            erosion=float(p.erosion), gain=float(p.gain),
            extinction_per_m=float(p.extinction_per_m))

    def measured_cover(self, columns: int = 256) -> float:
        """Fraction of columns holding cloud a camera cannot see through, on a fresh lattice."""
        axis = (np.arange(columns) + 0.25) / columns * self.weather_tile_m
        x, z = np.meshgrid(axis, axis, indexing="xy")
        cloudy = np.zeros(x.shape, dtype=bool)
        for y in self.base_m + (np.arange(20) + 0.5) / 20.0 * self.profile.thickness_m:
            cloudy |= self.density(x, y, z) > 0.5
        return float(cloudy.mean())
