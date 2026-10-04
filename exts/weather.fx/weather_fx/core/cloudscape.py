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

import dataclasses
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import clouds as _clouds

__all__ = ["CloudscapeProfile", "CLOUDSCAPE_TYPES", "Cloudscape", "sample_wrapped",
           "CLOUDSCAPE_SHAPE_KEYS", "cloudscape_from_state", "supports", "CloudPatch", "load_patches",
           "PATCH_DIRECTORIES"]

#: Where simulated cloud patches are looked for: the ones shipped with the extension, then the
#: ones ``tools/simulate_cloud_patches.py`` wrote for this user.
PATCH_DIRECTORIES = (pathlib.Path(__file__).resolve().parents[1] / "data" / "cloud_patches",
                     pathlib.Path.home() / ".cache" / "weather_fx" / "cloud_patches")
#: Empty margin each patch keeps on either side along the atlas's x axis, as a fraction of its
#: slot, so linear filtering never mixes two patches.
ATLAS_MARGIN = 0.02
#: How much of a patch's box its cloud fills across: the part that must stay inside a lattice cell.
PATCH_FOOTPRINT = 0.9
#: Tile of the coarser of the two noises that erode a patch's skin, metres; the finer is a quarter.
EROSION_TILE_M = 230.0
#: The skin erosion's strength swings over this many erosion tiles, between these two factors.
RAGGED_TILES = 6.0
#: The small clouds: their lattice's period as a share of the large one's, and how much further
#: into the weather map it reaches than the large clouds do.
SMALL_PERIOD = 0.4
SMALL_COVER = 1.5
RAGGED_LOW = 0.25
RAGGED_HIGH = 1.8


@dataclass(frozen=True)
class CloudPatch:
    """One simulated patch of cloud: density 0..1 on cubic cells, axis 1 up, base at index 0."""

    name: str
    density: np.ndarray
    voxel_m: float


def load_patches(directory: Any = None) -> List[CloudPatch]:
    """The patches in a directory (``*.npz`` from ``tools/simulate_cloud_patches.py``), by name;
    with no directory, those of the first of :data:`PATCH_DIRECTORIES` that holds any (a user's
    folder stands in for the shipped set only when that set is missing, never mixes into it).
    An empty list if there are none."""
    directories = [pathlib.Path(directory).expanduser()] if directory else list(PATCH_DIRECTORIES)
    found: Dict[str, CloudPatch] = {}
    for folder in directories:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.npz")):
            with np.load(path) as data:
                density = np.asarray(data["density"], dtype=np.float32)
                found[path.stem] = CloudPatch(path.stem, density / max(float(density.max()), 1e-9),
                                              float(data["voxel_m"]))
        if found:
            break
    return [found[name] for name in sorted(found)]


def load_towers(directory: Any = None) -> List[CloudPatch]:
    """The towering patches: the ``towers`` folder beside the patches :func:`load_patches` reads."""
    directories = [pathlib.Path(directory).expanduser()] if directory else list(PATCH_DIRECTORIES)
    for folder in directories:
        if folder.is_dir() and any(folder.glob("*.npz")):
            return load_patches(folder / "towers") if (folder / "towers").is_dir() else []
    return []


def _cell_hash(i: np.ndarray, j: np.ndarray, k: int) -> np.ndarray:
    """``gpu.cloud_march.cell_hash``: a number in 0..1 per lattice cell and channel."""
    with np.errstate(over="ignore"):
        i = np.asarray(i).astype(np.int64).astype(np.uint32)
        j = np.asarray(j).astype(np.int64).astype(np.uint32)
        n = i * np.uint32(73856093) ^ j * np.uint32(19349663) ^ np.uint32(k) * np.uint32(83492791)
        n = (n << np.uint32(13)) ^ n
        n = n * (n * n * np.uint32(15731) + np.uint32(789221)) + np.uint32(1376312589)
    return (n & np.uint32(0x7FFFFFFF)).astype(np.float64) / 2147483648.0


def _resize(a: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    """``a`` resampled to ``shape`` by linear interpolation along each axis in turn."""
    for axis, n in enumerate(shape):
        m = a.shape[axis]
        if m == n:
            continue
        at = (np.arange(n) + 0.5) * m / n - 0.5
        lo = np.clip(np.floor(at).astype(np.int64), 0, m - 1)
        hi = np.clip(lo + 1, 0, m - 1)
        f = np.clip(at - lo, 0.0, 1.0).astype(np.float32)
        f = f.reshape([-1 if k == axis else 1 for k in range(a.ndim)])
        a = np.take(a, lo, axis=axis) * (1.0 - f) + np.take(a, hi, axis=axis) * f
    return a


def _sample_clamped(texture: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Linear interpolation of a 3-D texture indexed ``[w, v, u]``, clamped at its edges."""
    out = 0.0
    index, weight = [], []
    for c, n in ((u, texture.shape[2]), (v, texture.shape[1]), (w, texture.shape[0])):
        f = np.clip(np.asarray(c, dtype=np.float64) * n - 0.5, 0.0, n - 1.0)
        i0 = np.minimum(np.floor(f).astype(np.int64), n - 2) if n > 1 else np.zeros(np.shape(f), np.int64)
        index.append((i0, np.minimum(i0 + 1, n - 1)))
        weight.append(f - i0)
    for corner in range(8):
        wgt = 1.0
        key = []
        for axis in range(3):
            bit = (corner >> axis) & 1
            wgt = wgt * (weight[axis] if bit else 1.0 - weight[axis])
            key.append(index[axis][bit])
        out = out + texture[key[2], key[1], key[0]] * wgt
    return out


@dataclass(frozen=True)
class CloudscapeProfile:
    """One genus as the function's constants."""

    name: str
    #: Depth of the layer the clouds may occupy, metres.
    thickness_m: float
    #: Visible extinction where the cloud holds the most water, at its top, per metre. Cumulus
    #: measure 0.05-0.12 in their cores; the base holds :attr:`water_base` of it.
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
    #: How fast density rises inside the cloud's outline, as ``1 - exp(-edge * depth)``: small is
    #: a soft, ragged skin (the base and the sides low down), large is a crisp one (the rising
    #: top). Between them the value follows the height, from ``crisp_from`` to ``crisp_to``.
    edge_base: float = 2.5
    edge_top: float = 40.0
    crisp_from: float = 0.1
    crisp_to: float = 0.55
    #: Simulated patches: the lattice's cell, metres (one patch, a group of clouds, per cell), and
    #: how much of a patch's density the small noise can eat at its base.
    patch_period_m: float = 3200.0
    #: How much of its lattice cell a cloud may fill, 0..1: with the period, the gap between clouds.
    patch_fill: float = 1.0
    #: The share of the fine lattice's cells that hold a small cloud.
    small_keep: float = 0.45
    #: The share of the large clouds that are towers (the patches in ``towers/``), 0..1.
    towers: float = 0.0
    patch_erosion: float = 0.5
    #: Liquid water at the base as a fraction of the top's. Lifted air condenses more the higher
    #: it goes (the adiabat), so a cloud is thin and grey at its base and dense at its top.
    water_base: float = 0.25
    #: The shape noise's lobes and its finer cells, as divisors of the tile.
    lobe_div: float = 4.0
    fine_div: float = 8.0


CLOUDSCAPE_TYPES: Dict[str, CloudscapeProfile] = {
    "cumulus": CloudscapeProfile(
        name="cumulus", thickness_m=1600.0, extinction_per_m=0.12, top_min=0.5, top_max=1.0,
        taper=0.6, spacing_m=2600.0, smallest_m=500.0, erosion=0.5, shape_tile_m=4200.0,
        lobe_div=6.0, fine_div=12.0),
    "congestus": CloudscapeProfile(
        name="congestus", towers=0.7, thickness_m=3800.0, extinction_per_m=0.07, top_min=0.3, top_max=1.0,
        taper=0.45, spacing_m=4200.0, smallest_m=800.0, erosion=0.36, shape_tile_m=6400.0),
    "stratocumulus": CloudscapeProfile(
        name="stratocumulus", thickness_m=550.0, extinction_per_m=0.045, top_min=0.75, top_max=1.0,
        taper=0.15, spacing_m=2200.0, smallest_m=500.0, erosion=0.30, shape_tile_m=3000.0),
    # A storm deck over the whole sky. What one sees from below is how thick it is: where the
    # deck is shallow it glows, under its deep cells it is dark, so its depth swings widely from
    # place to place while its base stays on one level.
    "storm": CloudscapeProfile(
        name="storm", thickness_m=3000.0, extinction_per_m=0.035, top_min=0.1, top_max=1.0,
        taper=0.25, spacing_m=3000.0, smallest_m=600.0, erosion=0.5, shape_tile_m=5200.0,
        lobe_div=5.0, fine_div=11.0, edge_base=6.0, water_base=0.7),
    "stratus": CloudscapeProfile(
        name="stratus", thickness_m=400.0, extinction_per_m=0.03, top_min=0.9, top_max=1.0,
        taper=0.0, spacing_m=6000.0, smallest_m=1500.0, erosion=0.12, shape_tile_m=5000.0),
}


#: Vertical optical depth above which a column counts toward the cover: a seventh of the light
#: from behind it gets through, and the sky's blue no longer shows.
CLOUDY_OPTICAL_DEPTH = 2.0
#: The same for simulated patches. Their thin, eroded parts are plainly cloud to an observer
#: (a third of the sky's light is already scattered at 0.4), and counting only the opaque cores
#: filled the sky with twice the cloud asked for.
PATCH_CLOUDY_OPTICAL_DEPTH = 0.4
#: Cover is the fraction of the sky an observer on the ground sees obscured, and a cumulus hides
#: sky with its sides as well as its base. The solve counts columns seen from straight below, so
#: for patches it aims at this fraction of the cover asked for. Measured on the look sheet's four
#: headings (22 degrees up, 60 degree lens): cover 0.35 leaves 0.3-0.4 of the pixels cloudy.
PATCH_PLAN_VIEW_FRACTION = 0.5


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
    #: Simulated patches of cloud. With any, they are the cloud: the weather map says which
    #: lattice cells hold one, and the noise function is not used. ``None`` or empty: the function.
    patches: Optional[List[CloudPatch]] = None
    #: Towering cumulus patches, placed in ``profile.towers`` of the large clouds' cells.
    towers: Optional[List[CloudPatch]] = None
    #: (nz, ny, slots * nx) float32: the patches side by side, indexed ``[w, v, u]``.
    patch_atlas: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    #: How full the two lattices of patches are, 0..2, as the cover solve found.
    patch_cover: float = field(init=False, default=0.0)
    patch_size_m: Tuple[float, float, float] = field(init=False, default=(1.0, 1.0, 1.0))
    tower_size_m: Tuple[float, float, float] = field(init=False, default=(1.0, 1.0, 1.0))

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
        if self.patches:
            self._build_atlas()
        self.coverage_bias = self._solve_cover()

    def _build_atlas(self) -> None:
        """One 3-D texture holding every patch side by side along x, each in a slot of the same
        number of cells. The cumulus patches share one box in metres (``patch_size_m``) and the
        towers another (``tower_size_m``); a tower is resampled into its slot."""
        patches = self.patches or []
        towers = self.towers or []
        voxel = float(patches[0].voxel_m)
        nx = max(q.density.shape[0] for q in patches)
        ny = max(q.density.shape[1] for q in patches)
        nz = max(q.density.shape[2] for q in patches)
        slot = int(round(nx / (1.0 - 2.0 * ATLAS_MARGIN)))
        atlas = np.zeros((slot * (len(patches) + len(towers)), ny, nz), dtype=np.float32)
        for k, q in enumerate(patches):
            if abs(q.voxel_m - voxel) > 1e-3 * voxel:
                raise ValueError("cloud patches must share one voxel size")
            sx, sy, sz = q.density.shape
            x0 = k * slot + (slot - sx) // 2
            z0 = (nz - sz) // 2
            atlas[x0:x0 + sx, :sy, z0:z0 + sz] = q.density
        self.patch_size_m = (nx * voxel, ny * voxel, nz * voxel)
        if towers:
            size = tuple(max(q.density.shape[axis] * q.voxel_m for q in towers) for axis in range(3))
            for k, q in enumerate(towers):
                # Centred in the towers' common box (standing on its floor), then that box
                # resampled to the slot's cells.
                cells = tuple(int(round(size[axis] / q.voxel_m)) for axis in range(3))
                box = np.zeros(cells, dtype=np.float32)
                sx, sy, sz = (min(q.density.shape[axis], cells[axis]) for axis in range(3))
                x0, z0 = (cells[0] - sx) // 2, (cells[2] - sz) // 2
                box[x0:x0 + sx, :sy, z0:z0 + sz] = q.density[:sx, :sy, :sz]
                x0 = (len(patches) + k) * slot + (slot - nx) // 2
                atlas[x0:x0 + nx, :, :] = _resize(box, (nx, ny, nz))
            self.tower_size_m = tuple(float(v) for v in size)
        self.patch_atlas = np.ascontiguousarray(atlas.transpose(2, 1, 0))

    def _solve_cover(self) -> float:
        """The coverage bias at which the fraction of cloudy columns is the cover asked for.

        A column is cloudy when its vertical optical depth passes :data:`CLOUDY_OPTICAL_DEPTH`:
        a cloud a camera looking up cannot see the sky through. Measured on a coarse lattice of
        columns and solved by bisection, since more coverage can only cover more columns.
        """
        if self.cover <= 0.0:
            return -1.0
        n = 112 if self.patches else 160
        axis = (np.arange(n) + 0.5) / n * self.weather_tile_m
        x, z = np.meshgrid(axis, axis, indexing="xy")
        lo, hi = (0.0, 2.0) if self.patches else (-1.0, 1.0)
        target = self.cover * (PATCH_PLAN_VIEW_FRACTION if self.patches else 1.0)
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            cloudy = self._column_optical_depth(x, z, mid, 14) > self._cloudy_optical_depth
            if float(cloudy.mean()) < target:
                lo = mid
            else:
                hi = mid
        if self.patches:
            # With patches the unknown is how full the two lattices are, 0..2.
            self.patch_cover = 0.5 * (lo + hi)
            return 0.0
        return 0.5 * (lo + hi)

    @property
    def _cloudy_optical_depth(self) -> float:
        return PATCH_CLOUDY_OPTICAL_DEPTH if self.patches else CLOUDY_OPTICAL_DEPTH

    def _column_optical_depth(self, x: np.ndarray, z: np.ndarray, bias: float, levels: int) -> np.ndarray:
        """Visible optical depth straight up through the layer, by the midpoint rule."""
        d = self.profile.thickness_m
        total = np.zeros(np.shape(x))
        for y in self.base_m + (np.arange(levels) + 0.5) / levels * d:
            total += self._density(x, y, z, bias)
        return total * d / levels * self.profile.extinction_per_m

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
        return self._density(x_m, y_m, z_m, self.patch_cover if self.patches else self.coverage_bias)

    def _density(self, x_m: Any, y_m: Any, z_m: Any, bias: float) -> np.ndarray:
        """The arithmetic the GPU kernel repeats (``gpu/cloud_march.py::cloud_density``).
        ``bias`` is the coverage bias for the function, the fraction of cells for patches."""
        if self.patches:
            return self._patch_density(x_m, y_m, z_m, bias)
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
        # The skin: crisp where the cloud is rising, soft at its base and low on its sides.
        sharp = p.edge_base + (p.edge_top - p.edge_base) * _smoothstep(p.crisp_from, p.crisp_to, hr)
        edge = 1.0 - np.exp(-sharp * eroded)
        # The body: liquid water grows with height above the base as the adiabat's does, and
        # extinction as its two-thirds power (the droplets grow too, their number does not). The
        # height is the layer's, not the cloud's own: a shallow cloud is a thin one.
        water = p.water_base + (1.0 - p.water_base) * np.clip(h, 0.0, 1.0) ** (2.0 / 3.0)
        return np.where(inside, edge * water, 0.0)

    def _patch_density(self, x_m: Any, y_m: Any, z_m: Any, cover: float) -> np.ndarray:
        """``gpu/cloud_march.py::patch_density``, line for line. Two lattices, the second offset
        by half a cell and filled only once the first is full (``cover`` runs 0..2)."""
        period = self.profile.patch_period_m
        d = self._lattice_density(x_m, y_m, z_m, min(cover, 1.0), 0, period)
        if cover > 1.0:
            d = np.maximum(d, self._lattice_density(x_m, y_m, z_m, cover - 1.0, 1, period))
        # A fine lattice of small clouds among the large ones.
        small = min(SMALL_COVER * cover, 1.0)
        return np.maximum(d, self._lattice_density(x_m, y_m, z_m, small, 2, SMALL_PERIOD * period))

    def _lattice_density(self, x_m: Any, y_m: Any, z_m: Any, cover: float, lattice: int, period: float) -> np.ndarray:
        p = self.profile
        x, y, z = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (x_m, y_m, z_m)))
        h = (y - self.base_m) / p.thickness_m
        count = len(self.patches or [])
        tall = len(self.towers or [])
        shift = 0.5 * period * lattice
        channel = 10 * lattice
        cx = np.floor((x - shift) / period)
        cz = np.floor((z - shift) / period)
        mx = (cx + 0.5) * period + shift
        mz = (cz + 0.5) * period + shift
        wc = sample_wrapped(self.weather, mx / self.weather_tile_m, mz / self.weather_tile_m)[..., 0]
        rank = (wc - (1.0 - cover)) / max(cover, 1.0e-3)
        present = rank > 0.0
        if lattice == 2:
            present = present & (_cell_hash(cx, cz, 7 + channel) <= p.small_keep)
        scale = (0.45 + 0.55 * np.sqrt(np.clip(rank, 0.0, None))) * (0.8 + 0.4 * _cell_hash(cx, cz, 2 + channel))
        # Some of the large clouds are towers: another set of patches in a taller box.
        tower = np.zeros(x.shape, dtype=bool)
        if tall and lattice != 2 and p.towers > 0.0:
            tower = _cell_hash(cx, cz, 8 + channel) < p.towers
        sx, sy, sz = (np.where(tower, t, c) for t, c in zip(self.tower_size_m, self.patch_size_m))
        angle = 6.2831853 * _cell_hash(cx, cz, 5 + channel)
        ca, sa = np.cos(angle), np.sin(angle)
        # The turned box has to fit its cell, or the cell's edge cuts the cloud flat.
        fx = PATCH_FOOTPRINT * (np.abs(ca) * sx + np.abs(sa) * sz)
        fz = PATCH_FOOTPRINT * (np.abs(sa) * sx + np.abs(ca) * sz)
        scale = np.minimum(scale, period / np.maximum(fx, fz)) * p.patch_fill
        lx = x - mx - (_cell_hash(cx, cz, 3 + channel) - 0.5) * np.maximum(period - scale * fx, 0.0)
        lz = z - mz - (_cell_hash(cx, cz, 4 + channel) - 0.5) * np.maximum(period - scale * fz, 0.0)
        u = (ca * lx + sa * lz) / (scale * sx) + 0.5
        v = (y - self.base_m) / (scale * sy)
        w = (-sa * lx + ca * lz) / (scale * sz) + 0.5
        inside = present & (h > 0.0) & (h < 1.0) & (u > 0.0) & (u < 1.0) & (v > 0.0) & (v < 1.0) & (w > 0.0) & (w < 1.0)
        pick = _cell_hash(cx, cz, 6 + channel)
        which = np.where(tower, count + np.minimum((pick * max(tall, 1)).astype(np.int64), max(tall, 1) - 1),
                         np.minimum((pick * count).astype(np.int64), count - 1))
        d = _sample_clamped(self.patch_atlas, (which + ATLAS_MARGIN + (1.0 - 2.0 * ATLAS_MARGIN) * np.clip(u, 0.0, 1.0)) / (count + tall),
                            np.clip(v, 0.0, 1.0), np.clip(w, 0.0, 1.0))
        d = d * _smoothstep(0.0, 0.08, np.minimum(np.minimum(u, 1.0 - u), np.minimum(w, 1.0 - w)))
        d = d * _smoothstep(0.0, 0.06, 0.5 - np.maximum(np.abs(x - mx), np.abs(z - mz)) / period)
        t = EROSION_TILE_M
        e1 = sample_wrapped(self.detail, x / t, y / t, z / t)
        f = t / 4.0
        e2 = sample_wrapped(self.detail, x / f + 0.37, y / f + 0.11, z / f + 0.73)
        noise = 0.65 * e1 + 0.35 * e2
        strength = p.patch_erosion * (1.0 - 0.55 * np.clip(v * 1.4, 0.0, 1.0)) * (0.3 + 0.7 * _smoothstep(0.0, 0.1, v))
        g = t * RAGGED_TILES
        ragged = sample_wrapped(self.detail, x / g + 0.19, y / g + 0.61, z / g + 0.43)
        strength = strength * (RAGGED_LOW + (RAGGED_HIGH - RAGGED_LOW) * _smoothstep(0.3, 0.7, ragged))
        eroded = _remap(d, noise * strength, 1.0) if p.patch_erosion > 0.0 else d
        return np.where(inside & (d > 0.0), eroded, 0.0)

    def kernel_constants(self) -> Dict[str, Any]:
        """Every scalar the GPU kernel needs, by the names its struct uses."""
        p = self.profile
        return dict(
            base_m=float(self.base_m), thickness_m=float(p.thickness_m),
            weather_tile_m=float(self.weather_tile_m), shape_tile_m=float(p.shape_tile_m),
            detail_tile_m=float(p.detail_tile_m), coverage_bias=float(self.coverage_bias),
            top_min=float(p.top_min), top_max=float(p.top_max), taper=float(p.taper),
            erosion=float(p.erosion), edge_base=float(p.edge_base), edge_top=float(p.edge_top),
            crisp_from=float(p.crisp_from), crisp_to=float(p.crisp_to),
            water_base=float(p.water_base), extinction_per_m=float(p.extinction_per_m),
            patch_on=1 if self.patches else 0, patch_count=len(self.patches or []),
            patch_size=tuple(float(v) for v in self.patch_size_m),
            patch_period_m=float(p.patch_period_m), patch_cover=float(self.patch_cover),
            patch_erosion=float(p.patch_erosion), patch_fill=float(p.patch_fill),
            small_keep=float(p.small_keep), tower_count=len(self.towers or []),
            tower_size=tuple(float(v) for v in self.tower_size_m),
            towers=float(p.towers) if self.towers else 0.0)

    def measured_cover(self, columns: int = 256) -> float:
        """Fraction of columns holding cloud a camera cannot see through, on a fresh lattice."""
        if self.cover <= 0.0:
            return 0.0
        axis = (np.arange(columns) + 0.25) / columns * self.weather_tile_m
        x, z = np.meshgrid(axis, axis, indexing="xy")
        bias = self.patch_cover if self.patches else self.coverage_bias
        return float((self._column_optical_depth(x, z, bias, 20) > self._cloudy_optical_depth).mean())

    # --- what another sensor's march reads (the contract a CloudField also meets) ---------------

    @property
    def thickness_m(self) -> float:
        """Depth of the layer the clouds may occupy, metres."""
        return float(self.profile.thickness_m)

    @property
    def finest_pitch_m(self) -> float:
        """The finest spacing a march has to resolve, metres: a patch's cell, or the detail
        noise's cell for the function. The skin erosion's smaller noise is finer than this; it
        tears the outline, and a march that steps over it misplaces an edge by less than a cell."""
        if self.patches:
            return float(min(q.voxel_m for q in self.patches))
        return float(self.profile.detail_tile_m / self.detail_cells)

    @property
    def optical_depth(self) -> float:
        """Visible optical depth of the median cloudy column, straight up (0 for a clear sky)."""
        cached = getattr(self, "_median_optical_depth", None)
        if cached is None:
            cached = 0.0
            if self.cover > 0.0:
                n = 96
                axis = (np.arange(n) + 0.25) / n * self.weather_tile_m
                x, z = np.meshgrid(axis, axis, indexing="xy")
                bias = self.patch_cover if self.patches else self.coverage_bias
                column = self._column_optical_depth(x, z, bias, 20)
                cloudy = column > self._cloudy_optical_depth
                cached = float(np.median(column[cloudy])) if cloudy.any() else 0.0
            self._median_optical_depth = cached
        return cached

    def slab_span(self, origin_m: Any, direction: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Where a ray enters and leaves the layer, as distances along it (``CloudField.slab_span``):
        ``exit <= enter`` for a ray that misses it; a ray starting inside enters at zero."""
        origin = np.asarray(origin_m, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        oy, dy = np.broadcast_arrays(origin[..., 1], direction[..., 1])
        steep = np.abs(dy) > 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            t_base = (self.base_m - oy) / dy
            t_top = (self.top_m - oy) / dy
        lo = np.where(steep, np.minimum(t_base, t_top), 0.0)
        hi = np.where(steep, np.maximum(t_base, t_top), 0.0)
        level_inside = (~steep) & (oy >= self.base_m) & (oy <= self.top_m)
        near = np.where(steep, np.maximum(lo, 0.0), np.where(level_inside, 0.0, 1.0))
        far = np.where(steep, np.maximum(hi, 0.0), np.where(level_inside, 1.0e6, 0.0))
        return near, far

    def optical_depth_toward(self, origin_m: Any, direction: Any, samples: int = 96) -> float:
        """Visible optical depth from a point through the layer along a direction (flat slab)."""
        o = np.asarray(origin_m, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        if self.cover <= 0.0 or abs(d[1]) < 1e-4:
            return 0.0
        t0 = (self.base_m - o[1]) / d[1]
        t1 = (self.top_m - o[1]) / d[1]
        lo, hi = max(min(t0, t1), 0.0), max(t0, t1)
        if hi <= lo:
            return 0.0
        t = lo + (np.arange(samples) + 0.5) / samples * (hi - lo)
        p = o[None, :] + t[:, None] * d[None, :]
        rho = self.density(p[:, 0], p[:, 1], p[:, 2])
        return float(rho.sum() * (hi - lo) / samples * self.extinction_per_m)


#: The cloud parameters a cloudscape's textures depend on.
CLOUDSCAPE_SHAPE_KEYS = ("enabled", "cover", "genus", "base_m", "temperature_c", "dewpoint_c", "seed",
                         "patches", "spacing_m", "cloud_fill", "small_clouds", "raggedness", "towers")
#: Genera drawn from simulated patches when any are installed.
PATCH_GENERA = ("cumulus", "congestus")

_CACHE: Dict[Tuple[Any, ...], "Cloudscape"] = {}


def supports(genus: str) -> bool:
    """Whether a genus has a cloudscape profile (cirrus does not yet)."""
    return genus in CLOUDSCAPE_TYPES


def cloudscape_from_state(state: Any, build: bool = True):
    """The cloudscape a state describes, or ``None`` for a clear sky or an unsupported genus.

    Cached on its shape keys like :func:`~weather_fx.core.clouds.cloud_field_from_state`;
    ``build=False`` never pays for one.
    """
    clouds = state.clouds
    if not (clouds.enabled and clouds.cover > 0.0 and supports(clouds.genus)):
        return None
    key = tuple(getattr(clouds, name) for name in CLOUDSCAPE_SHAPE_KEYS)
    cached = _CACHE.get(key)
    if cached is not None or not build:
        return cached
    base = clouds.base_m
    if base <= 0.0:
        base = _clouds.lifting_condensation_level_m(clouds.temperature_c, clouds.dewpoint_c)
    # Simulated patches are cumulus; the sheet genera stay the function.
    patches = load_patches(getattr(clouds, "patches", "") or None) if clouds.genus in PATCH_GENERA else []
    if getattr(clouds, "patches", "") == "none":
        patches = []
    profile = CLOUDSCAPE_TYPES[clouds.genus]
    # The owner's controls over the field of patches; each leaves the genus default at its own.
    spacing = float(getattr(clouds, "spacing_m", 0.0))
    profile = dataclasses.replace(
        profile,
        patch_period_m=spacing if spacing > 0.0 else profile.patch_period_m,
        patch_fill=float(getattr(clouds, "cloud_fill", profile.patch_fill)),
        small_keep=float(getattr(clouds, "small_clouds", profile.small_keep)),
        patch_erosion=profile.patch_erosion * float(getattr(clouds, "raggedness", 1.0)))
    towers = load_towers(getattr(clouds, "patches", "") or None) if patches else []
    share = float(getattr(clouds, "towers", 0.0))
    share = share if share > 0.0 else profile.towers
    if towers and share > 0.0:
        # The layer has to hold the tallest tower at the largest scale a cell allows.
        height = max(q.density.shape[1] * q.voxel_m for q in towers)
        widest = max(max(q.density.shape[0], q.density.shape[2]) * q.voxel_m for q in towers)
        tallest = min(1.2, profile.patch_period_m / (PATCH_FOOTPRINT * widest)) * profile.patch_fill * height
        profile = dataclasses.replace(profile, towers=min(share, 1.0),
                                      thickness_m=max(profile.thickness_m, float(np.ceil(tallest / 100.0) * 100.0)))
    else:
        towers = []
    built = Cloudscape(cover=float(clouds.cover), base_m=float(base),
                       profile=profile, seed=int(clouds.seed),
                       patches=patches or None, towers=towers or None)
    while len(_CACHE) >= 2:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = built
    return built
