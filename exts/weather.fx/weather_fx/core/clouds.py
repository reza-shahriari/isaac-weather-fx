"""A cloud that has an inside: a three-dimensional density field, not an extruded height map.

The model this replaces was a horizontal map of column depth extruded upward by a fixed vertical
profile. That is a *vertical extrusion*: the horizontal cross-section at every height is the same
shape, only scaled, so a cloud is a smooth mound standing on its base plane and getting narrower
as it rises. Rendered, it looks like a grey dune. A real cumulus does the opposite -- it has a
sharply flat base, bulges *outward* above it, and breaks into cauliflower lobes near the top,
whose silhouette at one height has little to do with the silhouette a hundred metres lower.

So the field here is genuinely ``density(x, y, z)``, built once on a periodic grid and sampled
trilinearly. Three pieces make the shape:

* **Band-limited fractal noise**, synthesised as an inverse FFT of a ``1 / f^beta`` spectrum, so
  the field is *exactly* periodic in all three axes -- sample ``i`` and sample ``i + n`` are the
  same number, not merely similar ones. A ray can run for kilometres and never find an edge. The
  spectrum is band-limited at the low end because a scale-free field puts cloud structure at every
  size including sizes no cumulus has.
* **A height profile that says how much area the cloud occupies at each height**, applied by
  taking that level's own quantile of the noise. This is what makes the shape right. Cutting the
  field at a level-dependent quantile changes which *set* of points survives as the cloud rises,
  so the silhouette genuinely changes with height. Scaling a density by a height profile -- which
  is what an extrusion does -- cannot: it leaves every cross-section the same shape, only fainter.
  Measured on the shipped cumulus, two levels a third of the depth apart share an intersection
  over union of about 0.2; an extrusion scores 1.0 by construction.
* **Erosion**, a higher-frequency term that bites into the field only where it is already near
  the threshold. That puts detail on the lobes and the edges, where a cloud has it, and leaves the
  optically thick core alone, where detail would not be visible anyway.

**Coverage means what it says.** ``cover`` is the fraction of sky the cloud hides from an observer
looking straight up, and the threshold that achieves it is *solved* on the built grid by bisection
rather than assumed from the noise's distribution. Every other model in this area quietly has
coverage as an input that comes out different.

**One field, every band.** The visible renderer and an infrared radiative-transfer march sample
the same array through the same :meth:`CloudField.density`. They cannot draw cloud in different
places, because there is only one cloud. That is the whole reason this lives in ``core`` next to
the physics rather than inside a renderer backend.

The model is a *morphology* -- the shape and the optical depth -- and the quantities a sensor
model needs (extinction per metre, optical depth along a path) follow from it by construction
rather than being authored twice. The microphysics is deliberately thin and derived: each genus
states a phase and an effective particle size, and the water content is whatever reproduces the
visible extinction (:class:`CloudMicrophysics`). There is no drop-size distribution, no adiabatic
liquid water profile and no entrainment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "CLOUD_TYPES",
    "CloudProfile",
    "CloudField",
    "cloud_profile",
    "lifting_condensation_level_m",
    "VolumeGrid",
    "CLOUD_SHAPE_KEYS",
    "cloud_field_from_state",
    "TilePlacement",
    "tile_placements",
    "stage_to_field",
    "volume_offset_m",
    "drift_velocity_m_s",
    "cloud_drift_at",
    "cloud_drift_from_state",
    "DriftTrack",
    "CloudMicrophysics",
    "microphysics_for",
    "liquid_extinction_per_m",
    "liquid_water_content_g_m3",
    "ice_extinction_per_m",
    "ice_water_content_g_m3",
    "DETAIL_STRENGTH",
]

@dataclass(frozen=True)
class CloudProfile:
    """How a genus of cloud is shaped in the vertical, and how thick it is.

    The shape is stated as **the fraction of the horizontal plane the cloud occupies at each
    height**, relative to its own widest level. That is the morphology, written down directly:
    "a cumulus is as wide as it ever gets right at its base, holds that to a third of its depth,
    and tapers to almost nothing at the top" is a sentence about occupied area, and this is that
    sentence as numbers.

    Writing it as an *area* rather than as a threshold offset matters. A threshold offset has to
    be compared against the noise's own distribution, so it fights the coverage solve and the
    spectral slope; an area is solved for exactly, at each level, by taking that level's own
    quantile. The first version of this model used offsets and squashed every cloud into the
    bottom half of its declared thickness, which is not a subtle error but it is an invisible one
    -- the clouds looked like clouds, they were simply all the wrong shape.
    """

    name: str
    #: Fractional heights at which ``area_points`` are given, ascending, spanning 0..1.
    height_points: Tuple[float, ...]
    #: Occupied fraction of the horizontal plane at each height, relative to the widest level.
    #: A flat base is this reaching its maximum immediately above u = 0.
    area_points: Tuple[float, ...]
    #: Typical geometric thickness of the genus, metres. ESTIMATED from standard cloud atlases.
    thickness_m: float
    #: Typical visible optical depth of a full-depth column. ESTIMATED.
    optical_depth: float
    #: How strongly the higher-frequency term eats into the edges. 0 leaves smooth blobs.
    erosion: float = 0.35
    #: Spectral slope of the fractal noise. Larger is smoother and more blobby.
    beta: float = 2.6
    #: Horizontal stretch applied along the wind, which is what makes cirrus streaks and
    #: stratocumulus rolls look like rolls rather than like spots.
    anisotropy: float = 1.0
    #: How billowy the shape is, 0..1: how strongly Worley (cellular) noise rounds the field
    #: into lobes. Cumulus towers are made of rising thermals, each a rounded bubble, and a
    #: fractal noise alone cannot make a convex bubble -- it makes smooth ridges, which read as
    #: smeared blobs or, cut by the height profile, as cones. 0 for a sheet or a streak.
    billow: float = 0.0
    #: How much of the shape comes from a **two-dimensional** coverage map (where the clouds are)
    #: rather than from the 3D noise, 0..1. Thresholding 3D noise level by level makes a sponge:
    #: every level is cut independently, so the inside of a cloud is riddled with holes. A real
    #: cumulus is a solid body with its structure on the outside. Mixing in a column term that is
    #: the same at every height makes the cores solid -- a column that is cloudy is cloudy all
    #: the way up, until the height profile narrows it -- and leaves the 3D noise to carve the
    #: edges. It is the coverage ("weather") map of Schneider's and Unreal's cloud renderers.
    columnar: float = 0.0
    #: Typical distance between neighbouring clouds, as a multiple of ``feature_m``. Sets both the
    #: longest wave the noise carries and the spacing of the coverage map's cells.
    spacing: float = 12.0
    #: Thermodynamic phase: ``"liquid"``, ``"ice"`` or ``"mixed"``. An infrared band cares far
    #: more than a visible one: ice and water absorb differently across 8-14 um.
    phase: str = "liquid"
    #: Effective radius of the droplets, micrometres (the ratio of the third to the second moment
    #: of the size distribution, which is what extinction and absorption scale with). ESTIMATED
    #: from the in-situ ranges in Miles, Verlinde & Clothiaux (2000).
    effective_radius_um: float = 10.0
    #: Generalised effective size of the ice crystals, micrometres, as Fu (1996) defines it.
    #: Only read when there is ice. ESTIMATED from Fu's observed 20-130 um range.
    ice_effective_diameter_um: float = 60.0
    #: Fraction of the visible extinction carried by ice. 0 for a warm cloud, 1 for cirrus.
    ice_fraction: float = 0.0


CLOUD_TYPES: Dict[str, CloudProfile] = {
    # Fair-weather cumulus: flat base, widest a third of the way up, cauliflower top.
    "cumulus": CloudProfile(
        name="cumulus",
        height_points=(0.0, 0.04, 0.30, 0.55, 0.75, 0.90, 1.0),
        area_points=(1.0, 1.0, 0.95, 0.80, 0.55, 0.26, 0.03),
        thickness_m=1200.0,
        optical_depth=18.0,
        erosion=0.45,
        beta=2.5,
        billow=1.0,
        columnar=0.75,
        spacing=5.0,
        phase="liquid",
        effective_radius_um=8.0,
    ),
    # Towering cumulus: deeper, narrower, and it keeps its width much higher up.
    "congestus": CloudProfile(
        name="congestus",
        height_points=(0.0, 0.03, 0.35, 0.70, 0.88, 1.0),
        area_points=(1.0, 1.0, 0.92, 0.78, 0.50, 0.08),
        thickness_m=3500.0,
        optical_depth=45.0,
        erosion=0.5,
        beta=2.4,
        billow=1.0,
        columnar=0.8,
        spacing=6.0,
        # Tall enough to cross the freezing level: the top glaciates.
        phase="mixed",
        effective_radius_um=12.0,
        ice_effective_diameter_um=70.0,
        ice_fraction=0.25,
    ),
    # Stratocumulus: a broken sheet with rolls, thin and nearly uniform in the vertical.
    "stratocumulus": CloudProfile(
        name="stratocumulus",
        height_points=(0.0, 0.10, 0.55, 0.85, 1.0),
        area_points=(1.0, 1.0, 0.96, 0.80, 0.30),
        thickness_m=600.0,
        optical_depth=12.0,
        erosion=0.25,
        beta=3.0,
        anisotropy=2.5,
        billow=0.6,
        columnar=0.45,
        spacing=6.0,
        phase="liquid",
        effective_radius_um=10.0,
    ),
    # Stratus: a featureless sheet. The threshold barely moves, so there is no morphology at all.
    "stratus": CloudProfile(
        name="stratus",
        height_points=(0.0, 0.1, 0.9, 1.0),
        area_points=(1.0, 1.0, 1.0, 0.85),
        thickness_m=400.0,
        optical_depth=9.0,
        erosion=0.08,
        beta=3.4,
        billow=0.1,
        columnar=0.3,
        phase="liquid",
        effective_radius_um=9.0,
    ),
    # Cirrus: high, thin, streaked hard along the wind, and optically shallow.
    "cirrus": CloudProfile(
        name="cirrus",
        height_points=(0.0, 0.25, 0.75, 1.0),
        area_points=(0.55, 1.0, 1.0, 0.55),
        thickness_m=1500.0,
        optical_depth=1.2,
        erosion=0.6,
        beta=2.2,
        anisotropy=6.0,
        phase="ice",
        ice_effective_diameter_um=60.0,
        ice_fraction=1.0,
    ),
}


def cloud_profile(name: str) -> CloudProfile:
    if name not in CLOUD_TYPES:
        raise ValueError(f"unknown cloud type {name!r}; known: {sorted(CLOUD_TYPES)}")
    return CLOUD_TYPES[name]


#: Density of liquid water, kg/m^3.
WATER_DENSITY_KG_M3 = 1000.0
#: Fu (1996) ice extinction fit, ``beta = IWC * (a0 + a1 / D_ge)`` with IWC in g/m^3, D_ge in um
#: and beta in 1/m (his eq. 3.9a). Valid for D_ge of about 20 to 130 um.
FU1996_A0 = -6.656e-3
FU1996_A1 = 3.686


def liquid_extinction_per_m(lwc_g_m3: Any, effective_radius_um: Any) -> Any:
    """Visible extinction of a water cloud from its liquid water content, per metre.

    Geometric optics (extinction efficiency 2, valid for droplets much larger than the
    wavelength): ``beta = 3 LWC / (2 rho_w r_e)``. With LWC in g/m^3 and r_e in um that is simply
    ``1.5 LWC / r_e`` per metre (Stephens 1978; Hansen & Travis 1974).
    """
    return 1.5 * np.asarray(lwc_g_m3, dtype=np.float64) / np.asarray(effective_radius_um, dtype=np.float64)


def liquid_water_content_g_m3(extinction_per_m: Any, effective_radius_um: Any) -> Any:
    """The inverse of :func:`liquid_extinction_per_m`: LWC in g/m^3 from visible extinction."""
    return np.asarray(extinction_per_m, dtype=np.float64) * np.asarray(effective_radius_um, dtype=np.float64) / 1.5


def ice_extinction_per_m(iwc_g_m3: Any, effective_diameter_um: Any) -> Any:
    """Visible extinction of an ice cloud from its ice water content, per metre (Fu 1996)."""
    d = np.asarray(effective_diameter_um, dtype=np.float64)
    return np.asarray(iwc_g_m3, dtype=np.float64) * (FU1996_A0 + FU1996_A1 / d)


def ice_water_content_g_m3(extinction_per_m: Any, effective_diameter_um: Any) -> Any:
    """The inverse of :func:`ice_extinction_per_m`: IWC in g/m^3 from visible extinction."""
    d = np.asarray(effective_diameter_um, dtype=np.float64)
    return np.asarray(extinction_per_m, dtype=np.float64) / (FU1996_A0 + FU1996_A1 / d)


@dataclass(frozen=True)
class CloudMicrophysics:
    """What a cloud is made of, at **unit density** of its field.

    The field is a morphology with a visible extinction; this is the same cloud stated as water.
    Everything scales linearly with :meth:`CloudField.density`, so the content at a point is
    ``density * liquid_water_g_m3`` (and likewise for ice). The water is **derived from** the
    visible extinction and the genus' particle sizes, not authored separately, so the visible
    render and an infrared band computed from these numbers describe the same cloud.

    Not modelled: the adiabatic growth of LWC and r_e with height above the base, drizzle, and
    any size distribution beyond its effective radius. An infrared consumer that needs more than
    a per-genus effective size should say so; the hooks are here.
    """

    phase: str
    #: Liquid droplet effective radius, um.
    effective_radius_um: float
    #: Ice generalised effective size (Fu 1996), um.
    ice_effective_diameter_um: float
    #: Fraction of the visible extinction carried by ice.
    ice_fraction: float
    #: Visible extinction at unit density, 1/m (the field's :attr:`CloudField.extinction_per_m`).
    extinction_per_m: float
    #: Liquid water content at unit density, g/m^3.
    liquid_water_g_m3: float
    #: Ice water content at unit density, g/m^3.
    ice_water_g_m3: float


def microphysics_for(profile: CloudProfile, extinction_per_m: float) -> CloudMicrophysics:
    """Split a visible extinction into liquid and ice water for a genus."""
    ice = min(max(float(profile.ice_fraction), 0.0), 1.0)
    beta = max(float(extinction_per_m), 0.0)
    return CloudMicrophysics(
        phase=profile.phase,
        effective_radius_um=float(profile.effective_radius_um),
        ice_effective_diameter_um=float(profile.ice_effective_diameter_um),
        ice_fraction=ice,
        extinction_per_m=beta,
        liquid_water_g_m3=float(liquid_water_content_g_m3((1.0 - ice) * beta, profile.effective_radius_um)),
        ice_water_g_m3=float(ice_water_content_g_m3(ice * beta, profile.ice_effective_diameter_um)) if ice > 0 else 0.0,
    )


def lifting_condensation_level_m(temperature_c: float, dewpoint_c: float) -> float:
    """Cloud base height from the surface temperature and dew point, metres.

    The meteorologist's rule of thumb -- 125 m per degree of spread -- which is the dry adiabatic
    lapse rate (9.8 K/km) minus the dew-point lapse rate (1.8 K/km) inverted. It is the reason a
    cumulus base is *flat*: every parcel rising from the same surface condenses at the same
    height, so the bases of a whole field line up along one level.
    """
    spread = max(temperature_c - dewpoint_c, 0.0)
    return 125.0 * spread


def _periodic_fractal_noise(
    shape: Tuple[int, int, int],
    beta: float,
    seed: int,
    *,
    cell_size_m: Tuple[float, float, float],
    min_wavelength_m: float,
    anisotropy: float = 1.0,
    max_wavelength_m: float = 0.0,
) -> np.ndarray:
    """Band-limited ``1/f^beta`` noise on a periodic grid, normalised to 0..1 by rank.

    Built in the Fourier domain so periodicity is exact rather than blended, and **rank-normalised**
    afterwards so the output is uniform on 0..1. That last step is what lets a coverage threshold
    mean a quantile: with raw Gaussian noise, "the top 8 % of the field" and "8 % of the area" are
    different statements, and the coverage a scene ends up with drifts with the spectral slope.

    **The spectrum is isotropic in metres, not in cells.** The grid is far finer vertically than
    horizontally -- 25 m against 60 m at the shipped settings -- so a spectrum written in cells
    gives a field that decorrelates over 150 m upward and 360 m sideways. It renders as fuzz
    rather than as towers, because a cumulus *is* vertically coherent: it is one parcel of air
    that rose. Converting each axis's frequency to cycles per metre before shaping the spectrum
    costs nothing and is the difference between a cloud and a cloud-coloured noise field.

    ``anisotropy`` then stretches the two horizontal axes against each other on purpose, which is
    what turns stratocumulus spots into rolls and cirrus into streaks.
    """
    nz, ny, nx = shape
    dz, dy, dx = cell_size_m
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(shape)
    spectrum = np.fft.rfftn(white)

    # Cycles per metre on each axis.
    fz = (np.fft.fftfreq(nz) / dz)[:, None, None]
    fy = (np.fft.fftfreq(ny) / dy)[None, :, None] * anisotropy
    fx = (np.fft.rfftfreq(nx) / dx)[None, None, :] / anisotropy
    radius = np.sqrt(fz**2 + fy**2 + fx**2)
    radius[0, 0, 0] = 1.0

    amplitude = radius ** (-beta / 2.0)
    # Band limit at the low end: a scale-free field carries structure at every size including
    # sizes no cumulus has, which is what made an earlier field render as specks in one band and
    # smeared kilometres in the other.
    cutoff = 1.0 / max(min_wavelength_m, 2.0 * max(dx, dy, dz))
    amplitude *= np.exp(-((radius / cutoff) ** 2))
    if max_wavelength_m > 0.0:
        # And at the high end of the wavelengths: a red spectrum puts most of its power in the
        # longest waves the tile allows, so a 30 % cover came out as one connected mass the size
        # of the tile with scraps around it, rather than a field of separate clouds.
        amplitude *= 1.0 - np.exp(-((radius * max_wavelength_m) ** 2))
    amplitude[0, 0, 0] = 0.0

    noise = np.fft.irfftn(spectrum * amplitude, s=shape, axes=(0, 1, 2))
    flat = noise.reshape(-1)
    ranks = np.empty(flat.size, dtype=np.float64)
    ranks[np.argsort(flat, kind="stable")] = np.arange(flat.size, dtype=np.float64)
    return (ranks / max(flat.size - 1, 1)).reshape(shape)


def _periodic_worley(
    shape: Tuple[int, int, int],
    lattice: Tuple[int, int, int],
    seed: int,
) -> np.ndarray:
    """Inverted cellular (Worley F1) noise on a periodic grid: 1 at a feature point, 0 far away.

    One jittered feature point per lattice cell; each sample takes its distance to the nearest
    point among the 27 cells around it, with the lattice wrapped so the result tiles exactly with
    the grid. ``lattice`` is the number of lattice cells along each axis (z, y, x); any whole
    number keeps the period. The result is ``1 - clip(F1, 0, 1)`` in lattice units, so the bright
    blobs are round, one per lattice cell -- the bubble a fractal noise cannot make.
    """
    nz, ny, nx = shape
    lz, ly, lx = (max(1, int(n)) for n in lattice)
    rng = np.random.default_rng(seed)
    points = rng.random((lz, ly, lx, 3)).astype(np.float32)
    # Sample positions in lattice units, at cell centres.
    pz = ((np.arange(nz, dtype=np.float32) + 0.5) * (lz / nz))[:, None, None]
    py = ((np.arange(ny, dtype=np.float32) + 0.5) * (ly / ny))[None, :, None]
    px = ((np.arange(nx, dtype=np.float32) + 0.5) * (lx / nx))[None, None, :]
    cz, cy, cx = np.floor(pz).astype(np.int64), np.floor(py).astype(np.int64), np.floor(px).astype(np.int64)
    fz, fy, fx = pz - cz, py - cy, px - cx
    best = np.full(shape, np.inf, dtype=np.float32)
    for oz in (-1, 0, 1):
        iz = (cz + oz) % lz
        for oy in (-1, 0, 1):
            iy = (cy + oy) % ly
            for ox in (-1, 0, 1):
                ix = (cx + ox) % lx
                p = points[iz, iy, ix]  # broadcasts to (nz, ny, nx, 3)
                dz = oz + p[..., 0] - fz
                dy = oy + p[..., 1] - fy
                dx = ox + p[..., 2] - fx
                np.minimum(best, dz * dz + dy * dy + dx * dx, out=best)
    return 1.0 - np.clip(np.sqrt(best), 0.0, 1.0)


def _worley_fbm(shape: Tuple[int, int, int], cell_size_m: Tuple[float, float, float],
                blob_m: float, seed: int, octaves: int = 3) -> np.ndarray:
    """Three octaves of inverted Worley noise, blobs about ``blob_m`` across at the first.

    Weights 0.625 / 0.25 / 0.125, as in Schneider's Perlin-Worley recipe: big round lobes with
    smaller lobes on them, which is the cauliflower.
    """
    extent = [n * c for n, c in zip(shape, cell_size_m)]
    out = np.zeros(shape, dtype=np.float32)
    weights = (0.625, 0.25, 0.125, 0.0625)[:octaves]
    total = sum(weights)
    for octave, weight in enumerate(weights):
        size = blob_m / (2 ** octave)
        # Whole lattice cells across each periodic axis; at least one, and never finer than two
        # grid cells, below which the blobs alias into noise.
        lattice = tuple(max(1, min(int(round(e / size)), n // 2)) for e, n in zip(extent, shape))
        out += weight * _periodic_worley(shape, lattice, seed + 31 * octave)
    return out / total


def _upsample2_periodic(a: np.ndarray) -> np.ndarray:
    """Double every axis of a periodic grid with linear interpolation between cell centres."""
    for axis in range(a.ndim):
        prev = np.roll(a, 1, axis=axis)
        nxt = np.roll(a, -1, axis=axis)
        even = 0.75 * a + 0.25 * prev
        odd = 0.75 * a + 0.25 * nxt
        stacked = np.stack([even, odd], axis=axis + 1)
        shape = list(a.shape)
        shape[axis] *= 2
        a = stacked.reshape(shape)
    return a


def _billow_field(shape: Tuple[int, int, int], cell_size_m: Tuple[float, float, float],
                  blob_m: float, seed: int) -> np.ndarray:
    """The Worley term for the field's shape, computed at half resolution and interpolated up.

    The lobes are several cells across and smooth, so half resolution loses nothing visible and
    costs an eighth: the full-resolution version took five seconds per octave at the default grid.
    Two octaves; the fine detail texture carries the small billows.
    """
    if all(n % 2 == 0 and n >= 8 for n in shape):
        coarse = tuple(n // 2 for n in shape)
        cells = tuple(2.0 * c for c in cell_size_m)
        return _upsample2_periodic(_worley_fbm(coarse, cells, blob_m, seed, octaves=2))
    return _worley_fbm(shape, cell_size_m, blob_m, seed, octaves=2)


def _perlin_worley(noise: np.ndarray, worley: np.ndarray, amount: float) -> np.ndarray:
    """Schneider's remap: dilate the fractal noise where the Worley noise is high.

    ``remap(noise, amount * (worley - 1), 1, 0, 1)``. Where a Worley blob sits, the lower bound
    drops and the noise is pushed up, so the level set bulges out into a round lobe; between blobs
    it is left alone. ``amount`` 0 is the identity.
    """
    low = amount * (worley.astype(np.float64) - 1.0)
    return (noise - low) / (1.0 - low)


def _rank_normalise(values: np.ndarray) -> np.ndarray:
    """Replace every value by its rank, scaled to 0..1: a uniform distribution, same order."""
    flat = values.reshape(-1)
    ranks = np.empty(flat.size, dtype=np.float64)
    ranks[np.argsort(flat, kind="stable")] = np.arange(flat.size, dtype=np.float64)
    return (ranks / max(flat.size - 1, 1)).reshape(values.shape)


def _linear_weights(position: np.ndarray, n: int) -> np.ndarray:
    """Periodic linear-interpolation weights: row k reads ``position[k]`` (in cells) from n cells."""
    i0 = np.floor(position).astype(np.int64)
    t = position - i0
    weights = np.zeros((position.size, n))
    rows = np.arange(position.size)
    np.add.at(weights, (rows, i0 % n), 1.0 - t)
    np.add.at(weights, (rows, (i0 + 1) % n), t)
    return weights


def _column_is_cloudy(density: np.ndarray) -> np.ndarray:
    """Which columns of a (level, z, x) density grid an observer below would call cloudy.

    A column counts when it holds at least one full level's worth of density, summed through its
    height: at the shipped extinction that is an optical depth of about two, a cloud you cannot
    see the sky through. The earlier rule -- some level above one half -- ignored columns made of
    many partly-dense cells, which are just as opaque, and a requested 30 % cover rendered as 45 %.
    """
    return density.sum(axis=0) >= 1.0


def _profile_area(profile: CloudProfile, u: np.ndarray) -> np.ndarray:
    """Relative occupied area at fractional height ``u``, interpolated from the profile."""
    return np.interp(
        np.clip(u, 0.0, 1.0),
        np.asarray(profile.height_points, dtype=np.float64),
        np.asarray(profile.area_points, dtype=np.float64),
    )


@dataclass
class MarchResult:
    """What one ray found on its way through the cloud. Arrays, shaped like the rays."""

    #: Optical depth accumulated along the ray, in the **visible** band. Another band scales it.
    optical_depth: np.ndarray
    #: ``exp(-optical_depth)``: how much of what is behind the cloud survives.
    transmittance: np.ndarray
    #: Height at which the ray's extinction is concentrated, weighted by what reaches the sensor.
    #: This is the height whose air temperature an infrared band should be emitting at -- the
    #: cloud does not emit from its base, it emits from wherever the ray stopped seeing through.
    emission_height_m: np.ndarray
    #: Single-plus-multiple-scattered radiance, relative to the incident sunlight. ``None``
    #: unless a sun direction was given.
    radiance: np.ndarray | None = None
    #: Skylight scattered toward the sensor, as weights on the radiance the cloud is bathed in
    #: from **above** (the sky) and from **below** (the ground and the low sky): the caller's
    #: ``ambient_above * L_above + ambient_below * L_below`` is the light the cloud adds. Each
    #: sample contributes ``albedo * T * (1 - exp(-dtau))`` split by its height in the slab, so
    #: the two sum to ``albedo * (1 - transmittance)`` exactly, and a cloud under a uniform sky
    #: adds back precisely what it removed: the white furnace (docs/physics-model.md §7.5).
    ambient_above: np.ndarray | None = None
    ambient_below: np.ndarray | None = None


#: Henyey-Greenstein asymmetry for cloud droplets in the visible. Strongly forward-scattering,
#: which is why a cloud between you and the sun has a bright rim.
CLOUD_ASYMMETRY_G = 0.85
#: How bright a fully self-shadowed part of a cloud is, as a fraction of a fully lit one. No
#: longer used by the march, whose shadows are the two-stream transmission along each point's
#: own sun chord (:func:`_diffuse_sunlight`); kept because the dome's lit/shadow colour blend
#: (``sky._cloud_colour``) maps the march's brightness onto it.
MS_SHADOW_FLOOR = 0.45
#: Single-scattering albedo of cloud droplets in the visible. Liquid water at 0.55 um has an
#: imaginary index near 2e-9, so a 10 um droplet absorbs about one part in 10^5 of what it
#: intercepts; 0.9999 is that rounded toward absorbing. Every step's in-scatter is multiplied
#: by it. (The path-traced volumes use 0.999 for a different reason: see
#: ``backends/viewport/clouds_volume.DROPLET_ALBEDO``.)
VISIBLE_DROPLET_ALBEDO = 0.9999
#: The smallest sun cosine the shading divides by: a sun 3 degrees up. Below it the plane-parallel
#: forms run away, and the light is the twilight sky's, which the ambient terms carry.
MIN_SUN_COSINE = 0.05
#: Coarsening of the light maps the march reads (:meth:`CloudField.sun_optical_depth`),
#: horizontally and vertically. Half resolution horizontally costs a quarter; the levels stay
#: whole because the march reads the depth fraction across a lit face, which a coarse level
#: smears: a uniform 50-deep layer read 0.785 against its two-stream 0.830 at half vertical
#: resolution, within 1 % at full. Measured on the production grid: a dome bake of 3.1 s with
#: both halved, 12.0 s with neither.
LIGHT_MAP_COARSEN = 2
LIGHT_MAP_VERTICAL_COARSEN = 1


def _diffuse_sunlight(
    tau_toward: Any, tau_away: Any, g: float = CLOUD_ASYMMETRY_G
) -> Tuple[np.ndarray, np.ndarray]:
    """The two streams of multiply-scattered sunlight at a point inside a cloud, in units of the
    horizontal illuminance over pi: ``(backward, forward)``.

    One-dimensional along the sunlight's own chord through the point, which is what lets a
    broken field shade itself: ``tau_toward`` is the optical depth from the point to the cloud's
    edge toward the sun, ``tau_away`` to its edge the other way. Across that chord, of scaled
    thickness ``tau* = (1 - g)(tau_toward + tau_away)``, the conservative two-stream (Eddington,
    delta-scaled) layer reflects ``R = tau*/(2 + tau*)`` and diffusely transmits
    ``T = 2/(2 + tau*) - exp(-tau)``. The stream running back toward the sun carries ``R`` out of
    the lit face and nothing out of the dark one; the stream running on with the sunlight carries
    nothing in at the lit face and ``T`` out of the dark one. Both are taken linear in the depth
    fraction ``f = tau_toward / chord``: ``R (1 - f)`` and ``T f``.

    In chord form these are the plane-parallel ``R = (1-g) tau_v / (2 mu0 + (1-g) tau_v)`` and
    its transmission exactly, since a plane-parallel chord is ``tau_v / mu0``. The march weights
    the backward stream where a ray enters through a lit face (the depth fraction rising along
    it), the forward one where it enters through a dark face, both along a grazing ray, and
    in-scatters the source ``S = I - dI/dtau`` that implies along the ray, so that a uniform
    layer returns exactly ``R`` from above and ``T`` from below, at any optical depth
    (docs/physics-model.md §7.5). A
    sun-facing flank is lit at the layer's reflectance, a base at its transmission, and a wisp,
    where both are of order tau, by little but its single scattering.
    """
    toward = np.asarray(tau_toward, dtype=np.float64)
    chord = toward + np.asarray(tau_away, dtype=np.float64)
    scaled = (1.0 - g) * chord
    reflected = scaled / (2.0 + scaled)
    transmitted = np.clip(2.0 / (2.0 + scaled) - np.exp(-chord), 0.0, 1.0)
    depth = np.where(chord > 1e-12, toward / np.maximum(chord, 1e-12), 0.0)
    return reflected * (1.0 - depth), transmitted * depth


#: How far a ray that runs level *inside* the slab is followed before it is called done. A ray
#: exactly parallel to the base never leaves through either plane, so it needs a length of its
#: own rather than an infinity that would make the step size meaningless.
_LEVEL_RAY_LENGTH_M = 40_000.0
#: How fast the march's step size grows along the ray. 0 would be uniform spacing; 3 puts about
#: a fifth of the samples in the first tenth of the path, which is where the transmittance is
#: still high enough for a sample to matter.
_MARCH_GROWTH = 3.0


#: Default strength of the fine detail, in threshold units (see :attr:`CloudField.detail_strength`).
DETAIL_STRENGTH = 0.04
#: Cells per side of the fine detail texture, and how much finer its cell is than the field's.
_DETAIL_CELLS = 64
_DETAIL_REFINE = 4


def _hash_unit(*arrays: Any, seed: int = 0) -> np.ndarray:
    """A deterministic uniform number in [0, 1) per element, from the exact bits of the inputs.

    SplitMix64 over the float64 bit patterns. Neighbouring rays whose directions differ in the
    last bit get unrelated values, which is the point: a jitter that varies *smoothly* with the
    direction (what the march used before) prints its own contour lines -- concentric rings --
    through every cloud. This one turns the same residue into per-ray noise.
    """
    arrays = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64) for a in arrays])
    shape = arrays[0].shape
    with np.errstate(over="ignore"):
        h = np.full(shape, np.uint64((0x9E3779B97F4A7C15 ^ (int(seed) * 0xD1B54A32D192ED03))
                                     & 0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
        for a in arrays:
            bits = np.array(a, dtype=np.float64, order="C").view(np.uint64)
            z = h ^ bits
            z = z + np.uint64(0x9E3779B97F4A7C15)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            h = z ^ (z >> np.uint64(31))
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


@dataclass
class CloudField:
    """A periodic three-dimensional cloud, sampled in metres.

    The horizontal origin is the observer's own column, so ``density(0, base + h, 0)`` is the
    cloud directly overhead. The tile is ``cells * cell_m`` across and wraps exactly, so there is
    no edge to reach and no cold band across the bottom of an oblique frame.
    """

    cover: float
    base_m: float
    profile: CloudProfile
    cell_m: float = 60.0
    cells: int = 256
    levels: int = 48
    seed: int = 0
    thickness_m: float = 0.0
    optical_depth: float = 0.0
    #: Softness of the cloud boundary, in threshold units. Zero gives a hard stencil, which is
    #: what made an earlier model show a step of tens of kelvin along every cloud edge.
    softness: float = 0.06
    #: Diameter of the smallest feature the field carries, metres. 400 m is the low end of the
    #: observed fair-weather cumulus size mode; below it a field is carrying structure that no
    #: cloud of this genus has, which one band renders as specks and the other smears.
    feature_m: float = 400.0
    erosion_scale: float = 1.0
    beta_scale: float = 1.0
    #: Scales the genus' :attr:`CloudProfile.billow`. 0 turns the Worley lobes off.
    billow_scale: float = 1.0
    #: Amplitude of the fine detail, in threshold units. The field's own grid is 60 m, so on its
    #: own every edge is a 60 m ramp and every cloud a soft blob. A second, four-times-finer
    #: periodic noise is added to the field's signed distance from its threshold *at sample
    #: time*, which only matters where that distance is small -- the edge -- so the core stays as
    #: it was and the boundary gets 15 m structure. The idea is the one real-time cloud renderers
    #: use (Schneider 2015, and Unreal's volumetric clouds): coarse shape, fine erosion. 0 turns
    #: it off and gives back the plain interpolated grid.
    detail_strength: float = DETAIL_STRENGTH
    _density: np.ndarray = field(init=False, repr=False)
    #: The eroded noise minus its level's threshold: > 0 inside the cloud. float32.
    _signed: np.ndarray = field(init=False, repr=False, default=None)
    #: The fine detail texture, 0..1, periodic in all three axes.
    _detail: np.ndarray = field(init=False, repr=False, default=None)
    _detail_cell_m: Tuple[float, float] = field(init=False, repr=False, default=(0.0, 0.0))
    _eroded: np.ndarray = field(init=False, repr=False, default=None)
    #: Per-level cut value. Held for inspection and for the tests that pin the morphology.
    _thresholds: np.ndarray = field(init=False, repr=False, default=None)
    #: The solved overall scale on the profile's relative areas.
    _area_scale: float = field(init=False, default=0.0, repr=False)
    _sun_cache: Dict[Tuple[float, ...], np.ndarray] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.cover <= 1.0:
            raise ValueError(f"cover is a fraction of sky, got {self.cover}")
        if self.base_m < 0.0:
            raise ValueError("the cloud base cannot be below the ground")
        if self.cells < 8 or self.levels < 4:
            raise ValueError("the grid needs at least 8 x 8 x 4 cells")
        if self.thickness_m <= 0.0:
            self.thickness_m = self.profile.thickness_m
        if self.optical_depth <= 0.0:
            self.optical_depth = self.profile.optical_depth
        self._build()

    # --- construction ---------------------------------------------------------------------

    def _build(self) -> None:
        shape = (self.levels, self.cells, self.cells)
        if self.cover <= 0.0:
            self._density = np.zeros(shape, dtype=np.float32)
            self._thresholds = np.ones(self.levels, dtype=np.float64)
            self._area_scale = 0.0
            self._signed = None
            self._detail = None
            return

        # Cell sizes in metres, which is the frame the spectrum below is shaped in.
        vertical_m = self.thickness_m / self.levels
        cell_size_m = (vertical_m, self.cell_m, self.cell_m)
        
        beta = self.profile.beta * self.beta_scale
        erosion = self.profile.erosion * self.erosion_scale

        spacing_m = self.profile.spacing * self.feature_m
        noise = _periodic_fractal_noise(
            shape,
            beta,
            self.seed,
            cell_size_m=cell_size_m,
            min_wavelength_m=self.feature_m,
            anisotropy=self.profile.anisotropy,
            max_wavelength_m=spacing_m,
        )
        columnar = min(max(self.profile.columnar, 0.0), 1.0)
        if columnar > 0.0:
            # The coverage map: a 2D field of blobs about two features across, broadcast up the
            # column. Built as a one-level slab of the same noise so it tiles with the field.
            column = _periodic_fractal_noise(
                (1, self.cells, self.cells),
                beta,
                self.seed + 1543,
                cell_size_m=(vertical_m, self.cell_m, self.cell_m),
                min_wavelength_m=2.0 * self.feature_m,
                anisotropy=self.profile.anisotropy,
                max_wavelength_m=spacing_m,
            )
            cellular = min(max(self.profile.billow, 0.0), 1.0)
            if cellular > 0.0:
                # A convective field is a set of separate thermals, each its own cloud: a cellular
                # noise gives every cloud a centre to grow round, spaced about spacing_m apart.
                lattice = max(1, int(round(self.cells * self.cell_m / spacing_m)))
                cells = _periodic_worley((1, self.cells, self.cells), (1, lattice, lattice),
                                         self.seed + 2711)
                # The fractal decides how big each cell's cloud grows, and leaves some empty:
                # equal clouds on a lattice would read as a pattern, not as weather.
                modulated = _rank_normalise(cells * (0.35 + column))
                column = cellular * modulated + (1.0 - cellular) * column
            noise = columnar * column + (1.0 - columnar) * noise
        billow = min(max(self.profile.billow * self.billow_scale, 0.0), 1.0)
        if billow > 0.0:
            # Lobes about one and a half features across: a turret, with smaller turrets on it.
            worley = _billow_field(shape, cell_size_m, 1.5 * self.feature_m, self.seed + 3571)
            noise = _perlin_worley(noise, worley, billow)
        detail = _periodic_fractal_noise(
            shape,
            max(beta - 1.2, 1.0),
            self.seed + 7919,
            cell_size_m=cell_size_m,
            min_wavelength_m=max(0.35 * self.feature_m, 2.5 * max(self.cell_m, vertical_m)),
            anisotropy=self.profile.anisotropy,
        )
        # Erosion bites where the field is already near its own edge and leaves the core alone:
        # `noise * (1 - noise)` peaks at the midpoint and vanishes at both extremes.
        eroded = noise - erosion * (detail - 0.5) * 4.0 * noise * (1.0 - noise)

        # The fine detail, read at the cell centres. It is not zero-sum on the cloud: near a
        # convex edge there is more outside than inside, so adding it dilates the cloud -- by a
        # third of the requested cover, measured. So the cover solve, the thresholds and the
        # extinction calibration all run on the field *with* the detail, and the thresholds come
        # out where they have to be for the cloud density() returns to have the cover asked for.
        self._build_detail(vertical_m, beta)
        jitter = self._detail_at_centres()
        solved = eroded if jitter is None else eroded + jitter
        self._eroded = solved

        u = (np.arange(self.levels, dtype=np.float64) + 0.5) / self.levels
        relative_area = _profile_area(self.profile, u)

        # Sorting each level once turns every later quantile lookup into an index, which is what
        # makes the coverage solve below affordable at 48 bisection steps.
        per_level = np.sort(solved.reshape(self.levels, -1), axis=1)
        self._area_scale = self._solve_area_scale(per_level, relative_area)
        self._thresholds = self._level_thresholds(per_level, relative_area, self._area_scale)

        softness = 2.0 * max(self.softness, 1e-6)
        # _density holds what density() returns at each cell centre, detail included, so the
        # calibration, the cover and the plain grid all describe the same cloud.
        self._density = np.clip(
            0.5 + (solved - self._thresholds[:, None, None]) / softness, 0.0, 1.0
        ).astype(np.float32)
        # The detail-free signed field, which density() interpolates before adding the detail
        # at the sample's own position. An empty level carries a threshold above its maximum;
        # clamp so interpolation toward it stays a boundary rather than a cliff of -inf.
        self._signed = np.maximum(eroded - self._thresholds[:, None, None], -1.0).astype(np.float32)

    def _detail_at_centres(self):
        """``detail_strength * (2 d - 1)`` at every cell centre, or None without detail."""
        if self._detail is None:
            return None
        half = self.half_extent_m
        dv = self.thickness_m / self.levels
        heights = self.base_m + (np.arange(self.levels) + 0.5) * dv
        axis = -half + (np.arange(self.cells) + 0.5) * self.cell_m
        # The centres are a regular grid, so the trilinear read separates into one linear
        # interpolation per axis: three small matrix products instead of 3 million gathers --
        # the same numbers as _sample_detail, forty times faster.
        n = self._detail.shape[0]
        detail_v, detail_h = self._detail_cell_m
        wy = _linear_weights((heights - self.base_m) / detail_v - 0.5, n)
        wh = _linear_weights(axis / detail_h - 0.5, n)
        d = np.tensordot(self._detail.astype(np.float64), wh, axes=([2], [1]))   # (i, j, x)
        d = np.tensordot(d, wh, axes=([1], [1]))                                # (i, x, z)
        d = np.tensordot(wy, d, axes=([1], [0]))                                # (y, x, z)
        return self.detail_strength * (2.0 * np.transpose(d, (0, 2, 1)) - 1.0)

    def _build_detail(self, vertical_m: float, beta: float) -> None:
        """The fine detail texture: small, periodic, and with a period that divides the tile.

        Its horizontal period is a whole fraction of the field's tile, so the detail wraps with
        the field and the tiles still meet without a seam. Vertically it just repeats.
        """
        if self.detail_strength <= 0.0:
            self._detail = None
            return
        tile = self.cells * self.cell_m
        repeats = max(1, int(round(tile / (_DETAIL_CELLS * self.cell_m / _DETAIL_REFINE))))
        horizontal = tile / repeats / _DETAIL_CELLS
        vertical = vertical_m / 2.0
        self._detail_cell_m = (vertical, horizontal)
        detail_shape = (_DETAIL_CELLS, _DETAIL_CELLS, _DETAIL_CELLS)
        detail = _periodic_fractal_noise(
            detail_shape,
            max(beta - 0.6, 1.4),
            self.seed + 104729,
            cell_size_m=(vertical, horizontal, horizontal),
            min_wavelength_m=4.0 * horizontal,
            anisotropy=self.profile.anisotropy,
        )
        billow = min(max(self.profile.billow * self.billow_scale, 0.0), 1.0)
        if billow > 0.0:
            # Small billows on the edge rather than fuzz: the same Perlin-Worley remap, at the
            # detail's own scale, then back to a uniform 0..1 so the strength keeps its meaning.
            worley = _worley_fbm(detail_shape, (vertical, horizontal, horizontal),
                                 16.0 * horizontal, self.seed + 7907)
            detail = _perlin_worley(detail, worley, billow)
            detail = _rank_normalise(detail)
        self._detail = detail.astype(np.float32)

    @staticmethod
    def _level_thresholds(
        per_level: np.ndarray, relative_area: np.ndarray, scale: float
    ) -> np.ndarray:
        """The value at each level above which exactly ``scale * relative_area`` of it lies."""
        n = per_level.shape[1]
        target = np.clip(scale * relative_area, 0.0, 1.0)
        index = np.clip(np.rint((1.0 - target) * (n - 1)).astype(np.int64), 0, n - 1)
        out = per_level[np.arange(per_level.shape[0]), index]
        # An area of exactly zero must mean *no cloud*, not "the single largest cell": nudge the
        # threshold above the level's maximum so the comparison is empty.
        return np.where(target <= 0.0, per_level[:, -1] + 1.0, out)

    def _solve_area_scale(self, per_level: np.ndarray, relative_area: np.ndarray) -> float:
        """Bisect the profile's overall scale until the *sky cover* comes out as asked.

        Cover is measured the way an observer measures it -- the fraction of vertical columns
        containing any cloud at all -- rather than as a fraction of occupied volume. The two are
        very different: a field of tall narrow towers fills little volume and hides much of the
        sky, and a model that confuses them reports four oktas while rendering eight.

        The relation is monotone (more area at every level can only cover more columns), which is
        what makes a bisection the right tool and what makes the answer unique.
        """
        eroded_shape = (self.levels, self.cells, self.cells)

        # Measured on every other column each way: a quarter of the work, and the cover of a
        # field with 400 m features does not change at that sampling.
        eroded = self._eroded[:, ::2, ::2]
        softness = 2.0 * max(self.softness, 1e-6)

        def covered(scale: float) -> float:
            thresholds = self._level_thresholds(per_level, relative_area, scale)
            density = np.clip(0.5 + (eroded - thresholds[:, None, None]) / softness, 0.0, 1.0)
            return float(np.mean(_column_is_cloudy(density)))

        assert self._eroded.shape == eroded_shape
        lo, hi = 0.0, 1.0
        if covered(hi) < self.cover:
            return hi
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if covered(mid) < self.cover:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # --- sampling -------------------------------------------------------------------------

    @property
    def top_m(self) -> float:
        return self.base_m + self.thickness_m

    @property
    def half_extent_m(self) -> float:
        return 0.5 * self.cells * self.cell_m

    @property
    def finest_pitch_m(self) -> float:
        """The finest spacing the field carries, metres: the smallest of a cell, a level and,
        with the edge detail on, the detail texture's own cell.

        A march has to sample at this spacing (twice per pitch, for a trilinear field) to see
        every feature the field holds. One sized by :attr:`cell_m` steps over the edge detail --
        the 15 m structure that is most of what makes a boundary look like cloud -- and on an
        oblique infrared ray that alone is a quarter kelvin of error. Public because the
        thermal-camera repo sizes its march by it; :attr:`HeroClouds.finest_pitch_m` answers the
        same question for the volume-asset source, so a march need not know which it is given.
        """
        pitch = min(self.cell_m, self.thickness_m / self.levels)
        if self._detail is not None:
            pitch = min(pitch, *self._detail_cell_m)
        return float(pitch)

    @property
    def extinction_per_m(self) -> float:
        """Visible extinction at unit density, per metre.

        ``optical_depth`` is calibrated on the **median cloudy column**, which is what the phrase
        means in meteorology and what a published figure for a genus refers to. Normalising on the
        mean over the whole field instead makes the number depend on the coverage: the same cloud
        at four oktas would be twice as thick as at eight, which is exactly backwards.
        """
        if self._density.size == 0:
            return 0.0
        columns = self._density.sum(axis=0) * (self.thickness_m / self.levels)
        cloudy = columns[columns > 0.05 * self.thickness_m * 0.01]
        if cloudy.size == 0:
            return 0.0
        median = float(np.median(cloudy))
        if median <= 1e-9:
            return 0.0
        return self.optical_depth / median

    @property
    def measured_cover(self) -> float:
        """The sky fraction the built field actually covers. Should equal ``cover``."""
        if self._density.size == 0:
            return 0.0
        return float(np.mean(_column_is_cloudy(self._density)))

    def density(self, x_m: Any, y_m: Any, z_m: Any) -> np.ndarray:
        """Normalised cloud density in 0..1 at a world position, trilinearly interpolated.

        ``y_m`` is height above the ground. Outside the slab the density is zero; horizontally the
        field wraps, so there is no edge. Broadcasting follows numpy's rules, so a whole ray or a
        whole frame of rays can be sampled in one call -- which is what makes a per-pixel march
        affordable.
        """
        if self._detail is None or self._signed is None:
            return self._sample(self._density, x_m, y_m, z_m)
        x = np.asarray(x_m, dtype=np.float64)
        y = np.asarray(y_m, dtype=np.float64)
        z = np.asarray(z_m, dtype=np.float64)
        x, y, z = np.broadcast_arrays(x, y, z)
        out = np.zeros(x.shape, dtype=np.float64)
        inside = (y >= self.base_m) & (y <= self.top_m)
        if not np.any(inside):
            return out
        xi, yi, zi = x[inside], y[inside], z[inside]
        signed = self._sample(self._signed, xi, yi, zi)
        # Only the band near the edge can be changed by the detail; everything else is already
        # decided, so the second lookup is paid only there.
        softness = max(self.softness, 1e-6)
        reach = self.detail_strength + softness
        edge = np.abs(signed) < reach
        if np.any(edge):
            signed[edge] += self.detail_strength * (
                2.0 * self._sample_detail(xi[edge], yi[edge], zi[edge]) - 1.0)
        out[inside] = np.clip(0.5 + signed / (2.0 * softness), 0.0, 1.0)
        return out

    def _sample_detail(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Trilinear read of the detail texture, wrapping on all three axes."""
        grid = self._detail
        n = grid.shape[0]
        dv, dh = self._detail_cell_m
        fi = (y - self.base_m) / dv - 0.5
        fj = z / dh - 0.5
        fk = x / dh - 0.5
        i0 = np.floor(fi).astype(np.int64)
        j0 = np.floor(fj).astype(np.int64)
        k0 = np.floor(fk).astype(np.int64)
        ti, tj, tk = fi - i0, fj - j0, fk - k0
        i0, i1 = i0 % n, (i0 + 1) % n
        j0, j1 = j0 % n, (j0 + 1) % n
        k0, k1 = k0 % n, (k0 + 1) % n
        c00 = grid[i0, j0, k0] * (1 - tk) + grid[i0, j0, k1] * tk
        c01 = grid[i0, j1, k0] * (1 - tk) + grid[i0, j1, k1] * tk
        c10 = grid[i1, j0, k0] * (1 - tk) + grid[i1, j0, k1] * tk
        c11 = grid[i1, j1, k0] * (1 - tk) + grid[i1, j1, k1] * tk
        return (c00 * (1 - tj) + c01 * tj) * (1 - ti) + (c10 * (1 - tj) + c11 * tj) * ti

    # --- microphysics ----------------------------------------------------------------------

    @property
    def microphysics(self) -> CloudMicrophysics:
        """The cloud as water: phase, particle sizes and water content at unit density."""
        return microphysics_for(self.profile, self.extinction_per_m)

    def liquid_water_content(self, x_m: Any, y_m: Any, z_m: Any) -> np.ndarray:
        """Liquid water content at a position, g/m^3. Zero outside the cloud."""
        return self.density(x_m, y_m, z_m) * self.microphysics.liquid_water_g_m3

    def ice_water_content(self, x_m: Any, y_m: Any, z_m: Any) -> np.ndarray:
        """Ice water content at a position, g/m^3. Zero outside the cloud and in a warm cloud."""
        return self.density(x_m, y_m, z_m) * self.microphysics.ice_water_g_m3

    def _sample(self, grid: np.ndarray, x_m: Any, y_m: Any, z_m: Any) -> np.ndarray:
        """Trilinear read of any grid shaped like the density: wrapping horizontally, clamped
        vertically, zero outside the slab. One implementation, so the light map and the density
        cannot end up sampled on two subtly different grids."""
        x = np.asarray(x_m, dtype=np.float64)
        y = np.asarray(y_m, dtype=np.float64)
        z = np.asarray(z_m, dtype=np.float64)
        x, y, z = np.broadcast_arrays(x, y, z)
        out = np.zeros(x.shape, dtype=np.float64)
        if self.cover <= 0.0:
            return out

        inside = (y >= self.base_m) & (y <= self.top_m)
        if not np.any(inside):
            return out

        # Cell centres sit at index + 0.5, so subtract it back off before interpolating.
        fi = (y[inside] - self.base_m) / self.thickness_m * self.levels - 0.5
        fj = z[inside] / self.cell_m + 0.5 * self.cells - 0.5
        fk = x[inside] / self.cell_m + 0.5 * self.cells - 0.5

        i0 = np.floor(fi).astype(np.int64)
        j0 = np.floor(fj).astype(np.int64)
        k0 = np.floor(fk).astype(np.int64)
        ti, tj, tk = fi - i0, fj - j0, fk - k0

        i0c = np.clip(i0, 0, self.levels - 1)
        i1c = np.clip(i0 + 1, 0, self.levels - 1)
        j0w, j1w = j0 % self.cells, (j0 + 1) % self.cells
        k0w, k1w = k0 % self.cells, (k0 + 1) % self.cells

        d = grid
        c00 = d[i0c, j0w, k0w] * (1 - tk) + d[i0c, j0w, k1w] * tk
        c01 = d[i0c, j1w, k0w] * (1 - tk) + d[i0c, j1w, k1w] * tk
        c10 = d[i1c, j0w, k0w] * (1 - tk) + d[i1c, j0w, k1w] * tk
        c11 = d[i1c, j1w, k0w] * (1 - tk) + d[i1c, j1w, k1w] * tk
        c0 = c00 * (1 - tj) + c01 * tj
        c1 = c10 * (1 - tj) + c11 * tj
        out[inside] = c0 * (1 - ti) + c1 * ti
        return out

    def volume_grid(self, up_axis: int = 1, refine: int = 1) -> "VolumeGrid":
        """The density as a voxel array in **stage axes**, ready for an OpenVDB fog volume.

        The field's own frame is the Y-up sky frame: ``x`` east, ``y`` up, ``z`` south. A Y-up
        stage is that frame. A Z-up stage is it turned by +90 degrees about X -- the same turn the
        dome light gets -- so field ``z`` (south) is stage ``-y`` and field ``y`` is stage ``z``.
        Getting this wrong would not look wrong: a random field is a random field either way round.
        It would only put the rendered cloud somewhere other than where the infrared march puts it.

        Values are normalised density in 0..1; multiply by :attr:`extinction_per_m` for the visible
        extinction. Voxel centres sit where :meth:`density` puts its cell centres, so the grid and
        the field agree to the sample.

        ``refine`` > 1 samples :meth:`density` -- detail included -- on a grid that many times
        finer horizontally (and vertically as far as needed to keep the voxels no taller than they
        are wide). At 1 the voxels are the field's own cells, which are too coarse to show the
        fine detail, though they still hold what :meth:`density` returns at their centres.
        """
        refine = max(1, int(refine))
        dv = self.thickness_m / self.levels
        half = self.half_extent_m
        cell = self.cell_m
        grid = self._density  # (level, z, x)
        if (refine > 1 or self._detail is not None) and self.cover > 0.0:
            # With detail on, the voxels are sampled from density() even at the field's own
            # resolution, so the volume still holds exactly what every other consumer samples.
            cell = self.cell_m / refine
            vertical_refine = 1 if refine == 1 else max(1, int(math.ceil(dv / cell - 1e-9)))
            dv = dv / vertical_refine
            n = self.cells * refine
            axis = -half + (np.arange(n) + 0.5) * cell
            zz, xx = np.meshgrid(axis, axis, indexing="ij")
            grid = np.empty((self.levels * vertical_refine, n, n), dtype=np.float32)
            for level in range(grid.shape[0]):
                height = self.base_m + (level + 0.5) * dv
                grid[level] = self.density(xx, np.full_like(xx, height), zz)
        if up_axis == 1:
            array = np.transpose(grid, (2, 0, 1))                 # (x, y_up, z)
            voxel = (cell, dv, cell)
            first = (-half + 0.5 * cell, self.base_m + 0.5 * dv, -half + 0.5 * cell)
        elif up_axis == 2:
            array = np.transpose(grid, (2, 1, 0))[:, ::-1, :]     # (x, y = -z_field, z_up)
            voxel = (cell, cell, dv)
            first = (-half + 0.5 * cell, -half + 0.5 * cell, self.base_m + 0.5 * dv)
        else:
            raise ValueError("up_axis is 1 (Y) or 2 (Z)")
        return VolumeGrid(
            values=np.ascontiguousarray(array, dtype=np.float32),
            voxel_m=tuple(float(v) for v in voxel),  # type: ignore[arg-type]
            first_centre_m=tuple(float(v) for v in first),  # type: ignore[arg-type]
            tile_m=float(self.cells * self.cell_m),
        )

    def optical_depth_toward(self, origin_m: Any, direction: Any, steps: int = 64) -> float:
        """Visible optical depth from a point toward a direction (the field's frame, metres).

        What decides whether the ground at ``origin_m`` is in a cloud's shadow: the sun's beam
        reaching it is ``exp(-tau)`` of what it would be in clear air.
        """
        o = np.asarray(origin_m, dtype=np.float64).reshape(3)
        d = np.asarray(direction, dtype=np.float64).reshape(3)
        d = d / max(float(np.linalg.norm(d)), 1e-12)
        if d[1] <= 1e-3 or self.cover <= 0.0:
            return 0.0 if d[1] > 1e-3 else float("inf")
        t0 = max((self.base_m - o[1]) / d[1], 0.0)
        t1 = max((self.top_m - o[1]) / d[1], 0.0)
        if t1 <= t0:
            return 0.0
        t = t0 + (np.arange(steps) + 0.5) / steps * (t1 - t0)
        p = o[None, :] + t[:, None] * d[None, :]
        rho = self.density(p[:, 0], p[:, 1], p[:, 2])
        return float(rho.sum() * (t1 - t0) / steps * self.extinction_per_m)

    def column_optical_depth(self, x_m: Any = 0.0, z_m: Any = 0.0, samples: int = 64) -> Any:
        """Vertical optical depth through the column over ``(x, z)``. The overhead answer."""
        heights = self.base_m + (np.arange(samples) + 0.5) / samples * self.thickness_m
        x = np.broadcast_to(np.asarray(x_m, dtype=np.float64), ())
        z = np.broadcast_to(np.asarray(z_m, dtype=np.float64), ())
        d = self.density(np.full(samples, float(x)), heights, np.full(samples, float(z)))
        return float(np.sum(d) * (self.thickness_m / samples) * self.extinction_per_m)

    # --- lighting and marching ---------------------------------------------------------------

    def sun_transmittance(
        self, sun_direction: Any, *, steps: int = 24, coarsen: int = 2
    ) -> np.ndarray:
        """Transmittance from every grid cell **toward the sun**: ``exp(-sun_optical_depth)``.

        This is the one quantity that turns a cloud from a cut-out into a cloud. Without it every
        cloudy sample is equally bright, the alpha saturates the moment the optical depth passes
        about 3, and the result is a flat white shape with a hard edge -- which is what a viewer
        reads as "grey blob". With it, a flank facing the sun is bright, the base is dark because
        a kilometre of its own cloud is between it and the sun, and a thin edge glows because
        almost nothing is.

        For sampling *between* cells use :meth:`sun_optical_depth` and exponentiate after: this
        grid interpolated directly is biased bright wherever a cell's neighbour is clear.
        """
        return np.exp(-self.sun_optical_depth(sun_direction, steps=steps, coarsen=coarsen)
                      ).astype(np.float32)

    def sun_optical_depth(
        self, sun_direction: Any, *, steps: int = 24, coarsen: int = 2,
        vertical_coarsen: Optional[int] = None,
    ) -> np.ndarray:
        """Visible optical depth from every grid cell to the cloud's edge **toward the sun**,
        cached per direction. float32, shaped like the density.

        Stored and interpolated as an optical depth, not as a transmittance. The two differ at
        every cloud boundary: halfway between a cell under 40 of optical depth and a clear one,
        interpolated transmittance is 0.5 -- an optical depth of 0.7 -- so a cloud's base read
        as almost unshadowed. Measured on a cumulus field at a 58 degree sun, the interpolated
        transmittance put the depth above cloud bases at 0.6 of a fine reference integration;
        interpolated depth agrees with it (docs/physics-model.md §7.5).

        Computed on a grid coarsened by ``coarsen`` horizontally and ``vertical_coarsen``
        (default: the same) vertically, and sampled back up. The light field varies far more
        smoothly than the density does -- it is an integral of it -- except across a cloud's lit
        face, where the depth runs from nothing to several optical depths within one coarse
        level; the march reads the depth *fraction* there, so it keeps the levels at full
        resolution (:data:`LIGHT_MAP_VERTICAL_COARSEN`).
        """
        direction = np.asarray(sun_direction, dtype=np.float64)
        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
        cv = int(coarsen if vertical_coarsen is None else vertical_coarsen)
        key = tuple(np.round(direction, 4)) + (int(steps), int(coarsen), cv)
        cached = self._sun_cache.get(key)
        if cached is not None:
            return cached

        nz = max(self.levels // cv, 2)
        nc = max(self.cells // coarsen, 4)
        heights = self.base_m + (np.arange(nz) + 0.5) / nz * self.thickness_m
        axis = (np.arange(nc) + 0.5) / nc * (self.cells * self.cell_m) - self.half_extent_m
        y, z, x = np.meshgrid(heights, axis, axis, indexing="ij")

        # Step far enough to leave the slab even for a low sun, but cap the cost.
        sin_el = max(abs(float(direction[1])), 0.12)
        span = self.thickness_m / sin_el
        step = span / steps
        optical = np.zeros(y.shape, dtype=np.float64)
        for k in range(steps):
            t = (k + 0.5) * step
            optical += self.density(x + direction[0] * t, y + direction[1] * t, z + direction[2] * t)
        optical = (optical * step * self.extinction_per_m).astype(np.float32)

        if coarsen > 1 or cv > 1:
            optical = np.repeat(
                np.repeat(np.repeat(optical, cv, 0), coarsen, 1), coarsen, 2
            )[: self.levels, : self.cells, : self.cells]
            if optical.shape != self._density.shape:
                pad = [(0, t - s) for s, t in zip(optical.shape, self._density.shape)]
                optical = np.pad(optical, pad, mode="edge")
        self._sun_cache[key] = optical
        return optical

    def slab_span(self, origin_m: Any, direction: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Where a ray enters and leaves the cloud slab, as distances along it.

        Returns ``(enter, exit)`` with ``exit <= enter`` for rays that miss it entirely, which is
        the caller's cue to skip them. Rays that start inside the slab enter at zero.
        """
        origin = np.asarray(origin_m, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        oy = origin[..., 1]
        dy = direction[..., 1]
        oy, dy = np.broadcast_arrays(oy, dy)

        near = np.zeros(dy.shape, dtype=np.float64)
        far = np.zeros(dy.shape, dtype=np.float64)
        steep = np.abs(dy) > 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            t_base = (self.base_m - oy) / dy
            t_top = (self.top_m - oy) / dy
        lo = np.where(steep, np.minimum(t_base, t_top), 0.0)
        hi = np.where(steep, np.maximum(t_base, t_top), 0.0)
        # A level ray never crosses either plane: it is either inside the slab for its whole
        # length or outside it for all of it.
        level_inside = (~steep) & (oy >= self.base_m) & (oy <= self.top_m)
        near = np.where(steep, np.maximum(lo, 0.0), np.where(level_inside, 0.0, 1.0))
        far = np.where(steep, np.maximum(hi, 0.0), np.where(level_inside, _LEVEL_RAY_LENGTH_M, 0.0))
        return near, far

    def march(
        self,
        origin_m: Any,
        direction: Any,
        *,
        sun_direction: Any = None,
        steps: int = 64,
        ambient: float = 0.0,
        max_path_m: float = 8000.0,
        jitter_seed: int = 0,
    ) -> MarchResult:
        """Integrate the field along a ray, for whichever band is asking.

        One march serves both. An infrared band takes ``optical_depth`` (scaled by its own
        extinction ratio) and ``emission_height_m``; a visible band takes ``radiance``. They
        cannot disagree about where the cloud is, because they are the same integral over the same
        array -- which is the property the whole module exists to guarantee.

        ``emission_height_m`` is the transmittance-weighted mean height of the extinction along
        the ray. It is the height an infrared band should be emitting at: a cloud does not radiate
        from its base, it radiates from wherever the ray stopped being able to see through, and on
        an oblique ray through a broken field that can be most of a kilometre higher.

        ``radiance`` is the sunlight the cloud sends toward the sensor, in units of the
        horizontal illuminance over pi -- the radiance of a white Lambertian ground in the same
        sun. Each step in-scatters, with the energy-conserving weight ``T (1 - e^-dtau)`` times
        the droplet albedo, two lights (docs/physics-model.md §7.5):

        * the direct beam scattered once, through a Henyey-Greenstein phase function at
          :data:`CLOUD_ASYMMETRY_G` and the sample's transmittance toward the sun -- exact for a
          thin cloud, and the silver lining;
        * the multiply-scattered light of :func:`_diffuse_sunlight`, the two-stream field along
          the sunlight's chord through the sample -- the stream leaving whichever face the ray
          looks into, in-scattered as the source ``S = I - dI/dtau`` it implies along the
          ray -- which gives a uniform
          deck its two-stream albedo from
          above and its transmission from below at any optical depth, a sun-facing flank the
          layer's reflectance and a base its transmission.

        The sky's and the ground's light are ``ambient_above`` and ``ambient_below``, returned
        with or without a sun. ``ambient`` adds a constant to the sun's source and is 0; it used
        to stand in for them.
        """
        origin = np.asarray(origin_m, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        direction = direction / np.maximum(
            np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12
        )
        near, far = self.slab_span(origin, direction)
        shape = near.shape

        optical = np.zeros(shape, dtype=np.float64)
        transmittance = np.ones(shape, dtype=np.float64)
        height_weight = np.zeros(shape, dtype=np.float64)
        height_sum = np.zeros(shape, dtype=np.float64)
        radiance = None if sun_direction is None else np.zeros(shape, dtype=np.float64)
        ambient_above = np.zeros(shape, dtype=np.float64)
        ambient_below = np.zeros(shape, dtype=np.float64)

        hit = far > near
        if not np.any(hit) or self.cover <= 0.0:
            return MarchResult(
                optical_depth=optical,
                transmittance=transmittance,
                emission_height_m=np.full(shape, self.base_m),
                radiance=radiance,
                ambient_above=ambient_above,
                ambient_below=ambient_below,
            )

        toward = away = None
        if sun_direction is not None:
            sun = np.asarray(sun_direction, dtype=np.float64)
            sun = sun / max(float(np.linalg.norm(sun)), 1e-12)
            # The optical depth from every cell to the cloud's edge toward the sun and away from
            # it: together, the chord the sunlight takes through each point.
            toward = self.sun_optical_depth(sun, coarsen=LIGHT_MAP_COARSEN,
                                            vertical_coarsen=LIGHT_MAP_VERTICAL_COARSEN)
            away = self.sun_optical_depth(-sun, coarsen=LIGHT_MAP_COARSEN,
                                          vertical_coarsen=LIGHT_MAP_VERTICAL_COARSEN)
            g = CLOUD_ASYMMETRY_G
            cos_theta = np.sum(direction * sun, axis=-1)
            # Henyey-Greenstein, normalised so an isotropic phase function is 1. The direct
            # beam's single scattering, in units of the horizontal illuminance over pi -- the
            # unit the two-stream reflectance below is in -- is p / (4 mu_sun): exact as the
            # optical depth goes to zero, and the whole of the silver lining.
            mu_sun = max(abs(float(sun[1])), MIN_SUN_COSINE)
            single = (1.0 - g * g) / np.power(1.0 + g * g - 2.0 * g * cos_theta, 1.5)
            single = single / (4.0 * mu_sun)
            backward_prev = np.zeros(shape, dtype=np.float64)
            forward_prev = np.zeros(shape, dtype=np.float64)
            depth_entry = np.zeros(shape, dtype=np.float64)
            tau_since_entry = np.zeros(shape, dtype=np.float64)
            d_tau_prev = np.zeros(shape, dtype=np.float64)

        # A ray at one degree of elevation crosses 69 km of a 1.2 km slab, and marching that
        # with a fixed step count puts the samples 1.4 km apart -- coarser than the clouds. The
        # path is capped instead, which costs nothing real: at the shipped extinction a ray
        # accumulates unit optical depth in about 70 m of cloud, so anything beyond the cap is
        # behind a transmittance that has already underflowed. It is a **limit on the geometry,
        # not on the physics**, and it is the reason the step size stays near 100 m.
        ox, oy, oz = origin[..., 0], origin[..., 1], origin[..., 2]
        dx, dy, dz = direction[..., 0], direction[..., 1], direction[..., 2]
        span = np.where(hit, np.minimum(far - near, max_path_m), 0.0)

        # **Geometric steps, not uniform ones.** A ray at 60 degrees of elevation crosses 1.4 km
        # of slab and one at 5 degrees crosses the full cap, so a fixed step *count* samples the
        # second at ten times the spacing of the first -- coarser than the clouds themselves,
        # which is what draws the vertical streaks across the bottom of a frame. Placing the
        # sample boundaries geometrically puts them close together near the camera, where the
        # transmittance is still high and the detail is resolved, and lets them spread out where
        # the cloud is already behind an optical depth of ten.
        #
        # The jitter is what turns the residue from banding into noise: the boundaries are offset
        # by a per-ray fraction of a step. It is a **hash** of the ray's direction, bit for bit
        # (not its origin, so moving the observer by a whole tile still changes nothing), so it
        # is deterministic -- a frame is reproducible -- but uncorrelated
        # between neighbouring rays. It used to be a smooth function of the direction, and a
        # smooth jitter has contour lines: they showed up as concentric rings through every cloud
        # in the infrared band. ``jitter_seed`` changes the pattern, for averaging several frames.
        u = np.linspace(0.0, 1.0, steps + 1)
        growth = np.expm1(_MARCH_GROWTH * u) / math.expm1(_MARCH_GROWTH)
        jitter = (_hash_unit(dx, dy, dz, seed=jitter_seed)[..., None] - 0.5) / steps
        edges = span[..., None] * np.clip(growth + jitter, 0.0, 1.0)
        edges = np.sort(edges, axis=-1)
        centres = near[..., None] + 0.5 * (edges[..., 1:] + edges[..., :-1])
        widths = np.diff(edges, axis=-1)
        for k in range(steps):
            t = centres[..., k]
            step = widths[..., k]
            px, py, pz = ox + dx * t, oy + dy * t, oz + dz * t
            sigma = self.density(px, py, pz) * self.extinction_per_m
            d_tau = sigma * step
            # The weight is what *reaches the sensor* from this sample: extinction times the
            # transmittance back along the ray. Weighting by extinction alone would average in
            # cloud the sensor cannot see.
            weight = sigma * transmittance * step
            height_sum += weight * py
            height_weight += weight
            # Energy-conserving in-scatter of the skylight (Hillaire 2015): what this step
            # removes from the ray, ``T (1 - e^-dtau)``, times the albedo, comes back as the
            # light the medium is bathed in. Split by height: a sample at the top sees the sky,
            # one at the base the ground, and the diffusion profile in between is close to linear.
            scatter = VISIBLE_DROPLET_ALBEDO * transmittance * (1.0 - np.exp(-d_tau))
            height = np.clip((py - self.base_m) / max(self.thickness_m, 1e-9), 0.0, 1.0)
            ambient_above += scatter * height
            ambient_below += scatter * (1.0 - height)
            if radiance is not None:
                tau_toward = self._sample(toward, px, py, pz)
                tau_away = self._sample(away, px, py, pz)
                backward, forward = _diffuse_sunlight(tau_toward, tau_away)
                chord = tau_toward + tau_away
                depth = np.where(chord > 1e-12, tau_toward / np.maximum(chord, 1e-12), 0.0)
                inside = d_tau > 0.0
                both = inside & (d_tau_prev > 0.0)
                gap = np.maximum(0.5 * (d_tau + d_tau_prev), 1e-9)
                # Which stream travels toward the viewer. Going into the cloud through its lit
                # face the depth fraction rises along the ray, and what comes back out is the
                # backward stream; through its dark face it falls, and it is the forward one. A
                # grazing ray, along which it hardly changes, reads both: the face's own value.
                # So I = backward + forward, less the stream travelling away from the viewer, by
                # how clearly the ray crosses the profile (u = +-1 straight through a layer).
                # Judged from the change since the ray entered this cloud, not from one step: the
                # light maps' quadrature makes the depth fraction a staircase at the scale of a
                # light-map step, and a step's change flickers between nothing and a jump.
                depth_entry = np.where(both, depth_entry, depth)
                tau_since_entry = np.where(both, tau_since_entry + gap, 0.0)
                crossing = chord * (depth - depth_entry) / np.maximum(tau_since_entry, gap)
                u = np.where(both, np.tanh(2.0 * crossing), 0.0)
                keep_forward = 1.0 - np.maximum(u, 0.0)
                keep_backward = 1.0 - np.maximum(-u, 0.0)
                stream = keep_backward * backward + keep_forward * forward
                # The source that stream implies: dI/dtau = I - S along the ray (tau increasing
                # away from the viewer), so S = I - dI/dtau. Straight through a uniform layer it
                # returns the face value exactly at any depth; a grazing ray, with no gradient,
                # gets its own optical depth's share and no more. Differenced only between two
                # samples in cloud (outside it the stream is not zero but undefined), and with
                # this sample's choice of stream applied to both, so that the choice itself never
                # reads as a gradient.
                before = keep_backward * backward_prev + keep_forward * forward_prev
                slope = np.where(both, (stream - before) / gap, 0.0)
                diffuse = np.maximum(stream - slope, 0.0)
                radiance += scatter * (single * np.exp(-tau_toward) + diffuse + ambient)
                backward_prev = np.where(inside, backward, 0.0)
                forward_prev = np.where(inside, forward, 0.0)
                d_tau_prev = d_tau
            optical += d_tau
            transmittance = transmittance * np.exp(-d_tau)

        emission_height = np.where(
            height_weight > 1e-12,
            height_sum / np.maximum(height_weight, 1e-12),
            self.base_m,
        )
        return MarchResult(
            optical_depth=optical,
            transmittance=transmittance,
            emission_height_m=emission_height,
            radiance=radiance,
            ambient_above=ambient_above,
            ambient_below=ambient_below,
        )


@dataclass(frozen=True)
class VolumeGrid:
    """A cloud's density on a voxel grid in stage axes and metres. See :meth:`CloudField.volume_grid`."""

    values: np.ndarray                          # (nx, ny, nz) float32, 0..1
    voxel_m: Tuple[float, float, float]
    first_centre_m: Tuple[float, float, float]  # world position of voxel (0, 0, 0)'s centre
    tile_m: float                               # the field repeats every tile_m horizontally

    @property
    def low_m(self) -> Tuple[float, float, float]:
        return tuple(c - 0.5 * v for c, v in zip(self.first_centre_m, self.voxel_m))  # type: ignore[return-value]

    @property
    def high_m(self) -> Tuple[float, float, float]:
        return tuple(lo + n * v for lo, n, v in zip(self.low_m, self.values.shape, self.voxel_m))  # type: ignore[return-value]


#: The cloud parameters that change the field's *shape*. Anything else (colour, density scale,
#: march steps) can change without paying for a rebuild.
CLOUD_SHAPE_KEYS = (
    "enabled", "cover", "genus", "base_m", "temperature_c", "dewpoint_c", "thickness_m",
    "optical_depth", "feature_m", "erosion_scale", "beta_scale", "cells", "levels", "cell_m",
    "seed", "detail_strength", "billow_scale",
)

_FIELD_CACHE: Dict[Tuple[Any, ...], "CloudField"] = {}


def cloud_field_from_state(state: Any, build: bool = True):
    """The cloud field a state describes, or ``None`` for a clear sky. Cached on its shape keys.

    A field of the default size takes seconds to synthesise, and the dome, the volume and any
    sensor model all want the same one: building it once per shape is the difference between a
    colour change being instant and it costing a rebuild. The cache holds the last two shapes.
    ``build=False`` returns the cached field or ``None`` and never pays for one -- for callers on
    the UI thread.
    """
    clouds = state.clouds
    if not (clouds.enabled and clouds.cover > 0.0):
        return None
    key = tuple(getattr(clouds, name) for name in CLOUD_SHAPE_KEYS)
    cached = _FIELD_CACHE.get(key)
    if cached is not None or not build:
        return cached
    base = clouds.base_m
    if base <= 0.0:
        base = lifting_condensation_level_m(clouds.temperature_c, clouds.dewpoint_c)
    built = CloudField(
        cover=float(clouds.cover),
        base_m=float(base),
        profile=cloud_profile(clouds.genus),
        cell_m=float(clouds.cell_m),
        cells=int(clouds.cells),
        levels=int(clouds.levels),
        seed=int(clouds.seed),
        thickness_m=float(clouds.thickness_m),
        optical_depth=float(clouds.optical_depth),
        feature_m=float(clouds.feature_m),
        erosion_scale=float(clouds.erosion_scale),
        beta_scale=float(clouds.beta_scale),
        detail_strength=float(getattr(clouds, "detail_strength", DETAIL_STRENGTH)),
        billow_scale=float(getattr(clouds, "billow_scale", 1.0)),
    )
    while len(_FIELD_CACHE) >= 2:
        _FIELD_CACHE.pop(next(iter(_FIELD_CACHE)))
    _FIELD_CACHE[key] = built
    return built


@dataclass(frozen=True)
class TilePlacement:
    """One copy of a periodic :class:`VolumeGrid`, shifted by whole tiles. Metres, stage axes."""

    name: str
    first_centre_m: Tuple[float, float, float]
    low_m: Tuple[float, float, float]
    high_m: Tuple[float, float, float]


def tile_placements(grid: VolumeGrid, tiles: int, up_axis: int = 1) -> list:
    """``tiles x tiles`` copies of ``grid`` around the origin, shifted along the two horizontal
    axes. Because the field is exactly periodic, neighbouring copies meet without a seam."""
    tiles = max(1, int(tiles))
    horizontal = (0, 2) if up_axis == 1 else (0, 1)
    half = tiles // 2
    out = []
    for a in range(-half, tiles - half):
        for b in range(-half, tiles - half):
            shift = [0.0, 0.0, 0.0]
            shift[horizontal[0]] = a * grid.tile_m
            shift[horizontal[1]] = b * grid.tile_m
            out.append(TilePlacement(
                name=f"Tile_{a - -half}_{b - -half}",
                first_centre_m=tuple(c + s for c, s in zip(grid.first_centre_m, shift)),  # type: ignore[arg-type]
                low_m=tuple(c + s for c, s in zip(grid.low_m, shift)),  # type: ignore[arg-type]
                high_m=tuple(c + s for c, s in zip(grid.high_m, shift)),  # type: ignore[arg-type]
            ))
    return out


def stage_to_field(point: Any, up_axis: int = 1) -> np.ndarray:
    """A stage-axes position (metres) in the field's Y-up frame: the inverse of the dome's turn."""
    p = np.asarray(point, dtype=np.float64)
    if up_axis == 1:
        return p.copy()
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def volume_offset_m(drift_m: Any, anchor_m: Any, tile_m: float, up_axis: int = 1) -> np.ndarray:
    """Where to put the tiled volumes: the wind's drift, plus whole tiles to keep the anchor central.

    The field is exactly periodic, so shifting the volumes by a whole tile changes nothing a camera
    can see -- which is what lets them follow a camera across any distance with no rebuild and no
    seam. The drift is the continuous part: the cloud blowing past. Both in stage axes, metres;
    the vertical component is always zero.
    """
    drift = np.asarray(drift_m, dtype=np.float64)
    anchor = np.asarray(anchor_m, dtype=np.float64)
    offset = np.zeros(3)
    for axis in ((0, 2) if up_axis == 1 else (0, 1)):
        tiles = math.floor((anchor[axis] - drift[axis]) / tile_m + 0.5)
        offset[axis] = drift[axis] + tiles * tile_m
    return offset


def drift_velocity_m_s(speed_mps: float, direction_deg: float, factor: float, up_axis: int = 1) -> np.ndarray:
    """Horizontal cloud velocity in stage axes: the surface wind scaled to cloud level.

    ``direction_deg`` follows the wind section's convention: the direction the wind blows
    *toward*, from +X. Wind speed grows with height through the boundary layer, so cloud-level
    wind is typically one and a half to two times the surface value; ``factor`` is that ratio.
    """
    a = math.radians(direction_deg)
    v = np.zeros(3)
    h0, h1 = (0, 2) if up_axis == 1 else (0, 1)
    v[h0] = speed_mps * factor * math.cos(a)
    v[h1] = speed_mps * factor * math.sin(a)
    return v


def cloud_drift_at(elapsed_s: float, speed_mps: float, direction_deg: float, factor: float,
                   up_axis: int = 1, frame: str = "stage") -> np.ndarray:
    """How far a steady wind has carried the clouds after ``elapsed_s`` seconds, metres.

    A pure function of time and the wind, so anyone -- the viewport, a headless render, the
    infrared model -- gets the same drift for the same moment without having to have watched
    every frame in between. ``frame="stage"`` gives stage axes (what the volumes move by);
    ``frame="field"`` gives the field's own Y-up frame, which is what a sensor marching
    :class:`CloudField` subtracts from its sample positions.
    """
    drift = drift_velocity_m_s(speed_mps, direction_deg, factor, up_axis) * float(elapsed_s)
    return _in_frame(drift, up_axis, frame)


def cloud_drift_from_state(state: Any, elapsed_s: float, up_axis: int = 1,
                           frame: str = "stage") -> np.ndarray:
    """:func:`cloud_drift_at` with the wind a :class:`~weather_fx.core.state.WeatherState` holds.

    ``elapsed_s`` is weather time: the manager's ``time``, which is wall time scaled by
    ``general.time_scale``. For a wind that has not changed since time zero this is exactly the
    drift the viewport draws; see :class:`DriftTrack` for a wind changed along the way.
    """
    wind = state.wind
    return cloud_drift_at(elapsed_s, wind.speed_mps, wind.direction_deg,
                          state.clouds.wind_factor, up_axis, frame)


def _in_frame(drift_stage: np.ndarray, up_axis: int, frame: str) -> np.ndarray:
    if frame == "stage":
        return drift_stage
    if frame == "field":
        return stage_to_field(drift_stage, up_axis)
    raise ValueError("frame is 'stage' or 'field'")


@dataclass
class DriftTrack:
    """The drift as a piecewise-linear function of time: one segment per steady wind.

    A steady wind set before the clock starts is one segment from ``t = 0``, and :meth:`at` is
    then exactly :func:`cloud_drift_at`. When the wind changes at time ``t``, :meth:`rebase` starts
    a new segment from wherever the clouds were at ``t``, so they do not jump (a wind first set
    ten minutes into a session starts from there rather than teleporting the clouds ten minutes'
    worth; :meth:`reset` makes it one segment from zero again). Every value is still a
    function of time -- evaluating it at any ``t`` in the current segment needs no history of
    frames, and neither does a replay that applies the same wind changes at the same times.
    """

    velocity_m_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    origin_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    since_s: float = 0.0

    def at(self, elapsed_s: float) -> np.ndarray:
        """Drift in stage axes at weather time ``elapsed_s``, metres."""
        return self.origin_m + self.velocity_m_s * (float(elapsed_s) - self.since_s)

    def rebase(self, elapsed_s: float, velocity_m_s: Any) -> None:
        """A new wind from ``elapsed_s`` on, continuing from the drift reached at that moment."""
        velocity = np.asarray(velocity_m_s, dtype=np.float64).reshape(3)
        if np.array_equal(velocity, self.velocity_m_s):
            return
        if float(elapsed_s) <= self.since_s and not np.any(self.origin_m):
            # The clock has not moved since this segment began (a wind set before time starts):
            # replace the segment rather than stacking a zero-length one in front of it.
            self.velocity_m_s = velocity
            return
        self.origin_m = self.at(elapsed_s)
        self.since_s = float(elapsed_s)
        self.velocity_m_s = velocity

    def reset(self, velocity_m_s: Any = (0.0, 0.0, 0.0)) -> None:
        """Back to one segment from ``t = 0``."""
        self.velocity_m_s = np.asarray(velocity_m_s, dtype=np.float64).reshape(3)
        self.origin_m = np.zeros(3)
        self.since_s = 0.0
