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

Nothing here is a microphysical model: there is no drop-size distribution, no adiabatic liquid
water profile and no entrainment. It is a *morphology* -- the shape and the optical depth -- and
the quantities a sensor model needs (extinction per metre, optical depth along a path) follow from
it by construction rather than being authored twice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

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
    ),
}


def cloud_profile(name: str) -> CloudProfile:
    if name not in CLOUD_TYPES:
        raise ValueError(f"unknown cloud type {name!r}; known: {sorted(CLOUD_TYPES)}")
    return CLOUD_TYPES[name]


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
    amplitude[0, 0, 0] = 0.0

    noise = np.fft.irfftn(spectrum * amplitude, s=shape, axes=(0, 1, 2))
    flat = noise.reshape(-1)
    ranks = np.empty(flat.size, dtype=np.float64)
    ranks[np.argsort(flat, kind="stable")] = np.arange(flat.size, dtype=np.float64)
    return (ranks / max(flat.size - 1, 1)).reshape(shape)


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


#: Henyey-Greenstein asymmetry for cloud droplets in the visible. Strongly forward-scattering,
#: which is why a cloud between you and the sun has a bright rim.
CLOUD_ASYMMETRY_G = 0.85
#: Octaves of multiple scattering. A cumulus is brilliant white *because* light bounces inside it
#: many times, and single scattering cannot produce that: a thick cloud scatters almost all the
#: light that enters it, but a single-scatter integral attenuates every sample by the direct
#: sunlight reaching it, which deep inside is nothing. Rendered that way a sunlit cumulus comes
#: out **darker than the sky behind it**, which is what the first version of this model did.
#:
#: Each octave re-integrates with the sun's optical depth scaled by ``a``, its contribution by
#: ``b`` and the phase function's eccentricity by ``c``, standing in for light that has already
#: been scattered ``n`` times and so is both less attenuated and less directional. The
#: contributions are normalised to sum to one, so the model cannot make light.
MS_OCTAVES = 4
MS_ATTENUATION = 0.55
MS_CONTRIBUTION = 0.5
MS_ECCENTRICITY = 0.5
#: How bright a fully self-shadowed part of a cloud is, as a fraction of a fully lit one. Not
#: zero: the shadowed side of a cumulus is grey, not black, because it is still lit by light that
#: reached it the long way through the cloud and by the sky and ground around it.
MS_SHADOW_FLOOR = 0.45
#: How far a ray that runs level *inside* the slab is followed before it is called done. A ray
#: exactly parallel to the base never leaves through either plane, so it needs a length of its
#: own rather than an infinity that would make the step size meaningless.
_LEVEL_RAY_LENGTH_M = 40_000.0
#: How fast the march's step size grows along the ray. 0 would be uniform spacing; 3 puts about
#: a fifth of the samples in the first tenth of the path, which is where the transmittance is
#: still high enough for a sample to matter.
_MARCH_GROWTH = 3.0


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
    _density: np.ndarray = field(init=False, repr=False)
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
            return

        # Cell sizes in metres, which is the frame the spectrum below is shaped in.
        vertical_m = self.thickness_m / self.levels
        cell_size_m = (vertical_m, self.cell_m, self.cell_m)
        
        beta = self.profile.beta * self.beta_scale
        erosion = self.profile.erosion * self.erosion_scale

        noise = _periodic_fractal_noise(
            shape,
            beta,
            self.seed,
            cell_size_m=cell_size_m,
            min_wavelength_m=self.feature_m,
            anisotropy=self.profile.anisotropy,
        )
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
        self._eroded = eroded

        u = (np.arange(self.levels, dtype=np.float64) + 0.5) / self.levels
        relative_area = _profile_area(self.profile, u)

        # Sorting each level once turns every later quantile lookup into an index, which is what
        # makes the coverage solve below affordable at 48 bisection steps.
        per_level = np.sort(eroded.reshape(self.levels, -1), axis=1)
        self._area_scale = self._solve_area_scale(per_level, relative_area)
        self._thresholds = self._level_thresholds(per_level, relative_area, self._area_scale)

        signed = eroded - self._thresholds[:, None, None]
        self._density = np.clip(
            0.5 + signed / (2.0 * max(self.softness, 1e-6)), 0.0, 1.0
        ).astype(np.float32)

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

        def covered(scale: float) -> float:
            thresholds = self._level_thresholds(per_level, relative_area, scale)
            # Reconstruct occupancy from the sorted array's own statistics rather than the field:
            # a column is covered when any level exceeds its threshold, which needs the field, so
            # this uses the cached one built by the caller.
            occupied = self._eroded > thresholds[:, None, None]
            return float(np.mean(occupied.any(axis=0)))

        assert self._eroded.shape == eroded_shape
        lo, hi = 0.0, 1.0
        if covered(hi) < self.cover:
            return hi
        for _ in range(40):
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
        return float(np.mean(self._density.max(axis=0) > 0.5))

    def density(self, x_m: Any, y_m: Any, z_m: Any) -> np.ndarray:
        """Normalised cloud density in 0..1 at a world position, trilinearly interpolated.

        ``y_m`` is height above the ground. Outside the slab the density is zero; horizontally the
        field wraps, so there is no edge. Broadcasting follows numpy's rules, so a whole ray or a
        whole frame of rays can be sampled in one call -- which is what makes a per-pixel march
        affordable.
        """
        return self._sample(self._density, x_m, y_m, z_m)

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

    def volume_grid(self, up_axis: int = 1) -> "VolumeGrid":
        """The density as a voxel array in **stage axes**, ready for an OpenVDB fog volume.

        The field's own frame is the Y-up sky frame: ``x`` east, ``y`` up, ``z`` south. A Y-up
        stage is that frame. A Z-up stage is it turned by +90 degrees about X -- the same turn the
        dome light gets -- so field ``z`` (south) is stage ``-y`` and field ``y`` is stage ``z``.
        Getting this wrong would not look wrong: a random field is a random field either way round.
        It would only put the rendered cloud somewhere other than where the infrared march puts it.

        Values are normalised density in 0..1; multiply by :attr:`extinction_per_m` for the visible
        extinction. Voxel centres sit where :meth:`density` puts its cell centres, so the grid and
        the field agree to the sample.
        """
        dv = self.thickness_m / self.levels
        half = self.half_extent_m
        grid = self._density  # (level, z, x)
        if up_axis == 1:
            array = np.transpose(grid, (2, 0, 1))                 # (x, y_up, z)
            voxel = (self.cell_m, dv, self.cell_m)
            first = (-half + 0.5 * self.cell_m, self.base_m + 0.5 * dv, -half + 0.5 * self.cell_m)
        elif up_axis == 2:
            array = np.transpose(grid, (2, 1, 0))[:, ::-1, :]     # (x, y = -z_field, z_up)
            voxel = (self.cell_m, self.cell_m, dv)
            first = (-half + 0.5 * self.cell_m, -half + 0.5 * self.cell_m, self.base_m + 0.5 * dv)
        else:
            raise ValueError("up_axis is 1 (Y) or 2 (Z)")
        return VolumeGrid(
            values=np.ascontiguousarray(array, dtype=np.float32),
            voxel_m=tuple(float(v) for v in voxel),  # type: ignore[arg-type]
            first_centre_m=tuple(float(v) for v in first),  # type: ignore[arg-type]
            tile_m=float(self.cells * self.cell_m),
        )

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
        """Transmittance from every grid cell **toward the sun**, cached per direction.

        This is the one quantity that turns a cloud from a cut-out into a cloud. Without it every
        cloudy sample is equally bright, the alpha saturates the moment the optical depth passes
        about 3, and the result is a flat white shape with a hard edge -- which is what a viewer
        reads as "grey blob". With it, a flank facing the sun is bright, the base is dark because
        a kilometre of its own cloud is between it and the sun, and a thin edge glows because
        almost nothing is.

        Computed on a grid coarsened by ``coarsen`` in each axis and sampled back up. The light
        field varies far more smoothly than the density does -- it is an integral of it -- so a
        half-resolution light map is visually indistinguishable and eight times cheaper.
        """
        direction = np.asarray(sun_direction, dtype=np.float64)
        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
        key = tuple(np.round(direction, 4))
        cached = self._sun_cache.get(key)
        if cached is not None:
            return cached

        nz = max(self.levels // coarsen, 2)
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
        optical *= step * self.extinction_per_m
        transmittance = np.exp(-optical).astype(np.float32)

        if coarsen > 1:
            transmittance = np.repeat(
                np.repeat(np.repeat(transmittance, coarsen, 0), coarsen, 1), coarsen, 2
            )[: self.levels, : self.cells, : self.cells]
            if transmittance.shape != self._density.shape:
                pad = [(0, t - s) for s, t in zip(transmittance.shape, self._density.shape)]
                transmittance = np.pad(transmittance, pad, mode="edge")
        self._sun_cache[key] = transmittance
        return transmittance

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
        ambient: float = 0.30,
        max_path_m: float = 8000.0,
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

        ``radiance`` is relative to the incident sunlight: the in-scattered light at each sample,
        weighted by the transmittance back to the sensor, with the sun's own transmittance into
        the cloud giving the self-shadowing and ``ambient`` standing in for the sky and the
        ground. A Henyey-Greenstein phase function at :data:`CLOUD_ASYMMETRY_G` supplies the
        forward-scattering rim.
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

        hit = far > near
        if not np.any(hit) or self.cover <= 0.0:
            return MarchResult(
                optical_depth=optical,
                transmittance=transmittance,
                emission_height_m=np.full(shape, self.base_m),
                radiance=radiance,
            )

        light = None
        phase = None
        if sun_direction is not None:
            sun = np.asarray(sun_direction, dtype=np.float64)
            sun = sun / max(float(np.linalg.norm(sun)), 1e-12)
            light = self.sun_transmittance(sun)
            cos_theta = np.sum(direction * sun, axis=-1)
            # One phase value per octave, at a progressively less directional g. Normalised so an
            # isotropic phase function is 1: the dome's absolute scale is an exposure choice and
            # carrying 1/4pi through it helps nobody.
            octave_weight = []
            octave_phase = []
            octave_power = []
            norm = sum(MS_CONTRIBUTION**n for n in range(MS_OCTAVES))
            for n in range(MS_OCTAVES):
                g = CLOUD_ASYMMETRY_G * MS_ECCENTRICITY**n
                octave_phase.append(
                    (1.0 - g * g) / np.power(1.0 + g * g - 2.0 * g * cos_theta, 1.5)
                )
                octave_weight.append(MS_CONTRIBUTION**n / norm)
                octave_power.append(MS_ATTENUATION**n)
            phase = (octave_weight, octave_phase, octave_power)

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
        # by a per-ray fraction of a step, derived from the ray's own direction so it is
        # deterministic and a frame is reproducible.
        u = np.linspace(0.0, 1.0, steps + 1)
        growth = np.expm1(_MARCH_GROWTH * u) / math.expm1(_MARCH_GROWTH)
        jitter = (
            np.modf(np.abs(dx) * 71.3 + np.abs(dy) * 131.7 + np.abs(dz) * 197.1)[0][..., None]
            - 0.5
        ) / steps
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
            if radiance is not None:
                sun_t = np.clip(self._sample(light, px, py, pz), 1e-6, 1.0)
                weights, phases, powers = phase
                scattered = np.zeros_like(sun_t)
                for w, ph, a in zip(weights, phases, powers):
                    # `sun_t ** a` is `exp(-a * tau_sun)`: the same path, less attenuated, which
                    # is what a photon that has already bounced sees.
                    scattered = scattered + w * (sun_t**a) * ph
                radiance += transmittance * (1.0 - np.exp(-d_tau)) * (scattered + ambient)
            optical += d_tau
            transmittance = transmittance * np.exp(-d_tau)

        if radiance is not None:
            # --- energy from two-stream, detail from the march ------------------------------
            # The march above gives the *shape* of the light -- which flank faces the sun, where
            # the rim glows, how deep the shadow is -- but it cannot give the right amount of it.
            # Its value is bounded by the mean of the scattered term, so a cloud of optical depth
            # 60 comes out no brighter than one of depth 20, where the real albedos are 0.89 and
            # 0.72. Multiple scattering is exactly the part a forward march does not carry.
            #
            # So the magnitude comes from the two-stream reflectance of a conservatively
            # scattering layer, R = (1-g) tau / (2 mu0 + (1-g) tau), and the march supplies a
            # normalised modulation around it: 1 where the sample is fully lit by an isotropic
            # phase function, more on a forward-scattering rim, less in shadow.
            intercepted = 1.0 - transmittance
            mu_sun = max(abs(float(sun[1])), 0.05)
            g = CLOUD_ASYMMETRY_G
            albedo = ((1.0 - g) * optical) / (2.0 * mu_sun + (1.0 - g) * optical)
            # The march's mean scattered value along this ray, against what the same ray would
            # have got with nothing shadowing it. The reference is per-ray because it carries the
            # phase function, so a rim facing the sun is compared against its own bright limit
            # rather than against an average taken over the frame -- which would make the result
            # depend on how much cloud happened to be in shot.
            weights, phases, _ = phase
            fully_lit = sum(w * ph for w, ph in zip(weights, phases)) + ambient
            shading = np.where(
                intercepted > 1e-6, radiance / np.maximum(intercepted, 1e-6), 0.0
            )
            modulation = np.clip(shading / np.maximum(fully_lit, 1e-9), 0.0, 1.0)
            radiance = (
                intercepted
                * albedo
                * (MS_SHADOW_FLOOR + (1.0 - MS_SHADOW_FLOOR) * modulation)
            )

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
    "seed",
)

_FIELD_CACHE: Dict[Tuple[Any, ...], "CloudField"] = {}


def cloud_field_from_state(state: Any):
    """The cloud field a state describes, or ``None`` for a clear sky. Cached on its shape keys.

    A field of the default size takes seconds to synthesise, and the dome, the volume and any
    sensor model all want the same one: building it once per shape is the difference between a
    colour change being instant and it costing a rebuild. The cache holds the last two shapes.
    """
    clouds = state.clouds
    if not (clouds.enabled and clouds.cover > 0.0):
        return None
    key = tuple(getattr(clouds, name) for name in CLOUD_SHAPE_KEYS)
    cached = _FIELD_CACHE.get(key)
    if cached is not None:
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
