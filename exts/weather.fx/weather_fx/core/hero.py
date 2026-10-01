"""Hero clouds: a few hand-made (sculpted or simulated) cloud volumes placed in the sky.

A procedural field is right for a whole sky; a close, single cumulus is where it shows its
limits, because film clouds are sculpted or simulated at centimetre-to-metre detail. A hero
cloud is one of those -- an OpenVDB file such as the Walt Disney Animation Studios cloud
(CC BY-SA 3.0) -- scaled to a real cloud's size and placed a few times around the observer.

This module is the engine-free half: the voxels as a numpy array (the backend reads the VDB with
Kit's OpenVDB), the scale, the placements, the drift, and :meth:`HeroClouds.density` -- so an
infrared march sees exactly the hero clouds the path tracer draws, the same way it sees the
procedural field.

Frames follow :mod:`weather_fx.core.clouds`: the field frame is Y up, ``x`` east, ``z`` south,
metres, horizontally centred on the observer's column. Positions wrap within ``spread_m`` around
the anchor, so the clouds that drift away come back on the far side instead of leaving the sky.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

__all__ = ["HeroAsset", "HeroClouds", "prepare_hero_asset", "hero_positions", "HERO_KEYS"]

#: The cloud parameters a rebuild of the hero clouds depends on.
HERO_KEYS = ("hero_vdb", "hero_size_m", "hero_max_voxels")


@dataclass(frozen=True)
class HeroAsset:
    """One cloud's density on a voxel grid, in the field frame, scaled to metres."""

    #: ``(nx, ny, nz)`` float32 in 0..1, axis 1 up.
    values: np.ndarray
    #: Edge of one (cubic) voxel, metres.
    voxel_m: float

    @property
    def size_m(self) -> Tuple[float, float, float]:
        return tuple(n * self.voxel_m for n in self.values.shape)  # type: ignore[return-value]

    def volume_grid(self, up_axis: int = 1):
        """The voxels in **stage axes** with their lower corner at the origin: ``(values,
        voxel_m, size)``. A Z-up stage is the field turned +90 degrees about X, as for
        :meth:`weather_fx.core.clouds.CloudField.volume_grid`: field ``z`` is stage ``-y``."""
        if up_axis == 1:
            array = self.values
        elif up_axis == 2:
            array = np.transpose(self.values, (0, 2, 1))[:, ::-1, :]
        else:
            raise ValueError("up_axis is 1 (Y) or 2 (Z)")
        array = np.ascontiguousarray(array, dtype=np.float32)
        size = tuple(n * self.voxel_m for n in array.shape)
        return array, self.voxel_m, size


def prepare_hero_asset(values: Any, size_m: float, max_voxels: int = 256,
                       source_up_axis: int = 1) -> HeroAsset:
    """Turn a VDB's dense density array into a :class:`HeroAsset` of a real cloud's size.

    ``values`` is ``(nx, ny, nz)`` in the file's own index order; ``source_up_axis`` says which
    axis is up in it (film assets are Y up). The array is box-filtered down until no axis exceeds
    ``max_voxels`` (a quarter-resolution Disney cloud is 100 million voxels, far more than the
    path tracer needs at the distance a cloud is seen from), normalised so its densest voxel is
    1, and given the voxel size that makes its **widest horizontal extent** ``size_m``. The
    voxels stay cubic: a cloud's proportions are the thing that makes it look like one.
    """
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or min(array.shape) < 2:
        raise ValueError("a hero cloud needs a 3D density array")
    if source_up_axis == 2:
        array = np.transpose(array, (0, 2, 1))[:, :, ::-1]   # (x, z_up, -y) -> Y up
    elif source_up_axis != 1:
        raise ValueError("source_up_axis is 1 (Y) or 2 (Z)")
    factor = max(1, int(math.ceil(max(array.shape) / max(int(max_voxels), 8))))
    if factor > 1:
        array = _box_downsample(array, factor)
    peak = float(array.max())
    if peak <= 0.0:
        raise ValueError("the hero cloud's density is empty")
    array = np.clip(array / peak, 0.0, 1.0).astype(np.float32)
    widest = max(array.shape[0], array.shape[2])
    return HeroAsset(values=array, voxel_m=float(size_m) / widest)


def _box_downsample(array: np.ndarray, factor: int) -> np.ndarray:
    """Mean over ``factor``-cubed blocks (padded with zeros to a whole number of blocks)."""
    pad = [(0, (-n) % factor) for n in array.shape]
    padded = np.pad(array, pad)
    nx, ny, nz = (n // factor for n in padded.shape)
    return padded.reshape(nx, factor, ny, factor, nz, factor).mean(axis=(1, 3, 5))


def hero_positions(count: int, seed: int, spread_m: float, min_gap_m: float) -> np.ndarray:
    """``(count, 2)`` horizontal positions (field ``x``, ``z``) inside a ``spread_m`` square.

    Deterministic for a seed. Kept at least ``min_gap_m`` apart where the square allows it --
    two hero clouds overlapping read as one malformed cloud.
    """
    rng = np.random.default_rng(int(seed) + 7717)
    half = 0.5 * float(spread_m)
    points = []
    for _ in range(max(0, int(count))):
        best, best_gap = None, -1.0
        for _ in range(64):
            candidate = rng.uniform(-half, half, 2)
            gap = min((_wrapped_distance(candidate, p, spread_m) for p in points), default=math.inf)
            if gap >= min_gap_m:
                best = candidate
                break
            if gap > best_gap:
                best, best_gap = candidate, gap
        points.append(best)
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def _wrapped_distance(a: np.ndarray, b: np.ndarray, spread_m: float) -> float:
    d = np.abs(a - b)
    d = np.minimum(d, spread_m - d)
    return float(np.hypot(d[0], d[1]))


@dataclass(frozen=True)
class HeroClouds:
    """Several copies of one :class:`HeroAsset`, placed, drifting and wrapping around an anchor."""

    asset: HeroAsset
    #: Height of the clouds' lowest voxel above the ground, metres.
    base_m: float
    #: ``(count, 2)`` field ``x``, ``z`` of each cloud's horizontal centre at zero drift.
    positions_m: np.ndarray
    #: Side of the square the clouds wrap within, metres.
    spread_m: float
    #: Visible extinction at unit density, per metre.
    extinction_per_m: float

    @property
    def count(self) -> int:
        return int(self.positions_m.shape[0])

    @property
    def finest_pitch_m(self) -> float:
        """The finest spacing these clouds carry, metres: one voxel of the asset.

        The same question :attr:`CloudField.finest_pitch_m` answers for the procedural field, so
        a march that sizes its steps by it works on either source.
        """
        return float(self.asset.voxel_m)

    def centres_m(self, drift_m: Any = (0.0, 0.0, 0.0), anchor_m: Any = (0.0, 0.0, 0.0)) -> np.ndarray:
        """``(count, 3)`` centre of each cloud's base in the field frame, after the drift.

        Each position is carried by the drift and then wrapped into the ``spread_m`` square
        around the anchor's column, so the set follows the observer across any distance and a
        cloud blown out of one side comes back on the other.
        """
        drift = np.asarray(drift_m, dtype=np.float64).reshape(3)
        anchor = np.asarray(anchor_m, dtype=np.float64).reshape(3)
        horizontal = self.positions_m + drift[[0, 2]] - anchor[[0, 2]]
        wrapped = (horizontal + 0.5 * self.spread_m) % self.spread_m - 0.5 * self.spread_m
        out = np.empty((self.count, 3))
        out[:, 0] = wrapped[:, 0] + anchor[0]
        out[:, 1] = self.base_m
        out[:, 2] = wrapped[:, 1] + anchor[2]
        return out

    def lower_corners_m(self, drift_m: Any = (0.0, 0.0, 0.0),
                        anchor_m: Any = (0.0, 0.0, 0.0)) -> np.ndarray:
        """``(count, 3)`` lower corner of each cloud's voxel box in the field frame."""
        sx, _, sz = self.asset.size_m
        corners = self.centres_m(drift_m, anchor_m)
        corners[:, 0] -= 0.5 * sx
        corners[:, 2] -= 0.5 * sz
        return corners

    def stage_lower_corners_m(self, drift_m: Any, anchor_m: Any, up_axis: int = 1) -> np.ndarray:
        """``(count, 3)`` lower corner of each box in **stage axes**, matching
        :meth:`HeroAsset.volume_grid` (drift and anchor still in the field frame)."""
        corners = self.lower_corners_m(drift_m, anchor_m)
        if up_axis == 1:
            return corners
        sz = self.asset.size_m[2]
        # Field (x, y_up, z_south) -> stage (x, -z, y); the box's low stage y is at field z + sz.
        return np.stack([corners[:, 0], -(corners[:, 2] + sz), corners[:, 1]], axis=-1)

    def density(self, x_m: Any, y_m: Any, z_m: Any, drift_m: Any = (0.0, 0.0, 0.0),
                anchor_m: Any = (0.0, 0.0, 0.0)) -> np.ndarray:
        """Normalised density at field-frame positions, the densest of the clouds covering each.

        Trilinear, zero outside every box. Unlike the procedural field this is *not* offset by
        subtracting the drift from the sample position: the hero clouds wrap around the anchor,
        so pass the same ``drift_m`` and ``anchor_m`` the renderer used.
        """
        x = np.asarray(x_m, dtype=np.float64)
        y = np.asarray(y_m, dtype=np.float64)
        z = np.asarray(z_m, dtype=np.float64)
        x, y, z = np.broadcast_arrays(x, y, z)
        out = np.zeros(x.shape, dtype=np.float64)
        values = self.asset.values
        voxel = self.asset.voxel_m
        shape = np.asarray(values.shape)
        for corner in self.lower_corners_m(drift_m, anchor_m):
            fi = (x - corner[0]) / voxel - 0.5
            fj = (y - corner[1]) / voxel - 0.5
            fk = (z - corner[2]) / voxel - 0.5
            inside = ((fi > -1.0) & (fi < shape[0]) & (fj > -1.0) & (fj < shape[1])
                      & (fk > -1.0) & (fk < shape[2]))
            if not np.any(inside):
                continue
            out[inside] = np.maximum(out[inside], _trilinear_zero(values, fi[inside],
                                                                  fj[inside], fk[inside]))
        return out

    def optical_depth_toward(self, origin_m: Any, direction: Any, length_m: float = 20000.0,
                             steps: int = 256, drift_m: Any = (0.0, 0.0, 0.0),
                             anchor_m: Optional[Any] = None) -> float:
        """Visible optical depth along a ray, for a sun shadow or a sensor's transmittance."""
        o = np.asarray(origin_m, dtype=np.float64).reshape(3)
        d = np.asarray(direction, dtype=np.float64).reshape(3)
        d = d / max(float(np.linalg.norm(d)), 1e-12)
        t = (np.arange(steps) + 0.5) / steps * float(length_m)
        p = o[None, :] + t[:, None] * d[None, :]
        rho = self.density(p[:, 0], p[:, 1], p[:, 2], drift_m, o if anchor_m is None else anchor_m)
        return float(rho.sum() * float(length_m) / steps * self.extinction_per_m)


def _trilinear_zero(grid: np.ndarray, fi: np.ndarray, fj: np.ndarray, fk: np.ndarray) -> np.ndarray:
    """Trilinear read of ``grid`` at fractional cell-centre coordinates, zero outside it."""
    padded = np.pad(grid, 1)
    fi, fj, fk = fi + 1.0, fj + 1.0, fk + 1.0
    limit = np.asarray(padded.shape) - 2
    i0 = np.clip(np.floor(fi).astype(np.int64), 0, limit[0])
    j0 = np.clip(np.floor(fj).astype(np.int64), 0, limit[1])
    k0 = np.clip(np.floor(fk).astype(np.int64), 0, limit[2])
    ti, tj, tk = (np.clip(fi - i0, 0, 1), np.clip(fj - j0, 0, 1), np.clip(fk - k0, 0, 1))
    g = padded
    c00 = g[i0, j0, k0] * (1 - tk) + g[i0, j0, k0 + 1] * tk
    c01 = g[i0, j0 + 1, k0] * (1 - tk) + g[i0, j0 + 1, k0 + 1] * tk
    c10 = g[i0 + 1, j0, k0] * (1 - tk) + g[i0 + 1, j0, k0 + 1] * tk
    c11 = g[i0 + 1, j0 + 1, k0] * (1 - tk) + g[i0 + 1, j0 + 1, k0 + 1] * tk
    return (c00 * (1 - tj) + c01 * tj) * (1 - ti) + (c10 * (1 - tj) + c11 * tj) * ti
