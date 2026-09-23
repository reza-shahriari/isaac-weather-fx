"""Camera-following particle volume with toroidal wrapping (numpy only)."""
from __future__ import annotations

import math
from typing import Callable, Dict

import numpy as np

from .physics import horizontal_axes

# sampler(rng, n) -> {"diameter_m": (n,), "fall_speed_mps": (n,)}
Sampler = Callable[[np.random.Generator, int], Dict[str, np.ndarray]]


class ParticleField:
    """Particles live in world space; the box around the anchor wraps them.

    Because positions are stored in world space, moving the camera produces
    correct parallax, and a drop leaving the box re-enters on the other side,
    so the box never empties.
    """

    def __init__(self, up_axis: int = 2, seed: int = 0):
        self.up_axis = up_axis
        self.h_axes = horizontal_axes(up_axis)
        self.rng = np.random.default_rng(seed)
        self.positions = np.zeros((0, 3))
        self.attrs: Dict[str, np.ndarray] = {"diameter_m": np.zeros(0), "fall_speed_mps": np.zeros(0)}
        self.phase = np.zeros(0)

    @property
    def count(self) -> int:
        return len(self.positions)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def clear(self) -> None:
        self.resize(0, np.zeros(3), np.ones(3), None)

    def resize(self, n: int, anchor, half_extents, sampler: Sampler) -> None:
        n = max(int(n), 0)
        cur = self.count
        if n < cur:
            self.positions = self.positions[:n]
            self.phase = self.phase[:n]
            self.attrs = {k: v[:n] for k, v in self.attrs.items()}
        elif n > cur:
            k = n - cur
            anchor = np.asarray(anchor, dtype=float)
            half = np.asarray(half_extents, dtype=float)
            new_pos = anchor + (self.rng.random((k, 3)) * 2.0 - 1.0) * half
            new_attrs = sampler(self.rng, k)
            self.positions = np.concatenate([self.positions, new_pos])
            self.phase = np.concatenate([self.phase, self.rng.random(k) * 2.0 * math.pi])
            self.attrs = {key: np.concatenate([self.attrs[key], np.asarray(new_attrs[key], dtype=float)])
                          for key in self.attrs}

    def resample(self, sampler: Sampler) -> None:
        """Redraw per-particle attributes, keeping positions."""
        if self.count:
            new_attrs = sampler(self.rng, self.count)
            self.attrs = {k: np.asarray(new_attrs[k], dtype=float) for k in self.attrs}

    def recenter(self, anchor, half_extents) -> None:
        """Scatter all particles uniformly in the box (e.g. after a teleport)."""
        anchor = np.asarray(anchor, dtype=float)
        half = np.asarray(half_extents, dtype=float)
        self.positions = anchor + (self.rng.random((self.count, 3)) * 2.0 - 1.0) * half

    def step(self, dt: float, anchor, half_extents, wind_velocity, t: float = 0.0,
             velocity_scale: float = 1.0, sway_amplitude: float = 0.0, sway_frequency: float = 0.0) -> np.ndarray:
        """Advance particles. Velocities are in m/s; ``velocity_scale`` converts to stage units/s."""
        n = self.count
        if n == 0:
            return np.zeros((0, 3))
        vel = np.broadcast_to(np.asarray(wind_velocity, dtype=float), (n, 3)).copy()
        vel[:, self.up_axis] -= self.attrs["fall_speed_mps"]
        if sway_amplitude > 0.0 and sway_frequency > 0.0:
            w = 2.0 * math.pi * sway_frequency
            a0, a1 = self.h_axes
            vel[:, a0] += sway_amplitude * w * np.cos(w * t + self.phase)
            vel[:, a1] += sway_amplitude * w * np.sin(0.87 * w * t + 1.7 * self.phase)
        self.positions += vel * (dt * velocity_scale)
        self.wrap(anchor, half_extents)
        return vel

    def wrap(self, anchor, half_extents) -> None:
        anchor = np.asarray(anchor, dtype=float)
        half = np.asarray(half_extents, dtype=float)
        rel = self.positions - anchor
        rel = np.mod(rel + half, 2.0 * half) - half
        self.positions = anchor + rel


def rotation_between(a, b):
    """Quaternion (w, x, y, z) rotating unit vector ``a`` onto unit vector ``b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / max(np.linalg.norm(b), 1e-12)
    d = float(np.dot(a, b))
    if d < -0.999999:  # opposite: rotate 180 deg around any perpendicular axis
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return (0.0, *axis)
    c = np.cross(a, b)
    q = np.array([1.0 + d, *c])
    q /= np.linalg.norm(q)
    return tuple(float(x) for x in q)
