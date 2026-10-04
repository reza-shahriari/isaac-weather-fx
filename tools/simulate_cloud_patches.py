"""Grow a patch of cumulus with Blender's gas solver and save it as a weather-fx cloud patch.

    blender -b --python tools/simulate_cloud_patches.py -- --seed 21 --wind 1.5 --dissolve 70
    blender -b --python tools/simulate_cloud_patches.py -- --convert <cache dir> --frame 75 --name w21

Run by Blender, headless; no Isaac Sim. Warm bubbles of several sizes are released near the
ground for the first frames and rise by buoyancy, and the solver's own turbulence rolls them into
turrets. ``--wind`` adds a wind that grows with height, so the clouds lean and have a growing and
a decaying side; ``--dissolve`` lets the smoke evaporate (thin edges first) over that many
frames. One Blender unit is 100 m: the domain is 3.2 km wide and 2 km tall, and at the default
resolution a cell is 10 m.

What is kept is one frame's density above the condensation level, as a patch file
(``<name>.npz``: ``density`` uint8 ``(nx, ny, nz)`` with axis 1 up and 255 the densest cell,
``voxel_m``) in ``~/.cache/weather_fx/cloud_patches``, which is where the extension looks for
them. A patch is a shape, not a picture: the renderer marches it, and so can an infrared one.

This is a smoke solver, not a cloud-resolving model: it has no water vapour and no condensation.
The patch carries where the cloud is; how much water it holds and how warm it is come from the
height above the base, in the extension.
"""
import argparse
import math
import os
import random
import shutil
import sys
import time

import bpy
import numpy as np

try:
    import openvdb
except ImportError:
    import pyopenvdb as openvdb

UNIT_M = 100.0
#: Fraction of the cloud's height, from the ground, that is below the condensation level.
CONDENSATION = 0.3
#: A patch's base is the lowest level where its cloud covers this share of its widest level.
BASE_AREA = 0.9

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--out", default=os.path.expanduser("~/.cache/weather_fx/cloud_patches"))
ap.add_argument("--name", default=None)
ap.add_argument("--convert", default=None, help="an existing cache directory: no simulation")
ap.add_argument("--res", type=int, default=320)
ap.add_argument("--frames", type=int, default=90)
ap.add_argument("--frame", type=int, default=75, help="the frame kept")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--bubbles", type=int, default=9)
ap.add_argument("--wind", type=float, default=0.0)
ap.add_argument("--dissolve", type=int, default=0)
ap.add_argument("--keep-cache", action="store_true")
a = ap.parse_args(argv)


def simulate(cache: str) -> None:
    random.seed(a.seed)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, a.frames

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 10.0))
    domain = bpy.context.active_object
    domain.scale = (32.0, 32.0, 20.0)
    bpy.ops.object.transform_apply(scale=True)
    mod = domain.modifiers.new("Fluid", "FLUID")
    mod.fluid_type = "DOMAIN"
    d = mod.domain_settings
    d.domain_type = "GAS"
    d.resolution_max = a.res
    d.use_adaptive_domain = False
    d.alpha = -0.2            # smoke itself is slightly light
    d.beta = 8.0              # heat is what lifts it
    d.vorticity = 0.45
    d.use_noise = False
    # Open sides and top: smoke the wind carries to a wall leaves the domain. Against a closed
    # wall it piles up into a flat face, which a cloud does not have.
    for side in ("front", "back", "left", "right", "top"):
        setattr(d, "use_collision_border_" + side, False)
    d.use_dissolve_smoke = a.dissolve > 0
    if a.dissolve > 0:
        d.dissolve_speed = a.dissolve
        d.use_dissolve_smoke_log = True
    d.cache_type = "ALL"
    d.cache_data_format = "OPENVDB"
    d.cache_directory = cache
    d.cache_frame_start, d.cache_frame_end = 1, a.frames

    for _ in range(a.bubbles):
        r = random.uniform(0.9, 3.6)
        x, y = random.uniform(-9.5, 9.5), random.uniform(-9.5, 9.5)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=r, location=(x, y, 1.2 + r))
        o = bpy.context.active_object
        o.scale = (1.0, random.uniform(0.7, 1.3), 0.6)
        f = o.modifiers.new("Fluid", "FLUID")
        f.fluid_type = "FLOW"
        f = f.flow_settings
        f.flow_type = "SMOKE"
        f.flow_behavior = "INFLOW"
        f.density = 1.0
        f.temperature = random.uniform(1.0, 5.0)
        f.surface_distance = 0.5
        f.volume_density = 1.0
        stop = random.randint(14, 45)
        f.use_inflow = True
        f.keyframe_insert("use_inflow", frame=1)
        f.keyframe_insert("use_inflow", frame=stop)
        f.use_inflow = False
        f.keyframe_insert("use_inflow", frame=stop + 1)

    if a.wind > 0.0:
        # A wind field above the domain, fading with distance below it: the tops are carried
        # further than the bases.
        bpy.ops.object.effector_add(type="WIND", location=(0.0, 0.0, 24.0), rotation=(0.0, math.radians(90.0), 0.0))
        w = bpy.context.active_object.field
        w.strength = a.wind
        w.flow = 0.0
        w.falloff_type = "SPHERE"
        w.falloff_power = 1.0

    t = time.time()
    with bpy.context.temp_override(object=domain, active_object=domain, scene=scene):
        bpy.ops.fluid.bake_all()
    print(f"[patch] baked {a.frames} frames at res {a.res} in {time.time() - t:.0f} s", flush=True)


def round_off(values: np.ndarray, seed: int) -> np.ndarray:
    """Taper a cloud that reaches the side of its box, so the box never cuts it flat.

    Wind carries part of the smoke out through the domain's open wall, and the trim puts the
    box's side right there. Toward such a side the density is eroded, not faded: a threshold
    rises to the cloud's peak at the wall and only what is denser than it stays, so the cloud
    ends in its own billows. The threshold's distance from the wall wobbles over a few hundred
    metres, so the end is not a plane either.
    """
    rng = np.random.default_rng(seed)
    peak = float(values.max())
    out = values / peak
    for axis in (0, 2):
        n = out.shape[axis]
        other = 2 - axis
        for side in (0, 1):
            face = np.take(out, range(4) if side == 0 else range(n - 4, n), axis=axis)
            if face.max() < 0.1:
                continue
            width = 0.3 * n
            along = np.arange(n, dtype=np.float32) if side == 0 else np.arange(n - 1, -1, -1, dtype=np.float32)
            a_o = np.arange(out.shape[other], dtype=np.float32)[None, :]
            a_y = np.arange(out.shape[1], dtype=np.float32)[:, None]
            wobble = np.zeros((out.shape[1], out.shape[other]), dtype=np.float32)
            for _ in range(6):
                k = rng.uniform(0.02, 0.09, 2)
                wobble += np.sin(k[0] * a_y + k[1] * a_o + rng.uniform(0.0, 6.283))
            wobble *= 0.3 / 6.0 ** 0.5
            shape = [1, 1, 1]
            shape[axis] = n
            e = (along / width).reshape(shape)
            wob = wobble[None, :, :] if axis == 0 else wobble.T[:, :, None]
            t = np.clip(e + wob * np.clip(1.5 - e, 0.0, 1.0), 0.0, 1.0)
            threshold = 1.0 - t * t * (3.0 - 2.0 * t)
            out = np.clip((out - threshold) / np.maximum(1.0 - threshold, 1.0e-3), 0.0, 1.0)
    return out * peak


def flat_base(values: np.ndarray) -> np.ndarray:
    """Cut the patch where its cloud is first wide, so it stands on a flat base.

    The solver's smoke rises as stems under heads, and a cut low in the stems leaves clouds
    with narrow round bottoms. A cumulus is the part of a thermal above its condensation level,
    which is one height for the whole field: the cut goes at the lowest level where the cloud
    covers ``BASE_AREA`` of the widest it gets, and everything under it is dropped.
    """
    solid = values > 0.15 * values.max()
    area = solid.sum(axis=(0, 2)).astype(np.float64)
    level = int(np.argmax(area >= BASE_AREA * area.max()))
    return values[:, level:, :]


def trim(values: np.ndarray) -> np.ndarray:
    """What holds cloud, with a two-cell empty margin at the sides and the top."""
    solid = values > 0.004 * values.max()
    keep = []
    for axis in range(3):
        other = tuple(k for k in range(3) if k != axis)
        idx = np.flatnonzero(solid.any(axis=other))
        lo = 0 if axis == 1 else max(int(idx.min()) - 2, 0)
        keep.append(slice(lo, min(int(idx.max()) + 3, values.shape[axis])))
    return values[tuple(keep)]


def convert(cache: str, name: str) -> None:
    path = os.path.join(cache, "data", f"fluid_data_{a.frame:04d}.vdb")
    grid = openvdb.read(path, "density")
    (i0, j0, k0), (i1, j1, k1) = grid.evalActiveVoxelBoundingBox()
    values = np.zeros((i1 - i0 + 1, j1 - j0 + 1, k1 - k0 + 1), dtype=np.float32)
    grid.copyToArray(values, ijk=(i0, j0, k0))
    voxel_m = float(grid.transform.voxelSize()[0]) * UNIT_M
    values = np.ascontiguousarray(values.transpose(0, 2, 1))            # Blender is Z up
    values = values[:, int(CONDENSATION * values.shape[1]):, :]
    values = trim(flat_base(values))
    values = round_off(values, a.seed)
    density = np.round(255.0 * np.clip(values / values.max(), 0.0, 1.0)).astype(np.uint8)
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, name + ".npz")
    np.savez_compressed(out, density=density, voxel_m=np.float32(voxel_m),
                        source=np.array(f"blender {bpy.app.version_string} gas solver; seed {a.seed}, "
                                        f"wind {a.wind}, dissolve {a.dissolve}, frame {a.frame}"))
    print(f"[patch] wrote {out}: {density.shape} cells of {voxel_m:.1f} m, "
          f"{os.path.getsize(out) / 1e6:.1f} MB", flush=True)


if a.convert:
    convert(a.convert, a.name or os.path.basename(os.path.normpath(a.convert)))
else:
    name = a.name or f"cumulus_s{a.seed}_w{a.wind:g}_d{a.dissolve}"
    cache = os.path.join(a.out, "_cache_" + name)
    simulate(cache)
    convert(cache, name)
    if not a.keep_cache:
        shutil.rmtree(cache, ignore_errors=True)
