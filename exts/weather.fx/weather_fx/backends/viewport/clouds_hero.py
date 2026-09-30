"""Hero clouds under the path tracer: an OpenVDB cloud asset, placed a few times in the sky.

The asset is read once with Kit's OpenVDB (``omni.volume``), box-filtered to a sensible size,
scaled to ``clouds.hero_size_m`` and written back as a single fog volume in stage axes; every
hero cloud is a box that points at that one file and is moved by a translate only -- the recipe
the procedural volumes use (ADR 0144: a translate carries the density texture, a scale breaks
it). The placements, drift and wrap come from :mod:`weather_fx.core.hero`, so an infrared model
that asks :meth:`HeroCloudEffect.hero_clouds` samples the same clouds the renderer draws.

Real-time mode cannot draw volumes, so hero clouds exist only while the path tracer runs.
"""
from __future__ import annotations

import logging
import os
import pathlib
import tempfile
from typing import Any, List, Optional, Set

import numpy as np

from ...core.clouds import lifting_condensation_level_m, stage_to_field
from ...core.hero import HERO_KEYS, HeroClouds, hero_positions, prepare_hero_asset
from ...core.jobs import LatestJob
from ..base import Effect
from .clouds_volume import (
    VolumeSettingsLease,
    _author_volume_box,
    _remove_files,
    set_volume_inputs,
)
from .render_mode import RenderModeWatcher, cloud_path

log = logging.getLogger("weather_fx")

HERO_ROOT = "/WeatherFX/HeroClouds"
#: Refuse to densify a grid bigger than this many voxels (a full-resolution Disney cloud is
#: billions); the error names the smaller files to use instead.
MAX_DENSE_VOXELS = 400_000_000


class HeroCloudEffect(Effect):
    name = "hero_clouds"

    def __init__(self, directory: Optional[str] = None):
        self._directory = pathlib.Path(directory) if directory else None
        self._job = LatestJob("weather_fx-hero")
        self._watcher: Optional[RenderModeWatcher] = None
        self._built_key: Optional[tuple] = None
        self._files: List[pathlib.Path] = []
        self._shaders: List[str] = []
        self._boxes: List[str] = []
        self._asset = None
        self._grid: Optional[dict] = None
        self._layout_key: Optional[tuple] = None
        self._clouds: Optional[HeroClouds] = None
        self._active = False
        self._last_corners = None
        self._error: Optional[str] = None

    # --- lifecycle ---------------------------------------------------------------------

    def attach(self, context) -> None:
        super().attach(context)
        self._watcher = RenderModeWatcher()

    def apply_state(self, state: Any, changed: Set[str]) -> None:
        stage = self.context.stage()
        if stage is None:
            return
        clouds = state.clouds
        path = os.path.expanduser(clouds.hero_vdb.strip()) if clouds.hero_vdb else ""
        wanted = (state.general.enabled and clouds.enabled and clouds.hero_count > 0
                  and bool(path) and cloud_path(state) == "volume")
        if not wanted:
            if self._active or self._built_key is not None:
                self._teardown()
            return
        if not os.path.isfile(path):
            if self._error != path:
                log.warning("weather_fx: hero cloud file %s does not exist", path)
            self._error = path
            self._teardown()
            return
        self._error = None
        VolumeSettingsLease.acquire(self.name, int(clouds.volume_bounces))
        self._active = True

        key = self._asset_key(state, path)
        if key != self._built_key and key != self._job.running_key:
            up_axis, mpu = self.context.up_axis(), self.context.meters_per_unit()
            directory = self._directory or pathlib.Path(tempfile.gettempdir())
            size_m, max_voxels = float(clouds.hero_size_m), int(clouds.hero_max_voxels)

            def build():
                return _build_hero(path, size_m, max_voxels, up_axis, mpu, directory, key)

            if state.general.time_source == "manual":
                self._job.cancel()
                self._install(stage, key, build(), state)
            else:
                self._job.submit(key, build)
        elif self._asset is not None:
            self._layout(stage, state)
        self._update_materials(stage, state)

    def update(self, dt: float, t: float) -> None:
        if self._watcher is not None and self._watcher.changed():
            self.apply_state(self.context.state, {"clouds"})
        self._place()
        try:
            done = self._job.poll()
        except Exception as exc:
            log.error("weather_fx: loading the hero cloud failed: %s", exc)
            return
        if done is None:
            return
        key, result = done
        stage = self.context.stage()
        state = self.context.state
        clouds = state.clouds
        path = os.path.expanduser(clouds.hero_vdb.strip()) if clouds.hero_vdb else ""
        if stage is None or not self._active or key != self._asset_key(state, path):
            _remove_files(result["files"])
            return
        self._install(stage, key, result, state)

    def detach(self) -> None:
        self._teardown()

    def stats(self) -> dict:
        return {
            "active": self._active,
            "clouds": len(self._boxes),
            "voxels": None if self._asset is None else list(self._asset.values.shape),
            "size_m": None if self._asset is None else [round(v, 1) for v in self._asset.size_m],
            "building": self._job.busy,
            "missing_file": self._error,
        }

    def hero_clouds(self) -> Optional[HeroClouds]:
        """The hero clouds as the renderer draws them (field frame), or None when there are none."""
        return self._clouds

    # --- building ----------------------------------------------------------------------

    def _asset_key(self, state: Any, path: str) -> tuple:
        clouds = state.clouds
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None
        return (path, mtime) + tuple(getattr(clouds, k) for k in HERO_KEYS[1:]) + (
            self.context.up_axis(), round(self.context.meters_per_unit(), 9))

    def _install(self, stage: Any, key: tuple, result: dict, state: Any) -> None:
        previous = self._files
        self._files = list(result["files"])
        self._asset = result["asset"]
        self._grid = result["grid"]
        self._built_key = key
        self._boxes, self._shaders = [], []
        self._layout(stage, state)
        _remove_files([f for f in previous if f not in self._files])

    def _layout(self, stage: Any, state: Any) -> None:
        """(Re)author one box per hero cloud and place them. Cheap: the file is shared."""
        from pxr import Usd

        clouds = state.clouds
        base = float(clouds.hero_base_m)
        if base <= 0.0:
            base = float(clouds.base_m) or lifting_condensation_level_m(
                clouds.temperature_c, clouds.dewpoint_c)
        positions = hero_positions(int(clouds.hero_count), int(clouds.seed),
                                   float(clouds.hero_spread_m), 1.2 * self._asset.size_m[0])
        self._clouds = HeroClouds(asset=self._asset, base_m=base, positions_m=positions,
                                  spread_m=float(clouds.hero_spread_m),
                                  extinction_per_m=float(clouds.hero_extinction_per_m))
        layout = (self._built_key, int(clouds.hero_count), int(clouds.seed),
                  float(clouds.hero_spread_m))
        if self._layout_key != layout or len(self._boxes) != self._clouds.count:
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                if stage.GetPrimAtPath(HERO_ROOT):
                    stage.RemovePrim(HERO_ROOT)
                stage.DefinePrim("/WeatherFX", "Xform")
                stage.DefinePrim(HERO_ROOT, "Xform")
                self._boxes, self._shaders = [], []
                for i in range(self._clouds.count):
                    prim_path = f"{HERO_ROOT}/Hero_{i}"
                    # A non-zero shift makes the box author its translate op, which _place moves.
                    tile = dict(self._grid, shift=(1e-3, 0.0, 0.0))
                    self._shaders.append(_author_volume_box(stage, prim_path, tile))
                    self._boxes.append(prim_path)
            self._layout_key = layout
            self._last_corners = None
        self._place()
        self._update_materials(stage, state)

    def _place(self) -> None:
        """Move each box to its drifted, wrapped place. Translates only (ADR 0144)."""
        if self._clouds is None or not self._boxes:
            return
        stage = self.context.stage()
        if stage is None:
            return
        from pxr import Gf, Usd, UsdGeom

        up_axis = self.context.up_axis()
        mpu = self.context.meters_per_unit()
        drift = stage_to_field(self.context.cloud_drift_m, up_axis)
        anchor = stage_to_field(np.asarray(self.context.anchor(), dtype=np.float64) * mpu, up_axis)
        corners = self._clouds.stage_lower_corners_m(drift, anchor, up_axis) / mpu
        if self._last_corners is not None and np.allclose(corners, self._last_corners, atol=1e-3):
            return
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for path, corner in zip(self._boxes, corners):
                prim = stage.GetPrimAtPath(path)
                if not prim:
                    continue
                xform = UsdGeom.Xformable(prim)
                ops = xform.GetOrderedXformOps()
                op = ops[0] if ops else xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                op.Set(Gf.Vec3d(*(float(v) for v in corner)))
        self._last_corners = corners

    def _update_materials(self, stage: Any, state: Any) -> None:
        if self._shaders:
            set_volume_inputs(stage, self._shaders, float(state.clouds.hero_extinction_per_m),
                              self.context.meters_per_unit(), state.clouds)

    def _teardown(self) -> None:
        self._job.cancel()
        stage = self.context.stage() if hasattr(self, "context") else None
        if stage is not None:
            try:
                from pxr import Usd

                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    if stage.GetPrimAtPath(HERO_ROOT):
                        stage.RemovePrim(HERO_ROOT)
            except Exception:
                log.exception("weather_fx: could not remove the hero clouds")
        _remove_files(self._files)
        self._files, self._shaders, self._boxes = [], [], []
        self._built_key = None
        self._layout_key = None
        self._asset, self._clouds, self._last_corners = None, None, None
        VolumeSettingsLease.release(self.name)
        self._active = False


# --------------------------------------------------------------------------- worker side

def read_vdb_density(path: str, max_dense_voxels: int = MAX_DENSE_VOXELS):
    """``(values, voxel_size)`` of the density grid in an OpenVDB file, as a dense numpy array.

    Takes the grid named ``density``, or failing that the first float grid. ``values`` is
    ``(nx, ny, nz)`` over the active bounding box; ``voxel_size`` is the file's own, unused for
    the scale (a hero cloud is scaled to ``hero_size_m``) but reported.
    """
    from .vdb import import_openvdb

    openvdb = import_openvdb()
    if openvdb is None:
        raise RuntimeError("OpenVDB is not importable; enable the omni.volume extension")
    names = [getattr(m, "name", None) or m["name"] for m in openvdb.readAllGridMetadata(path)]
    name = "density" if "density" in names else (names[0] if names else None)
    if name is None:
        raise RuntimeError(f"{path} holds no grids")
    grid = openvdb.read(path, name)
    (i0, j0, k0), (i1, j1, k1) = grid.evalActiveVoxelBoundingBox()
    shape = (i1 - i0 + 1, j1 - j0 + 1, k1 - k0 + 1)
    if int(np.prod(shape, dtype=np.int64)) > max_dense_voxels:
        raise RuntimeError(
            f"{os.path.basename(path)} is {shape[0]}x{shape[1]}x{shape[2]} voxels, too large to "
            "load densely; use a lower-resolution file (for the Disney cloud: wdas_cloud_eighth.vdb "
            "or wdas_cloud_quarter.vdb)")
    values = np.zeros(shape, dtype=np.float32)
    grid.copyToArray(values, ijk=(i0, j0, k0))
    voxel = grid.transform.voxelSize()
    return values, tuple(float(v) for v in voxel)


def _build_hero(path: str, size_m: float, max_voxels: int, up_axis: int, mpu: float,
                directory: pathlib.Path, key: tuple) -> dict:
    """Runs in the worker thread: read, filter, scale, write the asset once for every cloud."""
    import hashlib

    from .vdb import import_openvdb, write_fog_volume

    values, _ = read_vdb_density(path)
    asset = prepare_hero_asset(values, size_m, max_voxels)
    array, voxel_m, size = asset.volume_grid(up_axis)
    to_units = 1.0 / mpu
    voxel = voxel_m * to_units
    digest = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:12]
    out = directory / f"weather_fx_hero_{os.getpid()}_{digest}.vdb"
    # Lower corner at the local origin; each box is moved to its place by a translate.
    write_fog_volume(import_openvdb(), out, array, (voxel, voxel, voxel),
                     (0.5 * voxel, 0.5 * voxel, 0.5 * voxel))
    grid = {"file": str(out), "low": (0.0, 0.0, 0.0),
            "high": tuple(s * to_units for s in size)}
    return {"files": [out], "asset": asset, "grid": grid}
