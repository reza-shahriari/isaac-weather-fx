"""Clouds as real 3D volumes, for the path tracer.

The cloud field in :mod:`weather_fx.core.clouds` is voxelised to OpenVDB and placed in the stage
as volumes, so a cloud has a position, an inside, parallax as the camera moves, and a shadow on
the ground -- none of which a picture on the dome can have. It is the same field the dome march
and an infrared sensor model integrate.

**The recipe is the one measured to render** (the thermal-camera project's ADR 0144), not the
USD-textbook one: a ``UsdVol.Volume`` with an OpenVDB field never rendered there. What does is a
mesh box whose vertices sit at the grid's world bounds with **no transform**, marked
``primvars:isVolume``, bound to ``OmniVolumeDensity`` with the file on ``volume_density_texture``,
plus the path tracer's non-uniform-volume settings. Each clause was found by its failure.

**The clouds move and never run out.** The boxes have no transform of their own; their parent,
``/WeatherFX/Clouds``, carries one translate: the wind's drift (``context.cloud_drift_m``) plus
whole tiles chosen to keep the camera in the central tile. The field is periodic, so a whole-tile
jump is invisible, and a translate carries the texture with the box -- only a *scale* stretches it.

**Nothing slow runs on the UI thread.** Building the field and writing the files happens in a
worker; the prims are authored from ``update`` when it finishes, and the previous clouds stay up
until then. Colour, density and forward scattering are material inputs and change instantly. With
``general.time_source = "manual"`` the work is done synchronously instead, so a synthetic-data
frame always has the clouds its state describes.
"""
from __future__ import annotations

import logging
import os
import pathlib
import tempfile
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ...core.clouds import (
    CLOUD_SHAPE_KEYS,
    cloud_field_from_state,
    tile_placements,
    volume_offset_m,
)
from ...core.jobs import LatestJob
from ..base import Effect
from .render_mode import RenderModeWatcher, cloud_path

log = logging.getLogger("weather_fx")

CLOUDS_ROOT = "/WeatherFX/Clouds"

#: Path-tracer settings a heterogeneous volume needs (ADR 0144). The master switch is
#: ``ptvol/enabled``; without it a volume is invisible. Set as given.
VOLUME_SETTINGS = {
    "/rtx/pathtracing/ptvol/enabled": True,
    "/rtx/pathtracing/ptvol/transmittanceMethod": 0,
    "/rtx/pathtracing/volumesAOV": True,
}
#: Limits raised to at least these values and **never lowered**. A cloud of optical depth 18 is a
#: medium light crosses by hundreds of scattering events; every limit below that ends a path early
#: and ends it dark, which is exactly the grey, faded look. The collision counts bound the
#: delta-tracking steps through the heterogeneous grid (for camera and for shadow rays); the
#: renderer's own default of 1024 was being *lowered* to 128 here, which cut thick clouds off.
#: The volume bounce counts come from ``clouds.volume_bounces``.
VOLUME_FLOORS = {
    "/rtx/pathtracing/ptvol/maxCollisionCount": 1024,
    "/rtx/pathtracing/ptvol/maxLightCollisionCount": 256,
    "/rtx/pathtracing/maxBounces": 16,
}
#: Settings that take ``clouds.volume_bounces``.
VOLUME_BOUNCE_SETTINGS = ("/rtx/pathtracing/ptvol/maxBounces", "/rtx/pathtracing/maxVolumeBounces")

#: Single-scattering albedo of cloud droplets in the visible. Liquid water at 0.55 um absorbs so
#: little that the true value is 0.99999; it was 0.96 here, and at a few hundred scattering events
#: per path that 4 % loss per bounce is what turned the body of a thick cloud grey. 0.999 keeps a
#: sliver of absorption for the bounce limit to lean on.
DROPLET_ALBEDO = 0.999


class VolumeSettingsLease:
    """The path-tracer volume settings, shared by every effect that draws a volume.

    The procedural cloud volume and the hero clouds each need them; each holding its own copy of
    "the user's original values" would let one of them restore the renderer's defaults -- and so
    switch volumes off -- while the other is still drawing. So the originals are captured once,
    each owner holds a lease with the bounce depth it wants, the settings follow the largest lease,
    and they are restored only when the last lease goes.
    """

    _leases: Dict[str, int] = {}
    _originals: Dict[str, Any] = {}

    @classmethod
    def acquire(cls, owner: str, bounces: int) -> None:
        cls._leases[owner] = int(bounces)
        cls._apply()

    @classmethod
    def release(cls, owner: str) -> None:
        if cls._leases.pop(owner, None) is None:
            return
        if cls._leases:
            cls._apply()
            return
        try:
            import carb.settings

            settings = carb.settings.get_settings()
            for path, value in cls._originals.items():
                settings.set(path, value)
        except Exception:
            log.exception("weather_fx: could not restore the path-tracer volume settings")
        cls._originals.clear()

    @classmethod
    def _apply(cls) -> None:
        try:
            import carb.settings
        except ImportError:
            return
        settings = carb.settings.get_settings()
        for path, value in VOLUME_SETTINGS.items():
            current = settings.get(path)
            if current is None:
                continue
            cls._originals.setdefault(path, current)
            if current != value:
                settings.set(path, value)
        floors = dict(VOLUME_FLOORS)
        bounces = max(cls._leases.values(), default=0)
        for path in VOLUME_BOUNCE_SETTINGS:
            floors[path] = bounces
        for path, floor in floors.items():
            current = settings.get(path)
            if current is None:
                continue
            original = cls._originals.setdefault(path, current)
            # Raise to the floor, never below what the user had; a lowered volume_bounces goes
            # back down as far as the user's own value.
            wanted = max(int(original), int(floor))
            if int(current) != wanted:
                settings.set(path, wanted)


class CloudVolumeEffect(Effect):
    name = "cloud_volume"

    def __init__(self, directory: Optional[str] = None):
        self._directory = pathlib.Path(directory) if directory else None
        self._job = LatestJob("weather_fx-clouds")
        self._watcher: Optional[RenderModeWatcher] = None
        self._built_key: Optional[tuple] = None
        self._files: List[pathlib.Path] = []
        self._shaders: List[str] = []
        self._extinction_per_m = 0.0
        self._active = False
        self._voxels = 0
        self._tile_m = 0.0
        self._placed = None

    # --- lifecycle ---------------------------------------------------------------------

    def attach(self, context) -> None:
        super().attach(context)
        self._watcher = RenderModeWatcher()

    def apply_state(self, state: Any, changed: Set[str]) -> None:
        stage = self.context.stage()
        if stage is None:
            return
        wanted = (state.general.enabled and state.clouds.enabled and state.clouds.cover > 0.0
                  and cloud_path(state) == "volume")
        if not wanted:
            if self._active or self._built_key is not None:
                self._teardown()
            return
        self._enable_settings(state)
        self._active = True

        key = self._shape_key(state)
        if key != self._built_key and key != self._job.running_key:
            snapshot = state.copy()
            up_axis, mpu = self.context.up_axis(), self.context.meters_per_unit()
            directory = self._directory or pathlib.Path(tempfile.gettempdir())

            def build():
                return _build_volumes(snapshot, up_axis, mpu, directory, key)

            if state.general.time_source == "manual":
                self._job.cancel()
                self._install(stage, key, build(), state)
            else:
                self._job.submit(key, build)
        self._update_materials(stage, state)

    def update(self, dt: float, t: float) -> None:
        if self._watcher is not None and self._watcher.changed():
            self.apply_state(self.context.state, {"clouds"})
        self._place()
        try:
            done = self._job.poll()
        except Exception:
            log.exception("weather_fx: building the cloud volumes failed")
            return
        if done is None:
            return
        key, result = done
        stage = self.context.stage()
        state = self.context.state
        if stage is None or not self._active or key != self._shape_key(state):
            _remove_files(result["files"])  # superseded while it was being built
            return
        self._install(stage, key, result, state)

    def detach(self) -> None:
        self._teardown()

    def stats(self) -> dict:
        return {
            "active": self._active,
            "tiles": len(self._files),
            "active_voxels": self._voxels,
            "extinction_per_m": round(self._extinction_per_m, 6),
            "offset": None if self._placed is None else [round(float(v), 2) for v in self._placed],
            "building": self._job.busy,
        }

    # --- building ----------------------------------------------------------------------

    def _shape_key(self, state: Any) -> tuple:
        clouds = state.clouds
        return (tuple(getattr(clouds, k) for k in CLOUD_SHAPE_KEYS), int(clouds.volume_tiles),
                int(clouds.volume_detail),
                self.context.up_axis(), round(self.context.meters_per_unit(), 9))

    def _install(self, stage: Any, key: tuple, result: dict, state: Any) -> None:
        from pxr import Usd

        previous = self._files
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            if stage.GetPrimAtPath(CLOUDS_ROOT):
                stage.RemovePrim(CLOUDS_ROOT)
            stage.DefinePrim("/WeatherFX", "Xform")
            stage.DefinePrim(CLOUDS_ROOT, "Xform")
            self._shaders = [
                _author_volume_box(stage, f"{CLOUDS_ROOT}/{tile['name']}", tile)
                for tile in result["tiles"]
            ]
        self._files = list(result["files"])
        self._tile_m = float(result["tile_m"])
        self._placed = None
        self._place()
        self._extinction_per_m = float(result["extinction_per_m"])
        self._voxels = int(result["voxels"])
        self._built_key = key
        _remove_files([f for f in previous if f not in self._files])
        self._update_materials(stage, state)

    def _place(self) -> None:
        """Move the volumes: the wind's drift, plus whole tiles to keep the camera in the middle.

        One translate on the root, so it costs nothing per frame. It is a translation only: the
        density texture is read in the box's local frame, so a translate carries the cloud with
        the box, where a scale would stretch the texture (ADR 0144's failure was a scale).
        """
        if not self._shaders or self._tile_m <= 0.0:
            return
        stage = self.context.stage()
        if stage is None:
            return
        mpu = self.context.meters_per_unit()
        up_axis = self.context.up_axis()
        anchor_m = np.asarray(self.context.anchor(), dtype=np.float64) * mpu
        offset = volume_offset_m(self.context.cloud_drift_m, anchor_m, self._tile_m, up_axis) / mpu
        if self._placed is not None and np.allclose(offset, self._placed, atol=1e-3):
            return
        from pxr import Gf, Usd, UsdGeom

        with Usd.EditContext(stage, stage.GetSessionLayer()):
            prim = stage.GetPrimAtPath(CLOUDS_ROOT)
            if not prim:
                return
            xform = UsdGeom.Xformable(prim)
            ops = xform.GetOrderedXformOps()
            op = ops[0] if ops else xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            op.Set(Gf.Vec3d(*(float(v) for v in offset)))
        self._placed = offset

    def _update_materials(self, stage: Any, state: Any) -> None:
        """Colour, density and phase: material inputs, so they never cost a rebuild."""
        if not self._shaders:
            return
        # The grid holds normalised density; the path tracer integrates it per stage unit.
        set_volume_inputs(stage, self._shaders, self._extinction_per_m,
                          self.context.meters_per_unit(), state.clouds)

    def _teardown(self) -> None:
        self._job.cancel()
        stage = self.context.stage() if hasattr(self, "context") else None
        if stage is not None:
            try:
                from pxr import Usd

                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    if stage.GetPrimAtPath(CLOUDS_ROOT):
                        stage.RemovePrim(CLOUDS_ROOT)
            except Exception:
                log.exception("weather_fx: could not remove the cloud volumes")
        _remove_files(self._files)
        self._files, self._shaders = [], []
        self._built_key = None
        self._restore_settings()
        self._active = False

    # --- render settings ---------------------------------------------------------------

    def _enable_settings(self, state: Any) -> None:
        VolumeSettingsLease.acquire(self.name, int(state.clouds.volume_bounces))

    def _restore_settings(self) -> None:
        VolumeSettingsLease.release(self.name)


# --------------------------------------------------------------------------- worker side

def _build_volumes(state: Any, up_axis: int, mpu: float, directory: pathlib.Path, key: tuple) -> dict:
    """Runs in the worker thread: build the field, voxelise it, write it once for every tile."""
    import hashlib

    from .vdb import import_openvdb, write_fog_volume

    openvdb = import_openvdb()
    if openvdb is None:
        raise RuntimeError(
            "OpenVDB is not importable; enable the omni.volume extension, or set "
            "clouds.render_path = 'dome'"
        )
    field = cloud_field_from_state(state)
    grid = field.volume_grid(up_axis, refine=int(state.clouds.volume_detail))
    digest = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:12]
    to_units = 1.0 / mpu
    voxel = tuple(v * to_units for v in grid.voxel_m)
    # **One file for every tile.** A translate on a box carries its density texture with it (the
    # drift already relies on that), so each tile is the same grid shifted by whole tiles. Nine
    # copies of the file cost nine writes -- held under the interpreter lock by the OpenVDB
    # binding, which is UI time -- and nine textures on the GPU for no information.
    path = directory / f"weather_fx_cloud_{os.getpid()}_{digest}.vdb"
    voxels = write_fog_volume(openvdb, path, grid.values, voxel,
                              tuple(c * to_units for c in grid.first_centre_m))
    files = [path]
    tiles = []
    for placement in tile_placements(grid, state.clouds.volume_tiles, up_axis):
        tiles.append({
            "name": placement.name,
            "file": str(path),
            "low": tuple(c * to_units for c in grid.low_m),
            "high": tuple(c * to_units for c in grid.high_m),
            "shift": tuple((p - g) * to_units for p, g in zip(placement.low_m, grid.low_m)),
        })
    return {"tiles": tiles, "files": files, "voxels": voxels, "tile_m": grid.tile_m,
            "extinction_per_m": field.extinction_per_m}


def set_volume_inputs(stage: Any, shaders: List[str], extinction_per_m: float, mpu: float,
                      clouds: Any) -> None:
    """Density, colour and phase of a cloud volume's material: inputs, so never a rebuild."""
    from pxr import Gf, Sdf, Usd, UsdShade

    scale = float(extinction_per_m) * float(mpu) * float(clouds.density_scale)
    albedo = Gf.Vec3f(*(DROPLET_ALBEDO * float(c) for c in clouds.lit_color))
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path in shaders:
            shader = UsdShade.Shader(stage.GetPrimAtPath(path))
            if not shader:
                continue
            shader.CreateInput("volume_density_scale", Sdf.ValueTypeNames.Float).Set(scale)
            tint = shader.CreateInput("volume_albedo", Sdf.ValueTypeNames.Color3f)
            tint.Set(albedo)
            tint.GetAttr().SetColorSpace("raw")
            shader.CreateInput("directional_bias", Sdf.ValueTypeNames.Float).Set(
                float(clouds.phase_bias))


def _remove_files(files) -> None:
    for path in files:
        try:
            pathlib.Path(path).unlink()
        except OSError:
            pass


def _author_volume_box(stage: Any, prim_path: str, tile: dict) -> str:
    """A mesh box at the grid's bounds, bound to ``OmniVolumeDensity``. Returns the shader path.

    The box's points are the grid's own bounds, because the density texture is sampled in the
    prim's local frame and the grid's transform is in those units. The tile's place comes from a
    **translate only**: it carries the texture with the box, where a scale would stretch it
    (ADR 0144's failure).
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    x0, y0, z0 = tile["low"]
    x1, y1, z1 = tile["high"]
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    corners = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
               (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    mesh.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(
        [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7])
    # Authored, not computed: a prim with no bounds is culled before anything opens the file.
    mesh.CreateExtentAttr([Gf.Vec3f(x0, y0, z0), Gf.Vec3f(x1, y1, z1)])
    xformable = UsdGeom.Xformable(mesh.GetPrim())
    xformable.ClearXformOpOrder()
    shift = tile.get("shift", (0.0, 0.0, 0.0))
    if any(abs(v) > 0.0 for v in shift):
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*shift))
    UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "isVolume", Sdf.ValueTypeNames.Bool).Set(True)
    material = UsdShade.Material.Define(stage, f"{prim_path}/Material")
    shader = UsdShade.Shader.Define(stage, f"{prim_path}/Material/Shader")
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath("OmniVolumeDensity.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniVolumeDensity", "mdl")
    shader.CreateInput("volume_density_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(tile["file"]))
    for output in (material.CreateSurfaceOutput("mdl"), material.CreateDisplacementOutput("mdl"),
                   material.CreateVolumeOutput("mdl")):
        output.ConnectToSource(shader.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
    return str(shader.GetPath())
