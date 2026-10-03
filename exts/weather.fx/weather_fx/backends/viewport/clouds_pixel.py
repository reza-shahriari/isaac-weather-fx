"""Clouds marched per camera pixel, drawn in the viewport every frame.

The cloud is :class:`~weather_fx.core.cloudscape.Cloudscape`, a function. Each frame the GPU march
(:mod:`weather_fx.gpu.cloud_march`) evaluates it along every pixel's ray of the active camera,
composes it over the clear sky through the air in front of it, and the result goes straight from
the GPU into a dynamic texture. That texture is the emission of one quad that rides with the
camera at the far end of its frustum and fills it exactly.

So the sky the camera sees is the march at the screen's own resolution, in every direction, and
scene geometry, being nearer than the quad, is drawn in front of it by the renderer itself. It
works the same under RTX Real-Time and the path tracer, because an emissive surface is the same
thing to both. The dome light stays, clear of cloud, to light the scene.

**Measured, not assumed** (Isaac Sim 6, Kit 110, RTX A6000):

* an OmniPBR emission of texture value ``v`` at ``emissive_intensity`` ``pi * I`` renders exactly
  as bright as a dome light of texture value ``v`` at intensity ``I``, in both render modes;
* ``DynamicTextureProvider.set_bytes_data_from_gpu`` takes 0.08 ms for 1280 x 720 RGBA32F, and
  expects the array on CUDA device 0 -- which must be the renderer's GPU, so on a machine with
  two cards set ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` and ``active_gpu`` to the same index.

Everything is authored in the session layer under ``/WeatherFX/CloudLayer`` and removed on detach.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Optional, Set

import numpy as np

from ...core import layer_tables
from ...core.cloudscape import CLOUDSCAPE_SHAPE_KEYS, cloudscape_from_state
from ...core.clouds import stage_to_field
from ...core.jobs import LatestJob
from ...core.sky import conditions_from_state
from ..base import Effect
from .render_mode import cloud_path

log = logging.getLogger("weather_fx")

LAYER_ROOT = "/WeatherFX/CloudLayer"
QUAD_PATH = LAYER_ROOT + "/Quad"
MATERIAL_PATH = LAYER_ROOT + "/Material"
TEXTURE_NAME = "weather_fx_cloud_layer"
#: OmniPBR ``emissive_intensity`` per unit of dome-light intensity, for equal radiance (measured).
EMISSIVE_PER_DOME = math.pi
#: Where the quad sits, as a fraction of the camera's far clipping distance.
QUAD_DEPTH = 0.9
#: The sun's movement, degrees, after which the sky and air tables are rebuilt.
TABLE_STEP_DEG = 0.25


class CloudLayerEffect(Effect):
    name = "cloud_layer"

    def __init__(self) -> None:
        self._job = LatestJob("weather_fx-cloudscape")
        self._renderer = None
        self._renderer_key = None
        self._tables = None
        self._tables_key = None
        self._lighting = None
        self._conditions = None
        self._provider = None
        self._authored = False
        self._quad_key = None
        self._intensity = None
        self._frame_ms = 0.0
        self._size = (0, 0)
        self._failed = False

    # --- lifecycle ---------------------------------------------------------------------

    def _wanted(self, state: Any) -> bool:
        return bool(state.general.enabled and state.sky.enabled and state.clouds.enabled
                    and state.clouds.cover > 0.0 and cloud_path(state) == "pixel")

    def apply_state(self, state: Any, changed: Set[str]) -> None:
        if not self._wanted(state) or self._failed:
            self._remove()
            return
        key = tuple(getattr(state.clouds, name) for name in CLOUDSCAPE_SHAPE_KEYS)
        if key != self._renderer_key and key != self._job.running_key:
            snapshot = state.copy()
            if state.general.time_source == "manual":
                self._job.cancel()
                self._install(key, cloudscape_from_state(snapshot))
            else:
                self._job.submit(key, lambda: cloudscape_from_state(snapshot))
        self._conditions = conditions_from_state(state, build_cloud=False)

    def update(self, dt: float, t: float) -> None:
        state = self.context.state
        if not self._wanted(state) or self._failed:
            if self._authored:
                self._remove()
            return
        try:
            done = self._job.poll()
        except Exception:
            log.exception("weather_fx: building the cloudscape failed")
            done = None
        if done is not None:
            self._install(*done)
        if self._renderer is None:
            return
        try:
            self._draw(state)
        except Exception:
            log.exception("weather_fx: the per-pixel cloud layer failed; turning it off")
            self._failed = True
            self._remove()

    def detach(self) -> None:
        self._job.cancel()
        self._remove()
        self._renderer = None
        self._renderer_key = None
        self._tables = None
        self._provider = None

    def stats(self) -> dict:
        return {"active": self._authored, "frame_ms": round(self._frame_ms, 2),
                "resolution": list(self._size), "building": self._job.busy,
                "ready": self._renderer is not None, "failed": self._failed}

    # --- the cloud and its tables --------------------------------------------------------

    def _install(self, key: Any, cloudscape: Any) -> None:
        if cloudscape is None:
            self._renderer, self._renderer_key = None, key
            return
        from ...gpu import cloud_march

        self._renderer = cloud_march.CloudRenderer(cloudscape, device="cuda:0")
        self._renderer_key = key

    def _refresh_tables(self, state: Any, height_m: float) -> None:
        conditions = conditions_from_state(state, build_cloud=False)
        body = layer_tables.lit_body(conditions)
        key = (round(body.elevation_deg / TABLE_STEP_DEG), round(body.azimuth_deg / TABLE_STEP_DEG),
               body is conditions.sun, round(float(state.sky.turbidity), 3),
               repr(state.sky.ground_albedo), state.sky.model,
               round(float(state.sky.exposure_scale), 4), round(height_m / 250.0))
        if key == self._tables_key:
            return
        from ...gpu import cloud_march

        sky = layer_tables.sky_table(conditions)
        air_in, air_tr = layer_tables.air_tables(conditions, observer_height_m=max(height_m, 2.0))
        light = layer_tables.lighting_for(conditions)
        self._tables = cloud_march.SkyTables(sky, air_in, air_tr, light["azimuth_rad"],
                                             layer_tables.AIR_NEAR_M, layer_tables.AIR_FAR_M, "cuda:0")
        self._lighting = cloud_march.Lighting(light["sun_direction"], light["sun_rgb"],
                                              light["above_rgb"], light["below_rgb"])
        self._tables_key = key
        self._conditions = conditions

    # --- one frame -------------------------------------------------------------------------

    def _camera(self):
        """``(prim, width, height)`` of the camera the layer is drawn for."""
        stage = self.context.stage()
        path, size = None, (1280, 720)
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                path = str(viewport.camera_path)
                size = tuple(int(v) for v in viewport.resolution)
        except Exception:
            pass
        follow = self.context.state.general.follow_prim
        if follow and stage.GetPrimAtPath(follow).GetTypeName() == "Camera":
            path = follow
        if not path:
            return None, size
        prim = stage.GetPrimAtPath(path)
        return (prim if prim and prim.GetTypeName() == "Camera" else None), size

    def _draw(self, state: Any) -> None:
        from pxr import Gf, Usd, UsdGeom

        from ...gpu import cloud_march

        start = time.perf_counter()
        stage = self.context.stage()
        prim, (width, height) = self._camera()
        if stage is None or prim is None or width < 8 or height < 8:
            return
        scale = float(getattr(state.clouds, "layer_scale", 1.0))
        width, height = max(8, int(width * scale)), max(8, int(height * scale))
        camera = UsdGeom.Camera(prim)
        focal = float(camera.GetFocalLengthAttr().Get() or 18.0)
        aperture = float(camera.GetHorizontalApertureAttr().Get() or 20.955)
        clip = camera.GetClippingRangeAttr().Get() or Gf.Vec2f(0.1, 1.0e6)
        world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        mpu = self.context.meters_per_unit()
        up_axis = self.context.up_axis()

        def direction(x: float, y: float, z: float) -> np.ndarray:
            v = world.TransformDir(Gf.Vec3d(x, y, z))
            v = stage_to_field(np.array([v[0], v[1], v[2]], dtype=np.float64), up_axis)
            return v / max(float(np.linalg.norm(v)), 1e-12)

        t = world.ExtractTranslation()
        position_m = np.array([t[0], t[1], t[2]], dtype=np.float64) * mpu
        origin = stage_to_field(position_m - self.context.cloud_drift_m, up_axis)
        self._refresh_tables(state, float(origin[1]))

        hfov = 2.0 * math.degrees(math.atan(0.5 * aperture / focal))
        view = cloud_march.Camera(width, height, hfov, 0.0, 0.0,
                                  position_m=tuple(float(v) for v in origin),
                                  pose=(direction(0, 0, -1), direction(1, 0, 0), direction(0, 1, 0)))
        dome = getattr(self.context, "sky_dome", None) or {}
        gains = tuple(dome.get("gains", (1.0, 1.0, 1.0)))
        layer = self._renderer.render_layer(
            view, self._lighting, self._tables, gains=gains, flip=False,
            density_scale=float(state.clouds.density_scale))

        import omni.ui as ui

        if self._provider is None:
            self._provider = ui.DynamicTextureProvider(TEXTURE_NAME)
        self._provider.set_bytes_data_from_gpu(layer.ptr, [width, height],
                                               format=ui.TextureFormat.RGBA32_SFLOAT)
        intensity = (EMISSIVE_PER_DOME * float(dome.get("exposure", 1.0))
                     * float(state.sky.exposure_scale))
        self._author(stage, world, focal, aperture, float(clip[1]), width / height, intensity)
        self._size = (width, height)
        self._frame_ms = (time.perf_counter() - start) * 1000.0

    # --- authoring -------------------------------------------------------------------------

    def _author(self, stage: Any, world: Any, focal: float, aperture: float, far: float,
                aspect: float, intensity: float) -> None:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        with Usd.EditContext(stage, stage.GetSessionLayer()):
            if not self._authored:
                stage.DefinePrim("/WeatherFX", "Xform")
                stage.DefinePrim(LAYER_ROOT, "Xform")
                quad = UsdGeom.Mesh.Define(stage, QUAD_PATH)
                quad.CreateFaceVertexCountsAttr([4])
                quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
                quad.CreateDoubleSidedAttr(True)
                UsdGeom.PrimvarsAPI(quad.GetPrim()).CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
                ).Set([(0, 0), (1, 0), (1, 1), (0, 1)])
                for name in ("primvars:doNotCastShadows", "primvars:invisibleToSecondaryRays"):
                    quad.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Bool).Set(True)
                material = UsdShade.Material.Define(stage, MATERIAL_PATH)
                shader = UsdShade.Shader.Define(stage, MATERIAL_PATH + "/Shader")
                shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
                shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
                shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
                shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set((0, 0, 0))
                shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.0)
                shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(1.0)
                shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
                shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set((1, 1, 1))
                shader.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(
                    Sdf.AssetPath(f"dynamic://{TEXTURE_NAME}"))
                shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(float(intensity))
                material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
                UsdShade.MaterialBindingAPI.Apply(quad.GetPrim()).Bind(material)
                UsdGeom.Xformable(quad).AddTransformOp()
                self._authored = True
                self._quad_key = None
                self._intensity = float(intensity)
            quad = UsdGeom.Mesh(stage.GetPrimAtPath(QUAD_PATH))
            key = (round(focal, 4), round(aperture, 4), round(far, 2), round(aspect, 5))
            if key != self._quad_key:
                depth = QUAD_DEPTH * far
                half_w = depth * 0.5 * aperture / focal
                half_h = half_w / aspect
                quad.GetPointsAttr().Set([(-half_w, -half_h, -depth), (half_w, -half_h, -depth),
                                          (half_w, half_h, -depth), (-half_w, half_h, -depth)])
                quad.CreateExtentAttr([(-half_w, -half_h, -depth), (half_w, half_h, -depth)])
                self._quad_key = key
            UsdGeom.Xformable(quad).GetOrderedXformOps()[0].Set(Gf.Matrix4d(world))
            if self._intensity is None or abs(intensity - self._intensity) > 1e-6 * max(intensity, 1e-9):
                shader = UsdShade.Shader(stage.GetPrimAtPath(MATERIAL_PATH + "/Shader"))
                shader.GetInput("emissive_intensity").Set(float(intensity))
                self._intensity = float(intensity)

    def _remove(self) -> None:
        if not self._authored:
            return
        self._authored = False
        self._quad_key = None
        stage = self.context.stage() if hasattr(self, "context") else None
        if stage is None:
            return
        try:
            from pxr import Usd

            with Usd.EditContext(stage, stage.GetSessionLayer()):
                if stage.GetPrimAtPath(LAYER_ROOT):
                    stage.RemovePrim(LAYER_ROOT)
        except Exception:
            log.exception("weather_fx: could not remove the cloud layer")
