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
#: How much of the ground's upward light a full cover takes away.
GROUND_SHADE = 0.8
#: How much of the clear air's light in front of the cloud a full cover takes away (the sun's
#: beam does not reach the air under a deck); it goes as the square of the cover, since under
#: scattered cloud most of the air on a line of sight is still in the sun.
AIR_SHADE = 0.85
VEIL_PATH = LAYER_ROOT + "/Veil"
VEIL_MATERIAL_PATH = LAYER_ROOT + "/VeilMaterial"
VEIL_TEXTURE_NAME = "weather_fx_cloud_veil"
VEIL_OPACITY_TEXTURE_NAME = "weather_fx_cloud_veil_opacity"
#: Where the veil sits in front of the camera, metres (and never inside the near clip).
VEIL_DEPTH_M = 0.5
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
        self._depth = None
        self._veil_providers = None
        self._veil_key = None
        self._veil_visible = False
        self._veil_failed = False

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
        self._veil_providers = None
        self._drop_depth()

    def stats(self) -> dict:
        return {"active": self._authored, "frame_ms": round(self._frame_ms, 2),
                "resolution": list(self._size), "building": self._job.busy,
                "ready": self._renderer is not None, "failed": self._failed,
                "veil": self._veil_visible}

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
               round(float(state.sky.exposure_scale), 4), round(height_m / 250.0),
               round(float(state.clouds.cover), 2))
        if key == self._tables_key:
            return
        from ...gpu import cloud_march

        sky = layer_tables.sky_table(conditions)
        air_in, air_tr = layer_tables.air_tables(conditions, observer_height_m=max(height_m, 2.0))
        light = layer_tables.lighting_for(conditions)
        self._tables = cloud_march.SkyTables(sky, air_in, air_tr, light["azimuth_rad"],
                                             layer_tables.AIR_NEAR_M, layer_tables.AIR_FAR_M, "cuda:0")
        # The ground under the cloud is lit by what the cloud lets through: under an overcast it
        # sends up a fraction of what it does under a clear sky, and the underside of the deck
        # is grey, not the colour of sunlit ground.
        shaded = 1.0 - GROUND_SHADE * min(max(float(state.clouds.cover), 0.0), 1.0)
        self._lighting = cloud_march.Lighting(light["sun_direction"], light["sun_rgb"], light["above_rgb"],
                                              tuple(float(v) * shaded for v in light["below_rgb"]))
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
        if follow and stage.GetPrimAtPath(follow).GetTypeName() == "Camera" and follow != path:
            path = follow
            # Not the viewport's camera, so not the viewport's frame either: the layer has to
            # have the shape of what renders this camera, or its quad leaves the frame's top and
            # bottom bare. A render product of the camera says it; failing that, its apertures.
            size = self._frame_of(stage, follow, size)
        if not path:
            return None, size
        prim = stage.GetPrimAtPath(path)
        return (prim if prim and prim.GetTypeName() == "Camera" else None), size

    @staticmethod
    def _frame_of(stage: Any, camera_path: str, fallback: Any) -> Any:
        """``(width, height)`` of the frame a camera is rendered at."""
        from pxr import Usd, UsdGeom

        render = stage.GetPrimAtPath("/Render")
        if render:
            for prim in Usd.PrimRange(render):
                if prim.GetTypeName() != "RenderProduct":
                    continue
                targets = prim.GetRelationship("camera").GetTargets()
                resolution = prim.GetAttribute("resolution").Get() if targets else None
                if targets and str(targets[0]) == camera_path and resolution:
                    return int(resolution[0]), int(resolution[1])
        camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
        wide = camera.GetHorizontalApertureAttr().Get()
        tall = camera.GetVerticalApertureAttr().Get()
        if wide and tall:
            return int(fallback[0]), max(8, int(round(fallback[0] * float(tall) / float(wide))))
        return fallback

    # --- the scene's depth, for the cloud in front of objects ----------------------------------

    def _drop_depth(self) -> None:
        for annotator in (self._depth or {}).values():
            try:
                annotator.detach()
            except Exception:
                pass
        self._depth = None

    def _scene_depth(self, state: Any, stage: Any, camera_path: str):
        """Distance from the camera to the nearest surface per pixel, as a Warp array on the
        renderer's GPU, from the frame before; None when it is not to be had (nothing renders
        this camera, no replicator, the first frame). It is read from whichever render product
        of this camera delivers one: the viewport's, or one a script made. The veil quad is not
        in it: a surface that is partly see-through leaves no depth."""
        if self._veil_failed or not bool(getattr(state.clouds, "veil_scene", True)):
            self._drop_depth()
            return None
        try:
            import omni.replicator.core as rep

            products = []
            try:
                from omni.kit.viewport.utility import get_active_viewport

                viewport = get_active_viewport()
                if viewport is not None and str(viewport.camera_path) == camera_path:
                    products.append(str(viewport.render_product_path))
            except Exception:
                pass
            render = stage.GetPrimAtPath("/Render")
            if render:
                from pxr import Usd

                for prim in Usd.PrimRange(render):
                    if prim.GetTypeName() == "RenderProduct" and str(prim.GetPath()) not in products:
                        targets = prim.GetRelationship("camera").GetTargets()
                        if targets and str(targets[0]) == camera_path:
                            products.append(str(prim.GetPath()))
            if self._depth is None:
                self._depth = {}
            for product in [p for p in self._depth if p not in products]:
                self._depth.pop(product).detach()
            found = None
            for product in products:
                annotator = self._depth.get(product)
                if annotator is None:
                    annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera", device="cuda")
                    annotator.attach([product])
                    self._depth[product] = annotator
                    continue
                data = annotator.get_data()
                if found is None and data is not None and len(getattr(data, "shape", ())) == 2 and data.shape[0] >= 8:
                    found = data
            return found
        except Exception:
            log.exception("weather_fx: no scene depth; the cloud will not be drawn over objects")
            self._veil_failed = True
            self._drop_depth()
            return None

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
        depth = self._scene_depth(state, stage, str(prim.GetPath()))
        layer = self._renderer.render_layer(
            view, self._lighting, self._tables, gains=gains, flip=False,
            density_scale=float(state.clouds.density_scale),
            depth=depth, depth_scale=mpu, depth_max_m=0.8 * QUAD_DEPTH * float(clip[1]) * mpu,
            air_light=1.0 - AIR_SHADE * min(max(float(state.clouds.cover), 0.0), 1.0) ** 2)
        veil = self._renderer.veil

        import omni.ui as ui

        if self._provider is None:
            self._provider = ui.DynamicTextureProvider(TEXTURE_NAME)
        self._provider.set_bytes_data_from_gpu(layer.ptr, [width, height],
                                               format=ui.TextureFormat.RGBA32_SFLOAT)
        intensity = (EMISSIVE_PER_DOME * float(dome.get("exposure", 1.0))
                     * float(state.sky.exposure_scale))
        self._author(stage, world, focal, aperture, float(clip[1]), width / height, intensity)
        if veil is not None:
            if self._veil_providers is None:
                self._veil_providers = (ui.DynamicTextureProvider(VEIL_TEXTURE_NAME),
                                        ui.DynamicTextureProvider(VEIL_OPACITY_TEXTURE_NAME))
            for provider, array in zip(self._veil_providers, veil):
                provider.set_bytes_data_from_gpu(array.ptr, [width, height], format=ui.TextureFormat.RGBA32_SFLOAT)
        self._author_veil(stage, world, focal, aperture, max(VEIL_DEPTH_M / mpu, 2.0 * float(clip[0])),
                          width / height, intensity, veil is not None)
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

    def _author_veil(self, stage: Any, world: Any, focal: float, aperture: float, depth: float,
                     aspect: float, intensity: float, show: bool) -> None:
        """The veil: a quad just in front of the camera that emits the cloud lying in front of
        the scene's surfaces and is as opaque as that cloud, so the renderer itself blends it
        over them, the same in both render modes."""
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        with Usd.EditContext(stage, stage.GetSessionLayer()):
            prim = stage.GetPrimAtPath(VEIL_PATH)
            if not prim:
                if not show:
                    return
                quad = UsdGeom.Mesh.Define(stage, VEIL_PATH)
                quad.CreateFaceVertexCountsAttr([4])
                quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
                quad.CreateDoubleSidedAttr(True)
                UsdGeom.PrimvarsAPI(quad.GetPrim()).CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
                ).Set([(0, 0), (1, 0), (1, 1), (0, 1)])
                for name in ("primvars:doNotCastShadows", "primvars:invisibleToSecondaryRays"):
                    quad.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Bool).Set(True)
                material = UsdShade.Material.Define(stage, VEIL_MATERIAL_PATH)
                shader = UsdShade.Shader.Define(stage, VEIL_MATERIAL_PATH + "/Shader")
                shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
                shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
                shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
                shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set((0, 0, 0))
                shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.0)
                shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(1.0)
                shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
                shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set((1, 1, 1))
                shader.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(
                    Sdf.AssetPath(f"dynamic://{VEIL_TEXTURE_NAME}"))
                shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(float(intensity))
                shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
                shader.CreateInput("enable_opacity_texture", Sdf.ValueTypeNames.Bool).Set(True)
                shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(
                    Sdf.AssetPath(f"dynamic://{VEIL_OPACITY_TEXTURE_NAME}"))
                shader.CreateInput("opacity_mode", Sdf.ValueTypeNames.Int).Set(1)    # the mean of r, g, b
                shader.CreateInput("opacity_threshold", Sdf.ValueTypeNames.Float).Set(0.0)
                material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
                UsdShade.MaterialBindingAPI.Apply(quad.GetPrim()).Bind(material)
                UsdGeom.Xformable(quad).AddTransformOp()
                self._veil_key = None
                self._veil_visible = True
                prim = quad.GetPrim()
            quad = UsdGeom.Mesh(prim)
            if show != self._veil_visible:
                UsdGeom.Imageable(prim).GetVisibilityAttr().Set(
                    UsdGeom.Tokens.inherited if show else UsdGeom.Tokens.invisible)
                self._veil_visible = show
            if not show:
                return
            key = (round(focal, 4), round(aperture, 4), round(depth, 5), round(aspect, 5))
            if key != self._veil_key:
                half_w = depth * 0.5 * aperture / focal
                half_h = half_w / aspect
                quad.GetPointsAttr().Set([(-half_w, -half_h, -depth), (half_w, -half_h, -depth),
                                          (half_w, half_h, -depth), (-half_w, half_h, -depth)])
                quad.CreateExtentAttr([(-half_w, -half_h, -depth), (half_w, half_h, -depth)])
                self._veil_key = key
            UsdGeom.Xformable(quad).GetOrderedXformOps()[0].Set(Gf.Matrix4d(world))
            shader = UsdShade.Shader(stage.GetPrimAtPath(VEIL_MATERIAL_PATH + "/Shader"))
            current = shader.GetInput("emissive_intensity").Get()
            if current is None or abs(intensity - current) > 1e-6 * max(intensity, 1e-9):
                shader.GetInput("emissive_intensity").Set(float(intensity))

    def _remove(self) -> None:
        if not self._authored:
            return
        self._authored = False
        self._quad_key = None
        self._veil_key = None
        self._veil_visible = False
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
