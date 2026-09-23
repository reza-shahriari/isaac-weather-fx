"""Rain and snow as a PointInstancer in the session layer, following the camera.

The instancer is real geometry, so every RTX camera sees it (viewport,
Replicator render products, ROS camera publishers).
"""
from __future__ import annotations

import logging
import math

import numpy as np

from ...core.particles import ParticleField, rotation_between
from ...core.physics import (
    gust_factor,
    rain_number_density,
    raindrop_terminal_velocity,
    sample_drop_diameters,
    wind_vector,
)
from ...runtime.context import SESSION_ROOT
from ..base import Effect

log = logging.getLogger("weather_fx")

# Params that change the per-particle distribution (require resampling).
_RAIN_SAMPLER_KEYS = ("rate_mm_h", "drop_min_diameter_mm", "drop_max_diameter_mm")
_SNOW_SAMPLER_KEYS = ("flake_min_diameter_mm", "flake_max_diameter_mm", "fall_speed_mps", "fall_speed_jitter")
_MATERIAL_KEYS = ("color", "opacity", "material", "cast_shadows")


class PrecipitationEffect(Effect):
    def __init__(self, kind: str):
        if kind not in ("rain", "snow"):
            raise ValueError(kind)
        self.kind = kind
        self.name = kind
        self.path = f"{SESSION_ROOT}/{kind.capitalize()}"
        self._stage = None
        self._instancer = None
        self._proto = None
        self._shader = None
        self._material_key = None
        self._field = None
        self._params = None
        self._sampler_key = None
        self._seed = None
        self._active = False
        self._last_quat = None
        self._last_scale_key = None
        self._physical_density = 0.0

    # ------------------------------------------------------------------ helpers
    def _half_extents(self, p, mpu):
        half = np.full(3, p.volume_radius_m / mpu)
        half[self._field.up_axis] = 0.5 * p.volume_height_m / mpu
        return half

    def _volume_m3(self, p):
        return (2.0 * p.volume_radius_m) ** 2 * p.volume_height_m

    def _target_count(self, p):
        if self.kind == "rain":
            self._physical_density = rain_number_density(p.rate_mm_h, p.drop_min_diameter_mm, p.drop_max_diameter_mm)
            n = self._physical_density * self._volume_m3(p) * p.density_scale
        else:
            self._physical_density = p.number_density_m3
            n = self._physical_density * self._volume_m3(p)
        return int(min(p.max_particles, n))

    def _sampler(self, p):
        if self.kind == "rain":
            def sample(rng, n):
                d_mm = sample_drop_diameters(rng, n, p.rate_mm_h, p.drop_min_diameter_mm, p.drop_max_diameter_mm)
                return {"diameter_m": d_mm * 1e-3, "fall_speed_mps": raindrop_terminal_velocity(d_mm)}
        else:
            def sample(rng, n):
                d_mm = rng.uniform(p.flake_min_diameter_mm, p.flake_max_diameter_mm, n)
                jitter = 1.0 + p.fall_speed_jitter * (rng.random(n) * 2.0 - 1.0)
                return {"diameter_m": d_mm * 1e-3, "fall_speed_mps": np.maximum(p.fall_speed_mps * jitter, 0.05)}
        return sample

    # ------------------------------------------------------------------ USD
    def _prims_valid(self, stage):
        try:
            return (self._instancer is not None and stage == self._stage
                    and self._instancer.GetPrim().IsValid())
        except Exception:
            return False

    def _build_prims(self, stage, seed):
        from pxr import Gf, Usd, UsdGeom

        up = 2 if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z else 1
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            UsdGeom.Scope.Define(stage, SESSION_ROOT)
            inst = UsdGeom.PointInstancer.Define(stage, self.path)
            UsdGeom.Scope.Define(stage, f"{self.path}/Prototypes")
            if self.kind == "rain":
                geom = UsdGeom.Cylinder.Define(stage, f"{self.path}/Prototypes/Drop")
                geom.CreateRadiusAttr(0.5)
                geom.CreateHeightAttr(1.0)
                geom.CreateAxisAttr(UsdGeom.Tokens.z if up == 2 else UsdGeom.Tokens.y)
            else:
                geom = UsdGeom.Sphere.Define(stage, f"{self.path}/Prototypes/Flake")
                geom.CreateRadiusAttr(0.5)
            geom.CreateExtentAttr([Gf.Vec3f(-0.5), Gf.Vec3f(0.5)])
            inst.CreatePrototypesRel().SetTargets([geom.GetPath()])
        self._stage = stage
        self._instancer = inst
        self._proto = geom
        self._shader = None
        self._material_key = None
        self._field = ParticleField(up_axis=up, seed=seed)
        self._last_quat = None
        self._last_scale_key = None

    def _update_material(self, p):
        from pxr import Gf, Sdf, Usd, UsdShade

        stage = self._stage
        key = (p.material,)
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            if key != self._material_key:
                mat_path = f"{self.path}/Looks/{p.material}"
                mat = UsdShade.Material.Define(stage, mat_path)
                shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
                if p.material == "glass":
                    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
                    shader.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
                    shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
                    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
                    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
                    mat.CreateVolumeOutput("mdl").ConnectToSource(out)
                    mat.CreateDisplacementOutput("mdl").ConnectToSource(out)
                else:
                    shader.CreateIdAttr("UsdPreviewSurface")
                    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.15)
                    out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                    mat.CreateSurfaceOutput().ConnectToSource(out)
                UsdShade.MaterialBindingAPI.Apply(self._proto.GetPrim()).Bind(mat)
                self._shader = shader
                self._material_key = key
            color = Gf.Vec3f(*p.color)
            if p.material == "glass":
                self._shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(color)
                self._shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.333)
            else:
                self._shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
                self._shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(p.opacity))
            for prim in (self._instancer.GetPrim(), self._proto.GetPrim()):
                prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(not p.cast_shadows)

    def _set_visible(self, visible):
        if self._instancer is None or not self._prims_valid(self.context.stage()):
            return
        from pxr import Usd, UsdGeom

        stage = self._stage
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            img = UsdGeom.Imageable(self._instancer.GetPrim())
            img.MakeVisible() if visible else img.MakeInvisible()

    def _write(self, wind_mps, full):
        from pxr import Gf, Usd, Vt

        stage, inst, field, p = self._stage, self._instancer, self._field, self._params
        mpu = self.context.meters_per_unit()
        n = field.count
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            if full:
                inst.GetProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(np.zeros(n, dtype=np.int32)))
            inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(field.positions.astype(np.float32)))

            d_units = field.attrs["diameter_m"] / mpu
            if self.kind == "snow":
                if full:
                    scales = np.repeat(d_units[:, None], 3, axis=1)
                    inst.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(scales.astype(np.float32)))
                return

            # Rain: streak aligned with the mean velocity, length = speed * exposure.
            up = field.up_axis
            up_vec = np.zeros(3)
            up_vec[up] = 1.0
            fall = field.attrs["fall_speed_mps"]
            mean_vel = wind_mps - up_vec * (float(fall.mean()) if n else 7.0)
            quat = rotation_between(up_vec, -mean_vel)
            scale_key = (round(float(np.linalg.norm(wind_mps)), 2), p.streak_exposure_s, p.streak_width_scale)
            if full or scale_key != self._last_scale_key:
                speed = np.sqrt(fall ** 2 + float(np.dot(wind_mps, wind_mps)))
                scales = np.repeat((d_units * p.streak_width_scale)[:, None], 3, axis=1)
                scales[:, up] = np.maximum(speed * p.streak_exposure_s / mpu, d_units)
                inst.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(scales.astype(np.float32)))
                self._last_scale_key = scale_key
            if full or self._last_quat is None or _quat_angle(quat, self._last_quat) > math.radians(0.5):
                # Fabric rejects half quaternions ("Unsupported type during VtValue extraction"),
                # so use the float orientationsf attribute (USD 23.11+) when available.
                if hasattr(inst, "CreateOrientationsfAttr"):
                    q = Gf.Quatf(quat[0], Gf.Vec3f(quat[1], quat[2], quat[3]))
                    inst.CreateOrientationsfAttr().Set(Vt.QuatfArray([q] * n))
                else:
                    q = Gf.Quath(quat[0], Gf.Vec3h(quat[1], quat[2], quat[3]))
                    inst.GetOrientationsAttr().Set(Vt.QuathArray([q] * n))
                self._last_quat = quat

    # ------------------------------------------------------------------ lifecycle
    def apply_state(self, state, changed):
        p = getattr(state, self.kind)
        active = state.general.enabled and p.enabled
        if not active:
            self._active = False
            self._set_visible(False)
            return
        stage = self.context.stage()
        if stage is None:
            return
        if not self._prims_valid(stage):
            self._build_prims(stage, state.general.seed)
            self._params = None
            self._sampler_key = None
        if self._seed != state.general.seed:
            self._field.reseed(state.general.seed)
            self._seed = state.general.seed

        old = self._params
        self._params = p
        if old is None or any(getattr(old, k) != getattr(p, k) for k in _MATERIAL_KEYS):
            self._update_material(p)

        keys = _RAIN_SAMPLER_KEYS if self.kind == "rain" else _SNOW_SAMPLER_KEYS
        sampler_key = tuple(getattr(p, k) for k in keys)
        sampler = self._sampler(p)
        if self._sampler_key is not None and sampler_key != self._sampler_key:
            self._field.resample(sampler)
        self._sampler_key = sampler_key

        mpu = self.context.meters_per_unit()
        anchor = self.context.anchor()
        half = self._half_extents(p, mpu)
        self._field.resize(self._target_count(p), anchor, half, sampler)
        if old is not None and (old.volume_radius_m != p.volume_radius_m or old.volume_height_m != p.volume_height_m):
            self._field.recenter(anchor, half)
        self._field.wrap(anchor, half)

        self._active = True
        self._write(self._wind(state, self.context.time), full=True)
        self._set_visible(True)

    def _wind(self, state, t):
        w = state.wind
        v = wind_vector(w.speed_mps, w.direction_deg, w.vertical_mps, self._field.up_axis)
        return v * gust_factor(t, w.gust_strength, w.gust_period_s)

    def update(self, dt, t):
        if not self._active:
            return
        stage = self.context.stage()
        if stage is None:
            return
        state = self.context.state
        if not self._prims_valid(stage):  # new stage opened: rebuild
            self.apply_state(state, {self.kind})
            return
        p = self._params
        mpu = self.context.meters_per_unit()
        wind = self._wind(state, t)
        sway_amp = getattr(p, "sway_amplitude_m", 0.0)
        sway_freq = getattr(p, "sway_frequency_hz", 0.0)
        self._field.step(dt, self.context.anchor(), self._half_extents(p, mpu), wind, t,
                         velocity_scale=1.0 / mpu, sway_amplitude=sway_amp, sway_frequency=sway_freq)
        self._write(wind, full=False)

    def detach(self):
        self._active = False
        stage = self._stage
        if stage is not None:
            try:
                from pxr import Usd

                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    if stage.GetPrimAtPath(self.path):
                        stage.RemovePrim(self.path)
            except Exception:
                log.exception("weather_fx: could not remove %s", self.path)
        self._instancer = None
        self._stage = None

    def stats(self):
        if not self._active or self._field is None:
            return {"active": False}
        key = "physical_drops_per_m3" if self.kind == "rain" else "flakes_per_m3"
        return {"active": True, "particles": self._field.count, key: round(self._physical_density, 2)}


def _quat_angle(q1, q2):
    d = abs(sum(a * b for a, b in zip(q1, q2)))
    return 2.0 * math.acos(min(1.0, d))
