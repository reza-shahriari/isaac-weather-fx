"""The cloud layer marched per camera pixel on the GPU.

Every pixel's ray crosses the cloud shell between the layer's base and top (a shell round the
planet, so the layer curves down at the horizon), and at each step evaluates the same density
function :mod:`weather_fx.core.cloudscape` defines, from the same three textures. Where there is
cloud, the step scatters:

* **sunlight**, scattered once (the phase function times the beam that reached the point) and
  many times (the delta-Eddington two-stream field along the sun's chord through the point, as
  the dome's march since ADR 0180: ``R (1 - f)`` of the beam at a lit face, ``T f`` at a dark one),
  the optical depth toward the sun and away from it each read along six growing steps;
* **sky light and ground light**, the clear sky's hemisphere means from above and from below,
  mixed by the two-stream diffuse transmittance of the cloud above and below the point (the
  weights sum to one, so a white cloud under a uniform sky vanishes: the furnace).

Steps integrate energy-conservingly (``S (1 - e^{-sigma ds})``) and the result is premultiplied:
``(scattered, transmittance, mean distance)`` per pixel, for the caller to compose over the
clear sky through the air in front of the cloud. Units are the sky model's (cd/m2, before the
dome's exposure).

The kernel is written once and holds for both bands: an infrared caller replaces the two lights
with the step's thermal emission and keeps the march.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from . import ensure_warp

wp = ensure_warp()

__all__ = ["Camera", "Lighting", "CloudRenderer", "camera_rays", "SkyTables"]

#: Mean radius of the Earth, metres: the cloud shell's curvature.
PLANET_RADIUS_M = 6_371_000.0
#: Henyey-Greenstein asymmetry of cloud droplets in the visible, and the back lobe's share.
CLOUD_G = 0.85
BACK_LOBE = 0.15
#: Single-scattering albedo of the droplets in the visible.
ALBEDO = 0.9999
#: Step inside and just past a cloud, as a fraction of the clear-air step.
FINE_STEP = 0.3
#: Longest step inside cloud, in mean free paths of the density at the sample: a crisp cloud top
#: stops light within ten metres, and a step that long would put the whole pixel's light at one
#: random depth under the surface (grain). Nor is a step shorter than a pixel's footprint.
SKIN_STEP = 0.8
#: Two-stream diffuse transmittance through optical depth tau: ``1 / (1 + 0.75 (1 - g) tau)``.
DIFFUSE_K = 0.75 * (1.0 - CLOUD_G)


@dataclass
class Camera:
    """A pinhole camera in the field's frame: metres, +Y up, azimuth 0 toward -Z, 90 toward +X."""

    width: int
    height: int
    hfov_deg: float
    elevation_deg: float
    azimuth_deg: float
    position_m: Tuple[float, float, float] = (0.0, 2.0, 0.0)
    roll_deg: float = 0.0
    #: ``(forward, right, up)`` unit vectors that replace the elevation and azimuth: a camera
    #: posed by a renderer's own matrix.
    pose: Optional[Tuple[Any, Any, Any]] = None

    def basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.pose is not None:
            return tuple(np.asarray(v, dtype=np.float64) for v in self.pose)  # type: ignore[return-value]
        el, az = math.radians(self.elevation_deg), math.radians(self.azimuth_deg)
        forward = np.array([math.cos(el) * math.sin(az), math.sin(el), -math.cos(el) * math.cos(az)])
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= max(np.linalg.norm(right), 1e-9)
        up = np.cross(right, forward)
        if self.roll_deg:
            r = math.radians(self.roll_deg)
            right, up = right * math.cos(r) + up * math.sin(r), up * math.cos(r) - right * math.sin(r)
        return forward, right, up


def camera_rays(camera: Camera) -> np.ndarray:
    """Unit directions of every pixel, ``(height, width, 3)``, pixel centres, row 0 at the top."""
    forward, right, up = camera.basis()
    focal = 0.5 * camera.width / math.tan(math.radians(camera.hfov_deg) / 2.0)
    xs = (np.arange(camera.width) + 0.5 - camera.width / 2.0) / focal
    ys = (camera.height / 2.0 - np.arange(camera.height) - 0.5) / focal
    d = forward[None, None, :] + xs[None, :, None] * right[None, None, :] + ys[:, None, None] * up[None, None, :]
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


@dataclass
class Lighting:
    """What lights the cloud, in the sky model's units before exposure."""

    sun_direction: Tuple[float, float, float]
    #: The direct beam's illuminance normal to itself, per channel (the sky model's 1.6e5 x
    #: daylight x the sun's colour).
    sun_rgb: Tuple[float, float, float]
    #: Hemisphere-mean clear radiance from above and from below (``sky.cloud_ambient_rgb``).
    above_rgb: Tuple[float, float, float]
    below_rgb: Tuple[float, float, float]


@wp.struct
class Layer:
    base_m: float
    thickness_m: float
    weather_tile_m: float
    shape_tile_m: float
    detail_tile_m: float
    coverage_bias: float
    top_min: float
    top_max: float
    taper: float
    erosion: float
    edge_base: float
    edge_top: float
    crisp_from: float
    crisp_to: float
    water_base: float
    extinction_per_m: float


@wp.struct
class March:
    origin: wp.vec3
    forward: wp.vec3
    right: wp.vec3
    up: wp.vec3
    focal_px: float
    width: int
    height: int
    sun_dir: wp.vec3
    sun_mu: float
    sun_rgb: wp.vec3
    above_rgb: wp.vec3
    below_rgb: wp.vec3
    max_distance_m: float
    step_min_m: float
    step_growth: float
    max_steps: int


@wp.func
def smoothstep(lo: float, hi: float, v: float) -> float:
    t = wp.clamp((v - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@wp.func
def remap01(v: float, lo: float, hi: float) -> float:
    return wp.clamp((v - lo) / wp.max(hi - lo, 1.0e-4), 0.0, 1.0)


@wp.func
def cloud_density(
    x: float, y: float, z: float, L: Layer,
    weather: wp.Texture2D, shape: wp.Texture3D, detail: wp.Texture3D,
) -> float:
    """``Cloudscape._density``, line for line."""
    h = (y - L.base_m) / L.thickness_m
    if h <= 0.0 or h >= 1.0:
        return 0.0
    w = wp.texture_sample(weather, wp.vec2f(x / L.weather_tile_m, z / L.weather_tile_m), dtype=wp.vec2f)
    coverage = wp.clamp(w[0] + L.coverage_bias, 0.0, 1.0)
    if coverage <= 0.0:
        return 0.0
    top = L.top_min + (L.top_max - L.top_min) * w[1]
    hr = h / top
    gradient = smoothstep(0.0, 0.07, hr) * (1.0 - smoothstep(0.35, 1.0, hr))
    if gradient <= 0.0:
        return 0.0
    coverage = coverage * (1.0 - L.taper * wp.pow(wp.clamp(hr, 0.0, 1.0), 1.5))
    s = 1.0 / L.shape_tile_m
    shp = wp.texture_sample(shape, wp.vec3f(x * s, y * s, z * s), dtype=float)
    base = remap01(shp * gradient, 1.0 - coverage, 1.0)
    if base <= 0.0:
        return 0.0
    d = 1.0 / L.detail_tile_m
    det = wp.texture_sample(detail, wp.vec3f(x * d, y * d, z * d), dtype=float)
    blend = wp.clamp(hr * 5.0, 0.0, 1.0)
    modifier = det * (1.0 - blend) + (1.0 - det) * blend
    eroded = remap01(base, modifier * L.erosion, 1.0)
    sharp = L.edge_base + (L.edge_top - L.edge_base) * smoothstep(L.crisp_from, L.crisp_to, hr)
    edge = 1.0 - wp.exp(-sharp * eroded)
    water = L.water_base + (1.0 - L.water_base) * wp.pow(wp.clamp(h, 0.0, 1.0), 2.0 / 3.0)
    return edge * water


@wp.func
def altitude_of(p: wp.vec3) -> float:
    """Height above the planet's surface of a point in the observer's tangent frame."""
    return wp.length(wp.vec3(p[0], p[1] + PLANET_RADIUS_M, p[2])) - PLANET_RADIUS_M


@wp.func
def sphere_far(o: wp.vec3, d: wp.vec3, radius: float) -> float:
    """Distance to the far intersection with the sphere of ``radius`` about the planet's centre
    (the observer's tangent frame), or -1 if the ray misses it."""
    oc = wp.vec3(o[0], o[1] + PLANET_RADIUS_M, o[2])
    b = wp.dot(oc, d)
    c = wp.dot(oc, oc) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return -1.0
    return -b + wp.sqrt(disc)


@wp.func
def sphere_near(o: wp.vec3, d: wp.vec3, radius: float) -> float:
    oc = wp.vec3(o[0], o[1] + PLANET_RADIUS_M, o[2])
    b = wp.dot(oc, d)
    c = wp.dot(oc, oc) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return -1.0
    return -b - wp.sqrt(disc)


@wp.func
def henyey_greenstein(cos_theta: float, g: float) -> float:
    denom = 1.0 + g * g - 2.0 * g * cos_theta
    return (1.0 - g * g) / (4.0 * 3.14159265 * wp.pow(denom, 1.5))


@wp.func
def phase(cos_theta: float, g: float) -> float:
    return (1.0 - BACK_LOBE) * henyey_greenstein(cos_theta, g) + BACK_LOBE * henyey_greenstein(cos_theta, -0.3 * g)


@wp.func
def sun_optical_depth(
    p: wp.vec3, sun: wp.vec3, L: Layer,
    weather: wp.Texture2D, shape: wp.Texture3D, detail: wp.Texture3D,
) -> float:
    """Optical depth toward the sun over eight steps that double in length (8 m to 1 km), so a
    lobe's own bumps shade its flank and the cloud behind shades its base."""
    tau = float(0.0)
    t = float(0.0)
    step = float(8.0)
    for _ in range(8):
        q = p + sun * (t + 0.5 * step)
        tau += cloud_density(q[0], altitude_of(q), q[2], L, weather, shape, detail) * step
        t += step
        step *= 2.0
    return tau * L.extinction_per_m


@wp.func
def column_optical_depth(
    p: wp.vec3, alt: float, span_m: float, L: Layer,
    weather: wp.Texture2D, shape: wp.Texture3D, detail: wp.Texture3D,
) -> float:
    """Optical depth straight up (``span_m`` > 0) or down from ``p`` to the layer's edge, from
    three samples at the sixth, the half and the five sixths of the span."""
    tau = float(0.0)
    for k in range(3):
        f = (float(k) + 0.5) / 3.0
        tau += cloud_density(p[0], alt + f * span_m, p[2], L, weather, shape, detail)
    return tau * wp.abs(span_m) / 3.0 * L.extinction_per_m


@wp.func
def two_stream(tau_sun: float, tau_away: float) -> float:
    """The multiply-scattered sunlight at a point, as a fraction of a white ground's radiance in
    the same sun: the delta-Eddington two-stream field of the chord through the point, read as
    the back stream ``R (1 - f)`` plus the forward stream ``T f`` (docs/physics-model.md section
    7.5, ADR 0180), with ``f`` the point's depth along the chord."""
    tau_c = tau_sun + tau_away
    if tau_c <= 1.0e-6:
        return 0.0
    tau_s = (1.0 - CLOUD_G) * tau_c
    r = tau_s / (2.0 + tau_s)
    t = 2.0 / (2.0 + tau_s) - wp.exp(-tau_c)
    f = tau_sun / tau_c
    return wp.max(r * (1.0 - f) + t * f, 0.0)


@wp.func
def depth_falloff(tau_sun: float) -> float:
    """How the multiply-scattered light falls off with optical depth from the lit surface: the
    mean of Wrenninge's attenuation octaves (``exp(-b^i tau)``, b = 1/2, six octaves), 1 at the
    surface and about 0.4 six optical depths in. The two-stream field gives the slab's mean; this
    is what makes a bump's shadowed flank darker than its lit crown."""
    total = float(0.0)
    b = float(1.0)
    for _ in range(6):
        total += wp.exp(-b * tau_sun)
        b *= 0.5
    return total / 6.0


@wp.func
def diffuse_transmittance(tau: float) -> float:
    return 1.0 / (1.0 + DIFFUSE_K * tau)


@wp.func
def hash01(i: int, j: int) -> float:
    n = wp.uint32(i) * wp.uint32(1973) + wp.uint32(j) * wp.uint32(9277) + wp.uint32(26699)
    n = (n << wp.uint32(13)) ^ n
    n = n * (n * n * wp.uint32(15731) + wp.uint32(789221)) + wp.uint32(1376312589)
    return float(n & wp.uint32(0x7FFFFFFF)) / 2147483648.0


@wp.kernel
def march_clouds(
    M: March, L: Layer,
    weather: wp.Texture2D, shape: wp.Texture3D, detail: wp.Texture3D,
    out_rgb: wp.array2d(dtype=wp.vec3), out_t: wp.array2d(dtype=float), out_dist: wp.array2d(dtype=float),
):
    j, i = wp.tid()
    px = (float(i) + 0.5 - 0.5 * float(M.width)) / M.focal_px
    py = (0.5 * float(M.height) - float(j) - 0.5) / M.focal_px
    d = wp.normalize(M.forward + M.right * px + M.up * py)
    o = M.origin

    scattered = wp.vec3(0.0, 0.0, 0.0)
    transmittance = float(1.0)
    weighted_t = float(0.0)
    weight = float(0.0)

    inner = PLANET_RADIUS_M + L.base_m
    outer = PLANET_RADIUS_M + L.thickness_m + L.base_m
    alt0 = altitude_of(o)
    t_start = float(0.0)
    t_end = float(-1.0)
    if alt0 < L.base_m:
        # Below the layer: the ray must clear the ground, then enters at the inner shell.
        if sphere_near(o, d, PLANET_RADIUS_M) <= 0.0:
            t_start = sphere_far(o, d, inner)
            t_end = sphere_far(o, d, outer)
    elif alt0 < L.base_m + L.thickness_m:
        t_start = 0.0
        t_in = sphere_near(o, d, inner)
        t_end = sphere_far(o, d, outer)
        if t_in > 0.0:
            t_end = t_in
    else:
        t_start = sphere_near(o, d, outer)
        if t_start > 0.0:
            t_end = sphere_far(o, d, outer)
            t_in = sphere_near(o, d, inner)
            if t_in > 0.0:
                t_end = t_in
    t_end = wp.min(t_end, M.max_distance_m)

    if t_end > t_start and t_start >= 0.0:
        step = wp.max(M.step_min_m, M.step_growth * t_start)
        t = t_start + hash01(i, j) * step
        cos_sun = wp.dot(d, M.sun_dir)
        count = int(0)
        # Metres marched since the last sample that held cloud; "far" before the first one.
        clear_m = float(1.0e9)
        while t < t_end and transmittance > 0.004 and count < M.max_steps:
            coarse = wp.max(M.step_min_m, M.step_growth * t)
            # Coarse steps through clear air. Within one coarse step of a cloud the step is
            # short enough to find a crisp surface: half a mean free path of the densest cloud,
            # but no shorter than a pixel's footprint.
            footprint = wp.max(t / M.focal_px, 1.0)
            fine = wp.min(coarse, wp.max(wp.min(coarse * FINE_STEP, SKIN_STEP / L.extinction_per_m), footprint))
            near = clear_m < coarse
            step = coarse
            if near:
                step = fine
            p = o + d * t
            alt = altitude_of(p)
            dens = cloud_density(p[0], alt, p[2], L, weather, shape, detail)
            if dens > 0.002:
                if not near:
                    # First contact at a coarse step: back up and resample finely from there.
                    t -= coarse - fine
                    clear_m = 0.0
                    continue
                clear_m = 0.0
                sigma = dens * L.extinction_per_m
                step = wp.min(coarse, wp.max(wp.min(coarse * FINE_STEP, SKIN_STEP / sigma), footprint))
                tau_sun = sun_optical_depth(p, M.sun_dir, L, weather, shape, detail)
                tau_away = sun_optical_depth(p, -M.sun_dir, L, weather, shape, detail)
                sun = (phase(cos_sun, CLOUD_G) * wp.exp(-tau_sun)
                       + M.sun_mu / 3.14159265 * two_stream(tau_sun, tau_away) * depth_falloff(tau_sun))
                # Sky and ground light through the cloud above and below this point.
                h = (alt - L.base_m) / L.thickness_m
                tau_up = column_optical_depth(p, alt, (1.0 - h) * L.thickness_m, L, weather, shape, detail)
                tau_dn = column_optical_depth(p, alt, -h * L.thickness_m, L, weather, shape, detail)
                # The diffuse field inside a conservative cloud is a mix of what comes in from
                # above and from below, weighted by how much of each reaches the point; the
                # weights sum to one, so under a uniform sky the cloud returns exactly the sky.
                w_up = diffuse_transmittance(tau_up)
                w_dn = diffuse_transmittance(tau_dn)
                ambient = (M.above_rgb * w_up + M.below_rgb * w_dn) / (w_up + w_dn)
                source = (M.sun_rgb * sun + ambient) * ALBEDO
                absorbed = 1.0 - wp.exp(-sigma * step)
                scattered += source * (transmittance * absorbed)
                weighted_t += transmittance * absorbed * t
                weight += transmittance * absorbed
                transmittance *= 1.0 - absorbed
            else:
                clear_m += step
            t += step
            count += 1

    out_rgb[j, i] = scattered
    out_t[j, i] = transmittance
    if weight > 0.0:
        out_dist[j, i] = weighted_t / weight
    else:
        out_dist[j, i] = 0.0


@wp.struct
class Compose:
    light_azimuth: float
    air_near_m: float
    air_log_span: float
    gains: wp.vec3
    scale: float
    flip: int


@wp.func
def row_of_elevation(elevation: float) -> float:
    """``core.layer_tables.row_of_elevation``: rows packed toward the horizon."""
    e = elevation / 1.5707963
    return 0.5 + 0.5 * wp.sign(e) * wp.sqrt(wp.abs(e))


@wp.kernel
def compose_layer(
    M: March, K: Compose,
    scattered: wp.array2d(dtype=wp.vec3), trans: wp.array2d(dtype=float), dist: wp.array2d(dtype=float),
    sky: wp.Texture2D, air_in: wp.Texture3D, air_tr: wp.Texture3D,
    out: wp.array2d(dtype=wp.vec4),
):
    """``sky T + air C + (1 - T) inscatter`` per pixel, white-balanced and scaled, as RGBA."""
    j, i = wp.tid()
    px = (float(i) + 0.5 - 0.5 * float(M.width)) / M.focal_px
    py = (0.5 * float(M.height) - float(j) - 0.5) / M.focal_px
    d = wp.normalize(M.forward + M.right * px + M.up * py)
    elevation = wp.asin(wp.clamp(d[1], -1.0, 1.0))
    azimuth = wp.atan2(d[0], -d[2])
    v = row_of_elevation(elevation)
    c4 = wp.texture_sample(sky, wp.vec2f(azimuth / 6.2831853, v), dtype=wp.vec4f)
    colour = wp.vec3(c4[0], c4[1], c4[2])
    t = trans[j, i]
    if t < 0.999:
        rel = azimuth - K.light_azimuth
        rel = wp.abs(rel - 6.2831853 * wp.floor((rel + 3.14159265) / 6.2831853))
        w = wp.log(wp.max(dist[j, i], K.air_near_m) / K.air_near_m) / K.air_log_span
        uvw = wp.vec3f(rel / 3.14159265, v, wp.clamp(w, 0.0, 1.0))
        ins4 = wp.texture_sample(air_in, uvw, dtype=wp.vec4f)
        tr4 = wp.texture_sample(air_tr, uvw, dtype=wp.vec4f)
        c = scattered[j, i]
        colour = (colour * t + wp.vec3(c[0] * tr4[0], c[1] * tr4[1], c[2] * tr4[2])
                  + wp.vec3(ins4[0], ins4[1], ins4[2]) * (1.0 - t))
    row = j
    if K.flip != 0:
        row = M.height - 1 - j
    out[row, i] = wp.vec4(colour[0] * K.gains[0] * K.scale, colour[1] * K.gains[1] * K.scale,
                          colour[2] * K.gains[2] * K.scale, 1.0)


class SkyTables:
    """The clear sky and the air in front of the cloud, on the GPU (``core.layer_tables``)."""

    def __init__(self, sky: np.ndarray, air_in: np.ndarray, air_tr: np.ndarray, light_azimuth_rad: float,
                 air_near_m: float, air_far_m: float, device: str) -> None:
        clamp = wp.TextureAddressMode.CLAMP
        wrap = wp.TextureAddressMode.WRAP
        linear = wp.TextureFilterMode.LINEAR
        with wp.ScopedDevice(device):
            self.sky = wp.Texture2D(np.ascontiguousarray(sky, dtype=np.float32), filter_mode=linear,
                                    address_mode_u=wrap, address_mode_v=clamp)
            self.air_in = wp.Texture3D(np.ascontiguousarray(air_in, dtype=np.float32),
                                       filter_mode=linear, address_mode=clamp)
            self.air_tr = wp.Texture3D(np.ascontiguousarray(air_tr, dtype=np.float32),
                                       filter_mode=linear, address_mode=clamp)
        self.light_azimuth_rad = float(light_azimuth_rad)
        self.air_near_m = float(air_near_m)
        self.air_log_span = math.log(float(air_far_m) / float(air_near_m))


class CloudRenderer:
    """Holds a cloudscape's textures on one GPU and marches cameras through it."""

    def __init__(self, cloudscape: Any, device: Optional[str] = None) -> None:
        wp.config.log_level = getattr(wp, "LOG_WARNING", 2)
        wp.init()
        self.device = device or ("cuda:0" if wp.get_cuda_device_count() else "cpu")
        self.cloudscape = cloudscape
        wrap = wp.TextureAddressMode.WRAP
        linear = wp.TextureFilterMode.LINEAR
        with wp.ScopedDevice(self.device):
            self.weather = wp.Texture2D(np.ascontiguousarray(cloudscape.weather, dtype=np.float32),
                                        filter_mode=linear, address_mode=wrap)
            self.shape = wp.Texture3D(np.ascontiguousarray(cloudscape.shape, dtype=np.float32),
                                      filter_mode=linear, address_mode=wrap)
            self.detail = wp.Texture3D(np.ascontiguousarray(cloudscape.detail, dtype=np.float32),
                                       filter_mode=linear, address_mode=wrap)
        self.layer = Layer()
        for key, value in cloudscape.kernel_constants().items():
            setattr(self.layer, key, value)
        self._buffers = None
        self._layer = None

    def render(
        self, camera: Camera, lighting: Lighting, *, max_distance_m: float = 80_000.0,
        step_min_m: float = 24.0, step_growth: float = 0.012, max_steps: int = 768,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(scattered rgb, transmittance, mean distance)``, each ``(height, width[, 3])``."""
        rgb, trans, dist, _ = self._march(camera, lighting, max_distance_m, step_min_m, step_growth, max_steps)
        return rgb.numpy().astype(np.float64), trans.numpy().astype(np.float64), dist.numpy().astype(np.float64)

    def render_layer(
        self, camera: Camera, lighting: Lighting, tables: SkyTables, *,
        gains: Tuple[float, float, float] = (1.0, 1.0, 1.0), scale: float = 1.0, flip: bool = False,
        density_scale: float = 1.0, max_distance_m: float = 80_000.0, step_min_m: float = 24.0,
        step_growth: float = 0.012, max_steps: int = 768,
    ):
        """The finished layer for one camera, left on the GPU: a ``(height, width)`` ``vec4``
        Warp array of the cloud composed over the clear sky through the air, times ``gains`` and
        ``scale``. The array is reused between calls of one size; hand it to a texture before
        the next call."""
        rgb, trans, dist, m = self._march(camera, lighting, max_distance_m, step_min_m, step_growth,
                                          max_steps, density_scale)
        k = Compose()
        k.light_azimuth = tables.light_azimuth_rad
        k.air_near_m, k.air_log_span = tables.air_near_m, tables.air_log_span
        k.gains = wp.vec3(*[float(g) for g in gains])
        k.scale = float(scale)
        k.flip = 1 if flip else 0
        shape = (camera.height, camera.width)
        with wp.ScopedDevice(self.device):
            if self._layer is None or self._layer.shape != shape:
                self._layer = wp.zeros(shape, dtype=wp.vec4)
            wp.launch(compose_layer, dim=shape,
                      inputs=[m, k, rgb, trans, dist, tables.sky, tables.air_in, tables.air_tr, self._layer])
            wp.synchronize_device(self.device)
        return self._layer

    def _march(self, camera: Camera, lighting: Lighting, max_distance_m: float, step_min_m: float,
               step_growth: float, max_steps: int, density_scale: float = 1.0):
        forward, right, up = camera.basis()
        m = March()
        m.origin = wp.vec3(*[float(v) for v in camera.position_m])
        m.forward, m.right, m.up = wp.vec3(*forward), wp.vec3(*right), wp.vec3(*up)
        m.focal_px = 0.5 * camera.width / math.tan(math.radians(camera.hfov_deg) / 2.0)
        m.width, m.height = int(camera.width), int(camera.height)
        m.sun_dir = wp.vec3(*[float(v) for v in lighting.sun_direction])
        m.sun_mu = max(float(lighting.sun_direction[1]), 0.0)
        m.sun_rgb = wp.vec3(*[float(v) for v in lighting.sun_rgb])
        m.above_rgb = wp.vec3(*[float(v) for v in lighting.above_rgb])
        m.below_rgb = wp.vec3(*[float(v) for v in lighting.below_rgb])
        m.max_distance_m = float(max_distance_m)
        m.step_min_m, m.step_growth, m.max_steps = float(step_min_m), float(step_growth), int(max_steps)
        shape = (camera.height, camera.width)
        layer = self.layer
        if density_scale != 1.0:
            layer = Layer()
            for key, value in self.cloudscape.kernel_constants().items():
                setattr(layer, key, value)
            layer.extinction_per_m = float(self.cloudscape.extinction_per_m * density_scale)
        with wp.ScopedDevice(self.device):
            if self._buffers is None or self._buffers[0].shape != shape:
                self._buffers = (wp.zeros(shape, dtype=wp.vec3), wp.zeros(shape, dtype=float),
                                 wp.zeros(shape, dtype=float))
            rgb, trans, dist = self._buffers
            wp.launch(march_clouds, dim=shape,
                      inputs=[m, layer, self.weather, self.shape, self.detail, rgb, trans, dist])
            wp.synchronize_device(self.device)
        return rgb, trans, dist, m

    def density(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """The kernel's density at given points: the test that it is the numpy function."""
        x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
        n = x.size
        with wp.ScopedDevice(self.device):
            px = wp.array(x.reshape(-1).astype(np.float32), dtype=float)
            py = wp.array(y.reshape(-1).astype(np.float32), dtype=float)
            pz = wp.array(z.reshape(-1).astype(np.float32), dtype=float)
            out = wp.zeros(n, dtype=float)
            wp.launch(_density_probe, dim=n, inputs=[px, py, pz, self.layer, self.weather, self.shape, self.detail, out])
            wp.synchronize_device(self.device)
            return out.numpy().reshape(x.shape).astype(np.float64)


@wp.kernel
def _density_probe(
    xs: wp.array(dtype=float), ys: wp.array(dtype=float), zs: wp.array(dtype=float), L: Layer,
    weather: wp.Texture2D, shape: wp.Texture3D, detail: wp.Texture3D, out: wp.array(dtype=float),
):
    i = wp.tid()
    out[i] = cloud_density(xs[i], ys[i], zs[i], L, weather, shape, detail)
