"""Render the per-pixel cloud layer (``core.cloudscape`` marched by ``gpu.cloud_march``) for a
camera on the ground, lit and composed by the sky model. No Isaac Sim; a CUDA GPU.

    python tools/render_cloudscape.py still --out captures/cloudscape --hour 14.5 --cover 0.35
    python tools/render_cloudscape.py pan --out captures/cloudscape/pan --frames 72 --mp4 pan.mp4
    python tools/render_cloudscape.py day --out captures/cloudscape/day --frames 48 --mp4 day.mp4

``still`` writes one frame per listed azimuth; ``pan`` turns the camera through a full circle at
one hour (the "rotate the camera" test); ``day`` holds the camera and runs the clock.

Composition per pixel: ``sky * T + air * C + (1 - T) * inscatter``, where ``(C, T)`` are the
march's premultiplied cloud light and transmittance, ``sky`` the clear sky in that direction,
and ``(inscatter, air)`` the atmosphere's aerial perspective to the cloud's mean distance. Tone:
auto-exposed to the frame (its 99.5th percentile at 0.9) through the ACES filmic curve and sRGB,
a stand-in for a camera, not the RTX tonemapper.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "exts" / "weather.fx"))

from weather_fx.core import cloudscape as cloudscape_mod  # noqa: E402
from weather_fx.core import sky as sky_mod  # noqa: E402
from weather_fx.core.clouds import lifting_condensation_level_m  # noqa: E402
from weather_fx.core.state import WeatherState  # noqa: E402
from weather_fx.gpu import cloud_march  # noqa: E402

TONE_GAIN = 0.7


def srgb(linear: np.ndarray) -> np.ndarray:
    v = np.clip(linear, 0.0, 1.0)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)


def aces(x: np.ndarray) -> np.ndarray:
    """Narkowicz's fit of the ACES filmic curve, on linear values with 1.0 near white."""
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def tone(radiance: np.ndarray, metered: np.ndarray) -> np.ndarray:
    """A camera's exposure, not the dome's: the 99th percentile luminance of the metered pixels
    (the frame outside 20 degrees of the sun) is placed at 0.9 before the curve, as an auto-exposed
    camera photographing clouds would, so a white cloud is white and the sky keeps its blue."""
    luma = radiance @ sky_mod.LUMA
    sample = luma[metered] if np.any(metered) else luma
    scale = 0.9 / max(float(np.percentile(sample, 99.0)), 1e-9)
    return (255.0 * srgb(aces(radiance * scale)) + 0.5).astype(np.uint8)


class Scene:
    """One sky and one cloud layer, with everything the composite needs cached per hour."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.state = WeatherState().with_updates(
            "sky", enabled=True, hour_utc=float(args.hour), turbidity=float(args.turbidity),
            latitude_deg=48.14, longitude_deg=11.58, date_utc=args.date,
        ).with_updates(
            "clouds", enabled=True, cover=float(args.cover), genus=args.genus,
            temperature_c=float(args.temperature), dewpoint_c=float(args.dewpoint),
        )
        self.base_m = lifting_condensation_level_m(float(args.temperature), float(args.dewpoint))
        t = time.time()
        self.cloudscape = cloudscape_mod.Cloudscape(
            cover=float(args.cover), base_m=float(self.base_m),
            profile=cloudscape_mod.CLOUDSCAPE_TYPES[args.genus], seed=int(args.seed))
        self.renderer = cloud_march.CloudRenderer(self.cloudscape, device=f"cuda:{args.gpu}")
        print(f"cloudscape built in {time.time() - t:.1f} s: base {self.base_m:.0f} m, "
              f"cover asked {args.cover}, measured {self.cloudscape.measured_cover():.3f}", flush=True)
        self._hour = None

    def at_hour(self, hour: float):
        if self._hour == hour:
            return
        self._hour = hour
        self.state = self.state.with_updates("sky", hour_utc=float(hour))
        self.conditions = sky_mod.conditions_from_state(self.state, build_cloud=False)
        c = self.conditions
        lit = c.sun if c.sun.elevation_deg > 0.0 else c.moon
        tint = sky_mod.sun_colour(lit.elevation_deg, c.turbidity)
        if lit is c.sun:
            source = 1.6e5 * c.daylight * tint
        else:
            source = 1.6e4 * c.moon_lux / max(math.sin(math.radians(max(lit.elevation_deg, 1.0))), 0.02) \
                * np.asarray(sky_mod.MOONLIGHT_TINT) * tint * 1e-5
        above, below = sky_mod.cloud_ambient_rgb(c)
        self.lighting = cloud_march.Lighting(
            tuple(float(v) for v in lit.direction()), tuple(float(v) for v in source),
            tuple(float(v) for v in above), tuple(float(v) for v in below))
        self.exposure = sky_mod.dome_exposure(sky_mod.environment_map(c, height=128), c)

    def frame(self, camera: cloud_march.Camera) -> tuple:
        c = self.conditions
        t = time.time()
        scattered, transmittance, distance = self.renderer.render(
            camera, self.lighting, step_min_m=float(self.args.step_m),
            step_growth=float(self.args.step_growth))
        gpu_s = time.time() - t
        directions = cloud_march.camera_rays(camera)
        sky = sky_mod.sky_radiance_rgb(directions, c) / max(float(c.exposure_scale), 1e-12)
        cloudy = transmittance < 0.999
        inscatter = np.zeros_like(sky)
        air = np.ones_like(sky)
        if np.any(cloudy) and c.model == "atmosphere":
            inscatter[cloudy], air[cloudy] = sky_mod.aerial_perspective_rgb(
                directions[cloudy], distance[cloudy], c, observer_height_m=float(camera.position_m[1]))
        out = (sky * transmittance[..., None] + air * scattered
               + (1.0 - transmittance)[..., None] * inscatter)
        sun = np.asarray(self.lighting.sun_direction)
        metered = directions @ sun < math.cos(math.radians(20.0))
        return tone(out, metered), dict(gpu_s=round(gpu_s, 3), total_s=round(time.time() - t, 2),
                                              cloud_fraction=round(float((transmittance < 0.5).mean()), 3))


def encode(frames_dir: pathlib.Path, out: pathlib.Path, fps: float) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", str(frames_dir / "frame_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "18", str(out)]
    subprocess.run(cmd, check=True)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("still", "pan", "day"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hfov", type=float, default=70.0)
    ap.add_argument("--elevation", type=float, default=22.0)
    ap.add_argument("--azimuths", type=float, nargs="*", default=[0.0, 90.0, 180.0, 270.0])
    ap.add_argument("--hour", type=float, default=14.5)
    ap.add_argument("--start-hour", type=float, default=6.0)
    ap.add_argument("--end-hour", type=float, default=20.5)
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--mp4", default=None)
    ap.add_argument("--date", default="2024-06-21")
    ap.add_argument("--turbidity", type=float, default=2.8)
    ap.add_argument("--cover", type=float, default=0.35)
    ap.add_argument("--genus", default="cumulus", choices=sorted(cloudscape_mod.CLOUDSCAPE_TYPES))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=22.0)
    ap.add_argument("--dewpoint", type=float, default=12.0)
    ap.add_argument("--step-m", type=float, default=24.0)
    ap.add_argument("--step-growth", type=float, default=0.012)
    args = ap.parse_args()

    from PIL import Image

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = Scene(args)
    log = []

    def shoot(index: int, hour: float, azimuth: float, name: str) -> None:
        scene.at_hour(hour)
        cam = cloud_march.Camera(args.width, args.height, args.hfov, args.elevation, azimuth)
        image, info = scene.frame(cam)
        Image.fromarray(image).save(out / name)
        info.update(index=index, hour_utc=round(hour, 3), azimuth_deg=round(azimuth, 2),
                    sun_elevation_deg=round(float(scene.conditions.sun.elevation_deg), 2),
                    sun_azimuth_deg=round(float(scene.conditions.sun.azimuth_deg), 2), file=name)
        log.append(info)
        print(json.dumps(info), flush=True)

    if args.mode == "still":
        for k, az in enumerate(args.azimuths):
            shoot(k, args.hour, az, f"still_az{int(round(az)):03d}.png")
    elif args.mode == "pan":
        for k in range(args.frames):
            shoot(k, args.hour, 360.0 * k / args.frames, f"frame_{k:04d}.png")
    else:
        for k in range(args.frames):
            hour = args.start_hour + (args.end_hour - args.start_hour) * k / max(args.frames - 1, 1)
            shoot(k, hour, args.azimuths[0], f"frame_{k:04d}.png")
    json.dump(dict(args=vars(args), base_m=scene.base_m, frames=log), open(out / "summary.json", "w"), indent=1)
    if args.mp4 and args.mode != "still":
        encode(out, out / args.mp4, args.fps)


if __name__ == "__main__":
    main()
