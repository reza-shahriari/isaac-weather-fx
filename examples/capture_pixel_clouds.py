"""The per-pixel cloud layer inside Isaac Sim: turn the camera, change the hour, time the frames.

    ./python.sh examples/capture_pixel_clouds.py                       # stills, both render modes
    ./python.sh examples/capture_pixel_clouds.py --pan 90 --day 60    # plus the two clips

Everything here is rendered by Isaac Sim's own renderer through the extension
(``clouds.render_path = "pixel"``): the cloud is marched on the GPU for the viewport's camera each
frame and drawn on a quad at the far end of its frustum (``backends/viewport/clouds_pixel.py``).
The script writes, under ``--out``:

* ``heading_<mode>_<azimuth>.png``: one sky seen north, east, south and west, in RTX Real-Time and
  in the path tracer -- the "turn the camera" test;
* ``hour_<mode>_<hh>.png``: one heading at several hours;
* ``pan/`` and ``day/`` frames and their mp4 (with ``--pan`` and ``--day``), in Real-Time;
* ``report.json``: milliseconds per frame for the cloud layer and for the whole app update.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "exts", "weather.fx"))

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--out", default=os.path.join(REPO, "captures", "pixel_clouds"))
parser.add_argument("--gpu", default="0")
parser.add_argument("--resolution", type=int, nargs=2, default=(1280, 720))
parser.add_argument("--genus", default="cumulus")
parser.add_argument("--cover", type=float, default=0.35)
parser.add_argument("--hour", type=float, default=14.5)
parser.add_argument("--tilt", type=float, default=22.0)
parser.add_argument("--pan", type=int, default=0, help="frames of a full turn (0: skip)")
parser.add_argument("--day", type=int, default=0, help="frames of a day at one heading (0: skip)")
parser.add_argument("--pt-subframes", type=int, default=64)
parser.add_argument("--skip-path-traced", action="store_true")
args, _ = parser.parse_known_args()

# Kit picks a Vulkan device by PCI order; make CUDA's device 0 the same card, since the cloud
# layer's texture has to be handed over on the renderer's GPU.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True, "active_gpu": int(args.gpu), "multi_gpu": False})

import carb.settings  # noqa: E402
import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf  # noqa: E402

extensions = omni.kit.app.get_app().get_extension_manager()
extensions.add_path(os.path.join(REPO, "exts"))
extensions.set_extension_enabled_immediate("omni.warp.core", True)
extensions.set_extension_enabled_immediate("weather.fx", True)
extensions.set_extension_enabled_immediate("omni.replicator.core", True)

import omni.replicator.core as rep  # noqa: E402

from weather_fx import api as weather  # noqa: E402

sys.path.insert(0, HERE)
from gallery_scene import EYE_M, SKY_CAMERA, _direction, _look_at, build_scene  # noqa: E402

settings = carb.settings.get_settings()
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
build_scene(stage)
try:
    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    viewport.camera_path = SKY_CAMERA
    viewport.resolution = tuple(args.resolution)
except Exception as exc:  # the layer falls back to general.follow_prim
    print(f"[pixel] no viewport to aim ({exc!r})")

product = rep.create.render_product(SKY_CAMERA, tuple(args.resolution))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(product)

wx = weather.get_controller()
wx.clear()
wx.set_general(time_source="manual", seed=42)
wx.follow(SKY_CAMERA)
wx.set_site(48.14, 11.58)
wx.set_time(date_utc="2024-06-21")
wx.configure(sky={"enabled": True, "hour_utc": args.hour},
             clouds={"enabled": True, "render_path": "pixel", "cover": args.cover,
                     "genus": args.genus, "seed": 3})


def aim(azimuth_deg: float, tilt_deg: float) -> None:
    eye = Gf.Vec3d(*EYE_M)
    _look_at(stage.GetPrimAtPath(SKY_CAMERA), eye, eye + 3000.0 * _direction(azimuth_deg, tilt_deg))


def pump(n: int) -> None:
    for _ in range(n):
        wx.step(1.0 / 60.0)
        simulation_app.update()


def shoot(path: str, subframes: int) -> None:
    pump(4)
    rep.orchestrator.step(rt_subframes=subframes)
    Image.fromarray(np.asarray(rgb.get_data())[..., :3]).save(path)
    print(f"[pixel] wrote {path}", flush=True)


def encode(folder: str, name: str, fps: int) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                    os.path.join(folder, "frame_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", os.path.join(folder, name)], check=True)
    print(f"[pixel] wrote {os.path.join(folder, name)}", flush=True)


os.makedirs(args.out, exist_ok=True)
report = {"args": vars(args)}
modes = [("RaytracedLighting", "rt", 8)] + ([] if args.skip_path_traced else [("PathTracing", "pt", args.pt_subframes)])
aim(0.0, args.tilt)
pump(30)
report["layer_after_start"] = wx.stats().get("viewport", {}).get("cloud_layer")

for mode, tag, subframes in modes:
    settings.set("/rtx/rendermode", mode)
    pump(20)
    wx.set_time(hour_utc=args.hour)
    for azimuth in (0.0, 90.0, 180.0, 270.0):
        aim(azimuth, args.tilt)
        shoot(os.path.join(args.out, f"heading_{tag}_{int(azimuth):03d}.png"), subframes)
    aim(300.0, args.tilt)
    for hour in (7.0, 11.0, 17.0, 19.0):
        wx.set_time(hour_utc=hour)
        shoot(os.path.join(args.out, f"hour_{tag}_{int(hour):02d}.png"), subframes)

# Frame times in Real-Time, the camera turning so every frame is a new march.
settings.set("/rtx/rendermode", "RaytracedLighting")
wx.set_time(hour_utc=args.hour)
pump(20)
layer_ms, start = [], time.perf_counter()
frames = 120
for k in range(frames):
    aim(3.0 * k, args.tilt)
    wx.step(1.0 / 60.0)
    simulation_app.update()
    layer_ms.append(wx.stats().get("viewport", {}).get("cloud_layer", {}).get("frame_ms", 0.0))
elapsed = time.perf_counter() - start
report["timing"] = {"frames": frames, "app_update_ms": round(1000.0 * elapsed / frames, 2),
                    "fps": round(frames / elapsed, 1),
                    "cloud_layer_ms_median": round(float(np.median(layer_ms)), 2),
                    "cloud_layer_ms_p95": round(float(np.percentile(layer_ms, 95)), 2)}
report["layer"] = wx.stats().get("viewport", {}).get("cloud_layer")
print("[pixel]", json.dumps(report["timing"]), flush=True)

if args.pan:
    folder = os.path.join(args.out, "pan")
    os.makedirs(folder, exist_ok=True)
    for k in range(args.pan):
        aim(360.0 * k / args.pan, args.tilt)
        shoot(os.path.join(folder, f"frame_{k:04d}.png"), 8)
    encode(folder, "pan.mp4", 15)
if args.day:
    folder = os.path.join(args.out, "day")
    os.makedirs(folder, exist_ok=True)
    aim(300.0, args.tilt)
    for k in range(args.day):
        wx.set_time(hour_utc=5.5 + (20.3 - 5.5) * k / max(args.day - 1, 1))
        shoot(os.path.join(folder, f"frame_{k:04d}.png"), 8)
    encode(folder, "day.mp4", 10)

with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=1, default=str)
weather.shutdown_controller()
simulation_app.close()
