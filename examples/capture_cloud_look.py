"""The cloud's look on one sheet: a fixed set of views in RTX Real-Time and in the path tracer.

    ./python.sh examples/capture_cloud_look.py --out captures/cloud_look/step1 --label "step 1"

The sheet is what a change to the cloud is judged on, so the views never change between runs:
four headings at mid-afternoon, a long lens on one cloud, and a low sun seen nearly head on. The
top row is RTX Real-Time, the bottom row the path tracer; both come from Isaac Sim's own renderer
through the extension (``clouds.render_path = "pixel"``). The script writes, under ``--out``:

* ``<mode>_<view>.png``: every frame at full size;
* ``sheet.jpg``: all of them on one labelled sheet;
* ``report.json``: milliseconds per frame for the cloud layer and for the whole app update in
  Real-Time, with the camera turning so every frame is a new march.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "exts", "weather.fx"))

#: ``(name, hour UTC, azimuth, tilt, focal length mm)``. 18 mm is the gallery's 60 degree lens.
VIEWS = (
    ("az000", 14.5, 0.0, 22.0, 18.0),
    ("az090", 14.5, 90.0, 22.0, 18.0),
    ("az180", 14.5, 180.0, 22.0, 18.0),
    ("az270", 14.5, 270.0, 22.0, 18.0),
    ("zoom", 14.5, 90.0, 16.0, 50.0),
    ("lowsun", 18.0, 300.0, 15.0, 18.0),
)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--out", default=os.path.join(REPO, "captures", "cloud_look", "latest"))
parser.add_argument("--label", default="")
parser.add_argument("--gpu", default="0")
parser.add_argument("--resolution", type=int, nargs=2, default=(1280, 720))
parser.add_argument("--genus", default="cumulus")
parser.add_argument("--cover", type=float, default=0.35)
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--patches", default="", help="a folder of cloud patches, or none for the noise function")
parser.add_argument("--layer-scale", type=float, default=0.75)
parser.add_argument("--views", nargs="*", default=[v[0] for v in VIEWS])
parser.add_argument("--rt-subframes", type=int, default=8)
parser.add_argument("--pt-subframes", type=int, default=48)
parser.add_argument("--skip-path-traced", action="store_true")
parser.add_argument("--skip-timing", action="store_true")
parser.add_argument("--pan", type=int, default=0, help="frames of a full turn in Real-Time, as pan.mp4 (0: skip)")
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
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

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
    print(f"[look] no viewport to aim ({exc!r})")

product = rep.create.render_product(SKY_CAMERA, tuple(args.resolution))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(product)

wx = weather.get_controller()
wx.clear()
wx.set_general(time_source="manual", seed=42)
wx.follow(SKY_CAMERA)
wx.set_site(48.14, 11.58)
wx.set_time(date_utc="2024-06-21")
wx.configure(sky={"enabled": True, "hour_utc": VIEWS[0][1]},
             clouds={"enabled": True, "render_path": "pixel", "cover": args.cover,
                     "genus": args.genus, "seed": args.seed, "patches": args.patches,
                     "layer_scale": args.layer_scale})

camera = UsdGeom.Camera(stage.GetPrimAtPath(SKY_CAMERA))


def aim(azimuth_deg: float, tilt_deg: float, focal_mm: float = 18.0) -> None:
    eye = Gf.Vec3d(*EYE_M)
    camera.GetFocalLengthAttr().Set(float(focal_mm))
    _look_at(stage.GetPrimAtPath(SKY_CAMERA), eye, eye + 3000.0 * _direction(azimuth_deg, tilt_deg))


def pump(n: int) -> None:
    for _ in range(n):
        wx.step(1.0 / 60.0)
        simulation_app.update()


os.makedirs(args.out, exist_ok=True)
views = [v for v in VIEWS if v[0] in set(args.views)]
modes = [("RaytracedLighting", "rt", "REAL-TIME", args.rt_subframes)]
if not args.skip_path_traced:
    modes.append(("PathTracing", "pt", "PATH-TRACED", args.pt_subframes))
report = {"args": vars(args), "views": [list(v) for v in views]}
aim(views[0][2], views[0][3], views[0][4])
pump(30)

def time_real_time() -> None:
    """Frame times in Real-Time, the camera turning so every frame is a new march. Run before
    the path tracer has been on: Real-Time is slower for a while after it."""
    wx.set_time(hour_utc=views[0][1])
    pump(20)
    layer_ms, count = [], 90
    start = time.perf_counter()
    for k in range(count):
        aim(4.0 * k, 22.0)
        wx.step(1.0 / 60.0)
        simulation_app.update()
        layer_ms.append(wx.stats().get("viewport", {}).get("cloud_layer", {}).get("frame_ms", 0.0))
    elapsed = time.perf_counter() - start
    report["timing"] = {"frames": count, "app_update_ms": round(1000.0 * elapsed / count, 2),
                        "fps": round(count / elapsed, 1),
                        "cloud_layer_ms_median": round(float(np.median(layer_ms)), 2),
                        "cloud_layer_ms_p95": round(float(np.percentile(layer_ms, 95)), 2)}
    print("[look]", json.dumps(report["timing"]), flush=True)


frames = {}
for mode, tag, title, subframes in modes:
    settings.set("/rtx/rendermode", mode)
    pump(20)
    hour = None
    for name, view_hour, azimuth, tilt, focal in views:
        if view_hour != hour:
            wx.set_time(hour_utc=view_hour)
            hour = view_hour
            pump(12)
        aim(azimuth, tilt, focal)
        pump(20)    # the layer's running mean over frames settles
        rep.orchestrator.step(rt_subframes=subframes)
        image = np.asarray(rgb.get_data())[..., :3].copy()
        path = os.path.join(args.out, f"{tag}_{name}.png")
        Image.fromarray(image).save(path)
        frames[(tag, name)] = (image, f"{title}  {name}  {view_hour:g} h UTC")
        print(f"[look] wrote {path}", flush=True)
    if tag == "rt" and not args.skip_timing:
        time_real_time()

# One sheet: a row per render mode, a column per view, each cell labelled.
cell_w = 640
cell_h = int(round(cell_w * args.resolution[1] / args.resolution[0]))
sheet = Image.new("RGB", (cell_w * len(views), cell_h * len(modes)), (0, 0, 0))
draw = ImageDraw.Draw(sheet)
for row, (_, tag, _, _) in enumerate(modes):
    for col, (name, *_rest) in enumerate(views):
        image, caption = frames[(tag, name)]
        cell = Image.fromarray(image).resize((cell_w, cell_h), Image.LANCZOS)
        sheet.paste(cell, (col * cell_w, row * cell_h))
        text = f"{args.label}  {caption}".strip()
        draw.rectangle([col * cell_w, row * cell_h, col * cell_w + 7 * len(text) + 8, row * cell_h + 15],
                       fill=(0, 0, 0))
        draw.text((col * cell_w + 4, row * cell_h + 2), text, fill=(255, 255, 0))
sheet_path = os.path.join(args.out, "sheet.jpg")
sheet.save(sheet_path, quality=90)
print(f"[look] wrote {sheet_path}", flush=True)

if args.pan:
    import subprocess

    folder = os.path.join(args.out, "pan")
    os.makedirs(folder, exist_ok=True)
    settings.set("/rtx/rendermode", "RaytracedLighting")
    wx.set_time(hour_utc=views[0][1])
    aim(0.0, 22.0)
    pump(30)
    for k in range(args.pan):
        # One app update per frame, as a viewport does: the layer has only its history to lean on.
        aim(360.0 * k / args.pan, 22.0)
        wx.step(1.0 / 30.0)
        simulation_app.update()
        rep.orchestrator.step(rt_subframes=2)
        Image.fromarray(np.asarray(rgb.get_data())[..., :3]).save(os.path.join(folder, f"frame_{k:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i",
                    os.path.join(folder, "frame_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", os.path.join(args.out, "pan.mp4")], check=True)
    print(f"[look] wrote {os.path.join(args.out, 'pan.mp4')}", flush=True)

report["layer"] = wx.stats().get("viewport", {}).get("cloud_layer")

with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=1, default=str)
weather.shutdown_controller()
simulation_app.close()
