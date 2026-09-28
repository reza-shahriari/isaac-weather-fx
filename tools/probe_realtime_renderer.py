"""Is RTX Real-Time black on this machine even without the weather extension? And why?

The weather debug run showed Real-Time black in every step, including the one with the extension
fully disabled and the stage's own sun back. This probe does not load weather.fx at all. It builds a
plain scene (ground, boxes, a sun, a white dome), waits for Real-Time to finish compiling its
shaders, then changes one renderer setting at a time and measures the frame.

    ./python.sh /path/to/isaac-weather-fx/tools/probe_realtime_renderer.py
    ./python.sh .../probe_realtime_renderer.py --gpu 1        # render on the second GPU instead

Send back ``captures/probe_realtime/report.txt``.
"""
import argparse
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

parser = argparse.ArgumentParser()
parser.add_argument("--out", default=os.path.join(REPO, "captures", "probe_realtime"))
parser.add_argument("--gpu", type=int, default=None, help="render on this GPU only")
parser.add_argument("--compile-wait", type=float, default=300.0,
                    help="seconds to wait for Real-Time's first non-black frame")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

config = {"headless": False, "multi_gpu": False, "width": 1280, "height": 720}
if args.gpu is not None:
    config["active_gpu"] = args.gpu
    config["physics_gpu"] = args.gpu
simulation_app = SimulationApp(config)

import carb.settings  # noqa: E402
import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

settings = carb.settings.get_settings()
os.makedirs(args.out, exist_ok=True)
LINES, RESULTS = [], []


def log(*parts):
    text = " ".join(str(p) for p in parts)
    print("[probe_rt]", text, flush=True)
    LINES.append(text)


def pump(frames):
    for _ in range(frames):
        simulation_app.update()


def viewport():
    from omni.kit.viewport.utility import get_active_viewport

    return get_active_viewport()


def set_mode(mode):
    try:
        viewport().set_hd_engine("rtx", mode)
    except Exception as exc:
        log(f"  set_hd_engine failed: {exc!r}")
    settings.set("/rtx/rendermode", mode)
    pump(20)
    log(f"  mode -> {mode}: viewport={getattr(viewport(), 'render_mode', None)}, "
        f"setting={settings.get('/rtx/rendermode')}")


def grab(name, settle=30, record=True):
    from omni.kit.viewport.utility import capture_viewport_to_file
    from PIL import Image

    pump(settle)
    path = os.path.join(args.out, f"{len(RESULTS):02d}_{name}.png")
    if os.path.exists(path):
        os.remove(path)
    capture_viewport_to_file(viewport(), path)
    size, stable = -1, 0
    for _ in range(600):
        simulation_app.update()
        if os.path.exists(path):
            now = os.path.getsize(path)
            stable = stable + 1 if now == size and now > 0 else 0
            size = now
            if stable >= 3:
                break
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    luma = image @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    h = luma.shape[0]
    entry = {"step": name, "top": float(luma[: h // 2].mean()), "bottom": float(luma[h // 2:].mean()),
             "black": float((luma < 2 / 255).mean()), "std": float(luma.std())}
    if record:
        RESULTS.append(entry)
        log(f"  {name:40s} top={entry['top']:.4f} bottom={entry['bottom']:.4f} "
            f"black={entry['black']:.3f} std={entry['std']:.4f}")
    else:
        os.remove(path)
    return entry


def with_setting(name, path, value, settle=60):
    old = settings.get(path)
    if old is None:
        log(f"  {name:40s} (setting {path} does not exist here)")
        return
    settings.set(path, value)
    grab(f"{name} [{path}={value}, was {old}]", settle)
    settings.set(path, old)
    pump(10)


def dump(prefixes):
    def flatten(prefix, value, out):
        if isinstance(value, dict):
            for key, item in value.items():
                flatten(f"{prefix}/{key}", item, out)
        else:
            out[prefix] = value

    for prefix in prefixes:
        flat = {}
        try:
            flatten(prefix, settings.get(prefix), flat)
        except Exception as exc:
            log(f"  could not read {prefix}: {exc!r}")
            continue
        log(f"  {prefix}: {len(flat)} settings")
        for key, value in sorted(flat.items()):
            LINES.append(f"    {key} = {repr(value)[:160]}")


def build_scene():
    context = omni.usd.get_context()
    context.new_stage()
    pump(10)
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.AddScaleOp().Set(Gf.Vec3f(50, 50, 0.05))
    ground.GetDisplayColorAttr().Set([Gf.Vec3f(0.5, 0.5, 0.5)])
    for i, x in enumerate(range(5, 45, 8)):
        box = UsdGeom.Cube.Define(stage, f"/World/Box_{i}")
        box.AddTranslateOp().Set(Gf.Vec3d(x, (i - 2) * 3, 1))
        box.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.2, 0.1)])
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(3000)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-45, 0, 30))
    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr(1000)
    dome.CreateColorAttr(Gf.Vec3f(0.45, 0.6, 0.85))
    camera = UsdGeom.Camera.Define(stage, "/World/ProbeCamera")
    UsdGeom.XformCommonAPI(camera).SetTranslate(Gf.Vec3d(-10, 0, 2.5))
    UsdGeom.XformCommonAPI(camera).SetRotate(Gf.Vec3f(95, 0, -90))
    viewport().camera_path = "/World/ProbeCamera"
    pump(30)
    return stage


def main():
    log("python", sys.version.split()[0])
    try:
        log("kit:", omni.kit.app.get_app().get_build_version())
    except Exception:
        pass
    manager = omni.kit.app.get_app().get_extension_manager()
    log("weather.fx enabled (should be False):", manager.is_extension_enabled("weather.fx"))
    for path in ("/renderer/activeGpu", "/renderer/multiGpu/enabled", "/renderer/multiGpu/maxGpuCount",
                 "/rtx/rendermode", "/app/renderer/resolution/width"):
        log(f"  {path} = {settings.get(path)}")

    stage = build_scene()

    log("")
    log("======== 1. path tracing, for reference")
    set_mode("PathTracing")
    grab("pt_reference", settle=120)

    log("")
    log("======== 2. Real-Time: wait for the shaders, then measure")
    set_mode("RealTimePathTracing")
    start = time.time()
    first = None
    while time.time() - start < args.compile_wait:
        entry = grab("wait", settle=60, record=False)
        if entry["black"] < 0.99:
            first = time.time() - start
            break
    log(f"  first non-black Real-Time frame after: {first if first is None else round(first, 1)} s "
        f"(None = never within {args.compile_wait:.0f} s)")
    dump(["/rtx/rtpt", "/rtx/post/aa", "/rtx/post/tonemap", "/rtx/post/histogram", "/rtx/post/dlss",
          "/rtx/rendermode", "/rtx/realTimePathTracing", "/rtx-transient/dlssg"])
    grab("rt_baseline", settle=60)

    log("")
    log("======== 3. Real-Time, one setting at a time")
    with_setting("auto exposure (aa) off", "/rtx/post/aa/autoExposureMode", 0)
    with_setting("auto exposure (aa) mode 2", "/rtx/post/aa/autoExposureMode", 2)
    with_setting("aa exposure x100", "/rtx/post/aa/exposure", 100.0)
    with_setting("histogram auto exposure on", "/rtx/post/histogram/enabled", True)
    with_setting("tonemapper linear", "/rtx/post/tonemap/op", 1)
    with_setting("tonemapper clamp", "/rtx/post/tonemap/op", 0)
    with_setting("film ISO x1000", "/rtx/post/tonemap/filmIso", 100000.0)
    with_setting("debug view: pre-tonemap", "/rtx/debugView/enablePreTonemap", True)
    with_setting("DLSS / anti-aliasing op 0 (off)", "/rtx/post/aa/op", 0)

    log("")
    log("======== 4. Real-Time, the lights")
    sun = stage.GetPrimAtPath("/World/Sun")
    dome = stage.GetPrimAtPath("/World/Dome")
    UsdGeom.Imageable(dome).MakeInvisible()
    grab("sun only (dome hidden)", settle=60)
    UsdGeom.Imageable(dome).MakeVisible()
    UsdGeom.Imageable(sun).MakeInvisible()
    grab("dome only (sun hidden)", settle=60)
    UsdGeom.Imageable(sun).MakeVisible()
    dome.GetAttribute("inputs:intensity").Set(100000.0)
    grab("dome at 100000", settle=60)
    dome.GetAttribute("inputs:intensity").Set(1000.0)

    log("")
    log("======== 5. the old real-time renderer, if this build still has it")
    set_mode("RaytracedLighting")
    grab("RaytracedLighting", settle=120)

    log("")
    log("======== summary")
    for r in RESULTS:
        log(f"  {r['step'][:80]:80s} top={r['top']:.4f} bottom={r['bottom']:.4f} "
            f"black={r['black']:.3f}")


try:
    main()
except Exception:
    log("!! stopped with an error:\n" + traceback.format_exc())
finally:
    kit_log = settings.get("/log/file")
    if kit_log and os.path.isfile(kit_log):
        seen, keep = set(), []
        with open(kit_log, errors="replace") as handle:
            for line in handle:
                if "[Warning]" in line or "[Error]" in line or "compilation" in line.lower():
                    body = line.split("] ", 2)[-1].strip()
                    if body not in seen:
                        seen.add(body)
                        keep.append(line.rstrip()[:300])
        LINES.append("")
        LINES.append(f"======== distinct warnings and errors in the Kit log ({len(keep)})")
        LINES.extend(keep[-200:])
    report = os.path.join(args.out, "report.txt")
    with open(report, "w") as handle:
        handle.write("\n".join(LINES) + "\n")
    print(f"\n[probe_rt] done. Send back: {report}", flush=True)
    simulation_app.close()
