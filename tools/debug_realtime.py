"""Find out why RTX Real-Time renders black with the weather on. Standalone, with a window.

Run with Isaac Sim's Python and send back the file it prints at the end (``report.txt``):

    ./python.sh /path/to/isaac-weather-fx/tools/debug_realtime.py
    ./python.sh .../debug_realtime.py --usd /path/to/your_scene.usd    # your own stage

It builds the demo scene (or opens yours), then captures the viewport in a series of steps: no
weather, sky only, sky and clouds, in Real-Time and in path tracing, and then a set of experiments
in Real-Time that each change one thing (fog off, dome brighter, dome untextured, sun brighter,
stage lights back, weather off). Each capture gets its mean brightness and the share of black
pixels, so the report shows which step turns the frame black without anyone reading an image.

Everything lands in ``captures/debug_realtime/``: report.txt (the thing to send), report.json,
one PNG per step, and a copy of the Kit log.
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

parser = argparse.ArgumentParser()
parser.add_argument("--usd", help="open this stage instead of building the demo scene")
parser.add_argument("--out", default=os.path.join(REPO, "captures", "debug_realtime"))
parser.add_argument("--headless", action="store_true",
                    help="no window (the viewport capture still works, but the window is closer "
                         "to what you see)")
parser.add_argument("--pt-frames", type=int, default=90, help="frames to accumulate in path tracing")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless, "multi_gpu": False,
                                "width": 1280, "height": 720})

import carb.settings  # noqa: E402
import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux  # noqa: E402

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.add_path(os.path.join(REPO, "exts"))
ext_manager.set_extension_enabled_immediate("weather.fx", True)

from weather_fx import api as weather  # noqa: E402
from weather_fx.backends.viewport.rtx_settings import FOG_SETTINGS  # noqa: E402
from weather_fx.runtime.diagnostics import format_report  # noqa: E402

settings = carb.settings.get_settings()
os.makedirs(args.out, exist_ok=True)
LINES = []
RESULTS = []


def log(*parts):
    text = " ".join(str(p) for p in parts)
    print("[debug_rt]", text, flush=True)
    LINES.append(text)


def pump(frames):
    for _ in range(frames):
        simulation_app.update()


def viewport():
    from omni.kit.viewport.utility import get_active_viewport

    return get_active_viewport()


# ----------------------------------------------------------------------------- renderer

def render_mode_report():
    vp = viewport()
    return {
        "viewport.hd_engine": getattr(vp, "hd_engine", None) if vp else None,
        "viewport.render_mode": getattr(vp, "render_mode", None) if vp else None,
        "/rtx/rendermode": settings.get("/rtx/rendermode"),
    }


def set_mode(mode):
    """Switch the way the viewport menu does, and the global setting too."""
    vp = viewport()
    try:
        vp.set_hd_engine("rtx", mode)
    except Exception as exc:
        log(f"  set_hd_engine('rtx', {mode!r}) failed: {exc!r}")
    settings.set("/rtx/rendermode", mode)
    pump(20)
    log(f"  render mode -> {mode}: {render_mode_report()}")


def is_path_traced():
    return "PathTracing" in str(render_mode_report())


# ----------------------------------------------------------------------------- captures

def capture(name, note=""):
    from omni.kit.viewport.utility import capture_viewport_to_file

    pump(args.pt_frames if is_path_traced() else 30)
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
    entry = {"step": name, "note": note, "file": os.path.basename(path), **render_mode_report()}
    try:
        from PIL import Image

        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        h = image.shape[0]
        luma = image @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        entry.update({
            "mean_rgb": [round(float(c), 3) for c in image.reshape(-1, 3).mean(0)],
            "mean_luma_top_half": round(float(luma[: h // 2].mean()), 4),
            "mean_luma_bottom_half": round(float(luma[h // 2:].mean()), 4),
            "max_luma": round(float(luma.max()), 3),
            "black_fraction": round(float((luma < 2.0 / 255.0).mean()), 3),
        })
    except Exception as exc:
        entry["error"] = f"could not read capture: {exc!r}"
    RESULTS.append(entry)
    log(f"  capture {name}: " + ", ".join(
        f"{k}={entry[k]}" for k in ("mean_luma_top_half", "mean_luma_bottom_half",
                                    "black_fraction", "error") if k in entry))
    return entry


# ----------------------------------------------------------------------------- dumps

def flatten(prefix, value, out):
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(f"{prefix}/{key}", item, out)
    else:
        out[prefix] = value


INTERESTING = ("fog", "tonemap", "exposure", "histogram", "dome", "useviewlighting", "sky",
               "ambient", "rendermode", "ptvol", "maxbounces", "indirectdiffuse", "directlighting",
               "shadows", "reflections")


def dump_rtx_settings(label):
    flat = {}
    for root in ("/rtx", "/rtx-transient"):
        try:
            flatten(root, settings.get(root), flat)
        except Exception as exc:
            log(f"  could not read {root}: {exc!r}")
    picked = {k: v for k, v in sorted(flat.items())
              if any(word in k.lower() for word in INTERESTING)}
    log(f"  rtx settings ({label}), {len(picked)} of {len(flat)}:")
    for key, value in picked.items():
        text = repr(value)
        LINES.append(f"    {key} = {text[:160]}")
    return picked


def dump_lights(label):
    stage = omni.usd.get_context().get_stage()
    log(f"  lights ({label}):")
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdLux.LightAPI):
            continue
        light = UsdLux.LightAPI(prim)
        visible = UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
        info = {
            "type": prim.GetTypeName(),
            "visible": visible,
            "intensity": light.GetIntensityAttr().Get(),
            "exposure": light.GetExposureAttr().Get(),
            "color": tuple(round(c, 3) for c in (light.GetColorAttr().Get() or ())),
        }
        if prim.IsA(UsdLux.DomeLight):
            asset = UsdLux.DomeLight(prim).GetTextureFileAttr().Get()
            info["texture"] = asset.path if asset else None
            info["texture_exists"] = bool(asset and asset.path and os.path.isfile(asset.path))
        attr = prim.GetAttribute("visibleInPrimaryRay")
        if attr and attr.HasAuthoredValue():
            info["visibleInPrimaryRay"] = attr.Get()
        if prim.IsA(UsdLux.DistantLight):
            info["angle"] = UsdLux.DistantLight(prim).GetAngleAttr().Get()
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            info["direction_world"] = tuple(round(c, 3) for c in xf.TransformDir(Gf.Vec3d(0, 0, -1)))
        LINES.append(f"    {prim.GetPath()}: {info}")


def dump_weather(label):
    wx = weather.get_controller()
    try:
        report = wx.diagnose(show=False)
        LINES.append(f"  diagnose ({label}):")
        LINES.extend("    " + line for line in format_report(report).splitlines())
    except Exception:
        LINES.append(f"  diagnose ({label}) failed:\n" + traceback.format_exc())


def wait_for_sky(seconds=240):
    wx = weather.get_controller()
    start = time.time()
    while time.time() - start < seconds:
        sky = wx.stats().get("viewport", {}).get("sky", {})
        if sky.get("texture") and not sky.get("baking"):
            return sky
        simulation_app.update()
    log("  !! timed out waiting for the sky bake")
    return wx.stats().get("viewport", {}).get("sky", {})


# ----------------------------------------------------------------------------- session-layer pokes

def session_set(path, attr, value, type_name):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim:
        log(f"  (no prim at {path})")
        return None
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        attribute = prim.GetAttribute(attr) or prim.CreateAttribute(attr, type_name)
        old = attribute.Get()
        attribute.Set(value)
    return old


# ----------------------------------------------------------------------------- the run

def build_demo_scene(stage):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.AddScaleOp().Set(Gf.Vec3f(50, 50, 0.05))
    ground.GetDisplayColorAttr().Set([Gf.Vec3f(0.5, 0.5, 0.5)])
    for i, x in enumerate(range(5, 45, 8)):
        box = UsdGeom.Cube.Define(stage, f"/World/Box_{i}")
        box.AddTranslateOp().Set(Gf.Vec3d(x, (i - 2) * 3, 1))
        box.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.2, 0.1)])
    sun = UsdLux.DistantLight.Define(stage, "/World/StageSun")
    sun.CreateIntensityAttr(3000)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-45, 0, 30))
    camera = UsdGeom.Camera.Define(stage, "/World/DebugCamera")
    UsdGeom.XformCommonAPI(camera).SetTranslate(Gf.Vec3d(-10, 0, 2.5))
    # Looking along +X, slightly up, so the frame holds sky and ground.
    UsdGeom.XformCommonAPI(camera).SetRotate(Gf.Vec3f(95, 0, -90))
    return "/World/DebugCamera"


def step(title):
    log("")
    log("=" * 8, title)


def main():
    import platform

    log("python", sys.version.split()[0], "|", platform.platform())
    try:
        log("isaac sim / kit:", omni.kit.app.get_app().get_build_version())
    except Exception:
        pass
    for ext in ("weather.fx", "omni.volume", "omni.kit.viewport.window"):
        log(f"extension {ext}: enabled={ext_manager.is_extension_enabled(ext)}")

    context = omni.usd.get_context()
    if args.usd:
        context.open_stage(args.usd)
        pump(60)
        camera = None
    else:
        context.new_stage()
        pump(10)
        camera = build_demo_scene(context.get_stage())
    stage = context.get_stage()
    log("stage:", args.usd or "demo scene", "| metersPerUnit:", UsdGeom.GetStageMetersPerUnit(stage),
        "| upAxis:", UsdGeom.GetStageUpAxis(stage))
    if camera:
        viewport().camera_path = camera
    pump(30)

    wx = weather.get_controller()
    lon = wx.state.sky.longitude_deg if hasattr(wx, "state") else 11.6
    noon_utc = (12.0 - lon / 15.0) % 24.0

    # ---------------------------------------------------------------- 1. no weather
    step("1. no weather (the stage's own light), Real-Time then path tracing")
    wx.configure(general={"enabled": True, "time_source": "wall"},
                 sky={"enabled": False}, clouds={"enabled": False}, fog={"enabled": False})
    pump(10)
    set_mode("RaytracedLighting")
    dump_rtx_settings("before any weather, RT")
    capture("rt_no_weather", "the stage's own sun only")
    set_mode("PathTracing")
    capture("pt_no_weather")

    # ---------------------------------------------------------------- 2. sky only
    step("2. weather sky only (no clouds, no fog), late morning")
    set_mode("RaytracedLighting")
    wx.configure(sky={"enabled": True, "hour_utc": (noon_utc - 1.5) % 24.0,
                      "hide_scene_lights": True},
                 clouds={"enabled": False}, fog={"enabled": False})
    sky = wait_for_sky()
    log("  sky stats:", sky)
    dump_lights("sky only")
    dump_weather("sky only, RT")
    capture("rt_sky_only")
    set_mode("PathTracing")
    capture("pt_sky_only")

    # ---------------------------------------------------------------- 3. sky and clouds
    step("3. sky and clouds (cumulus, auto render path)")
    wx.configure(clouds={"enabled": True, "cover": 0.45, "genus": "cumulus", "render_path": "auto"})
    wait_for_sky()
    pump(60)
    capture("pt_sky_clouds", "clouds should be volumes here")
    set_mode("RaytracedLighting")
    pump(30)
    sky = wait_for_sky()
    log("  sky stats:", sky)
    dump_lights("sky + clouds, RT")
    dump_weather("sky + clouds, RT")
    dump_rtx_settings("sky + clouds, RT")
    base = capture("rt_sky_clouds", "clouds should be in the dome here")

    # ---------------------------------------------------------------- 4. experiments in RT
    step("4. Real-Time experiments: one change at a time, each undone afterwards")
    fog_path = FOG_SETTINGS.get("enabled")
    if fog_path and settings.get(fog_path) is not None:
        old = settings.get(fog_path)
        settings.set(fog_path, False)
        capture("rt_exp_fog_forced_off", f"{fog_path} was {old}")
        settings.set(fog_path, old)

    dome = "/WeatherFX/Sky/Dome"
    old = session_set(dome, "inputs:intensity", 1000.0, Sdf.ValueTypeNames.Float)
    if old is not None:
        capture("rt_exp_dome_intensity_1000", f"weather dome intensity was {old}")
        session_set(dome, "inputs:intensity", old, Sdf.ValueTypeNames.Float)

    old_tex = session_set(dome, "inputs:texture:file", Sdf.AssetPath(""), Sdf.ValueTypeNames.Asset)
    if old_tex is not None:
        old_int = session_set(dome, "inputs:intensity", 1000.0, Sdf.ValueTypeNames.Float)
        capture("rt_exp_dome_no_texture", "plain white dome at 1000: tests the EXR in RT")
        session_set(dome, "inputs:texture:file", old_tex, Sdf.ValueTypeNames.Asset)
        session_set(dome, "inputs:intensity", old_int, Sdf.ValueTypeNames.Float)

    sun = "/WeatherFX/Sky/Sun"
    old = session_set(sun, "inputs:intensity", 3000.0, Sdf.ValueTypeNames.Float)
    if old is not None:
        capture("rt_exp_sun_3000", f"weather sun intensity was {old}")
        session_set(sun, "inputs:intensity", old, Sdf.ValueTypeNames.Float)

    clouds_root = stage.GetPrimAtPath("/WeatherFX/Clouds")
    if clouds_root:
        img = UsdGeom.Imageable(clouds_root)
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            img.MakeInvisible()
        capture("rt_exp_volumes_hidden", "the path-traced cloud boxes were still in the stage")
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            img.MakeVisible()

    wx.configure(sky={"hide_scene_lights": False})
    wait_for_sky()
    capture("rt_exp_stage_lights_back", "hide_scene_lights=False")
    wx.configure(sky={"hide_scene_lights": True})

    wx.configure(clouds={"enabled": False})
    wait_for_sky()
    capture("rt_exp_clouds_off")

    wx.configure(general={"enabled": False})
    pump(30)
    dump_lights("weather disabled")
    dump_rtx_settings("weather disabled, RT")
    capture("rt_exp_weather_disabled", "everything the extension did should be undone")
    wx.configure(general={"enabled": True})

    # ---------------------------------------------------------------- summary
    step("summary (mean brightness 0..1; black_fraction = share of pixels below 2/255)")
    log(f"  {'step':34s} {'renderer':22s} {'top':>7s} {'bottom':>7s} {'black':>6s}")
    for r in RESULTS:
        log(f"  {r['step']:34s} {str(r.get('viewport.render_mode')):22s} "
            f"{r.get('mean_luma_top_half', float('nan')):7.4f} "
            f"{r.get('mean_luma_bottom_half', float('nan')):7.4f} "
            f"{r.get('black_fraction', float('nan')):6.3f}  {r.get('note', '')}")
    return base


try:
    main()
except Exception:
    log("!! the run stopped with an error:\n" + traceback.format_exc())
finally:
    kit_log = settings.get("/log/file")
    if kit_log and os.path.isfile(kit_log):
        shutil.copy(kit_log, os.path.join(args.out, "kit.log"))
        # The warnings that matter are the renderer's; pull them into the report.
        with open(kit_log, errors="replace") as handle:
            keep = [line.rstrip() for line in handle
                    if any(w in line.lower() for w in ("weather", "rtx", "hydra", "dome", "texture",
                                                        "exr", "fog", "volume", "error"))]
        LINES.append("")
        LINES.append(f"======== kit log lines mentioning the renderer ({len(keep)}; last 150)")
        LINES.extend(keep[-150:])
    with open(os.path.join(args.out, "report.json"), "w") as handle:
        json.dump(RESULTS, handle, indent=2, default=str)
    report = os.path.join(args.out, "report.txt")
    with open(report, "w") as handle:
        handle.write("\n".join(LINES) + "\n")
    print(f"\n[debug_rt] done. Send back: {report}\n[debug_rt] (PNGs and kit.log are next to it)",
          flush=True)
    weather.shutdown_controller()
    simulation_app.close()
