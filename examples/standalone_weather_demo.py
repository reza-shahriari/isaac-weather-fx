"""Standalone demo: builds a small scene and cycles through weather presets.

Run with Isaac Sim's Python:
    ./python.sh /path/to/isaac-weather-fx/examples/standalone_weather_demo.py
    ./python.sh .../standalone_weather_demo.py --headless --manual
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--manual", action="store_true", help="deterministic stepping with wx.step(dt)")
parser.add_argument("--multi-gpu", action="store_true",
                    help="render on every GPU (fails with Vulkan import errors on mixed GPU models)")
parser.add_argument("--sky", default="auto", help="'auto' (Isaac Sim HDR sky), 'none', or an HDR path/URL")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless, "multi_gpu": args.multi_gpu})

import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402

EXT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "exts"))

# Option A: enable the extension (gives the UI panel when not headless).
try:
    manager = omni.kit.app.get_app().get_extension_manager()
    manager.add_path(EXT_FOLDER)
    manager.set_extension_enabled_immediate("weather.fx", True)
except Exception as exc:  # Option B: just import the package (API only, no UI)
    print(f"[demo] could not enable extension ({exc}); importing package directly")
    sys.path.insert(0, os.path.join(EXT_FOLDER, "weather.fx"))

from weather_fx import api as weather  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_scene import CAMERA_PATH, build_scene  # noqa: E402

# ---------------------------------------------------------------- scene
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
build_scene(stage, sky=args.sky)
try:
    from omni.kit.viewport.utility import get_active_viewport

    get_active_viewport().camera_path = CAMERA_PATH
except Exception:
    pass

# ---------------------------------------------------------------- weather
wx = weather.get_controller()
wx.follow(CAMERA_PATH)
if args.manual:
    wx.set_general(time_source="manual", seed=42)

schedule = [(0, "light_fog"), (200, "heavy_rain"), (500, "storm"), (800, "blizzard"), (1100, "clear")]
DT = 1.0 / 60.0
CYCLE = 1300
# Headless: one pass. With a window: repeat the cycle until the window is closed.
frame = 0
while simulation_app.is_running() and (frame < CYCLE or not args.headless):
    for start, preset in schedule:
        if frame % CYCLE == start:
            wx.apply_preset(preset)
            print(f"[demo] frame {frame}: {preset} -> {wx.stats()['viewport']}")
    if args.manual:
        wx.step(DT)
    simulation_app.update()
    frame += 1

weather.shutdown_controller()
simulation_app.close()
