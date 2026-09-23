"""Render one camera frame per weather preset into captures/ (headless, deterministic).

Run with Isaac Sim's Python:
    ./python.sh /path/to/isaac-weather-fx/examples/capture_presets.py
    ./python.sh .../capture_presets.py --presets heavy_rain storm --sky none
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

parser = argparse.ArgumentParser()
parser.add_argument("--presets", nargs="*", help="default: every registered preset")
parser.add_argument("--sky", default="auto", help="'auto' (Isaac Sim HDR sky), 'none', or an HDR path/URL")
parser.add_argument("--out", default=os.path.join(REPO, "captures"))
parser.add_argument("--resolution", type=int, nargs=2, default=(960, 540), metavar=("W", "H"))
parser.add_argument("--settle-frames", type=int, default=30, help="weather steps before each capture")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True, "multi_gpu": False})

import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402

manager = omni.kit.app.get_app().get_extension_manager()
manager.add_path(os.path.join(REPO, "exts"))
manager.set_extension_enabled_immediate("weather.fx", True)
manager.set_extension_enabled_immediate("omni.replicator.core", True)

import omni.replicator.core as rep  # noqa: E402
from weather_fx import api as weather  # noqa: E402

sys.path.insert(0, HERE)
from demo_scene import CAMERA_PATH, build_scene  # noqa: E402

omni.usd.get_context().new_stage()
build_scene(omni.usd.get_context().get_stage(), sky=args.sky)

render_product = rep.create.render_product(CAMERA_PATH, tuple(args.resolution))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(render_product)

wx = weather.get_controller()
wx.follow(CAMERA_PATH)
wx.set_general(time_source="manual", seed=42)

os.makedirs(args.out, exist_ok=True)
for preset in args.presets or wx.list_presets():
    wx.apply_preset(preset)
    for _ in range(args.settle_frames):
        wx.step(1.0 / 60.0)
        simulation_app.update()
    rep.orchestrator.step(rt_subframes=4)
    path = os.path.join(args.out, f"{preset}.png")
    Image.fromarray(np.asarray(rgb.get_data())[..., :3]).save(path)
    print(f"[capture] {path}")

weather.shutdown_controller()
simulation_app.close()
