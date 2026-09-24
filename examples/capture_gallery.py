"""Render the README gallery: one RGB frame per weather scenario, lit only by weather-fx.

    ./python.sh examples/capture_gallery.py
    ./python.sh examples/capture_gallery.py --scenarios golden_hour full_moon_night
    ./python.sh examples/capture_gallery.py --asset /path/to/drone.usdc --gpu 0

Every frame comes out of the same stage with the same two cameras, and **nothing in that stage
emits light**: the sun, the moon and the sky dome are the ones this extension authors under
``/WeatherFX/Sky``. So the gallery is a picture of the model rather than of a lighting rig, and a
scenario that looks wrong is the model being wrong.

The clock is **solved, not typed.** "Golden hour" is the hour at which the sun is actually four
degrees above the horizon at the site and date, found by scanning the ephemeris; the full-moon
frame is a real date on which a nearly full moon stands high while the sun is 20 degrees down. Both
would otherwise be a number somebody tuned once, in one month, at one latitude, and they would
quietly stop being true anywhere else.
"""
import argparse
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "exts", "weather.fx"))

SITE = (48.14, 11.58)  # Munich, the default site in `SkyParams`
BASE_DATE = "2024-06-21"

parser = argparse.ArgumentParser(description=__doc__)
# Not `captures/`, which is gitignored for ad-hoc runs: these frames are what the README is
# made of, so they are documentation and they are tracked.
parser.add_argument("--out", default=os.path.join(REPO, "docs", "images", "gallery"))
parser.add_argument("--scenarios", nargs="*", help="default: all of them, in order")
parser.add_argument("--asset", default=None, help="a USD to fly in the subject frames")
parser.add_argument("--resolution", type=int, nargs=2, default=(1600, 900), metavar=("W", "H"))
parser.add_argument("--rt-subframes", type=int, default=24)
parser.add_argument("--settle", type=int, default=24, help="app updates before each capture")
parser.add_argument("--gpu", default="0", help="GPU index, or 'all' to let Kit choose")
parser.add_argument("--skip-subject", action="store_true", help="sky frames only")
# Gallery quality. The defaults in `state.py` are sized for an interactive viewport; a still that
# someone will look at closely wants a finer cloud grid and a bigger dome, and can afford the bake.
parser.add_argument("--dome", type=int, default=1024, help="dome texture height")
parser.add_argument("--march-steps", type=int, default=160,
                    help="samples per ray in the cloud march (64 is the viewport default)")
parser.add_argument("--cell-m", type=float, default=45.0, help="cloud grid cell size")
parser.add_argument("--cells", type=int, default=384, help="cloud grid cells per side")
args, _ = parser.parse_known_args()

# --- the clock is solved from the ephemeris, so every caption is true ---------------------------
from weather_fx.core.celestial import moon_position, sun_position  # noqa: E402


def _utc(date_str: str, hour: float) -> dt.datetime:
    day = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    return day + dt.timedelta(hours=hour)


def hour_for_sun(target_deg: float, *, date_str: str = BASE_DATE, descending: bool = True) -> float:
    """The UTC hour at which the sun is nearest ``target_deg`` above the horizon.

    ``descending`` picks the afternoon crossing, which is the one with the warm light: the same
    elevation in the morning is a different photograph and a different colour temperature only
    because the aerosol has not been stirred yet, which this model does not claim to know.
    """
    lat, lon = SITE
    best, best_err = 12.0, 1e9
    previous = None
    for step in range(0, 24 * 30):  # two-minute resolution
        hour = step / 30.0
        elevation = sun_position(lat, lon, _utc(date_str, hour)).elevation_deg
        falling = previous is not None and elevation < previous
        previous = elevation
        if descending and not falling:
            continue
        if not descending and falling:
            continue
        err = abs(elevation - target_deg)
        if err < best_err:
            best, best_err = hour, err
    return best


def full_moon_night(*, search_days: int = 45) -> tuple:
    """A real (date, hour) with a nearly full moon high up and the sun well down."""
    lat, lon = SITE
    start = dt.datetime.strptime(BASE_DATE, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    best, best_score = (BASE_DATE, 1.0), -1e9
    for day in range(search_days):
        date_str = (start + dt.timedelta(days=day)).strftime("%Y-%m-%d")
        for step in range(0, 24 * 6):
            hour = step / 6.0
            when = _utc(date_str, hour)
            sun = sun_position(lat, lon, when)
            if sun.elevation_deg > -18.0:
                continue
            moon = moon_position(lat, lon, when)
            if moon.elevation_deg < 15.0:
                continue
            score = moon.illuminated_fraction * 100.0 + moon.elevation_deg
            if score > best_score:
                best, best_score = (date_str, hour), score
    return best


MOON_DATE, MOON_HOUR = full_moon_night()

# --- the scenarios -----------------------------------------------------------------------------
# Each is a complete `WeatherState` override. `clear()` runs first, so nothing leaks between them.
SCENARIOS: dict = {
    "clear_noon": {
        "caption": "Clear, high summer sun. Rayleigh gradient from a deep zenith to a pale horizon.",
        "sky": {"turbidity": 2.1, "hour_utc": hour_for_sun(58.0)},
        "clouds": {"enabled": False},
    },
    "fair_cumulus": {
        "caption": "Fair-weather cumulus, 28 % cover, bases on one level at the condensation height.",
        "sky": {"turbidity": 2.6, "hour_utc": hour_for_sun(50.0)},
        "clouds": {"enabled": True, "cover": 0.20, "genus": "cumulus",
                   "temperature_c": 24.0, "dewpoint_c": 11.0, "seed": 3},
    },
    "broken_cumulus": {
        "caption": "Broken cumulus, 55 % cover and a deeper deck — lit sides bright, bases grey.",
        "sky": {"turbidity": 3.0, "hour_utc": hour_for_sun(42.0)},
        "clouds": {"enabled": True, "cover": 0.38, "genus": "cumulus", "optical_depth": 26.0,
                   "temperature_c": 26.0, "dewpoint_c": 15.0, "seed": 11},
    },
    "towering_congestus": {
        "caption": "Congestus: the same field given height, so the towers shade their own bases.",
        "sky": {"turbidity": 3.2, "hour_utc": hour_for_sun(34.0)},
        "clouds": {"enabled": True, "cover": 0.32, "genus": "congestus", "feature_m": 900.0,
                   "temperature_c": 28.0, "dewpoint_c": 19.0, "seed": 5},
    },
    "cirrus_veil": {
        "caption": "Cirrus at 8 km: thin, high and barely attenuating — a whitened sky, not a grey one.",
        "sky": {"turbidity": 2.4, "hour_utc": hour_for_sun(38.0)},
        "clouds": {"enabled": True, "cover": 0.45, "genus": "cirrus", "cell_m": 220.0,
                   "feature_m": 2600.0, "seed": 2},
    },
    "overcast_stratocumulus": {
        "tilt_deg": 11.0,
        "caption": "Overcast stratocumulus. Shadowless, and about a quarter of the clear-sky light.",
        "sky": {"turbidity": 3.4, "hour_utc": hour_for_sun(36.0)},
        "clouds": {"enabled": True, "cover": 0.99, "genus": "stratocumulus",
                   "temperature_c": 14.0, "dewpoint_c": 12.0, "seed": 7},
    },
    "golden_hour": {
        "caption": "Sun four degrees up, solved from the ephemeris. Long airmass, warm beam, cool shadow.",
        "sky": {"turbidity": 3.6, "hour_utc": hour_for_sun(4.0)},
        "clouds": {"enabled": True, "cover": 0.26, "genus": "cumulus", "optical_depth": 20.0,
                   "temperature_c": 22.0, "dewpoint_c": 13.0, "seed": 17},
    },
    "blue_hour": {
        "caption": "Civil twilight, sun four degrees below. Daylight fading out, stars not yet in.",
        "sky": {"turbidity": 3.0, "hour_utc": hour_for_sun(-4.0)},
        "clouds": {"enabled": True, "cover": 0.22, "genus": "cumulus",
                   "temperature_c": 18.0, "dewpoint_c": 12.0, "seed": 17},
    },
    "full_moon_night": {
        "caption": "A real full-moon night: same Perez distribution, one source 400,000x fainter.",
        "sky": {"date_utc": MOON_DATE, "hour_utc": MOON_HOUR, "turbidity": 2.3,
                "star_intensity": 1.6, "exposure_scale": 3.0},
        "clouds": {"enabled": True, "cover": 0.18, "genus": "cumulus",
                   "temperature_c": 15.0, "dewpoint_c": 9.0, "seed": 23},
    },
    "morning_fog": {
        "tilt_deg": 2.5,
        "caption": "Radiation fog, 180 m visibility — a saturated surface layer, so no cloud above it.",
        "sky": {"turbidity": 4.5, "hour_utc": hour_for_sun(8.0, descending=False)},
        "clouds": {"enabled": False},
        "fog": {"enabled": True, "visibility_m": 180.0, "height_fog": True, "base_height_m": 0.0},
        "wind": {"speed_mps": 0.4},
    },
    "rain_squall": {
        "tilt_deg": 4.0,
        "caption": "22 mm/h from a deep deck: Marshall-Palmer drop sizes, streaks at 1/60 s exposure.",
        "sky": {"turbidity": 2.6, "hour_utc": hour_for_sun(28.0)},
        "clouds": {"enabled": True, "cover": 0.96, "genus": "stratocumulus",
                   "temperature_c": 15.0, "dewpoint_c": 13.5, "seed": 31},
        "fog": {"enabled": True, "visibility_m": 3200.0},
        "rain": {"enabled": True, "rate_mm_h": 22.0},
        "wind": {"speed_mps": 11.0, "direction_deg": 25.0, "gust_strength": 0.5},
    },
    "blizzard": {
        "tilt_deg": 4.0,
        "caption": "Sub-zero air, 260 m visibility.",
        "sky": {"turbidity": 3.0, "date_utc": "2024-01-18",
                "hour_utc": hour_for_sun(12.0, date_str="2024-01-18")},
        "clouds": {"enabled": True, "cover": 0.98, "genus": "stratocumulus",
                   "temperature_c": -7.0, "dewpoint_c": -9.0, "seed": 13},
        "fog": {"enabled": True, "visibility_m": 260.0},
        # The particle cap, not the density, is what a blizzard runs into: a 12 m volume at the
        # default 30,000 works out at under 7 flakes per cubic metre however high the density is
        # set. Shrink the volume and raise the cap, and the same physics reads as weather.
        "snow": {"enabled": True, "number_density_m3": 120.0, "fall_speed_mps": 1.1,
                 "volume_radius_m": 8.0, "volume_height_m": 8.0, "max_particles": 90000,
                 "flake_min_diameter_mm": 2.0, "flake_max_diameter_mm": 10.0,
                 "opacity": 0.95, "sway_amplitude_m": 0.5},
        "wind": {"speed_mps": 13.0, "direction_deg": -40.0, "gust_strength": 0.6},
    },
    "randomised": {
        "caption": "`wx.randomize(26)` — one integer, and every parameter drawn with its couplings.",
        "randomize": 26,
    },
}

# --- Kit --------------------------------------------------------------------------------------
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
from isaacsim import SimulationApp  # noqa: E402

_config = {"headless": True}
if args.gpu.strip().lower() not in {"all", "*"}:
    # Kit picks a *Vulkan* device, so CUDA_VISIBLE_DEVICES does not keep a render off the other
    # card: `active_gpu` and `multi_gpu` are the keys that become --/renderer/activeGpu=.
    _config.update(active_gpu=int(args.gpu), multi_gpu=False)
simulation_app = SimulationApp(_config)

import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402
from PIL import Image  # noqa: E402

extensions = omni.kit.app.get_app().get_extension_manager()
extensions.add_path(os.path.join(REPO, "exts"))
extensions.set_extension_enabled_immediate("weather.fx", True)
extensions.set_extension_enabled_immediate("omni.replicator.core", True)

import omni.replicator.core as rep  # noqa: E402

from weather_fx import api as weather  # noqa: E402

sys.path.insert(0, HERE)
from gallery_scene import (  # noqa: E402
    SKY_CAMERA,
    SUBJECT_CAMERA,
    SUBJECT_ROOT,
    aim_sky_camera,
    build_scene,
    pose_subject,
    reference_subject,
)

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
build_scene(stage)

subject = not args.skip_subject and args.asset is not None
if subject:
    reference_subject(stage, os.path.abspath(args.asset))
elif not args.skip_subject:
    print("[gallery] no --asset given: sky frames only")

shots = [("sky", SKY_CAMERA)] + ([("subject", SUBJECT_CAMERA)] if subject else [])
products = {}
for name, path in shots:
    product = rep.create.render_product(path, tuple(args.resolution))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(product)
    products[name] = annotator

wx = weather.get_controller()
wx.set_general(time_source="manual", seed=42)
wx.follow(SKY_CAMERA)  # both cameras stand at the same spot, so one anchor serves both

os.makedirs(args.out, exist_ok=True)
manifest = []
chosen = args.scenarios or list(SCENARIOS)
for name in chosen:
    if name not in SCENARIOS:
        print(f"[gallery] unknown scenario {name!r}; have {sorted(SCENARIOS)}", file=sys.stderr)
        continue
    spec = dict(SCENARIOS[name])
    caption = spec.pop("caption", "")
    tilt = spec.pop("tilt_deg", None)
    wx.clear()
    wx.set_general(time_source="manual", seed=42)
    wx.set_site(*SITE)
    wx.set_time(date_utc=BASE_DATE)
    if "randomize" in spec:
        print(f"[gallery] {name}: {wx.randomize(spec['randomize'])}")
    else:
        wx.configure(**spec)
    # Quality last, so a scenario cannot accidentally undercut it -- except where the scenario
    # deliberately asked for a coarser grid (cirrus is a 2.6 km feature; resolving it at 32 m
    # costs a minute and shows nothing).
    wx.set_sky(dome_resolution=args.dome)
    if wx.state.clouds.enabled:
        wx.set_clouds(march_steps=args.march_steps)
        if "cell_m" not in spec.get("clouds", {}):
            wx.set_clouds(cell_m=args.cell_m, cells=args.cells)

    conditions = wx.sky_conditions(build_cloud=True)
    series = wx.surface_weather(hours=24.0, step_s=3600.0)
    print(f"[gallery] {name}: {conditions.describe()}")

    # Frame whichever body is actually lighting the scene. A fixed bearing photographs half the
    # scenarios from behind the light, and a gallery shot into the shadow side of its own sky is
    # a gallery that says this model cannot do sunsets.
    lit_by = conditions.sun if conditions.sun.elevation_deg > -2.0 else conditions.moon
    aim_sky_camera(
        stage,
        lit_by.azimuth_deg,
        **({} if tilt is None else {"tilt_deg": float(tilt)}),
    )
    if subject:
        size = pose_subject(stage, lit_by.azimuth_deg)
        if name == chosen[0]:
            print(f"[gallery] subject: {tuple(round(float(v), 3) for v in size)} m")

    for _ in range(args.settle):
        wx.step(1.0 / 60.0)
        simulation_app.update()
    written = []
    for shot, annotator in products.items():
        # The aircraft is a speck at 18 mm and a distraction in a frame about the sky, so the sky
        # shots are taken without it. Visibility rather than a second stage: the weather, the
        # bake and the clock are then provably the same between the pair.
        if subject:
            UsdGeom.Imageable(stage.GetPrimAtPath(SUBJECT_ROOT)).GetVisibilityAttr().Set(
                UsdGeom.Tokens.inherited if shot == "subject" else UsdGeom.Tokens.invisible
            )
            simulation_app.update()
        rep.orchestrator.step(rt_subframes=args.rt_subframes)
        path = os.path.join(args.out, f"{name}_{shot}.png")
        Image.fromarray(np.asarray(annotator.get_data())[..., :3]).save(path)
        written.append(os.path.relpath(path, REPO))
        print(f"[gallery]   wrote {path}")

    manifest.append(
        {
            "scenario": name,
            "caption": caption,
            "conditions": conditions.describe(),
            "sun_elevation_deg": round(conditions.sun.elevation_deg, 2),
            "moon_elevation_deg": round(conditions.moon.elevation_deg, 2),
            "moon_illuminated": round(conditions.moon.illuminated_fraction, 3),
            "turbidity": round(conditions.turbidity, 2),
            "cloud_cover": (
                None if conditions.cloud is None else round(conditions.cloud.measured_cover, 3)
            ),
            "surface_weather": series.describe(),
            "images": written,
            "state": wx.state.to_dict(),
        }
    )

# Merge rather than overwrite: a run of two scenarios must not leave a manifest claiming the
# gallery is two frames. Keyed by scenario, newest wins, and the order of SCENARIOS is kept so the
# file reads in the same order as the README.
manifest_path = os.path.join(args.out, "gallery.json")
merged = {}
if os.path.exists(manifest_path):
    with open(manifest_path, encoding="utf-8") as handle:
        for entry in json.load(handle):
            merged[entry["scenario"]] = entry
for entry in manifest:
    merged[entry["scenario"]] = entry
ordered = [merged[name] for name in SCENARIOS if name in merged]
with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(ordered, handle, indent=2)
print(f"[gallery] wrote {manifest_path} ({len(ordered)} scenarios)")

weather.shutdown_controller()
simulation_app.close()
