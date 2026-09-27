"""Check the sky, the lights and the clouds in a running Isaac Sim. Paste into Window > Script Editor.

It walks through the checks below and prints what it found at each one. Watch the viewport while it
runs; every step says what you should be seeing.

1. one sun: the stage's own lights are hidden while the sky is on;
2. the path tracer: the clouds become volumes under /WeatherFX/Clouds;
3. a colour and a density change: the volumes update without a rebuild;
4. real-time: the volumes go away and the clouds are painted into the dome, baked in the background;
5. clean-up: switching the sky off puts the stage's lights and render settings back.
"""
import asyncio

import carb.settings
import omni.kit.app
import omni.usd
from pxr import UsdGeom, UsdLux

from weather_fx import api as weather

wx = weather.get_controller()
settings = carb.settings.get_settings()
app = omni.kit.app.get_app()


def foreign_lights():
    stage = omni.usd.get_context().get_stage()
    out = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdLux.LightAPI) and not str(prim.GetPath()).startswith("/WeatherFX"):
            visible = UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
            out.append((str(prim.GetPath()), "visible" if visible else "hidden"))
    return out


async def wait_until(predicate, seconds, label):
    for _ in range(int(seconds * 30)):
        if predicate():
            return True
        await app.next_update_async()
    print(f"   !! timed out waiting for {label}")
    return False


def viewport_stats():
    return wx.stats().get("viewport", {})


async def main():
    original_mode = settings.get("/rtx/rendermode")
    print("render mode at start:", original_mode)

    print("\n1. sky on, clouds on")
    wx.configure(general={"enabled": True, "time_source": "wall"},
                 sky={"enabled": True, "hour_utc": 9.5, "hide_scene_lights": True},
                 clouds={"enabled": True, "cover": 0.45, "genus": "cumulus", "render_path": "auto"})
    for _ in range(5):
        await app.next_update_async()
    print("   stage lights:", foreign_lights() or "none found")
    print("   -> expect every one of them 'hidden', and a single sun in the viewport")

    print("\n2. path tracer: clouds as volumes")
    settings.set("/rtx/rendermode", "PathTracing")
    ok = await wait_until(lambda: viewport_stats().get("cloud_volume", {}).get("tiles", 0) > 0,
                          120, "the cloud volumes")
    print("   cloud_volume:", viewport_stats().get("cloud_volume"))
    stage = omni.usd.get_context().get_stage()
    print("   /WeatherFX/Clouds exists:", bool(stage.GetPrimAtPath("/WeatherFX/Clouds")))
    print("   ptvol enabled:", settings.get("/rtx/pathtracing/ptvol/enabled"),
          " maxBounces:", settings.get("/rtx/pathtracing/maxBounces"))
    if ok:
        print("   -> expect 3D clouds with lit tops and dark bases; orbit the camera for parallax")
        print("      (path tracing needs a few seconds of samples to clean up)")

    print("\n3. colour and density: instant, no rebuild")
    wx.set_clouds(lit_color=(1.0, 0.75, 0.6), density_scale=2.0)
    await app.next_update_async()
    print("   building after the change:", viewport_stats().get("cloud_volume", {}).get("building"))
    print("   -> expect False, and warmer, denser clouds")
    await asyncio.sleep(3.0)
    wx.set_clouds(lit_color=(1.0, 1.0, 1.0), density_scale=1.0)

    print("\n4. real-time: clouds go into the dome")
    settings.set("/rtx/rendermode", "RaytracedLighting")
    await wait_until(lambda: viewport_stats().get("cloud_volume", {}).get("tiles", 1) == 0,
                     10, "the volumes to be removed")
    print("   cloud_volume:", viewport_stats().get("cloud_volume"))
    print("   sky baking in the background:", viewport_stats().get("sky", {}).get("baking"))
    await wait_until(lambda: not viewport_stats().get("sky", {}).get("baking", False),
                     180, "the dome bake")
    print("   sky:", {k: viewport_stats()["sky"].get(k)
                      for k in ("cloud_cover_in_dome", "texture", "sun_elevation_deg")})
    print("   -> expect the app to have stayed responsive while it baked, then clouds in the sky")

    print("\n5. sky off: everything restored")
    wx.set_sky(enabled=False)
    wx.set_clouds(enabled=False)
    for _ in range(5):
        await app.next_update_async()
    print("   stage lights:", foreign_lights() or "none found")
    print("   ptvol enabled:", settings.get("/rtx/pathtracing/ptvol/enabled"))
    settings.set("/rtx/rendermode", original_mode)
    print("\ndone -- please send this output, and a screenshot from steps 2 and 4")


asyncio.ensure_future(main())
