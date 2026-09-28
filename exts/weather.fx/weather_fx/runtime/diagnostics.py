"""What the extension has done to the stage and the renderer, in one report.

When a viewport shows something unexpected -- a black real-time frame, a missing sun -- the cause
is in state nobody can see from the panel: which renderer the extension thinks is running, which
prims it authored and whether they are visible, whether the dome's texture file exists, and what
the RTX fog settings actually hold. :func:`diagnose` gathers all of it so it can be pasted.
"""
from __future__ import annotations

import os
from typing import Any, Dict


def diagnose(manager: Any) -> Dict[str, Any]:
    import carb.settings
    import omni.usd
    from pxr import Usd, UsdGeom, UsdLux

    from ..backends.viewport.render_mode import (
        RENDER_MODE_SETTING,
        choose_cloud_path,
        current_render_mode,
    )
    from ..backends.viewport.rtx_settings import FOG_SETTINGS

    settings = carb.settings.get_settings()
    state = manager.state_ref
    stage = omni.usd.get_context().get_stage()
    mode = current_render_mode()
    report: Dict[str, Any] = {
        "render_mode_detected": mode,
        "render_mode_setting": settings.get(RENDER_MODE_SETTING),
        "cloud_path": choose_cloud_path(state.clouds.render_path, mode),
        "general": {"enabled": state.general.enabled, "time_source": state.general.time_source},
        "sky": {"enabled": state.sky.enabled, "hour_utc": state.sky.hour_utc,
                "aerial_perspective": state.sky.aerial_perspective},
        "clouds": {"enabled": state.clouds.enabled, "render_path": state.clouds.render_path},
        "fog_enabled": state.fog.enabled,
        "meters_per_unit": manager.context.meters_per_unit(),
        "up_axis": "Z" if manager.context.up_axis() == 2 else "Y",
        "sky_horizon_rgb": manager.context.sky_horizon_rgb,
    }

    prims = []
    if stage is not None and stage.GetPrimAtPath("/WeatherFX"):
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/WeatherFX")):
            entry = {"path": str(prim.GetPath()), "type": prim.GetTypeName()}
            if prim.IsA(UsdGeom.Imageable):
                entry["visible"] = (UsdGeom.Imageable(prim).ComputeVisibility()
                                    != UsdGeom.Tokens.invisible)
            if prim.HasAPI(UsdLux.LightAPI):
                light = UsdLux.LightAPI(prim)
                entry["intensity"] = light.GetIntensityAttr().Get()
                entry["color"] = tuple(light.GetColorAttr().Get() or ())
            if prim.IsA(UsdLux.DomeLight):
                asset = UsdLux.DomeLight(prim).GetTextureFileAttr().Get()
                path = asset.path if asset else ""
                entry["texture"] = path
                entry["texture_exists"] = bool(path) and os.path.isfile(path)
            prims.append(entry)
    report["weatherfx_prims"] = prims

    foreign = []
    if stage is not None:
        for prim in stage.Traverse():
            if prim.HasAPI(UsdLux.LightAPI) and not str(prim.GetPath()).startswith("/WeatherFX"):
                foreign.append({
                    "path": str(prim.GetPath()),
                    "visible": UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible,
                })
    report["stage_lights_outside_weatherfx"] = foreign
    report["rtx_fog"] = {key: settings.get(path) for key, path in FOG_SETTINGS.items()}
    report["path_tracer_volume_settings"] = {
        path: settings.get(path)
        for path in ("/rtx/pathtracing/ptvol/enabled", "/rtx/pathtracing/maxBounces")
    }
    report["stats"] = manager.stats()
    return report


def format_report(report: Dict[str, Any], indent: int = 0) -> str:
    lines = []
    pad = "  " * indent
    for key, value in report.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(format_report(value, indent + 1))
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            lines.append(f"{pad}{key}:")
            for item in value:
                lines.append(f"{pad}  - " + ", ".join(f"{k}={v}" for k, v in item.items()))
        else:
            lines.append(f"{pad}{key}: {value}")
    return "\n".join(lines)
