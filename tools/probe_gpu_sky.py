"""Probe: can this Isaac Sim run a GPU cloud march and feed a dome light from memory?

Paste into Window > Script Editor and watch the viewport for about ten seconds, first in
RTX - Real-Time, then again in RTX - Interactive (path tracing). Send back everything it prints and
what the sky did.

It checks the two things an Unreal-style cloud renderer needs:

1. Warp: a kernel over a 2048 x 1024 image on the GPU, timed.
2. A dome light whose texture is ``dynamic://...``, filled from memory with float data and
   updated every frame (no EXR files). If this works you will see the sky slowly cycle through
   colours; if it does not, the sky stays black or uniform.

Everything goes under /WeatherFX_Probe in the session layer and is removed at the end.
"""
import asyncio
import time

import numpy as np
import omni.kit.app
import omni.usd
from pxr import Sdf, Usd, UsdLux

W, H = 2048, 1024
NAME = "weather_fx_probe_sky"
app = omni.kit.app.get_app()


def probe_warp():
    try:
        import warp as wp
    except ImportError as exc:
        print("warp: NOT importable:", exc)
        return None
    wp.init()
    device = wp.get_preferred_device()
    print("warp:", wp.config.version, "device:", device)

    @wp.kernel
    def gradient(out: wp.array2d(dtype=wp.vec4), t: float):
        i, j = wp.tid()
        u = float(j) / float(W)
        v = float(i) / float(H)
        out[i, j] = wp.vec4(0.5 + 0.5 * wp.sin(6.28 * u + t), v, 0.5 + 0.5 * wp.cos(t), 1.0)

    image = wp.zeros((H, W), dtype=wp.vec4, device=device)
    wp.launch(gradient, dim=(H, W), inputs=[image, 0.0], device=device)
    wp.synchronize()
    start = time.perf_counter()
    for k in range(20):
        wp.launch(gradient, dim=(H, W), inputs=[image, float(k)], device=device)
    wp.synchronize()
    print(f"warp: 2048x1024 kernel {1000 * (time.perf_counter() - start) / 20:.3f} ms per launch")
    return wp, gradient, image


def make_provider():
    import omni.ui as ui

    provider = ui.DynamicTextureProvider(NAME)
    formats = [f for f in ("RGBA32_SFLOAT", "RGBA16_SFLOAT") if hasattr(ui.TextureFormat, f)]
    print("omni.ui float texture formats:", formats or "none")
    methods = [m for m in ("set_data_array", "set_bytes_data", "set_bytes_data_from_gpu")
               if hasattr(provider, m)]
    print("DynamicTextureProvider methods:", methods)
    return ui, provider, formats


def upload(ui, provider, formats, rgba):
    """Try the float paths first; report which one worked."""
    fmt = getattr(ui.TextureFormat, formats[0]) if formats else ui.TextureFormat.RGBA8_UNORM
    if formats and hasattr(provider, "set_data_array"):
        provider.set_data_array(rgba.astype(np.float32), [W, H], fmt)
        return "set_data_array/" + formats[0]
    if formats:
        provider.set_bytes_data(bytearray(rgba.astype(np.float32).tobytes()), [W, H], fmt)
        return "set_bytes_data/" + formats[0]
    data = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    provider.set_bytes_data(bytearray(data.tobytes()), [W, H], ui.TextureFormat.RGBA8_UNORM)
    return "set_bytes_data/RGBA8 (no float format)"


async def main():
    try:  # two dome lights at once would muddle what the probe shows
        from weather_fx import api as weather

        weather.get_controller().set_sky(enabled=False)
        weather.get_controller().set_clouds(enabled=False)
    except Exception:
        pass
    warp = probe_warp()
    ui, provider, formats = make_provider()

    stage = omni.usd.get_context().get_stage()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        dome = UsdLux.DomeLight.Define(stage, "/WeatherFX_Probe/Dome")
        dome.CreateTextureFileAttr().Set(Sdf.AssetPath(f"dynamic://{NAME}"))
        dome.CreateTextureFormatAttr().Set(UsdLux.Tokens.latlong)
        dome.CreateIntensityAttr(1.0)
        dome.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(True)

    v = np.linspace(0.0, 1.0, H)[:, None]
    u = np.linspace(0.0, 1.0, W)[None, :]
    path = None
    start = time.perf_counter()
    frames = 0
    while time.perf_counter() - start < 10.0:
        t = time.perf_counter() - start
        rgba = np.empty((H, W, 4), dtype=np.float32)
        rgba[..., 0] = 0.5 + 0.5 * np.sin(6.28 * u + t)
        rgba[..., 1] = v
        rgba[..., 2] = 0.5 + 0.5 * np.cos(t)
        rgba[..., 3] = 1.0
        try:
            path = upload(ui, provider, formats, rgba * 4.0)
        except Exception as exc:  # report and stop
            print("upload FAILED:", repr(exc))
            break
        frames += 1
        await app.next_update_async()
    print(f"uploaded {frames} frames via {path}; did the sky cycle through colours?")

    if warp is not None and hasattr(provider, "set_bytes_data_from_gpu") and formats:
        wp, gradient, image = warp
        try:
            wp.launch(gradient, dim=(H, W), inputs=[image, 1.0])
            wp.synchronize()
            provider.set_bytes_data_from_gpu(image.ptr, [W, H],
                                             getattr(ui.TextureFormat, "RGBA32_SFLOAT"), W * 16)
            print("set_bytes_data_from_gpu: accepted (zero-copy GPU path available)")
        except Exception as exc:
            print("set_bytes_data_from_gpu: FAILED:", repr(exc))
        for _ in range(60):
            await app.next_update_async()

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        stage.RemovePrim("/WeatherFX_Probe")
    print("probe done; /WeatherFX_Probe removed")


asyncio.ensure_future(main())
