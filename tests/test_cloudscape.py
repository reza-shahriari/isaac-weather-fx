"""The cloud layer as a function (``core.cloudscape``), and the GPU march that evaluates it.

The numpy definition is the reference. The GPU kernel repeats its arithmetic on the same
textures, and the test that matters most here is the one holding the two together: a layer the
infrared march reads on the CPU must be the layer the camera sees on the GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from weather_fx.core import cloudscape as C

SMALL = dict(weather_cells=128, shape_cells=48, detail_cells=24)


@pytest.fixture(scope="module")
def cumulus() -> C.Cloudscape:
    return C.Cloudscape(cover=0.35, base_m=1200.0, profile=C.CLOUDSCAPE_TYPES["cumulus"], seed=5, **SMALL)


def test_sample_wrapped_is_a_gpu_sampler() -> None:
    """Texel centres at (i + 0.5) / n, linear between them, wrapping at both ends."""
    tex = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float64)  # (v, u): one row of four
    assert C.sample_wrapped(tex, 0.125, 0.5) == pytest.approx(0.0)
    assert C.sample_wrapped(tex, 0.375, 0.5) == pytest.approx(1.0)
    assert C.sample_wrapped(tex, 0.25, 0.5) == pytest.approx(0.5)
    assert C.sample_wrapped(tex, 0.0, 0.5) == pytest.approx(1.5)   # halfway between 3 and 0
    assert C.sample_wrapped(tex, 1.125, 0.5) == pytest.approx(0.0)  # wraps


def test_the_layer_is_empty_outside_its_base_and_top(cumulus: C.Cloudscape) -> None:
    x = np.linspace(-20_000.0, 20_000.0, 300)
    assert not np.any(cumulus.density(x, cumulus.base_m - 1.0, x))
    assert not np.any(cumulus.density(x, cumulus.top_m + 1.0, x))
    assert np.any(cumulus.density(x, cumulus.base_m + 150.0, x) > 0.5)


def test_the_cover_asked_for_is_the_cover_measured() -> None:
    for cover in (0.15, 0.35, 0.6):
        layer = C.Cloudscape(cover=cover, base_m=1000.0, profile=C.CLOUDSCAPE_TYPES["cumulus"], seed=2, **SMALL)
        assert layer.measured_cover(192) == pytest.approx(cover, abs=0.04)


def test_a_cumulus_has_a_flat_base_and_narrows_toward_its_top(cumulus: C.Cloudscape) -> None:
    """Occupied area at 5 % of the depth is near its maximum; at 80 % it is a fraction of it."""
    axis = np.linspace(0.0, cumulus.weather_tile_m, 256, endpoint=False)
    x, z = np.meshgrid(axis, axis, indexing="xy")
    d = cumulus.profile.thickness_m
    area = [float((cumulus.density(x, cumulus.base_m + f * d, z) > 0.5).mean()) for f in (0.02, 0.06, 0.4, 0.8)]
    assert area[1] > 0.8 * max(area)
    assert area[3] < 0.35 * area[1]
    assert area[0] < area[1]


def test_the_layer_wraps_with_the_weather_map(cumulus: C.Cloudscape) -> None:
    tile = cumulus.weather_tile_m
    pts = np.random.default_rng(0).uniform(0.0, tile, (200, 2))
    y = cumulus.base_m + 200.0
    a = cumulus.density(pts[:, 0], y, pts[:, 1])
    # The shape and detail tiles divide the weather tile only by luck, so compare where all wrap.
    period = np.lcm.reduce([int(round(tile)), int(round(cumulus.profile.shape_tile_m)),
                            int(round(cumulus.profile.detail_tile_m))])
    b = cumulus.density(pts[:, 0] + period, y, pts[:, 1] - period)
    assert np.allclose(a, b, atol=1e-6)


@pytest.mark.skipif(not __import__("weather_fx.gpu", fromlist=["warp_available"]).warp_available(),
                    reason="no Warp")
def test_the_gpu_kernel_is_the_numpy_function(cumulus: C.Cloudscape) -> None:
    """Twenty thousand random points through the layer: the kernel's density is the numpy
    density to within float32 texture interpolation. This is the contract that lets an infrared
    march on the CPU and the camera's march on the GPU read one cloud."""
    from weather_fx.gpu import cloud_march

    wp = cloud_march.wp
    if wp.get_cuda_device_count() == 0:
        pytest.skip("no CUDA device")
    renderer = cloud_march.CloudRenderer(cumulus, device="cuda:0")
    rng = np.random.default_rng(3)
    n = 20_000
    x = rng.uniform(-30_000.0, 30_000.0, n)
    z = rng.uniform(-30_000.0, 30_000.0, n)
    y = rng.uniform(cumulus.base_m - 50.0, cumulus.top_m + 50.0, n)
    gpu = renderer.density(x, y, z)
    ref = cumulus.density(x, y, z)
    assert (ref > 0.0).mean() > 0.05
    # CUDA's texture units interpolate with 9-bit fixed-point weights (1/256 of a texel), and the
    # remaps and the gain of 3 steepen that into a few hundredths at the worst point.
    assert np.abs(gpu - ref).max() < 0.05
    assert np.abs(gpu - ref).mean() < 1e-3


@pytest.mark.skipif(not __import__("weather_fx.gpu", fromlist=["warp_available"]).warp_available(),
                    reason="no Warp")
def test_a_white_cloud_under_a_uniform_sky_is_invisible(cumulus: C.Cloudscape) -> None:
    """The furnace, on the GPU march: no sun, the same radiance above and below, and every pixel
    composes back to that radiance within the droplets' absorption and the march's step error."""
    from weather_fx.gpu import cloud_march

    if cloud_march.wp.get_cuda_device_count() == 0:
        pytest.skip("no CUDA device")
    renderer = cloud_march.CloudRenderer(cumulus, device="cuda:0")
    camera = cloud_march.Camera(192, 108, 70.0, 30.0, 45.0)
    light = cloud_march.Lighting((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1000.0,) * 3, (1000.0,) * 3)
    scattered, transmittance, _ = renderer.render(camera, light)
    out = scattered[..., 0] + transmittance * 1000.0
    cloudy = transmittance < 0.5
    assert cloudy.mean() > 0.1
    assert np.percentile(np.abs(out[cloudy] / 1000.0 - 1.0), 99) < 0.02
