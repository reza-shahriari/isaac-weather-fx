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
    assert np.any(cumulus.density(x, cumulus.base_m + 150.0, x) > 0.2)


def test_the_cover_asked_for_is_the_cover_measured() -> None:
    for cover in (0.15, 0.35, 0.6):
        layer = C.Cloudscape(cover=cover, base_m=1000.0, profile=C.CLOUDSCAPE_TYPES["cumulus"], seed=2, **SMALL)
        assert layer.measured_cover(192) == pytest.approx(cover, abs=0.04)


def test_a_cumulus_has_a_flat_base_and_narrows_toward_its_top(cumulus: C.Cloudscape) -> None:
    """Occupied area at 5 % of the depth is near its maximum; at 80 % it is a fraction of it."""
    axis = np.linspace(0.0, cumulus.weather_tile_m, 256, endpoint=False)
    x, z = np.meshgrid(axis, axis, indexing="xy")
    d = cumulus.profile.thickness_m
    area = [float((cumulus.density(x, cumulus.base_m + f * d, z) > 0.05).mean()) for f in (0.02, 0.06, 0.4, 0.8)]
    assert area[1] > 0.8 * max(area)
    assert area[3] < 0.35 * area[1]
    assert area[0] < area[1]


def test_the_cloud_is_thin_at_its_base_dense_at_its_top_and_nowhere_clipped_flat(cumulus: C.Cloudscape) -> None:
    """Liquid water grows with height above the base (extinction as its two-thirds power), so
    the densest cloud at 40 % of the layer's depth holds 1.7 times the densest at 8 %; and no
    part of the cloud sits on a ceiling, which is what made the earlier function a uniform solid
    with a blurred skin."""
    axis = np.linspace(0.0, cumulus.weather_tile_m, 384, endpoint=False)
    x, z = np.meshgrid(axis, axis, indexing="xy")
    p = cumulus.profile
    d = p.thickness_m
    low = cumulus.density(x, cumulus.base_m + 0.08 * d, z)
    high = cumulus.density(x, cumulus.base_m + 0.40 * d, z)
    water = lambda f: p.water_base + (1.0 - p.water_base) * f ** (2.0 / 3.0)  # noqa: E731
    assert high.max() == pytest.approx(water(0.40), rel=0.03)   # a crisp top reaches the body's value
    assert low.max() <= water(0.08) + 1e-9
    assert low.max() < 0.75 * high.max()
    cloudy = np.concatenate([low[low > 0.01], high[high > 0.01]])
    assert cloudy.size > 500
    assert (cloudy >= 0.999).mean() == 0.0
    # Graded: the middle half of the cloudy values spans a real range, not one number.
    q25, q75 = np.percentile(cloudy, [25, 75])
    assert q75 - q25 > 0.1


def _skin_m(layer: C.Cloudscape, fraction: float) -> float:
    """Median distance, along horizontal lines at one height, from a cloud's outline to where
    its density reaches half of what the body holds at that height."""
    p = layer.profile
    y = layer.base_m + fraction * p.thickness_m
    step = 2.0
    x = np.arange(0.0, layer.weather_tile_m, step)
    widths = []
    for z in np.linspace(0.0, layer.weather_tile_m, 48, endpoint=False):
        rho = layer.density(x, y, z)
        inside = rho > 0.01
        enters = np.flatnonzero(inside[1:] & ~inside[:-1]) + 1
        for i in enters:
            run = rho[i:i + 400]
            body = run.max()
            if body < 0.15:
                continue
            widths.append(float(np.argmax(run >= 0.5 * body)) * step)
    assert len(widths) > 20
    return float(np.median(widths))


def test_the_top_is_crisp_and_the_base_is_soft(cumulus: C.Cloudscape) -> None:
    """A rising cumulus top ends within metres; its base and its lower flanks are ragged. With
    the test's coarse textures the numbers are loose, the order is not."""
    high, low = _skin_m(cumulus, 0.40), _skin_m(cumulus, 0.04)
    print(f"skin: {high:.0f} m at 40 % of the depth, {low:.0f} m at 4 %")
    assert high < 0.5 * low
    assert high < 40.0


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




@pytest.fixture(scope="module")
def cirrus() -> C.Cloudscape:
    return C.Cloudscape(cover=0.4, base_m=9000.0, profile=C.CLOUDSCAPE_TYPES["cirrus"], seed=4, **SMALL)


def test_a_cirrus_is_high_thin_and_streaked_along_the_wind(cirrus: C.Cloudscape) -> None:
    """The ice genus: a base far above any condensation level of surface air, a median cloudy
    column near unit visible optical depth (the sky shows through, where a cumulus is opaque
    within its first hundred metres), and streaks -- the column depth stays correlated several
    times further along x, the wind's axis, than across it."""
    from weather_fx.core.clouds import lifting_condensation_level_m
    from weather_fx.core.state import WeatherState

    assert cirrus.base_m == 9000.0 and cirrus.top_m == 10_500.0
    assert 0.3 < cirrus.optical_depth < 3.0
    cumulus = C.CLOUDSCAPE_TYPES["cumulus"]
    assert cumulus.convective and not cirrus.profile.convective
    n = 160
    axis = (np.arange(n) + 0.5) / n * cirrus.weather_tile_m
    x, z = np.meshgrid(axis, axis, indexing="xy")
    tau = cirrus._column_optical_depth(x, z, cirrus.coverage_bias, 10)
    assert (tau > 0.1).mean() > 0.2

    def correlation(lag: int, axis: int) -> float:
        a = np.moveaxis(tau, axis, 0)
        return float(np.corrcoef(a[:-lag].ravel(), a[lag:].ravel())[0, 1])

    along_x, across = correlation(6, 1), correlation(6, 0)
    assert along_x > across + 0.2, (along_x, across)
    # From a state: the base defaults to the genus's own level and the mixed layer's top is the
    # LCL the surface air would have condensed at, which a thermal band lapses dry to.
    state = WeatherState().with_updates("clouds", enabled=True, cover=0.3, genus="cirrus", seed=4,
                                        base_m=0.0, temperature_c=20.0, dewpoint_c=10.0)
    built = C.cloudscape_from_state(state)
    assert built is not None and built.base_m == 9000.0
    assert built.mixed_layer_top_m == pytest.approx(lifting_condensation_level_m(20.0, 10.0))
    warm = C.cloudscape_from_state(state.with_updates("clouds", genus="stratus"))
    assert warm is not None and warm.mixed_layer_top_m is None
    assert warm.base_m == pytest.approx(lifting_condensation_level_m(20.0, 10.0))


def test_the_gpu_kernel_is_the_numpy_function_for_a_streaked_genus(cirrus: C.Cloudscape) -> None:
    """The stretch is in both evaluations: the kernel follows the numpy cirrus as it does the
    cumulus, so an infrared march on either side reads one streaked cloud."""
    from weather_fx.gpu import cloud_march

    wp = cloud_march.wp
    if wp.get_cuda_device_count() == 0:
        pytest.skip("no CUDA device")
    renderer = cloud_march.CloudRenderer(cirrus, device="cuda:0")
    rng = np.random.default_rng(5)
    n = 20_000
    x = rng.uniform(-60_000.0, 60_000.0, n)
    z = rng.uniform(-30_000.0, 30_000.0, n)
    y = rng.uniform(cirrus.base_m - 50.0, cirrus.top_m + 50.0, n)
    gpu, ref = renderer.density(x, y, z), cirrus.density(x, y, z)
    assert (ref > 0.0).mean() > 0.05
    error = np.abs(gpu - ref)
    assert np.percentile(error, 99) < 0.02 and error.mean() < 2e-3


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
    # CUDA's texture units interpolate with 9-bit fixed-point weights (1/256 of a texel). Where
    # the skin is crisp the density climbs to the body's value within a few hundredths of the
    # eroded shape, so at a point on a cloud's top that rounding is a visible fraction of it: the
    # worst point is loose, the field as a whole is not.
    error = np.abs(gpu - ref)
    assert error.max() < 0.25
    assert np.percentile(error, 99) < 0.02
    assert error.mean() < 2e-3


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


# --- the tables the GPU composite samples (core.layer_tables) ------------------------------------


def _noon():
    from weather_fx.core import sky as S
    from weather_fx.core.state import WeatherState

    state = WeatherState().with_updates("sky", enabled=True, hour_utc=11.0)
    return S.conditions_from_state(state, build_cloud=False)


def test_the_row_packing_inverts_and_puts_the_horizon_in_the_middle() -> None:
    from weather_fx.core import layer_tables as T

    v = np.linspace(0.0, 1.0, 41)
    assert np.allclose(T.row_of_elevation(T.elevation_of_row(v)), v)
    assert T.elevation_of_row(0.5) == pytest.approx(0.0)
    # Half the rows lie within 22.5 degrees of the horizon.
    assert np.degrees(T.elevation_of_row(0.75)) == pytest.approx(22.5)


def test_the_sky_table_is_the_sky_model_at_its_texel_centres() -> None:
    from weather_fx.core import layer_tables as T
    from weather_fx.core import sky as S

    conditions = _noon()
    table = T.sky_table(conditions, rows=32)
    assert table.shape == (32, 64, 4) and table.dtype == np.float32
    j, i = 22, 9
    el = float(T.elevation_of_row((j + 0.5) / 32))
    az = (i + 0.5) / 64 * 2.0 * np.pi
    d = np.array([[np.cos(el) * np.sin(az), np.sin(el), -np.cos(el) * np.cos(az)]])
    expected = S.sky_radiance_rgb(d, conditions)[0] / conditions.exposure_scale
    assert np.allclose(table[j, i, :3], expected, rtol=1e-5)


def test_the_air_is_clear_up_close_and_the_sky_far_away() -> None:
    """At the near slice the air transmits nearly everything and adds nearly nothing; at the far
    one, near the horizon, it has become most of the sky's own light."""
    from weather_fx.core import layer_tables as T

    conditions = _noon()
    inscatter, transmittance = T.air_tables(conditions, elevations=16, azimuths=8, distances=8)
    sky = T.sky_table(conditions, rows=16)
    low = 9                                   # the second row above the horizon
    # The nearest slice is 700 m of low air: about a tenth of the light is already lost in blue.
    assert transmittance[0, low, :, :3].min() > 0.85
    assert inscatter[0, low, :, 1].max() < 0.15 * sky[low, :, 1].mean()
    assert transmittance[-1, low, :, 1].max() < 0.5
    assert inscatter[-1, low, :, 1].mean() > 0.4 * sky[low, :, 1].mean()


def test_pixel_is_a_render_path_for_the_genera_it_knows() -> None:
    from weather_fx.backends.viewport.render_mode import choose_cloud_path

    assert choose_cloud_path("pixel", "RaytracedLighting", "cumulus") == "pixel"
    assert choose_cloud_path("pixel", "PathTracing", "stratocumulus") == "pixel"
    # Every genus the state offers has a profile, cirrus since the infrared asked for it.
    for genus in ("cumulus", "congestus", "stratocumulus", "stratus", "storm", "cirrus"):
        assert C.supports(genus)
        assert choose_cloud_path("pixel", "PathTracing", genus) == "pixel"
    assert choose_cloud_path("pixel", "RaytracedLighting", "nimbostratus") == "dome"


@pytest.mark.skipif(not __import__("weather_fx.gpu", fromlist=["warp_available"]).warp_available(),
                    reason="no Warp")
def test_the_gpu_composite_is_the_sky_where_there_is_no_cloud(cumulus: C.Cloudscape) -> None:
    """A camera looking where the layer is empty gets the sky table back, pixel for pixel, times
    the gains and the scale -- the composite adds nothing of its own."""
    from weather_fx.core import layer_tables as T
    from weather_fx.gpu import cloud_march

    if cloud_march.wp.get_cuda_device_count() == 0:
        pytest.skip("no CUDA device")
    conditions = _noon()
    light = T.lighting_for(conditions)
    a_in, a_tr = T.air_tables(conditions)
    tables = cloud_march.SkyTables(T.sky_table(conditions, rows=128), a_in, a_tr, light["azimuth_rad"],
                                   T.AIR_NEAR_M, T.AIR_FAR_M, "cuda:0")
    lighting = cloud_march.Lighting(light["sun_direction"], light["sun_rgb"], light["above_rgb"], light["below_rgb"])
    empty = C.Cloudscape(cover=0.0, base_m=1200.0, profile=C.CLOUDSCAPE_TYPES["cumulus"], seed=1, **SMALL)
    renderer = cloud_march.CloudRenderer(empty, device="cuda:0")
    camera = cloud_march.Camera(96, 54, 60.0, 35.0, 20.0)
    out = renderer.render_layer(camera, lighting, tables, gains=(1.0, 0.5, 2.0), scale=3.0).numpy()
    d = cloud_march.camera_rays(camera)
    from weather_fx.core import sky as S

    sky = S.sky_radiance_rgb(d, conditions) / conditions.exposure_scale * np.array([1.0, 0.5, 2.0]) * 3.0
    assert np.allclose(out[..., :3], sky, rtol=0.03)
    assert np.all(out[..., 3] == 1.0)
    # What another band's march is compared against: the frame's transmittance, all 1 here.
    assert renderer.last_transmittance().shape == (54, 96)
    assert np.all(renderer.last_transmittance() == 1.0)
    cloudy = cloud_march.CloudRenderer(cumulus, device="cuda:0")
    cloudy.render_layer(cloud_march.Camera(96, 54, 60.0, 35.0, 20.0), lighting, tables)
    trans = cloudy.last_transmittance()
    assert trans.shape == (54, 96) and trans.min() >= 0.0 and trans.max() <= 1.0
    assert (trans < 0.5).mean() > 0.05


def test_a_cloudscape_offers_what_another_sensors_march_reads(cumulus: C.Cloudscape) -> None:
    """The members an infrared march sizes and bounds itself by: the layer's span on a ray, the
    finest pitch, the thickness and the median cloudy column's optical depth."""
    assert cumulus.thickness_m == cumulus.top_m - cumulus.base_m
    assert 1.0 < cumulus.finest_pitch_m < 200.0
    assert cumulus.optical_depth > C.CLOUDY_OPTICAL_DEPTH
    up = np.array([0.0, 1.0, 0.0])
    near, far = cumulus.slab_span(np.zeros(3), up)
    assert float(near) == pytest.approx(cumulus.base_m) and float(far) == pytest.approx(cumulus.top_m)
    inside = np.array([0.0, cumulus.base_m + 100.0, 0.0])
    near, far = cumulus.slab_span(inside, up)
    assert float(near) == 0.0 and float(far) == pytest.approx(cumulus.thickness_m - 100.0)
    near, far = cumulus.slab_span(np.array([0.0, cumulus.top_m + 10.0, 0.0]), up)
    assert float(far) <= float(near)
