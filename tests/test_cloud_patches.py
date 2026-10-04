"""Simulated cloud patches as the cloudscape's cloud (``core.cloudscape``).

A patch is a density grid a fluid solver grew; the cloudscape places one in each lattice cell the
weather map keeps. The numpy placement is the reference an infrared march reads, the GPU kernel
repeats it, and these tests hold what the placement promises: the cloud stands on one base, the
cover asked for is the cover seen, the same state gives the same sky, and both evaluations agree.
"""
from __future__ import annotations

import numpy as np
import pytest

from weather_fx.core import cloudscape as C

SMALL = dict(weather_cells=128, shape_cells=32, detail_cells=24)


def _blob_patch(name: str, seed: int, shape=(44, 24, 40), voxel_m: float = 50.0) -> C.CloudPatch:
    """A few soft lumps standing on the patch's floor: a stand-in for a simulated patch."""
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(*(np.arange(n) + 0.5 for n in shape), indexing="ij")
    density = np.zeros(shape)
    for _ in range(4):
        cx, cz = rng.uniform(12, shape[0] - 12), rng.uniform(12, shape[2] - 12)
        r, top = rng.uniform(5, 9), rng.uniform(10, 20)
        density = np.maximum(density, np.clip(1.6 - np.hypot(x - cx, z - cz) / r, 0, 1) * (y < top))
    return C.CloudPatch(name, (density / density.max()).astype(np.float32), voxel_m)


@pytest.fixture(scope="module")
def patches():
    return [_blob_patch("a", 1), _blob_patch("b", 2), _blob_patch("c", 3)]


@pytest.fixture(scope="module")
def sky(patches) -> C.Cloudscape:
    return C.Cloudscape(cover=0.3, base_m=1200.0, profile=C.CLOUDSCAPE_TYPES["cumulus"], seed=5,
                        patches=patches, **SMALL)


def test_every_cloud_stands_on_the_one_base(sky: C.Cloudscape) -> None:
    """Nothing below the base or above the layer; cloud in the first cell above the base in a
    good share of the columns that hold any: the condensation level is one height for the field."""
    axis = np.linspace(0.0, sky.weather_tile_m, 400, endpoint=False)
    x, z = np.meshgrid(axis, axis, indexing="xy")
    assert not np.any(sky.density(x, sky.base_m - 1.0, z))
    assert not np.any(sky.density(x, sky.top_m + 1.0, z))
    floor = sky.density(x, sky.base_m + 15.0, z) > 0.0
    column = np.zeros(x.shape, dtype=bool)
    for h in np.linspace(15.0, 1100.0, 12):
        column |= sky.density(x, sky.base_m + h, z) > 0.0
    assert column.mean() > 0.02
    assert (floor & column).sum() > 0.6 * column.sum()


def test_the_cover_asked_for_is_the_cover_an_observer_sees(patches) -> None:
    """The solve aims the columns seen from straight below at the plan-view fraction of the
    cover, and more cover asked is more cloud. Beyond what two full lattices hold it saturates."""
    seen = []
    for cover in (0.1, 0.2, 0.3):
        layer = C.Cloudscape(cover=cover, base_m=1000.0, profile=C.CLOUDSCAPE_TYPES["cumulus"], seed=2,
                             patches=patches, **SMALL)
        seen.append(layer.measured_cover(192))
        if layer.patch_cover < 1.99:
            assert seen[-1] == pytest.approx(C.PATCH_PLAN_VIEW_FRACTION * cover, abs=0.03)
    assert seen[0] < seen[1] <= seen[2] + 1e-9


def test_the_same_state_is_the_same_sky_and_another_seed_is_another(patches) -> None:
    kw = dict(cover=0.25, base_m=1100.0, profile=C.CLOUDSCAPE_TYPES["cumulus"], patches=patches, **SMALL)
    a, b, c = C.Cloudscape(seed=7, **kw), C.Cloudscape(seed=7, **kw), C.Cloudscape(seed=8, **kw)
    pts = np.random.default_rng(0).uniform(0.0, 30_000.0, (4000, 2))
    y = 1100.0 + 300.0
    assert np.array_equal(a.density(pts[:, 0], y, pts[:, 1]), b.density(pts[:, 0], y, pts[:, 1]))
    assert not np.array_equal(a.density(pts[:, 0], y, pts[:, 1]), c.density(pts[:, 0], y, pts[:, 1]))


def test_no_patch_is_cut_flat_at_its_box(sky: C.Cloudscape) -> None:
    """Density falls to zero toward a box's sides: next to any empty point at mid height, the
    cloud is thin. A box clipped at its wall would put dense cloud beside empty air."""
    import dataclasses

    # Without the skin's small noise, which eats holes of its own on purpose.
    smooth = C.Cloudscape(cover=sky.cover, base_m=sky.base_m, seed=sky.seed, patches=sky.patches,
                          profile=dataclasses.replace(sky.profile, patch_erosion=0.0), **SMALL)
    y = smooth.base_m + 250.0
    x = np.arange(0.0, smooth.weather_tile_m, 12.5)
    worst = 0.0
    for z in np.linspace(0.0, smooth.weather_tile_m, 40, endpoint=False):
        rho = smooth.density(x, y, z)
        empty = rho == 0.0
        beside = np.maximum(np.where(empty[:-2], rho[1:-1], 0.0), np.where(empty[2:], rho[1:-1], 0.0))
        worst = max(worst, float(beside.max()))
    assert worst < 0.35


def test_the_shipped_patches_are_cloud_on_a_floor() -> None:
    shipped = C.load_patches(C.PATCH_DIRECTORIES[0])
    assert len(shipped) >= 6
    for patch in shipped:
        assert patch.density.dtype == np.float32
        assert patch.density.min() >= 0.0 and patch.density.max() == pytest.approx(1.0)
        assert patch.voxel_m == pytest.approx(10.0)
        assert patch.density.shape[1] * patch.voxel_m < C.CLOUDSCAPE_TYPES["cumulus"].thickness_m
        assert patch.density[:, 0, :].max() > 0.2          # cloud at the condensation level
        assert patch.density[:, -1, :].max() < 0.2         # and none cut off at the top


@pytest.mark.skipif(not __import__("weather_fx.gpu", fromlist=["warp_available"]).warp_available(),
                    reason="no Warp")
def test_the_gpu_kernel_places_the_patches_the_numpy_reference_does(sky: C.Cloudscape) -> None:
    from weather_fx.gpu import cloud_march

    if cloud_march.wp.get_cuda_device_count() == 0:
        pytest.skip("no CUDA device")
    renderer = cloud_march.CloudRenderer(sky, device="cuda:0")
    rng = np.random.default_rng(3)
    n = 30_000
    x = rng.uniform(-30_000.0, 30_000.0, n)
    z = rng.uniform(-30_000.0, 30_000.0, n)
    y = rng.uniform(sky.base_m - 50.0, sky.top_m + 50.0, n)
    gpu, ref = renderer.density(x, y, z), sky.density(x, y, z)
    assert (ref > 0.0).mean() > 0.01
    error = np.abs(gpu - ref)
    assert error.max() < 0.05
    assert error.mean() < 1e-3
