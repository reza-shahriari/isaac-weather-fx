"""Hero clouds: scaling an asset, placing, drifting and wrapping it, and sampling it."""
import numpy as np
import pytest

from weather_fx.core.hero import HeroAsset, HeroClouds, hero_positions, prepare_hero_asset


def _blob(nx=40, ny=24, nz=30):
    x, y, z = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny), np.linspace(-1, 1, nz),
                          indexing="ij")
    return np.clip(1.0 - (x**2 + y**2 + z**2), 0.0, None).astype(np.float32) * 3.0


def test_the_asset_is_scaled_to_its_widest_horizontal_extent_and_normalised():
    asset = prepare_hero_asset(_blob(), size_m=2000.0, max_voxels=256)
    assert max(asset.size_m[0], asset.size_m[2]) == pytest.approx(2000.0)
    assert asset.values.max() == pytest.approx(1.0)
    assert asset.size_m[1] == pytest.approx(2000.0 * 24 / 40)   # cubic voxels keep proportions


def test_a_big_asset_is_box_filtered_down():
    asset = prepare_hero_asset(_blob(200, 60, 120), size_m=1000.0, max_voxels=50)
    assert max(asset.values.shape) <= 50
    assert asset.voxel_m == pytest.approx(1000.0 / max(asset.values.shape[0], asset.values.shape[2]))


def test_a_z_up_source_is_turned_to_y_up():
    zup = np.zeros((10, 20, 30), dtype=np.float32)   # x, y, z_up: tall in z
    zup[5, 10, 25] = 1.0
    asset = prepare_hero_asset(zup, size_m=100.0, source_up_axis=2, max_voxels=64)
    assert asset.values.shape == (10, 30, 20)        # x, y_up, z_field
    i, j, k = np.unravel_index(np.argmax(asset.values), asset.values.shape)
    assert (i, j) == (5, 25)


def test_positions_are_deterministic_and_kept_apart():
    a = hero_positions(5, seed=3, spread_m=20000.0, min_gap_m=3000.0)
    np.testing.assert_array_equal(a, hero_positions(5, seed=3, spread_m=20000.0, min_gap_m=3000.0))
    assert a.shape == (5, 2) and np.all(np.abs(a) <= 10000.0)
    gaps = [np.hypot(*(a[i] - a[j])) for i in range(5) for j in range(i + 1, 5)]
    assert min(gaps) >= 3000.0


def _clouds(count=3):
    asset = prepare_hero_asset(_blob(), size_m=2000.0)
    return HeroClouds(asset=asset, base_m=1500.0,
                      positions_m=hero_positions(count, 1, 12000.0, 2500.0),
                      spread_m=12000.0, extinction_per_m=0.04)


def test_the_clouds_drift_and_wrap_around_the_anchor():
    clouds = _clouds()
    still = clouds.centres_m()
    moved = clouds.centres_m(drift_m=(500.0, 0.0, 0.0))
    np.testing.assert_allclose((moved[:, 0] - still[:, 0] + 6000.0) % 12000.0 - 6000.0, 500.0)
    far = clouds.centres_m(drift_m=(0, 0, 0), anchor_m=(50000.0, 0.0, -30000.0))
    assert np.all(np.abs(far[:, 0] - 50000.0) <= 6000.0)
    assert np.all(np.abs(far[:, 2] + 30000.0) <= 6000.0)
    assert np.all(far[:, 1] == 1500.0)


def test_the_density_is_the_asset_where_a_cloud_is_and_zero_elsewhere():
    clouds = _clouds(1)
    c = clouds.centres_m()[0]
    height = clouds.asset.size_m[1] / 2
    assert clouds.density(c[0], c[1] + height, c[2]) > 0.8   # the middle of the blob
    assert clouds.density(c[0] + 5000.0, c[1] + height, c[2]) == 0.0
    assert clouds.density(c[0], c[1] - 10.0, c[2]) == 0.0    # below the base


def test_the_stage_corners_match_the_voxel_layout_on_a_z_up_stage():
    clouds = _clouds(1)
    array, voxel, size = clouds.asset.volume_grid(2)
    corner = clouds.stage_lower_corners_m((0, 0, 0), (0, 0, 0), up_axis=2)[0]
    # The densest voxel, found in the stage array, must sit where the field says it is.
    i, j, k = np.unravel_index(np.argmax(array), array.shape)
    stage_point = corner + (np.array([i, j, k]) + 0.5) * voxel
    field_point = np.array([stage_point[0], stage_point[2], -stage_point[1]])
    assert clouds.density(*field_point) == pytest.approx(1.0, abs=0.05)


def test_optical_depth_through_a_cloud_is_positive_and_zero_beside_it():
    clouds = _clouds(1)
    c = clouds.centres_m()[0]
    through = clouds.optical_depth_toward((c[0], 0.0, c[2]), (0.0, 1.0, 0.0), anchor_m=(0, 0, 0))
    beside = clouds.optical_depth_toward((c[0] + 5000.0, 0.0, c[2]), (0.0, 1.0, 0.0),
                                         anchor_m=(0, 0, 0))
    assert through > 5.0 and beside == 0.0
