"""Drift as a function of time, microphysics per genus, the march's jitter and the fine detail."""
import math

import numpy as np
import pytest

from weather_fx.core.clouds import (
    CLOUD_TYPES,
    CloudField,
    DriftTrack,
    _hash_unit,
    cloud_drift_at,
    cloud_drift_from_state,
    cloud_profile,
    drift_velocity_m_s,
    ice_extinction_per_m,
    ice_water_content_g_m3,
    liquid_extinction_per_m,
    liquid_water_content_g_m3,
    stage_to_field,
)
from weather_fx.core.state import WeatherState


# --- drift is a function of time ---------------------------------------------------------------

def test_the_drift_is_the_wind_times_the_elapsed_time():
    v = drift_velocity_m_s(10.0, 30.0, 1.5, up_axis=2)
    np.testing.assert_allclose(cloud_drift_at(42.0, 10.0, 30.0, 1.5, up_axis=2), v * 42.0)
    np.testing.assert_allclose(cloud_drift_at(42.0, 10.0, 30.0, 1.5, up_axis=2, frame="field"),
                               stage_to_field(v * 42.0, 2))


def test_the_state_helper_reads_the_wind_and_the_cloud_factor():
    state = WeatherState().with_updates("wind", speed_mps=8.0, direction_deg=90.0)
    state = state.with_updates("clouds", wind_factor=2.0)
    np.testing.assert_allclose(cloud_drift_from_state(state, 10.0, up_axis=2), [0.0, 160.0, 0.0],
                               atol=1e-9)


def test_a_steady_wind_track_is_exactly_the_pure_function():
    """What the viewport draws and what a headless infrared render computes must be one number."""
    track = DriftTrack()
    v = drift_velocity_m_s(12.0, -45.0, 1.5, up_axis=1)
    track.rebase(0.0, v)  # the wind set before the clock starts
    for t in (0.0, 0.5, 17.25, 3600.0):
        np.testing.assert_array_equal(track.at(t), cloud_drift_at(t, 12.0, -45.0, 1.5, up_axis=1))


def test_a_wind_change_continues_from_where_the_clouds_are():
    track = DriftTrack()
    track.rebase(0.0, [10.0, 0.0, 0.0])
    before = track.at(20.0)
    track.rebase(20.0, [0.0, 0.0, 5.0])
    np.testing.assert_allclose(track.at(20.0), before)          # no jump
    np.testing.assert_allclose(track.at(30.0), [200.0, 0.0, 50.0])
    track.reset([1.0, 0.0, 0.0])
    np.testing.assert_allclose(track.at(7.0), [7.0, 0.0, 0.0])


def test_setting_the_same_wind_again_does_not_start_a_segment():
    track = DriftTrack()
    track.rebase(0.0, [3.0, 0.0, 0.0])
    track.rebase(50.0, [3.0, 0.0, 0.0])
    assert track.since_s == 0.0


# --- microphysics ------------------------------------------------------------------------------

def test_liquid_extinction_is_geometric_optics():
    # beta = 3 LWC / (2 rho_w r_e): 0.3 g/m^3 of 10 um droplets is 45 per km.
    assert liquid_extinction_per_m(0.3, 10.0) == pytest.approx(0.045)
    assert liquid_water_content_g_m3(0.045, 10.0) == pytest.approx(0.3)


def test_ice_extinction_follows_fu_1996_and_inverts():
    beta = ice_extinction_per_m(0.02, 60.0)
    assert beta == pytest.approx(0.02 * (-6.656e-3 + 3.686 / 60.0))
    assert ice_water_content_g_m3(beta, 60.0) == pytest.approx(0.02)


def test_every_genus_states_a_phase_and_particle_sizes():
    for name, profile in CLOUD_TYPES.items():
        assert profile.phase in ("liquid", "ice", "mixed"), name
        assert 0.0 <= profile.ice_fraction <= 1.0
        assert (profile.ice_fraction == 0.0) == (profile.phase == "liquid"), name
        assert (profile.ice_fraction == 1.0) == (profile.phase == "ice"), name
    assert cloud_profile("cirrus").phase == "ice"
    assert cloud_profile("cumulus").phase == "liquid"


def _field(genus, **kw):
    kw.setdefault("cells", 48)
    kw.setdefault("levels", 16)
    kw.setdefault("cell_m", 120.0)
    return CloudField(cover=0.4, base_m=1000.0, profile=cloud_profile(genus), seed=5, **kw)


@pytest.mark.parametrize("genus", ["cumulus", "congestus", "stratus", "cirrus"])
def test_the_water_reproduces_the_visible_extinction(genus):
    """Derived from the extinction, not authored beside it: the two cannot disagree."""
    field = _field(genus)
    m = field.microphysics
    beta = (liquid_extinction_per_m(m.liquid_water_g_m3, m.effective_radius_um)
            + (ice_extinction_per_m(m.ice_water_g_m3, m.ice_effective_diameter_um)
               if m.ice_water_g_m3 > 0 else 0.0))
    assert beta == pytest.approx(field.extinction_per_m, rel=1e-9)


def test_the_water_contents_are_in_the_observed_ranges():
    cumulus = _field("cumulus").microphysics
    assert cumulus.ice_water_g_m3 == 0.0
    assert 0.03 < cumulus.liquid_water_g_m3 < 1.5          # fair-weather cumulus: ~0.1-1 g/m^3
    cirrus = _field("cirrus").microphysics
    assert cirrus.liquid_water_g_m3 == 0.0
    assert 1e-4 < cirrus.ice_water_g_m3 < 0.2             # cirrus: ~0.001-0.1 g/m^3


def test_the_water_content_follows_the_density():
    field = _field("congestus")
    x, y, z = np.linspace(-2000, 2000, 50), np.full(50, 1500.0), np.zeros(50)
    rho = field.density(x, y, z)
    np.testing.assert_allclose(field.liquid_water_content(x, y, z),
                               rho * field.microphysics.liquid_water_g_m3)
    np.testing.assert_allclose(field.ice_water_content(x, y, z),
                               rho * field.microphysics.ice_water_g_m3)


# --- the march's jitter ------------------------------------------------------------------------

def test_the_jitter_is_deterministic_and_uncorrelated_between_neighbouring_rays():
    """A jitter that varies smoothly with direction prints concentric rings (ADR 0146)."""
    angle = np.linspace(0.3, 0.3001, 4096)
    dx, dy, dz = np.cos(angle), np.sin(angle), np.zeros_like(angle)
    a = _hash_unit(dx, dy, dz)
    np.testing.assert_array_equal(a, _hash_unit(dx, dy, dz))
    assert 0.0 <= a.min() and a.max() < 1.0
    assert abs(a.mean() - 0.5) < 0.03
    neighbours = np.corrcoef(a[:-1], a[1:])[0, 1]
    assert abs(neighbours) < 0.1
    assert not np.array_equal(a, _hash_unit(dx, dy, dz, seed=1))


def test_the_march_is_reproducible():
    field = _field("cumulus")
    d = np.stack([np.linspace(-0.5, 0.5, 64), np.full(64, 0.6), np.zeros(64)], axis=-1)
    o = np.zeros_like(d)
    first = field.march(o, d, steps=32).optical_depth
    np.testing.assert_array_equal(first, field.march(o, d, steps=32).optical_depth)


# --- the fine detail ---------------------------------------------------------------------------

def test_no_detail_is_the_plain_interpolated_grid():
    plain = _field("cumulus", detail_strength=0.0)
    x = np.linspace(-2500, 2500, 400)
    y = np.full_like(x, 1300.0)
    np.testing.assert_allclose(plain.density(x, y, 0.0), plain._sample(plain._density, x, y, 0.0))


def test_the_detail_tiles_with_the_field():
    field = _field("cumulus")
    tile = field.cells * field.cell_m
    rng = np.random.default_rng(1)
    x, z = rng.uniform(-3000, 3000, (2, 500))
    y = rng.uniform(field.base_m, field.top_m, 500)
    np.testing.assert_allclose(field.density(x, y, z), field.density(x + tile, y, z - 2 * tile),
                               atol=1e-6)


def test_the_detail_changes_the_edges_and_leaves_the_core_alone():
    field = _field("cumulus")
    plain = _field("cumulus", detail_strength=0.0)
    rng = np.random.default_rng(2)
    x, z = rng.uniform(-2500, 2500, (2, 20000))
    y = rng.uniform(field.base_m, field.top_m, 20000)
    signed = field._sample(field._signed, x, y, z)
    core = signed > field.detail_strength + field.softness
    assert np.all(field.density(x[core], y[core], z[core]) == 1.0)
    edge = np.abs(signed) < field.softness
    changed = np.abs(field.density(x[edge], y[edge], z[edge]) - plain.density(x[edge], y[edge], z[edge]))
    assert np.mean(changed > 0.1) > 0.3


@pytest.mark.parametrize("up_axis", [1, 2])
def test_a_refined_volume_holds_the_detailed_field(up_axis):
    field = _field("cumulus", cells=16, levels=8)
    grid = field.volume_grid(up_axis, refine=2)
    assert grid.voxel_m[0] == pytest.approx(field.cell_m / 2)
    assert grid.tile_m == pytest.approx(field.cells * field.cell_m)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, grid.values.shape, size=(300, 3))
    centres = np.asarray(grid.first_centre_m) + idx * np.asarray(grid.voxel_m)
    if up_axis == 1:
        fx, fy, fz = centres[:, 0], centres[:, 1], centres[:, 2]
    else:
        fx, fy, fz = centres[:, 0], centres[:, 2], -centres[:, 1]
    np.testing.assert_allclose(grid.values[idx[:, 0], idx[:, 1], idx[:, 2]],
                               field.density(fx, fy, fz), atol=1e-5)
    assert grid.low_m[1 if up_axis == 1 else 2] == pytest.approx(field.base_m)
    assert grid.high_m[1 if up_axis == 1 else 2] == pytest.approx(field.top_m)
