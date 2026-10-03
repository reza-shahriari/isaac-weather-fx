import math
import threading
import time

import numpy as np
import pytest

from weather_fx.backends.viewport.render_mode import choose_cloud_path
from weather_fx.core.clouds import CloudField, cloud_field_from_state, cloud_profile, tile_placements
from weather_fx.core.jobs import LatestJob
from weather_fx.core.sky import conditions_from_state, environment_map, latlong_directions
from weather_fx.core.state import WeatherState


# --- background jobs --------------------------------------------------------------------------

def test_a_job_runs_off_the_calling_thread_and_is_collected_once():
    job = LatestJob()
    caller = threading.get_ident()
    job.submit("a", lambda: threading.get_ident())
    key, worker = job.wait(5.0)
    assert key == "a" and worker != caller
    assert job.poll() is None


def test_submitting_while_busy_keeps_only_the_newest_pending_job():
    job = LatestJob()
    gate = threading.Event()
    ran = []

    def slow(name):
        def fn():
            gate.wait(5.0)
            ran.append(name)
            return name
        return fn

    job.submit(1, slow(1))
    time.sleep(0.05)
    job.submit(2, slow(2))
    job.submit(3, slow(3))  # replaces 2 before it starts
    gate.set()
    deadline = time.time() + 5.0
    while job.busy and time.time() < deadline:
        time.sleep(0.01)
    assert ran == [1, 3]
    assert job.poll() == (3, 3)


def test_a_failing_job_hands_its_exception_to_the_poller():
    job = LatestJob()
    job.submit("x", lambda: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        job.wait(5.0)


def test_cancel_discards_a_result_nobody_wants_any_more():
    job = LatestJob()
    gate = threading.Event()
    job.submit("old", lambda: gate.wait(5.0))
    job.cancel()
    gate.set()
    assert job.wait(5.0) is None


# --- which renderer draws the cloud ---------------------------------------------------------

@pytest.mark.parametrize("requested, mode, expected", [
    ("auto", "PathTracing", "volume"),
    ("auto", "RaytracedLighting", "dome"),
    ("auto", None, "dome"),
    ("volume", "RaytracedLighting", "volume"),
    ("dome", "PathTracing", "dome"),
])
def test_the_cloud_path_follows_the_renderer_unless_forced(requested, mode, expected):
    assert choose_cloud_path(requested, mode) == expected


# --- the voxel grid a volume is written from --------------------------------------------------

def _field(**kw):
    return CloudField(cover=0.4, base_m=900.0, profile=cloud_profile("cumulus"),
                      cells=32, levels=8, cell_m=100.0, seed=3, **kw)


@pytest.mark.parametrize("up_axis", [1, 2])
def test_every_voxel_centre_holds_what_the_field_holds_there(up_axis):
    """The volume the path tracer renders must be the field the dome and the IR band march."""
    field = _field()
    grid = field.volume_grid(up_axis)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, grid.values.shape, size=(200, 3))
    centres = np.asarray(grid.first_centre_m) + idx * np.asarray(grid.voxel_m)
    if up_axis == 1:
        fx, fy, fz = centres[:, 0], centres[:, 1], centres[:, 2]
    else:  # stage (x, y, z_up) = field (x, -z, y)
        fx, fy, fz = centres[:, 0], centres[:, 2], -centres[:, 1]
    expected = field.density(fx, fy, fz)
    got = grid.values[idx[:, 0], idx[:, 1], idx[:, 2]]
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_the_grid_bounds_are_the_slab():
    field = _field()
    grid = field.volume_grid(1)
    assert grid.low_m[1] == pytest.approx(field.base_m)
    assert grid.high_m[1] == pytest.approx(field.top_m)
    assert grid.high_m[0] - grid.low_m[0] == pytest.approx(grid.tile_m)


@pytest.mark.parametrize("up_axis, horizontal", [(1, (0, 2)), (2, (0, 1))])
def test_tiles_abut_exactly_and_centre_on_the_origin(up_axis, horizontal):
    grid = _field().volume_grid(up_axis)
    tiles = tile_placements(grid, 3, up_axis)
    assert len(tiles) == 9 and len({t.name for t in tiles}) == 9
    lows = sorted({round(t.low_m[horizontal[0]], 6) for t in tiles})
    assert np.allclose(np.diff(lows), grid.tile_m)
    centre = [0.5 * (t.low_m[horizontal[0]] + t.high_m[horizontal[0]]) for t in tiles]
    assert np.mean(centre) == pytest.approx(0.0, abs=1e-6)
    vertical = 1 if up_axis == 1 else 2
    assert all(t.low_m[vertical] == pytest.approx(grid.low_m[vertical]) for t in tiles)


def test_the_field_is_built_once_per_shape_and_reused_for_a_colour_change():
    state = WeatherState().with_updates("clouds", enabled=True, cells=32, levels=8, seed=11)
    first = cloud_field_from_state(state)
    recoloured = state.with_updates("clouds", lit_color=(1.0, 0.5, 0.5), density_scale=3.0)
    assert cloud_field_from_state(recoloured) is first
    assert cloud_field_from_state(state.with_updates("clouds", seed=12)) is not first
    assert cloud_field_from_state(state.with_updates("clouds", enabled=False)) is None


# --- the dome (real-time) path ----------------------------------------------------------------

def _cloudy(**clouds):
    state = WeatherState().with_updates("sky", hour_utc=10.0)
    values = dict(enabled=True, cover=0.5, cells=48, levels=12, cell_m=120.0, seed=5, dome_rows=32)
    values.update(clouds)
    return state.with_updates("clouds", **values)


def test_the_coarse_cloud_layer_leaves_the_clear_sky_sharp():
    """Upsampling the cloud's terms, not the composite, keeps the sky behind it at full res."""
    state = _cloudy(cover=0.01)
    cloudy = environment_map(conditions_from_state(state), height=64)
    clear = environment_map(conditions_from_state(state.with_updates("clouds", enabled=False)), 64)
    below = latlong_directions(64)[..., 1] < 0.0
    np.testing.assert_allclose(cloudy[below], clear[below], rtol=1e-5)


def test_the_shadow_colour_tints_the_cloud_and_not_the_clear_sky():
    base = environment_map(conditions_from_state(_cloudy(shadow_color=(1.0, 1.0, 1.0))), height=64)
    red = environment_map(conditions_from_state(_cloudy(shadow_color=(1.0, 0.2, 0.2))), height=64)
    changed = np.abs(red - base).sum(axis=-1) > 1e-6 * (base.sum(axis=-1) + 1.0)
    assert changed.any()
    # Where anything changed, it lost green and blue, not red.
    assert (red[changed][:, 1] <= base[changed][:, 1] + 1e-6).all()
    assert np.allclose(red[changed][:, 0], base[changed][:, 0], rtol=1e-4)


# --- drift, recentring and shadow -------------------------------------------------------------

from weather_fx.core.clouds import drift_velocity_m_s, stage_to_field, volume_offset_m  # noqa: E402


@pytest.mark.parametrize("up_axis", [1, 2])
def test_the_volumes_follow_the_camera_by_whole_tiles_and_keep_it_central(up_axis):
    tile = 15_360.0
    horizontal = (0, 2) if up_axis == 1 else (0, 1)
    drift = np.zeros(3)
    drift[horizontal[0]] = 1234.0
    for x in (0.0, 7_000.0, 8_000.0, 50_000.0, -123_456.0):
        anchor = np.zeros(3)
        anchor[horizontal[0]] = x
        anchor[horizontal[1]] = -0.4 * x
        offset = volume_offset_m(drift, anchor, tile, up_axis)
        # The camera is inside the central tile ...
        assert np.all(np.abs((anchor - offset)[list(horizontal)]) <= 0.5 * tile + 1e-6)
        # ... and the shift beyond the drift is whole tiles, so the cloud looks unchanged.
        extra = (offset - drift)[list(horizontal)] / tile
        assert np.allclose(extra, np.round(extra))
        assert offset[up_axis] == 0.0


def test_clouds_drift_downwind_at_the_cloud_level_speed():
    v = drift_velocity_m_s(10.0, 90.0, 1.5, up_axis=2)
    np.testing.assert_allclose(v, [0.0, 15.0, 0.0], atol=1e-9)
    assert np.all(drift_velocity_m_s(10.0, 0.0, 0.0) == 0.0)


def test_a_sun_behind_a_cloud_is_dimmed_and_a_clear_one_is_not():
    field = _field()
    sun = np.array([0.0, 1.0, 0.0])
    # Find a column that is cloudy and one that is clear, straight up.
    xs = np.linspace(-1500.0, 1500.0, 61)
    taus = [field.optical_depth_toward((x, 1.5, 0.0), sun) for x in xs]
    assert max(taus) > 1.0 and min(taus) == 0.0
    assert field.optical_depth_toward((0.0, 1.5, 0.0), (1.0, -0.2, 0.0)) == float("inf")


def test_stage_to_field_inverts_the_z_up_turn():
    p = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(stage_to_field(p, 1), p)
    np.testing.assert_allclose(stage_to_field(p, 2), [1.0, 3.0, -2.0])


def test_drifting_the_dome_by_a_whole_tile_changes_nothing():
    """The dome march honours the drift, and the field's period makes a full tile invisible."""
    from dataclasses import replace

    conditions = conditions_from_state(_cloudy())
    tile = conditions.cloud.cells * conditions.cloud.cell_m
    still = environment_map(conditions, height=32)
    moved = environment_map(replace(conditions, cloud_offset_m=(tile, 0.0, 0.0)), height=32)
    half = environment_map(replace(conditions, cloud_offset_m=(0.5 * tile, 0.0, 0.0)), height=32)
    np.testing.assert_allclose(moved, still, rtol=1e-4, atol=1e-6)
    assert not np.allclose(half, still, rtol=1e-3)


# --- a dome texel integrates the cloud it covers --------------------------------------------------


def _small_dome_terms(edge_rays: int):
    from weather_fx.core import sky as S
    from weather_fx.core.state import WeatherState

    state = WeatherState.from_dict({
        "general": {"time_source": "manual"},
        "sky": {"latitude_deg": 48.14, "longitude_deg": 11.58, "date_utc": "2024-06-21",
                "hour_utc": 12.97},
        "clouds": {"enabled": True, "cover": 0.35, "genus": "cumulus", "seed": 17,
                   "temperature_c": 22.0, "dewpoint_c": 12.0, "cells": 96, "levels": 32,
                   "cell_m": 80.0, "dome_edge_rays": edge_rays},
    }, strict=True)
    conditions = S.conditions_from_state(state)
    rows = 64
    directions = S.latlong_directions(rows)
    up = directions[..., 1] > 0.0
    transmit = np.ones(up.shape)
    added = np.zeros(up.shape + (3,))
    transmit[up], added[up] = S._cloud_terms(directions[up], conditions)
    single = transmit.copy()
    S._integrate_cloud_edges(transmit, added, up, conditions)
    return S, conditions, rows, directions, up, single, transmit


def test_a_dome_texel_an_edge_crosses_holds_the_mean_of_its_rays():
    """One ray through a texel's centre samples the cloud; the texel should hold its mean. Against
    6 x 6 rays per texel, the edge rays take the transmittance's static energy near the horizon
    to well under a fifth of one ray's, and leave every texel no edge crosses as it was."""
    S, conditions, rows, directions, up, single, refined = _small_dome_terms(2)
    theta = (np.arange(rows) + 0.5) * (math.pi / rows)
    phi = (np.arange(2 * rows) + 0.5) * (math.pi / rows)
    elevation = np.degrees(np.arcsin(directions[..., 1]))
    low = up & (elevation < 10.0)
    r, c = np.nonzero(low)
    n = 6
    offsets = (np.arange(n) + 0.5) / n - 0.5
    du, dv = np.meshgrid(offsets, offsets, indexing="ij")
    t = theta[r][:, None] + du.ravel() * (math.pi / rows)
    p = phi[c][:, None] + dv.ravel() * (math.pi / rows)
    rays = np.stack([np.sin(t) * np.cos(p), -np.sin(t) * np.sin(p), np.cos(t)], axis=-1)
    reference = S._cloud_terms(rays.reshape(-1, 3), conditions)[0].reshape(r.size, n * n).mean(1)
    before = np.mean((single[low] - reference) ** 2)
    after = np.mean((refined[low] - reference) ** 2)
    assert before > 0.0
    assert after < 0.2 * before
    unchanged = refined == single
    assert unchanged.any() and (~unchanged).any()


def test_one_edge_ray_turns_the_integration_off():
    _, _, _, _, _, single, refined = _small_dome_terms(1)
    assert np.array_equal(single, refined)
