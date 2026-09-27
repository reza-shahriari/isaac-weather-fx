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
