import numpy as np

from weather_fx.core.particles import ParticleField, rotation_between


def sampler(rng, n):
    return {"diameter_m": np.full(n, 0.002), "fall_speed_mps": np.full(n, 5.0)}


def inside(field, anchor, half):
    rel = field.positions - anchor
    return np.all(np.abs(rel) <= half + 1e-9)


def test_resize_up_and_down():
    f = ParticleField(seed=1)
    f.resize(100, np.zeros(3), np.ones(3), sampler)
    assert f.count == 100 and len(f.attrs["diameter_m"]) == 100 and len(f.phase) == 100
    f.resize(40, np.zeros(3), np.ones(3), sampler)
    assert f.count == 40 and len(f.attrs["fall_speed_mps"]) == 40


def test_particles_stay_in_box_and_wrap():
    f = ParticleField(up_axis=2, seed=2)
    half = np.array([5.0, 5.0, 2.0])
    anchor = np.array([10.0, -3.0, 1.0])
    f.resize(500, anchor, half, sampler)
    for _ in range(200):
        f.step(1 / 30, anchor, half, wind_velocity=[3.0, 0.0, 0.0], t=0.0)
        assert inside(f, anchor, half)


def test_follows_moving_anchor():
    f = ParticleField(seed=3)
    half = np.array([4.0, 4.0, 4.0])
    f.resize(200, np.zeros(3), half, sampler)
    new_anchor = np.array([100.0, 50.0, 0.0])
    f.step(1 / 60, new_anchor, half, wind_velocity=[0, 0, 0])
    assert inside(f, new_anchor, half)


def test_falls_along_up_axis_y_up():
    f = ParticleField(up_axis=1, seed=4)
    half = np.array([50.0, 50.0, 50.0])
    f.resize(10, np.zeros(3), half * 0.1, sampler)
    before = f.positions.copy()
    f.step(0.1, np.zeros(3), half, wind_velocity=[0, 0, 0])
    assert np.allclose(f.positions[:, 1] - before[:, 1], -0.5)
    assert np.allclose(f.positions[:, [0, 2]], before[:, [0, 2]])


def test_sway_moves_horizontally():
    f = ParticleField(up_axis=2, seed=5)
    f.resize(50, np.zeros(3), np.full(3, 10.0), sampler)
    before = f.positions.copy()
    f.step(0.05, np.zeros(3), np.full(3, 100.0), [0, 0, 0], t=0.3, sway_amplitude=0.5, sway_frequency=1.0)
    assert not np.allclose(f.positions[:, :2], before[:, :2])


def test_rotation_between():
    q = rotation_between([0, 0, 1], [1, 0, 0])
    w, x, y, z = q
    assert np.isclose(w, np.cos(np.pi / 4)) and np.isclose(y, np.sin(np.pi / 4))
    assert np.allclose(rotation_between([0, 0, 1], [0, 0, 1]), (1, 0, 0, 0))
    q = rotation_between([0, 0, 1], [0, 0, -1])
    assert np.isclose(q[0], 0.0)
