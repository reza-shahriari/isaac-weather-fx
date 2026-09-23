import math

import numpy as np

from weather_fx.core import physics as ph


def test_koschmieder():
    beta = ph.extinction_from_visibility(1000.0)
    assert math.isclose(beta, -math.log(0.05) / 1000.0)
    assert math.isclose(float(ph.transmittance(beta, 1000.0)), 0.05, rel_tol=1e-9)
    assert math.isclose(ph.visibility_from_extinction(beta), 1000.0)


def test_rain_density_grows_with_rate():
    d = [ph.rain_number_density(r, 0.5, 6.0) for r in (1, 10, 50)]
    assert d[0] < d[1] < d[2]


def test_drop_samples_in_bounds_and_mean_grows():
    rng = np.random.default_rng(0)
    light = ph.sample_drop_diameters(rng, 20000, 1.0, 0.5, 6.0)
    heavy = ph.sample_drop_diameters(rng, 20000, 50.0, 0.5, 6.0)
    assert light.min() >= 0.5 and light.max() <= 6.0
    assert heavy.mean() > light.mean()


def test_terminal_velocity():
    v = ph.raindrop_terminal_velocity([0.5, 1.0, 2.0, 5.0])
    assert np.all(np.diff(v) > 0)
    assert 6.0 < v[2] < 7.0  # ~6.5 m/s for a 2 mm drop


def test_wind_vector_axes():
    z_up = ph.wind_vector(10, 90, 1, up_axis=2)
    assert np.allclose(z_up, [0, 10, 1], atol=1e-9)
    y_up = ph.wind_vector(10, 90, 1, up_axis=1)
    assert np.allclose(y_up, [0, 1, 10], atol=1e-9)


def test_gust_factor():
    assert ph.gust_factor(3.0, 0.0, 5.0) == 1.0
    values = [ph.gust_factor(t, 0.5, 5.0) for t in np.linspace(0, 20, 200)]
    assert min(values) >= 0.0 and max(values) > 1.0
