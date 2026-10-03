"""The visible cloud's light, held to the criteria of the thermal-camera repo's physics spec.

docs/physics-model.md §7.5 there ("The visible companion's cloud") lists what a cloud has to do
before it looks like one, and each test here is one of them, measured rather than eyeballed:

* **white furnace** -- a cloud that does not absorb, under a sky that is the same in every
  direction, cannot be seen: it scatters back exactly what it removes;
* **thin edges** -- an edge too thin to shadow anything is never darker than the sky behind it;
* **a base darker than its sunlit flank**;
* **the albedo from above** -- a uniform deck reflects what a two-stream layer of the same
  optical depth reflects.

The earlier model failed the first, third and fourth (measured 2026-10-01): it had no sky light,
so a night cloud added nothing and an unlit one was black, and its brightness came from one
envelope applied to every base and top alike.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from weather_fx.core import sky as S
from weather_fx.core.clouds import CLOUD_ASYMMETRY_G, VISIBLE_DROPLET_ALBEDO, _diffuse_sunlight
from weather_fx.core.state import WeatherState

LUMA = np.array([0.2126, 0.7152, 0.0722])


def _conditions(hour: float = 12.97, cover: float = 0.35, genus: str = "cumulus", seed: int = 17):
    state = WeatherState.from_dict({
        "general": {"time_source": "manual"},
        "sky": {"latitude_deg": 48.14, "longitude_deg": 11.58, "date_utc": "2024-06-21",
                "hour_utc": hour, "turbidity": 2.6},
        "clouds": {"enabled": True, "cover": cover, "genus": genus, "seed": seed,
                   "temperature_c": 22.0, "dewpoint_c": 12.0, "cells": 96, "levels": 32,
                   "cell_m": 80.0},
    }, strict=True)
    return S.conditions_from_state(state)


def _directions(el_lo, el_hi, az_centre, az_half, n, seed):
    rng = np.random.default_rng(seed)
    el = np.radians(rng.uniform(el_lo, el_hi, n))
    az = np.radians(az_centre + rng.uniform(-az_half, az_half, n))
    return np.stack([np.cos(el) * np.sin(az), np.sin(el), -np.cos(el) * np.cos(az)], axis=-1)


@pytest.fixture(scope="module")
def noon():
    return _conditions()


def test_a_white_cloud_under_a_uniform_sky_vanishes(noon) -> None:
    """The white furnace. Sun off, the same radiance above and below: every cloudy ray composes to
    that radiance within the droplets' own absorption. The old model added nothing without a
    sun, so the same cloud was a black shape at the cloud's own transmittance."""
    directions = _directions(5.0, 89.0, 0.0, 180.0, 3000, seed=1)
    sky = np.array([1000.0, 1000.0, 1000.0])
    # The furnace is the cloud's own scattering, so the real atmosphere's air in front of it is
    # left out: under a sky this uniform, uniform air would add (1 - T_air) L0 and the cloud would
    # still vanish, but the table's air is a real noon's.
    furnace = dataclasses.replace(noon, aerial_perspective=False)
    transmittance, added = S._cloud_terms(directions, furnace, ambient=(sky, sky), sunlit=False)
    out = transmittance[:, None] * sky + added
    cloudy = transmittance < 0.99
    assert cloudy.sum() > 300
    assert np.max(np.abs(out[cloudy] / sky - 1.0)) < 0.01
    assert np.max(np.abs(out[cloudy] / sky - 1.0)) <= (1.0 - VISIBLE_DROPLET_ALBEDO) + 1e-9


def test_a_thin_edge_is_never_darker_than_the_sky_behind_it(noon) -> None:
    """An edge with tau < 0.3 above 20 degrees: what it scatters toward the eye -- sunlight once,
    and the sky's and the ground's light -- always outweighs the sky light it removes. The old
    shading drew the dark rim this forbids round thin edges (one was 0.92 of the sky behind).

    Outside the 20 degrees round the sun. Inside them a wisp whose sunward chord runs through
    its own cloud's body is in that cloud's shadow, and the clear sky it is compared with is the
    aureole -- air the same cloud also shadows, which this sky does not model. Such a wisp reads
    0.95 of the unshadowed aureole, and in a photograph it is the dark fringe against the glare.
    The bound was 15 degrees until the field's sizes were fixed (WX.6): that field put one such
    wisp at 17.5 degrees (tau 0.026, 0.993 of the aureole), and the aureole is that bright there."""
    directions = _directions(20.0, 89.0, 0.0, 180.0, 20000, seed=2)
    clear = S.sky_radiance_rgb(directions, S._without_cloud(noon)) / noon.exposure_scale
    transmittance, added = S._cloud_terms(directions, noon)
    tau = -np.log(np.clip(transmittance, 1e-12, 1.0))
    off_sun = directions @ noon.sun.direction() < math.cos(math.radians(20.0))
    thin = (tau > 0.02) & (tau < 0.3) & off_sun
    assert thin.sum() > 100
    ratio = ((transmittance[:, None] * clear + added) @ LUMA)[thin] / (clear @ LUMA)[thin]
    assert ratio.min() > 1.0


@pytest.mark.parametrize("hour", [8.0, 12.97])
def test_a_base_is_darker_than_its_sunlit_flank(hour: float) -> None:
    """Seen from below, a thick cloud sends down what it transmits; its flank, seen from the
    cloud's own height with the sun behind the observer, sends back what it reflects. Measured
    with each of its lights: the sun once and many times, the sky above, the ground below.

    For thick cloud, more than 30 deep along the view: that is what makes a base dark. A thin
    cumulus transmits half of a high sun diffusely (two-stream, tau 10) and its base is bright
    -- as small fair-weather cumulus at noon are -- and this model, one-dimensional along each
    sun chord, also leaves out the light a finite cloud loses through its sides."""
    conditions = _conditions(hour=hour)
    field = conditions.cloud
    sun_azimuth = conditions.sun.azimuth_deg
    up = _directions(60.0, 89.5, 0.0, 180.0, 40000, seed=3)
    t_base, base = S._cloud_terms(up, conditions)
    level = _directions(-3.0, 3.0, sun_azimuth + 180.0, 25.0, 20000, seed=4)
    rng = np.random.default_rng(5)
    origin = np.zeros_like(level)
    origin[:, 0] = rng.uniform(-3000.0, 3000.0, level.shape[0])
    origin[:, 1] = field.base_m + 0.4 * field.thickness_m
    origin[:, 2] = rng.uniform(-3000.0, 3000.0, level.shape[0])
    t_flank, flank = S._cloud_terms(level, conditions, origin_m=origin)
    base_l = (base @ LUMA)[t_base < math.exp(-30.0)]
    flank_l = (flank @ LUMA)[t_flank < math.exp(-30.0)]
    assert base_l.size > 1000 and flank_l.size > 1000
    # Measured 0.82 at a 44 degree sun and 0.89 at 58: the base is the darker by a clear margin.
    assert base_l.mean() < 0.95 * flank_l.mean()


def test_a_uniform_deck_seen_from_above_has_the_two_stream_albedo() -> None:
    """Looking down on an overcast deck, the sunlight term averages to the reflectance of a
    conservatively scattering layer, ``R = (1-g) tau / (2 mu0 + (1-g) tau)``, within 5 %. The old
    envelope-and-shadow model averaged 0.92 of it, and dropping its stand-in sky term took that to
    0.81."""
    conditions = _conditions(cover=1.0, genus="stratocumulus")
    field = conditions.cloud
    sun = conditions.sun.direction()
    mu0 = float(sun[1])
    rng = np.random.default_rng(6)
    n = 4000
    origin = np.zeros((n, 3))
    origin[:, 0] = rng.uniform(-3000.0, 3000.0, n)
    origin[:, 1] = field.top_m + 50.0
    origin[:, 2] = rng.uniform(-3000.0, 3000.0, n)
    down = _directions(-89.5, -80.0, 0.0, 180.0, n, seed=7)
    result = field.march(origin, down, sun_direction=sun, steps=128,
                         max_path_m=1.2 * field.thickness_m)
    tau = result.optical_depth * np.abs(down[:, 1])
    thick = tau > 5.0
    assert thick.sum() > 1000
    g = CLOUD_ASYMMETRY_G
    expected = (1.0 - g) * tau / (2.0 * mu0 + (1.0 - g) * tau)
    assert result.radiance[thick].mean() == pytest.approx(expected[thick].mean(), rel=0.05)


def test_the_two_streams_leave_the_faces_they_should() -> None:
    """Along the sunlight's chord: the stream running back toward the sun leaves the lit face at
    the layer's two-stream reflectance and the dark face at nothing; the one running on leaves the
    dark face at the diffuse transmission and enters the lit face at nothing. Linear between."""
    g = CLOUD_ASYMMETRY_G
    chord = 40.0
    scaled = (1.0 - g) * chord
    reflected = scaled / (2.0 + scaled)
    transmitted = 2.0 / (2.0 + scaled) - math.exp(-chord)
    back_lit, on_lit = _diffuse_sunlight(0.0, chord)
    back_dark, on_dark = _diffuse_sunlight(chord, 0.0)
    assert (back_lit, on_lit) == (pytest.approx(reflected), pytest.approx(0.0))
    assert (back_dark, on_dark) == (pytest.approx(0.0), pytest.approx(transmitted))
    back, on = _diffuse_sunlight(10.0, 30.0)
    assert (back, on) == (pytest.approx(0.75 * reflected), pytest.approx(0.25 * transmitted))


@pytest.mark.parametrize("depth", [0.5, 4.0, 50.0])
def test_a_uniform_layer_returns_its_two_stream_values_at_any_depth(depth: float) -> None:
    """Straight down onto a plane-parallel layer with the sun 50 degrees up, the multiply-scattered
    light is the layer's reflectance; straight up, its diffuse transmission -- thin or thick. The
    mean-intensity source this replaced read a 4-deep deck 37 % bright from above."""
    from weather_fx.core.clouds import CloudField, cloud_profile

    # 64 levels: the light maps are read between level centres, and at the slab's faces they
    # clamp to the outermost centre -- half a level of cloud. At 16 levels a 50-deep layer loses
    # 4 % to that; at 64, under 1 %. (A real cloud's faces are mostly inside the slab, where the
    # interpolation runs to the clear level beyond them.)
    deck = CloudField(cover=1.0, base_m=1000.0, profile=cloud_profile("stratus"), seed=3,
                      cells=32, levels=64, cell_m=100.0, optical_depth=depth,
                      detail_strength=0.0, billow_scale=0.0)
    deck._density[:] = 1.0  # a truly uniform layer: the identity under test, not the noise
    deck._signed = None
    deck._detail = None
    deck._sun_cache.clear()
    sigma = depth / deck.thickness_m
    object.__setattr__(deck, "optical_depth", depth)
    sun = np.array([0.0, math.sin(math.radians(50.0)), -math.cos(math.radians(50.0))])
    mu0 = float(sun[1])
    g = CLOUD_ASYMMETRY_G
    scaled = (1.0 - g) * depth
    reflected = scaled / (2.0 * mu0 + scaled)
    transmitted = 2.0 * mu0 / (2.0 * mu0 + scaled) - math.exp(-depth / mu0)
    assert deck.extinction_per_m == pytest.approx(sigma, rel=1e-6)
    above = deck.march(np.array([0.0, deck.top_m + 10.0, 0.0]), np.array([0.0, -1.0, 0.0]),
                       sun_direction=sun, steps=400)
    below = deck.march(np.array([0.0, 2.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                       sun_direction=sun, steps=400)
    single_back = float(above.radiance) - reflected
    assert float(above.radiance) == pytest.approx(reflected, rel=0.03, abs=0.01)
    assert float(below.radiance) - float(np.exp(-depth)) * 0 == pytest.approx(
        transmitted + _single_forward(depth, mu0, g), rel=0.05, abs=0.01)
    assert abs(single_back) < 0.03


def _single_forward(depth: float, mu0: float, g: float) -> float:
    """Single scattering of the direct beam seen straight up through a uniform layer, in the
    march's units: p(theta) / (4 mu0) times the integral of e^-(t/mu0) e^-(depth - t) dt."""
    cos_theta = mu0
    phase = (1 - g * g) / (1 + g * g - 2 * g * cos_theta) ** 1.5
    k = 1.0 / mu0 - 1.0
    integral = math.exp(-depth) * (math.expm1(-k * depth) / -k if abs(k) > 1e-9 else depth)
    return phase / (4.0 * mu0) * integral


# --- aerial perspective --------------------------------------------------------------------------


def test_the_air_in_front_of_a_point_far_up_the_sky_is_the_sky() -> None:
    """The aerial-perspective table is the sky view's own scattering integral kept at distances,
    so far enough along an upward ray -- 64 km, where the air above is a trace -- the air light
    is the sky itself: within 1.5 % from 30 degrees up. At 15 degrees 64 km reaches only 16.6 km
    of height, and the air above that still adds up to 4 % of the blue."""
    from weather_fx.core import atmosphere as A

    atm = A.atmosphere_for(2.8)
    for sun_el in (50.0, 15.0):
        view = A.sky_view(atm, sun_el)
        air = A.aerial_perspective(atm, sun_el)
        for elevations, tolerance in (((30.0, 60.0, 85.0), 0.015), ((15.0,), 0.05)):
            el = np.radians(np.array(elevations))
            az = np.radians(np.linspace(10.0, 180.0, el.size))
            direct, multiple, _ = air.at(el, az, np.full(el.size, A.AERIAL_PERSPECTIVE_MAX_M))
            assert np.allclose(direct + multiple, view.radiance(el, az), rtol=tolerance)


def test_contrast_falls_as_the_air_transmits_and_more_slowly_as_the_ray_climbs() -> None:
    """Koschmieder: a black object's contrast against the horizon is the air's transmittance to
    it -- exactly, in uniform air. Near the horizon at 5 km the table holds that to 5 %. Farther,
    the ray climbs out of the aerosol layer and the sky behind the object is dimmer than uniform
    air would make it, so the contrast stays above the transmittance, and more so with range:
    1.02, 1.07 and 1.16 of it at 5, 15 and 30 km, 0.5 degrees up."""
    from weather_fx.core import atmosphere as A

    atm = A.atmosphere_for(2.8)
    view, air = A.sky_view(atm, 40.0), A.aerial_perspective(atm, 40.0)
    el, az = np.radians([0.5]), np.radians([90.0])
    ratios = []
    for distance in (5_000.0, 15_000.0, 30_000.0):
        direct, multiple, transmittance = air.at(el, az, np.array([distance]))
        contrast = 1.0 - (direct + multiple) / view.radiance(el, az)
        ratios.append(float((contrast / transmittance).mean()))
    assert ratios[0] == pytest.approx(1.0, abs=0.05)
    assert 1.0 <= ratios[0] < ratios[1] < ratios[2] < 1.3


def test_a_far_cloud_takes_the_colour_of_the_horizon_and_a_near_one_keeps_its_own(noon) -> None:
    """With the air in front of it, a cloud 20 km off is much nearer the clear sky behind it than
    without; one overhead, about a kilometre off, hardly changes."""
    low = _directions(1.0, 4.0, 0.0, 180.0, 6000, seed=8)
    high = _directions(70.0, 89.0, 0.0, 180.0, 3000, seed=9)
    for directions, far in ((low, True), (high, False)):
        clear = S.sky_radiance_rgb(directions, S._without_cloud(noon)) @ LUMA
        hazy = S.sky_radiance_rgb(directions, noon) @ LUMA
        bare = S.sky_radiance_rgb(
            directions, dataclasses.replace(noon, aerial_perspective=False)) @ LUMA
        cloudy = np.abs(bare - clear) > 0.05 * clear
        assert cloudy.sum() > 100
        before = np.mean(np.abs(bare - clear)[cloudy])
        after = np.mean(np.abs(hazy - clear)[cloudy])
        if far:
            assert after < 0.6 * before
        else:
            assert after == pytest.approx(before, rel=0.15)
