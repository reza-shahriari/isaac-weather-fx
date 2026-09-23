"""The cloud field: shape, coverage, periodicity and light.

The tests that matter here are the ones that would pass for the *old* extruded model and fail for
a correct one, or the other way round. A cloud field is very easy to get plausibly wrong -- it
produces something cloud-shaped whatever you do -- so almost every assertion below is a contrast
against the specific wrong answer it replaces.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from weather_fx.core.clouds import (
    CLOUD_ASYMMETRY_G,
    CLOUD_TYPES,
    CloudField,
    cloud_profile,
    lifting_condensation_level_m,
)

# A small grid: these tests are about behaviour, not resolution, and the suite has to stay fast.
SMALL = dict(cells=96, levels=32, cell_m=80.0)


def field(**overrides) -> CloudField:
    kwargs = dict(
        cover=0.4, base_m=1000.0, profile=cloud_profile("cumulus"), seed=11, **SMALL
    )
    kwargs.update(overrides)
    return CloudField(**kwargs)


# --- coverage means what it says ---------------------------------------------------------------


@pytest.mark.parametrize("cover", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_the_sky_cover_that_comes_out_is_the_one_that_went_in(cover: float) -> None:
    """Coverage is an *input* here and is solved for, not a number that emerges from the noise's
    distribution and the spectral slope. Every model in this area quietly gets this wrong."""
    assert field(cover=cover).measured_cover == pytest.approx(cover, abs=0.02)


def test_zero_cover_is_a_clear_sky_and_costs_nothing() -> None:
    clear = field(cover=0.0)
    assert clear.measured_cover == 0.0
    assert float(np.max(clear.density(np.zeros(8), np.full(8, 1200.0), np.zeros(8)))) == 0.0


def test_cover_is_columns_not_volume() -> None:
    """A field of tall narrow towers fills little volume and hides much of the sky. Measuring the
    occupied *volume* fraction instead would report a quarter of the truth at four oktas."""
    f = field(cover=0.5)
    volume_fraction = float(np.mean(f._density > 0.5))
    assert f.measured_cover == pytest.approx(0.5, abs=0.02)
    assert volume_fraction < 0.25, "the two must not be confused; volume is far smaller"


# --- the shape is the whole point ----------------------------------------------------------------


def test_cross_sections_at_different_heights_are_different_shapes() -> None:
    """**The defect this model exists to fix.** An extruded height map has the same horizontal
    set at every level, only scaled, so any two levels that contain cloud score an intersection
    over union of 1.0. A real cumulus reorganises as it rises."""
    f = field()
    masks = [f._density[i] > 0.5 for i in (2, 10, 20, 28)]
    base = masks[0]

    def iou(a, b):
        return float((a & b).sum()) / max(float((a | b).sum()), 1.0)

    scores = [iou(base, m) for m in masks[1:]]
    assert all(s < 0.8 for s in scores), f"still an extrusion: {scores}"
    # And the change is progressive, not noise: further apart means less alike.
    assert scores[0] > scores[-1]


def test_nearby_levels_stay_coherent_because_a_tower_is_one_parcel_of_air() -> None:
    """The opposite failure: a spectrum written in *cells* rather than metres decorrelates over
    150 m vertically against 360 m horizontally, and the field renders as fuzz instead of towers."""
    f = field()
    a, b = f._density[10] > 0.5, f._density[12] > 0.5
    iou = float((a & b).sum()) / max(float((a | b).sum()), 1.0)
    assert iou > 0.55, f"vertical coherence lost: {iou}"


def test_a_cumulus_is_widest_at_its_base_and_tapers_to_its_top() -> None:
    """Which is what a flat-based convective cloud does, and the reason the profile is written as
    an occupied area per height rather than as a threshold offset."""
    f = field()
    areas = np.array([(f._density[i] > 0.5).mean() for i in range(f.levels)])
    assert areas[1] > areas[-2] * 2.0
    # Monotone enough: the top half must never be wider than the bottom half.
    assert areas[: f.levels // 2].mean() > areas[f.levels // 2 :].mean()


def test_the_base_is_flat_because_every_parcel_condenses_at_one_height() -> None:
    """Below the base there is no cloud at all, and immediately above it there is plenty. A model
    that ramps in over a few hundred metres has no flat base, and a cumulus field without flat
    bases does not read as a cumulus field."""
    f = field()
    below = f.density(np.zeros(64), np.full(64, f.base_m - 30.0), np.linspace(-2000, 2000, 64))
    assert float(np.max(below)) == 0.0
    just_above = np.array([(f._density[0] > 0.5).mean(), (f._density[1] > 0.5).mean()])
    assert just_above.min() > 0.5 * max(
        (f._density[i] > 0.5).mean() for i in range(f.levels)
    )


@pytest.mark.parametrize("name", sorted(CLOUD_TYPES))
def test_every_genus_builds_and_reports_the_cover_it_was_asked_for(name: str) -> None:
    f = field(profile=cloud_profile(name), cover=0.4)
    assert f.measured_cover == pytest.approx(0.4, abs=0.03)
    assert f.thickness_m == CLOUD_TYPES[name].thickness_m
    assert f.extinction_per_m > 0.0


def test_stratus_is_a_sheet_and_cumulus_is_not() -> None:
    """The genus has to actually change the morphology, not only the thickness."""
    sheet = field(profile=cloud_profile("stratus"))
    lumpy = field(profile=cloud_profile("cumulus"))

    def vertical_spread(f):
        areas = np.array([(f._density[i] > 0.5).mean() for i in range(f.levels)])
        return float(areas.std() / max(areas.mean(), 1e-9))

    assert vertical_spread(sheet) < vertical_spread(lumpy)


def test_an_unknown_genus_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ValueError, match="unknown cloud type"):
        cloud_profile("altocumulus lenticularis")


# --- the field has no edge -----------------------------------------------------------------------


def test_the_field_tiles_exactly_so_a_long_ray_never_finds_an_edge() -> None:
    """Built as an inverse FFT, so it is periodic to the bit rather than blended at a seam. An
    earlier model ran out of map at low elevation and put a hard cold band across the bottom of
    every oblique frame -- where a real cumulus field puts its densest wall."""
    f = field()
    period = f.cells * f.cell_m
    rng = np.random.default_rng(4)
    x = rng.uniform(-3000, 3000, 400)
    z = rng.uniform(-3000, 3000, 400)
    y = np.full(400, f.base_m + 0.4 * f.thickness_m)
    here = f.density(x, y, z)
    wrapped = f.density(x + period, y, z - 2 * period)
    assert np.allclose(here, wrapped, atol=1e-12)


def test_density_is_zero_outside_the_slab_and_bounded_inside_it() -> None:
    f = field()
    x = np.zeros(32)
    z = np.linspace(-1500, 1500, 32)
    assert float(np.max(f.density(x, np.full(32, f.base_m - 1.0), z))) == 0.0
    assert float(np.max(f.density(x, np.full(32, f.top_m + 1.0), z))) == 0.0
    inside = f.density(x, np.full(32, f.base_m + 0.5 * f.thickness_m), z)
    assert float(inside.min()) >= 0.0 and float(inside.max()) <= 1.0


# --- optical depth ---------------------------------------------------------------------------------


def test_optical_depth_is_calibrated_on_a_cloudy_column_not_on_the_whole_sky() -> None:
    """Otherwise the same cloud is twice as thick at four oktas as at eight, which is backwards.
    The median *cloudy* column is what a published optical depth for a genus means."""
    depths = {}
    for cover in (0.3, 0.8):
        f = field(cover=cover, optical_depth=20.0)
        columns = f._density.sum(axis=0) * (f.thickness_m / f.levels) * f.extinction_per_m
        depths[cover] = float(np.median(columns[columns > 0.5]))
    assert depths[0.3] == pytest.approx(depths[0.8], rel=0.25)
    assert depths[0.3] == pytest.approx(20.0, rel=0.25)


# --- marching, which both bands share -----------------------------------------------------------


def test_a_vertical_ray_through_a_cloudy_column_reproduces_that_column_s_depth() -> None:
    """The march and the column integral are the same quantity computed two ways; if they differ,
    one of the two bands is reading a different cloud from the other."""
    f = field()
    columns = f._density.sum(axis=0)
    j, k = np.unravel_index(int(np.argmax(columns)), columns.shape)
    x = (k - 0.5 * f.cells) * f.cell_m
    z = (j - 0.5 * f.cells) * f.cell_m
    direct = f.column_optical_depth(x, z, samples=256)
    marched = f.march(
        np.array([x, 0.0, z]), np.array([0.0, 1.0, 0.0]), steps=256
    ).optical_depth
    assert float(marched) == pytest.approx(direct, rel=0.05)


def test_a_ray_that_misses_the_slab_returns_a_clear_sky() -> None:
    f = field()
    down = f.march(np.array([0.0, 500.0, 0.0]), np.array([0.0, -1.0, 0.0]))
    assert float(down.optical_depth) == 0.0
    assert float(down.transmittance) == 1.0
    assert down.radiance is None


def test_an_oblique_ray_travels_further_through_the_cloud_than_a_vertical_one() -> None:
    """The reason the model is three-dimensional at all: at low elevation a ray crosses columns."""
    f = field(cover=0.9)
    origin = np.array([0.0, 2.0, 0.0])
    up = f.march(origin, np.array([0.0, 1.0, 0.0]), steps=128).optical_depth
    slant = f.march(
        origin, np.array([0.0, math.sin(math.radians(20.0)), -math.cos(math.radians(20.0))]),
        steps=128,
    ).optical_depth
    assert float(slant) > float(up)


def test_the_emission_height_rises_above_the_base_on_an_oblique_ray() -> None:
    """A cloud does not radiate from its base: it radiates from wherever the ray stopped being
    able to see through. On a slant through a broken field that is hundreds of metres higher, and
    an infrared band that assumes the base gets the temperature wrong by several kelvin."""
    f = field(cover=0.85)
    origin = np.array([0.0, 2.0, 0.0])
    result = f.march(
        origin,
        np.array([0.0, math.sin(math.radians(25.0)), -math.cos(math.radians(25.0))]),
        steps=128,
    )
    assert f.base_m < float(result.emission_height_m) < f.top_m


# --- light -------------------------------------------------------------------------------------


def test_a_thick_cloud_is_brighter_than_a_thin_one_which_single_scattering_cannot_do() -> None:
    """A forward march alone saturates: its value is bounded by the mean scattered term, so depth
    60 comes out no brighter than depth 20 although the real albedos are 0.89 and 0.72. The
    two-stream envelope is what restores that, and this test fails without it."""
    f = field(cover=0.8, cells=128, levels=32)
    sun = np.array([0.0, math.sin(math.radians(50.0)), -math.cos(math.radians(50.0))])
    rng = np.random.default_rng(1)
    dirs = np.stack(
        [rng.normal(0, 0.25, 4000), np.full(4000, 1.0), rng.normal(0, 0.25, 4000)], axis=-1
    )
    res = f.march(np.array([0.0, 2.0, 0.0]), dirs, sun_direction=sun, steps=64)
    thin = (res.optical_depth > 2.0) & (res.optical_depth < 6.0)
    thick = res.optical_depth > 40.0
    assert thin.any() and thick.any()
    assert res.radiance[thick].mean() > 1.5 * res.radiance[thin].mean()


def test_the_lit_side_approaches_the_two_stream_albedo_of_the_layer() -> None:
    """The energy is anchored on `R = (1 - g) tau / (2 mu0 + (1 - g) tau)`, the reflectance of a
    conservatively scattering layer, so a sunlit cumulus is *white* rather than the mid-grey a
    single-scattering march produces."""
    f = field(cover=0.85, cells=128, levels=32)
    elevation = 50.0
    sun = np.array([0.0, math.sin(math.radians(elevation)), -math.cos(math.radians(elevation))])
    rng = np.random.default_rng(2)
    dirs = np.stack(
        [rng.normal(0, 0.3, 6000), np.full(6000, 1.0), rng.normal(0, 0.3, 6000)], axis=-1
    )
    res = f.march(np.array([0.0, 2.0, 0.0]), dirs, sun_direction=sun, steps=64)
    band = (res.optical_depth > 25.0) & (res.optical_depth < 45.0)
    assert band.any()
    mu0 = math.sin(math.radians(elevation))
    tau = float(res.optical_depth[band].mean())
    expected = (1.0 - CLOUD_ASYMMETRY_G) * tau / (2.0 * mu0 + (1.0 - CLOUD_ASYMMETRY_G) * tau)
    assert float(res.radiance[band].max()) == pytest.approx(expected, rel=0.35)


def test_the_shadowed_side_is_darker_but_never_black() -> None:
    """Self-shadowing is what gives a cloud its shape; a floor is what keeps the shadow grey,
    because a real cloud's dark side is still lit the long way through and by the sky."""
    f = field(cover=0.85, cells=128, levels=32)
    sun = np.array([0.0, math.sin(math.radians(40.0)), -math.cos(math.radians(40.0))])
    rng = np.random.default_rng(3)
    dirs = np.stack(
        [rng.normal(0, 0.3, 6000), np.full(6000, 1.0), rng.normal(0, 0.3, 6000)], axis=-1
    )
    res = f.march(np.array([0.0, 2.0, 0.0]), dirs, sun_direction=sun, steps=64)
    lit = res.optical_depth > 20.0
    assert lit.any()
    values = res.radiance[lit]
    assert values.min() > 0.08, "shadow must not render black"
    # `MS_SHADOW_FLOOR` puts a ceiling of 1/0.45 = 2.2 on this ratio by construction, so the bar
    # is set inside it. A flat-shaded cloud -- the failure being guarded against -- scores 1.0.
    assert values.max() > 1.5 * values.min(), "and there must be real shading variation"


def test_no_sun_means_no_radiance_rather_than_a_silently_black_one() -> None:
    f = field()
    assert f.march(np.array([0.0, 2.0, 0.0]), np.array([0.0, 1.0, 0.0])).radiance is None


# --- the meteorology it is anchored on -----------------------------------------------------------


def test_the_cloud_base_follows_the_temperature_dewpoint_spread() -> None:
    """125 m per kelvin of spread: the dry adiabatic lapse rate less the dew-point one. It is the
    reason a whole field of cumulus has its bases on one level."""
    assert lifting_condensation_level_m(20.0, 20.0) == 0.0
    assert lifting_condensation_level_m(20.0, 12.0) == pytest.approx(1000.0)
    assert lifting_condensation_level_m(12.0, 20.0) == 0.0, "a negative spread is not a hole"


def test_a_field_refuses_impossible_inputs_rather_than_clamping_them() -> None:
    with pytest.raises(ValueError, match="fraction of sky"):
        field(cover=1.4)
    with pytest.raises(ValueError, match="below the ground"):
        field(base_m=-10.0)
    with pytest.raises(ValueError, match="at least"):
        field(cells=4)
