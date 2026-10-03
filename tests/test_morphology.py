"""The cloud field's shape from above, held to real clouds' (``core.morphology``).

First the rulers, on shapes whose answers are known, so that a field test failing means the field
and not the measurement. Then the field: the outline dimension, the size distribution and the
absence of streaks that satellite studies measure on real cumulus and stratocumulus.
"""
from __future__ import annotations

import numpy as np
import pytest

from weather_fx.core import morphology as M
from weather_fx.core.clouds import CloudField, cloud_profile

# --- the rulers --------------------------------------------------------------------------------


def _discs(radii_px, size=512, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    mask = np.zeros((size, size), dtype=bool)
    for r in radii_px:
        cy, cx = rng.integers(0, size, 2)
        dy = (y - cy + size // 2) % size - size // 2
        dx = (x - cx + size // 2) % size - size // 2
        mask |= dy * dy + dx * dx <= r * r
    return mask


def test_a_cloud_across_the_tile_edge_is_one_cloud() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:14, 30:] = True
    mask[10:14, :3] = True
    mask[20:22, 15:17] = True
    labels, count = M.label_periodic(mask)
    assert count == 2
    assert len(set(labels[10:14, 30:].ravel()) | set(labels[10:14, :3].ravel())) == 1


def test_smooth_discs_have_dimension_one() -> None:
    """``P ~ sqrt(A)`` for any family of one shape: D = 1, whatever the pixel staircase adds."""
    mask = _discs([6, 9, 13, 18, 25, 35, 6, 9, 13, 18, 25], size=1024)
    labels, count = M.label_periodic(mask)
    area, perimeter = M.areas_and_perimeters(labels, count, 15.0)
    assert M.area_perimeter_dimension(area, perimeter, 15.0) == pytest.approx(1.0, abs=0.06)


@pytest.mark.parametrize("b", [1.7, 2.5])
def test_the_size_exponent_is_recovered_from_a_power_law(b: float) -> None:
    rng = np.random.default_rng(3)
    lo, hi = 60.0, 800.0
    u = rng.random(4000)
    a = 1.0 - b
    sizes = (lo**a + u * (hi**a - lo**a)) ** (1.0 / a)
    fitted, used = M.size_exponent(sizes, lo, hi)
    assert used == 4000
    assert fitted == pytest.approx(b, abs=0.06)


def test_streaks_show_in_the_spectrum_and_blobs_do_not() -> None:
    blobs = _discs([8] * 120, size=512, seed=1)
    streaks = np.zeros((512, 512), dtype=bool)
    rng = np.random.default_rng(2)
    for col in rng.integers(0, 512, 40):
        streaks[:, col:col + 6] = True
    assert M.stripe_ratio(blobs, 30.0) < 1.5
    assert M.stripe_ratio(streaks, 30.0) > 5.0


def test_the_fringe_of_a_ramp_is_the_ramp() -> None:
    """A straight edge whose optical depth climbs 0 -> 2 over 20 pixels: 0.1 to 1 is 9 of them."""
    tau = np.tile(np.clip((np.arange(128) - 40.0) / 10.0, 0.0, 2.0), (64, 1))
    tau[:, 100:] = 0.0
    labels, count = M.label_periodic(tau >= 1.0)
    # Two outlines of 64 pixels each, and a 0.1 -> 1 band 9 pixels wide on the rising side only.
    assert M.fringe_width_m(tau, 15.0, labels, count) == pytest.approx(15.0 * 9 / 2, rel=0.12)


# --- the field ---------------------------------------------------------------------------------


def _plan(genus: str, cover: float, seed: int, pitch: float):
    field = CloudField(cover=cover, base_m=1000.0, profile=cloud_profile(genus), seed=seed)
    tau = M.plan_optical_depth(field, pitch)
    mask = tau >= M.MASK_OPTICAL_DEPTH
    labels, count = M.label_periodic(mask)
    return mask, M.areas_and_perimeters(labels, count, pitch)


def test_cumulus_has_the_outline_and_the_sizes_of_real_cumulus() -> None:
    """35 % fair-weather cumulus seen at 15 m, the resolution of the ASTER studies.

    D 1.3-1.4 (Lovejoy 1982: 1.35; Zhao & Di Girolamo 2007: 1.28) and b 1.7-2.0 between 60 and
    800 m (Neggers et al. 2003; Benner & Curry 1998), pooled over three seeds: 1.91. Before the
    coverage map carried features below 800 m and thermals of more than one size, b was 2.58:
    clouds about a kilometre across, and specks.
    """
    sizes = []
    for seed in (17, 3, 5):
        _, (area, perimeter) = _plan("cumulus", 0.35, seed, 15.0)
        assert 1.3 <= M.area_perimeter_dimension(area, perimeter, 15.0) <= 1.4
        sizes.append(np.sqrt(area))
    b, used = M.size_exponent(np.concatenate(sizes), 60.0, 800.0)
    assert used > 300
    assert 1.7 <= b <= 2.0


def test_stratocumulus_is_not_a_barcode() -> None:
    """Broken stratocumulus is cells (Wood 2012), and the field's stretch was along a fixed axis
    whatever the wind: a ratio of 4 along it. It must now sit in an isotropic field's own scatter,
    0.84-1.23 over four seeds of cumulus."""
    for seed in (17, 3):
        mask, _ = _plan("stratocumulus", 0.6, seed, 30.0)
        assert M.stripe_ratio(mask, 30.0) < 1.3
