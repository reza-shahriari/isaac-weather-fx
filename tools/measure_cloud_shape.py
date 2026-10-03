"""Measure the cloud field's shape from above against real clouds. Plain Python, no Isaac Sim.

    python tools/measure_cloud_shape.py                      # every genus, 15 m pixels
    python tools/measure_cloud_shape.py --genus cumulus --cover 0.35 --pitch 15 --png plan.png

For each genus and cover it builds the field for three seeds, takes the plan-view mask of columns
at visible optical depth >= 1 (``core.morphology``), and prints:

* ``D``: the area-perimeter dimension, per seed (real clouds 1.3-1.4);
* ``b``: the size exponent of ``n(l) ~ l^-b`` between 60 and 800 m, all seeds pooled (1.7-2.0);
* ``stripe``: spectral power along the field's z axis over the annulus mean, per seed (1 isotropic);
* ``fringe``: the mean width of the band between optical depth 0.1 and 1, metres;
* the cover the field measured, per seed.

``--png`` writes the first seed's plan view, two tiles side by side, so the tiling shows.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "exts" / "weather.fx"))

from weather_fx.core import clouds, morphology  # noqa: E402

CASES = (("cumulus", 0.15), ("cumulus", 0.35), ("congestus", 0.35), ("stratocumulus", 0.6),
         ("stratus", 0.9), ("cirrus", 0.4))


def measure(genus: str, cover: float, pitch: float, seeds=(17, 3, 5), png: str = "") -> dict:
    out = dict(genus=genus, cover=cover, D=[], stripe=[], fringe=[], measured_cover=[])
    sizes = []
    for i, seed in enumerate(seeds):
        field = clouds.CloudField(cover=cover, base_m=1000.0, profile=clouds.cloud_profile(genus),
                                  seed=seed)
        tau = morphology.plan_optical_depth(field, pitch)
        mask = tau >= morphology.MASK_OPTICAL_DEPTH
        labels, count = morphology.label_periodic(mask)
        area, perimeter = morphology.areas_and_perimeters(labels, count, pitch)
        out["D"].append(morphology.area_perimeter_dimension(area, perimeter, pitch))
        out["stripe"].append(morphology.stripe_ratio(mask, pitch))
        out["fringe"].append(morphology.fringe_width_m(tau, pitch, labels, count))
        out["measured_cover"].append(field.measured_cover)
        sizes.append(np.sqrt(area))
        if png and i == 0:
            from PIL import Image  # only for the picture

            shade = (255 * (1.0 - np.exp(-np.tile(tau, (1, 2)) / 4.0))).astype(np.uint8)
            Image.fromarray(shade).save(png)
    out["b"], out["clouds"] = morphology.size_exponent(np.concatenate(sizes), 60.0, 800.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--genus")
    ap.add_argument("--cover", type=float, default=0.35)
    ap.add_argument("--pitch", type=float, default=15.0)
    ap.add_argument("--png", default="")
    args = ap.parse_args()
    cases = [(args.genus, args.cover)] if args.genus else CASES
    for genus, cover in cases:
        start = time.time()
        r = measure(genus, cover, args.pitch, png=args.png)
        print(f"{genus:13s} {cover:.2f}  D {np.round(r['D'], 3)}  b {r['b']:.2f} ({r['clouds']} clouds)"
              f"  stripe {np.round(r['stripe'], 2)}  fringe {np.nanmean(r['fringe']):.0f} m"
              f"  cover {np.round(r['measured_cover'], 3)}  {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
