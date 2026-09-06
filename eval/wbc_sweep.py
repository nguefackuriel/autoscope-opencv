"""Fit the two parameters that trade over-counting against under-counting.

SEED_FRACTION decides how aggressively the watershed splits a blob. Low values
split a multi-lobed nucleus into its lobes and over-count. High values merge two
touching cells into one and under-count.

CLOSE_KERNEL_FULL rejoins the lobes of one nucleus before labelling. Same trade
from the other side.

Neither can be chosen by eye, so both get fitted here against ground truth, and
the chosen values are written back into segmentation.py with this file named as
the source.
"""

import argparse
import json
import random
from pathlib import Path

import cv2 as cv

from autoscope import annotations as ann, segmentation
from eval_helpers import load_pairs  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def score(pairs, seed_fraction: float, close_px: int, stain_min: float, l_dark: float = 0.0,
          dark_min: float = 0.0, area_min: int = 1800, stain_min_dark: float = 30.0) -> dict:
    segmentation.SEED_FRACTION = seed_fraction
    segmentation.CLOSE_KERNEL_FULL = close_px
    segmentation.STAIN_MIN = stain_min
    segmentation.L_DARK_MAX = l_dark
    segmentation.DARK_FRACTION_MIN = dark_min
    segmentation.AREA_MIN_FULL = area_min
    segmentation.STAIN_MIN_DARK = stain_min_dark

    hits = fa = miss = 0
    bias = 0
    total_truth = 0
    for img, truth in pairs:
        # Already normalised in load_pairs, so do not repeat it per grid point.
        found = segmentation.find_white_cells(img, normalise_stain=False)
        from wbc_eval import match

        h, f, m = match(found, truth.white_cells)
        hits += h
        fa += f
        miss += m
        bias += len(found) - len(truth.white_cells)
        total_truth += len(truth.white_cells)

    precision = hits / (hits + fa) if hits + fa else 0.0
    recall = hits / (hits + miss) if hits + miss else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "seed_fraction": seed_fraction,
        "close_px": close_px,
        "stain_min": stain_min,
        "l_dark_max": l_dark,
        "dark_fraction_min": dark_min,
        "area_min_full": area_min,
        "stain_min_dark": stain_min_dark,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "count_bias_pct": round(100 * bias / max(total_truth, 1), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    # 100, one image per patient, seed 0: that is the fit set every held-out
    # number in docs/ is measured against. Passing more images pulls held-out
    # patients into the fit and silently shrinks the held-out set, which is
    # what happened on 2 September 2026 with 120 before it was caught.
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--raw", action="store_true",
                    help="sweep without stain normalisation, for comparison")
    args = ap.parse_args()

    pairs = load_pairs(args.images, normalise=not args.raw)
    fit_patients = sorted({t.path.parent.name for _, t in pairs})
    print(f"fitting on {len(pairs)} images from {len(fit_patients)} patients, "
          f"stain normalisation {'off' if args.raw else 'on'}\n")

    # STAIN_MIN joined the grid on 2 September 2026. The stain statistics
    # behind the normalisation were wrong until then (results-stain.md), and
    # the corrected reference moves the background a - b level by about 8
    # units, so a threshold fitted on the old images no longer sits where it
    # was meant to. Sweeping it here keeps the choice measured.
    # L_DARK_MAX joined on 3 September 2026: a nucleus is dark as well as
    # purple, and on the pinkest slides only the darkness survives. 0 is the
    # purple-only rule, kept in the grid so the choice stays measured.
    # DARK_FRACTION_MIN joined on 6 September 2026: the false alarms are the
    # bright purple clumps, and a blob has to be mostly dark to count
    # (eval/wbc_fp_look.py). L_DARK_MAX stays at the 130 chosen on 3 September;
    # the gate needs the dark path to mean anything.
    # The first gate sweep (6 September, 48 points, wbc_sweep_dark_fraction.log)
    # found the gate lifts precision from 0.86 to 0.95 at equal recall, but
    # the rule kept it off: on the fit set the false alarms were balancing
    # the misses (recall 0.91), and removing them alone reads -6.6%. So the
    # second sweep pairs the gate with the two knobs that can give recall
    # back, the area floor (a small nucleus or a lobe fragment) and the
    # purple floor of the dark path, and keeps the no-gate point as control.
    results = []
    print(f"{'Ldark':>6} {'dark>=':>6} {'area':>6} {'sdark':>6} {'stain':>6} {'seed':>6} {'close':>6} {'prec':>7} {'recall':>7} {'F1':>7} {'bias %':>8}")
    grid = [(130.0, 0.0, 1800, 30.0, 65.0, 0.75, 24)]   # what ships, the control
    for dark_min in (0.3, 0.5):
        for area_min in (1200, 1500, 1800):
            for stain_min_dark in (20.0, 30.0):
                for seed_fraction in (0.55, 0.65, 0.75):
                    grid.append((130.0, dark_min, area_min, stain_min_dark, 65.0, seed_fraction, 24))
    for l_dark, dark_min, area_min, stain_min_dark, stain_min, seed_fraction, close_px in grid:
        r = score(pairs, seed_fraction, close_px, stain_min, l_dark, dark_min, area_min, stain_min_dark)
        results.append(r)
        print(f"{r['l_dark_max']:>6.0f} {r['dark_fraction_min']:>6.1f} {r['area_min_full']:>6d} {r['stain_min_dark']:>6.0f} "
              f"{r['stain_min']:>6.0f} {r['seed_fraction']:>6.2f} {r['close_px']:>6} "
              f"{r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f} "
              f"{r['count_bias_pct']:>+8.1f}", flush=True)

    best = max(results, key=lambda r: r["f1"])
    OUT.mkdir(exist_ok=True)
    (OUT / "wbc_sweep.json").write_text(json.dumps(
        {"best": best, "grid": results, "fit_patients": fit_patients}, indent=2))
    print(f"\nbest F1 {best['f1']:.3f} at l_dark={best['l_dark_max']:.0f} dark_min={best['dark_fraction_min']} area_min={best['area_min_full']} stain_min_dark={best['stain_min_dark']:.0f} stain_min={best['stain_min']:.0f} "
          f"seed_fraction={best['seed_fraction']} close_px={best['close_px']} "
          f"(bias {best['count_bias_pct']:+.1f}%)")

    # A detector that is right on average matters more than one with the best F1
    # when the number feeds a density, so report the least biased too.
    fair = min(results, key=lambda r: abs(r["count_bias_pct"]))
    print(f"least biased: {fair['count_bias_pct']:+.1f}% at dark_min={fair['dark_fraction_min']} area_min={fair['area_min_full']} "
          f"stain_min_dark={fair['stain_min_dark']:.0f} seed_fraction={fair['seed_fraction']} close_px={fair['close_px']} (F1 {fair['f1']:.3f})")


if __name__ == "__main__":
    main()
