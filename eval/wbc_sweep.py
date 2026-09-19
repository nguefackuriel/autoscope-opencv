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

# --- repo paths: run from anywhere, installed or not -------------------------
# Azure ML runs the code snapshot without `pip install -e .`, so the editable
# install that normally puts src/ on sys.path is absent and every `from
# autoscope import ...` fails. This depends on __file__ alone, so it survives
# any working directory, any launch style, and a missing install.
import sys as _sys, pathlib as _pl
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d)
                 for _d in ("src", "eval", "train")]
# ----------------------------------------------------------------------------

from autoscope import annotations as ann, segmentation
from eval_helpers import load_pairs  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def score(pairs, seed_fraction: float, close_px: int) -> dict:
    segmentation.SEED_FRACTION = seed_fraction
    segmentation.CLOSE_KERNEL_FULL = close_px

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
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "count_bias_pct": round(100 * bias / max(total_truth, 1), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=120)
    ap.add_argument("--raw", action="store_true",
                    help="sweep without stain normalisation, for comparison")
    args = ap.parse_args()

    pairs = load_pairs(args.images, normalise=not args.raw)
    fit_patients = sorted({t.path.parent.name for _, t in pairs})
    print(f"fitting on {len(pairs)} images from {len(fit_patients)} patients, "
          f"stain normalisation {'off' if args.raw else 'on'}\n")

    results = []
    print(f"{'seed':>6} {'close':>6} {'prec':>7} {'recall':>7} {'F1':>7} {'bias %':>8}")
    for seed_fraction in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
        for close_px in (24, 44, 64, 88):
            r = score(pairs, seed_fraction, close_px)
            results.append(r)
            print(f"{r['seed_fraction']:>6.2f} {r['close_px']:>6} {r['precision']:>7.3f} "
                  f"{r['recall']:>7.3f} {r['f1']:>7.3f} {r['count_bias_pct']:>+8.1f}")

    best = max(results, key=lambda r: r["f1"])
    OUT.mkdir(exist_ok=True)
    (OUT / "wbc_sweep.json").write_text(json.dumps(
        {"best": best, "grid": results, "fit_patients": fit_patients}, indent=2))
    print(f"\nbest F1 {best['f1']:.3f} at seed_fraction={best['seed_fraction']} "
          f"close_px={best['close_px']} (bias {best['count_bias_pct']:+.1f}%)")

    # A detector that is right on average matters more than one with the best F1
    # when the number feeds a density, so report the least biased too.
    fair = min(results, key=lambda r: abs(r["count_bias_pct"]))
    print(f"least biased: {fair['count_bias_pct']:+.1f}% at seed_fraction={fair['seed_fraction']} "
          f"close_px={fair['close_px']} (F1 {fair['f1']:.3f})")


if __name__ == "__main__":
    main()
