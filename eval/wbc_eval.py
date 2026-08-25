"""Score the white cell detector against ground truth.

Counting alone hides mistakes: a false positive and a missed cell cancel out and
the total looks perfect. So every detection is matched to a ground truth point
before anything is counted.

Matching is greedy nearest neighbour within a tolerance, and the tolerance is
one nucleus radius. Ground truth is a single click near the middle of a cell, so
a detection landing inside that cell is correct even when the click was off
centre.
"""

import argparse
import json
import random
from pathlib import Path

import cv2 as cv
import numpy as np

from autoscope import annotations as ann
from autoscope.segmentation import find_white_cells

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parent / "results"

# One nucleus radius. Measured median is 51 px, p90 is 61.
TOLERANCE_PX = 55.0


def match(found: list, truth: list, tolerance: float = TOLERANCE_PX) -> tuple[int, int, int]:
    """Greedy nearest matching. Returns hits, false alarms, misses."""
    if not found:
        return 0, 0, len(truth)
    if not truth:
        return 0, len(found), 0

    f = np.array([[c.x, c.y] for c in found])
    t = np.array([[a.x, a.y] for a in truth])
    d = np.linalg.norm(f[:, None, :] - t[None, :, :], axis=2)

    hits = 0
    used_f, used_t = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    for i, j in order:
        if d[i, j] > tolerance:
            break
        if i in used_f or j in used_t:
            continue
        used_f.add(int(i))
        used_t.add(int(j))
        hits += 1

    return hits, len(found) - hits, len(truth) - hits


def evaluate(images: int, seed: int = 0, holdout: bool = False) -> dict:
    """Score the detector. With holdout=True, skip every patient the sweep used.

    The parameters were fitted on ground truth, so reporting on the same images
    would flatter them. The honest number is the one from patients the fit never
    saw.
    """
    rng = random.Random(seed)
    patients = sorted(p for p in RAW.iterdir() if p.is_dir() and p.name.startswith("TF"))

    if holdout:
        sweep = OUT / "wbc_sweep.json"
        if not sweep.exists():
            raise SystemExit("run wbc_sweep.py first, holdout needs its patient list")
        fitted = set(json.loads(sweep.read_text()).get("fit_patients", []))
        patients = [p for p in patients if p.name not in fitted]
        print(f"holdout: {len(patients)} patients the fit never saw")

    rows = []
    for patient in patients:
        candidates = sorted(patient.glob("*.jpg"))
        if not candidates:
            continue
        for path in rng.sample(candidates, min(images, len(candidates))):
            img = cv.imread(str(path))
            if img is None:
                continue
            h, w = img.shape[:2]
            truth = ann.for_image(path, w, h)
            if truth is None:
                continue

            found = find_white_cells(img)
            hits, false_alarms, misses = match(found, truth.white_cells)
            rows.append({
                "patient": patient.name,
                "image": path.name,
                "found": len(found),
                "truth": len(truth.white_cells),
                "hits": hits,
                "false_alarms": false_alarms,
                "misses": misses,
            })
        if len(rows) % 50 == 0 and rows:
            print(f"  {len(rows)} images")

    hits = sum(r["hits"] for r in rows)
    fa = sum(r["false_alarms"] for r in rows)
    miss = sum(r["misses"] for r in rows)
    precision = hits / (hits + fa) if hits + fa else 0.0
    recall = hits / (hits + miss) if hits + miss else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    errors = np.array([r["found"] - r["truth"] for r in rows], dtype=float)
    truths = np.array([r["truth"] for r in rows], dtype=float)

    return {
        "images": len(rows),
        "patients": len({r["patient"] for r in rows}),
        "cells_in_truth": int(truths.sum()),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "count_bias": round(float(errors.mean()), 3),
        "count_mae": round(float(np.abs(errors).mean()), 3),
        "count_bias_pct": round(float(errors.sum() / max(truths.sum(), 1)) * 100, 2),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-patient", type=int, default=2)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--holdout", action="store_true",
                    help="only patients the parameter sweep never saw")
    args = ap.parse_args()

    report = evaluate(args.per_patient, holdout=args.holdout)
    OUT.mkdir(exist_ok=True)
    (OUT / f"wbc_eval_{args.tag}.json").write_text(json.dumps(report, indent=2))

    print(f"\n{report['images']} images, {report['patients']} patients, "
          f"{report['cells_in_truth']} annotated white cells\n")
    print(f"  precision      {report['precision']:.3f}")
    print(f"  recall         {report['recall']:.3f}")
    print(f"  F1             {report['f1']:.3f}")
    print(f"  count bias     {report['count_bias']:+.2f} cells per image "
          f"({report['count_bias_pct']:+.1f}%)")
    print(f"  count MAE      {report['count_mae']:.2f} cells per image")


if __name__ == "__main__":
    main()
