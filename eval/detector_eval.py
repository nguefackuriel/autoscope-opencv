"""Score a trained box detector the way everything else in this repo is scored.

WHY NOT mAP

The obvious metric for a detector is mAP at some IoU, and it would be the wrong
one here. The boxes these models are trained on came out of boxes.py, which
derives them from the stain. They are an algorithm's opinion, not ground truth.
Scoring IoU against them would mostly measure how faithfully the detector
imitates boxes.py, and a perfect score would mean we had reproduced the top-hat
generator we set out to beat.

So a detection is reduced to its box centre and matched against the annotated
centre with `wbc_eval.match`, greedy nearest, 15 px tolerance. Identical to
parasite_eval.py. That is what makes a number here directly comparable with the
patch network's held-out F1 of 0.763, which is the entire point of the exercise.

WHY TILED INFERENCE

The model was trained on 640 px tiles at native resolution, because a parasite
is ~10 px in a ~3000 px field. Inference has to see the same scale, so the field
is tiled the same way and detections are mapped back and deduplicated.

THE THRESHOLD MUST NOT BE CHOSEN HERE

Sweeping a confidence threshold on the held-out patients and reporting the best
one is fitting on them. `--split val` sweeps on fit patients held back from
training; `--split holdout` scores once, at a threshold already decided. The
sweep refuses to run on holdout without an explicit override.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2 as cv
import numpy as np


# --- repo paths: run from anywhere, installed or not -------------------------
# A remote runner executes the code snapshot without `pip install -e .`, so the
# editable install that normally puts src/ on sys.path is absent and every
# `from autoscope import ...` fails. True of AWS Batch and SageMaker as much as
# of the GPU box the detectors are trained on. This depends on __file__ alone,
# so it survives any working directory, any launch style, and a missing install.
import sys as _sys, pathlib as _pl
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d)
                 for _d in ("src", "eval", "train")]
# ----------------------------------------------------------------------------

from autoscope import annotations as ann, parasites as par
from autoscope.tiling import MIN_FOV, STRIDE_PX, TILE_PX, covered_tiles
from wbc_eval import match
import splits

# Patient images: resolved via splits.patient_root(), not assumed to sit
# directly under data/raw -- download.py may nest them one level down.
OUT = Path(__file__).resolve().parent / "results"

# Same tolerance as parasite_eval.py. Do not change it without changing that
# file too, or the numbers stop being comparable.
TOLERANCE_PX = 15.0

# Two detections closer than this are the same parasite seen in two overlapping
# tiles; the tiling, the dedupe and the backends live in autoscope.boxes since
# 11 September 2026, so the pipeline (parasites.find_parasites, the chain) runs
# a detector exactly the way this scorer does.
from autoscope.boxes import (DEDUPE_PX, Detection, FrcnnBackend, YoloBackend,  # noqa: E402,F401
                             dedupe, detect_field, tile_origins)

SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]


def evaluate(backend, patients, per_patient, seed, tile, stride, thresholds):
    """Score at every threshold in one pass, so the field is read once."""
    root = splits.patient_root()
    rows = {t: [] for t in thresholds}
    lowest = min(thresholds)

    for n, patient in enumerate(sorted(patients), 1):
        # Data cleaning: an image with no ground truth cannot score a detection
        # right or wrong, so it is excluded from the evaluation set rather than
        # sampled and then skipped. Sampling first would silently return fewer
        # images than asked for, and the shortfall would fall entirely on the
        # 37 patients that have unannotated fields.
        candidates = splits.annotated_images(root / patient)
        rng = splits.image_rng(seed, patient)
        if not candidates:
            continue
        for path in rng.sample(candidates, min(per_patient, len(candidates))):
            image = cv.imread(str(path))
            if image is None:
                continue
            h_full, w_full = image.shape[:2]
            truth = ann.for_image(path, w_full, h_full)
            if truth is None:
                continue

            prepared, mask, ox_fov, oy_fov = par._prepare(image, True, True)
            found = detect_field(backend, prepared, mask, tile, stride, lowest)

            shifted = [ann.Annotation("Parasite", p.x - ox_fov, p.y - oy_fov, p.radius)
                       for p in truth.parasites]
            for t in thresholds:
                sub = [d for d in found if d.score >= t]
                hits, fa, miss = match(sub, shifted, TOLERANCE_PX)
                rows[t].append({
                    "patient": patient, "image": path.name,
                    "found": len(sub), "truth": len(shifted),
                    "hits": hits, "false_alarms": fa, "misses": miss,
                })
        if n % 10 == 0:
            print(f"  {n} patients")

    report = {}
    for t, rs in rows.items():
        hits = sum(r["hits"] for r in rs)
        fa = sum(r["false_alarms"] for r in rs)
        miss = sum(r["misses"] for r in rs)
        precision = hits / (hits + fa) if hits + fa else 0.0
        recall = hits / (hits + miss) if hits + miss else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        errors = np.array([r["found"] - r["truth"] for r in rs], dtype=float)
        truths = np.array([r["truth"] for r in rs], dtype=float)
        report[f"{t:.2f}"] = {
            "images": len(rs),
            "patients": len({r["patient"] for r in rs}),
            "parasites_in_truth": int(truths.sum()),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "count_bias": round(float(errors.mean()), 3) if len(errors) else None,
            "count_mae": round(float(np.abs(errors).mean()), 3) if len(errors) else None,
            "count_bias_pct": round(float(errors.sum() / max(truths.sum(), 1)) * 100, 2),
        }
    return report, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["yolo", "frcnn"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", choices=["val", "holdout"], default="val")
    ap.add_argument("--threshold", type=float,
                    help="score once at this threshold. Required for holdout.")
    ap.add_argument("--sweep", action="store_true", help="report every threshold")
    ap.add_argument("--allow-holdout-sweep", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--per-patient", type=int, default=3)
    ap.add_argument("--tile", type=int, default=TILE_PX)
    ap.add_argument("--stride", type=int, default=STRIDE_PX)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.split == "holdout" and args.sweep and not args.allow_holdout_sweep:
        raise SystemExit(
            "Refusing to sweep thresholds on the held-out patients.\n"
            "Picking the best threshold there is fitting on them, and the\n"
            "generalisation claim in the report depends on not doing it.\n"
            "Sweep with --split val, then score holdout once at --threshold.")
    if args.split == "holdout" and not args.sweep and args.threshold is None:
        raise SystemExit("--threshold is required for holdout: decide it on val first")

    if args.split == "holdout":
        patients = splits.holdout_patients()
    else:
        _train, patients = splits.train_val(seed=args.seed)
    print(f"{args.split}: {len(patients)} patients")

    thresholds = SWEEP if args.sweep else [args.threshold or 0.25]
    backend = YoloBackend(args.weights) if args.backend == "yolo" \
        else FrcnnBackend(args.weights)

    report, rows = evaluate(backend, patients, args.per_patient, args.seed,
                            args.tile, args.stride, thresholds)

    tag = args.tag or f"{args.backend}_{args.split}"
    OUT.mkdir(exist_ok=True)
    payload = {
        "backend": args.backend, "weights": str(args.weights),
        "split": args.split, "split_source": "eval/results/wbc_sweep.json:fit_patients",
        "tolerance_px": TOLERANCE_PX, "tile_px": args.tile, "stride_px": args.stride,
        "per_patient": args.per_patient, "seed": args.seed,
        "by_threshold": report,
    }
    (OUT / f"detector_eval_{tag}.json").write_text(json.dumps(payload, indent=2))

    print(f"\n{'thr':>5} {'prec':>7} {'recall':>7} {'F1':>7} {'bias%':>8}")
    for t, r in sorted(report.items()):
        print(f"{t:>5} {r['precision']:>7.3f} {r['recall']:>7.3f} "
              f"{r['f1']:>7.3f} {r['count_bias_pct']:>+8.1f}")
    print("\nthe shipped pair on the same held-out patients (docs/results-parasites.md 15b):")
    print("  precision 0.814  recall 0.783  F1 0.798  count bias -3.8%  (99 images, 2 per patient)")
    print("  A better F1 with a worse count bias is not a better detector: the")
    print("  count is the denominator of every density. And F1 alone does not ship:")
    print("  run SKIP_TRAIN=1 MODEL=yolo WEIGHTS=<best> bash eval/chain.sh for the")
    print("  density per slide, the sparse band, the clean fields and the limit of detection.")
    print(f"written to {OUT / f'detector_eval_{tag}.json'}")


if __name__ == "__main__":
    main()
