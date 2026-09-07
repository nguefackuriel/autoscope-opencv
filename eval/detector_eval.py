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
patch network's held-out F1 of 0.762, which is the entire point of the exercise.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autoscope import annotations as ann, parasites as par
from wbc_eval import match
import splits

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parent / "results"

# Same tolerance as parasite_eval.py. Do not change it without changing that
# file too, or the numbers stop being comparable.
TOLERANCE_PX = 15.0

# Two detections closer than this are the same parasite seen in two overlapping
# tiles. Parasites are rarely within 20 px of each other, and the chromatin dot
# is ~10 px, so 12 merges duplicates without merging neighbours.
DEDUPE_PX = 12.0

SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]


class Detection:
    """Carries .x and .y so wbc_eval.match can consume it unchanged."""

    __slots__ = ("x", "y", "score")

    def __init__(self, x, y, score):
        self.x = float(x)
        self.y = float(y)
        self.score = float(score)


def tile_origins(size, tile, stride):
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def dedupe(dets, min_sep=DEDUPE_PX):
    """Greedy suppression by centre distance, strongest detection first.

    Distance rather than IoU because the metric is a centre distance: a
    duplicate 3 px away would otherwise score as one hit plus one false alarm
    and quietly halve precision.
    """
    kept = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all((d.x - k.x) ** 2 + (d.y - k.y) ** 2 > min_sep ** 2 for k in kept):
            kept.append(d)
    return kept


class YoloBackend:
    name = "yolo"

    def __init__(self, weights):
        from ultralytics import YOLO
        self.model = YOLO(weights)

    def __call__(self, tiles, conf):
        out = self.model.predict(tiles, conf=conf, verbose=False)
        results = []
        for r in out:
            b = r.boxes
            if b is None or len(b) == 0:
                results.append(np.zeros((0, 5), np.float32))
                continue
            xyxy = b.xyxy.cpu().numpy()
            scores = b.conf.cpu().numpy().reshape(-1, 1)
            results.append(np.hstack([xyxy, scores]).astype(np.float32))
        return results


class FrcnnBackend:
    name = "frcnn"

    def __init__(self, weights, device=None):
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
        from frcnn_model import build_model
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(num_classes=2)
        state = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.to(self.device).eval()

    def __call__(self, tiles, conf):
        torch = self.torch
        batch = [torch.from_numpy(cv.cvtColor(t, cv.COLOR_BGR2RGB))
                 .permute(2, 0, 1).float().div(255.0).to(self.device) for t in tiles]
        with torch.no_grad():
            preds = self.model(batch)
        results = []
        for p in preds:
            keep = p["scores"] >= conf
            xyxy = p["boxes"][keep].cpu().numpy()
            scores = p["scores"][keep].cpu().numpy().reshape(-1, 1)
            if len(xyxy) == 0:
                results.append(np.zeros((0, 5), np.float32))
            else:
                results.append(np.hstack([xyxy, scores]).astype(np.float32))
        return results


def detect_field(backend, prepared, mask, tile, stride, conf, batch=8, min_fov=0.9):
    """Run the detector over one whole field and return deduplicated centres."""
    height, width = prepared.shape[:2]
    crops, offsets = [], []
    for oy in tile_origins(height, tile, stride):
        for ox in tile_origins(width, tile, stride):
            if float((mask[oy:oy + tile, ox:ox + tile] > 0).mean()) < min_fov:
                continue
            crops.append(prepared[oy:oy + tile, ox:ox + tile])
            offsets.append((ox, oy))

    dets = []
    for i in range(0, len(crops), batch):
        chunk, chunk_off = crops[i:i + batch], offsets[i:i + batch]
        for arr, (ox, oy) in zip(backend(chunk, conf), chunk_off):
            for x1, y1, x2, y2, score in arr:
                dets.append(Detection((x1 + x2) / 2 + ox, (y1 + y2) / 2 + oy, score))
    return dedupe(dets)


def evaluate(backend, patients, per_patient, seed, tile, stride, thresholds):
    """Score at every threshold in one pass, so the field is read once."""
    rng = random.Random(seed)
    rows = {t: [] for t in thresholds}
    lowest = min(thresholds)

    for n, patient in enumerate(sorted(patients), 1):
        candidates = sorted((RAW / patient).glob("*.jpg"))
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
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--stride", type=int, default=512)
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
    print(f"\ncompare against the patch network: held-out F1 0.762, precision 0.841")
    print(f"written to {OUT / f'detector_eval_{tag}.json'}")


if __name__ == "__main__":
    main()
