"""Score the parasite detector end to end on patients the fit never saw.

Same discipline as wbc_eval.py: every detection is matched to a ground truth
centre before anything is counted, so a false alarm and a miss cannot cancel.

Tolerance is 15 px. The annotated centre is itself about 8 px off the
chromatin dot at the median (see annotations.py), and parasites are rarely
closer than 20 px to each other, so 15 is the widest tolerance that does not
let one detection claim two parasites.

Beyond precision and recall, the number that matters clinically is the RATIO:
parasites per white cell in the same image, because that is what becomes a
density. So the report also compares our ratio with the ground truth ratio,
image by image, and gives the Bland-Altman style bias and limits.
"""

import argparse
import json
import random
from pathlib import Path

import cv2 as cv
import numpy as np

from autoscope import annotations as ann
from autoscope.parasites import HYBRID_PATH, PatchScorer, Scorer, find_parasites
from autoscope.segmentation import find_white_cells
from wbc_eval import match

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parent / "results"
TOLERANCE_PX = 15.0
SWEEP = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]


def evaluate(per_patient: int, seed: int, holdout: bool, threshold: float,
             max_patients: int | None = None, model: str = "cnn") -> dict:
    rng = random.Random(seed)
    patients = sorted(p for p in RAW.iterdir() if p.is_dir() and p.name.startswith("TF"))
    if holdout:
        fitted = set(json.loads((OUT / "wbc_sweep.json").read_text())["fit_patients"])
        patients = [p for p in patients if p.name not in fitted]
        print(f"holdout: {len(patients)} patients the fit never saw")
    if max_patients:
        patients = patients[:max_patients]

    scorer = {"cnn": lambda: PatchScorer.load(), "hybrid": lambda: PatchScorer.load(HYBRID_PATH),
              "mlp": lambda: Scorer.load()}[model]()
    rows = []
    for patient in patients:
        candidates = sorted(patient.glob("*.jpg"))
        if not candidates:
            continue
        for path in rng.sample(candidates, min(per_patient, len(candidates))):
            img = cv.imread(str(path))
            if img is None:
                continue
            h, w = img.shape[:2]
            truth = ann.for_image(path, w, h)
            if truth is None:
                continue
            scored = find_parasites(img, threshold=0.0, model=model, scorer=scorer)
            found = [f for f in scored if f.score >= threshold]
            hits, fa, miss = match(found, truth.parasites, TOLERANCE_PX)
            wbc = len(find_white_cells(img))
            # the same image at other thresholds, for the sensitivity table
            at = {}
            for thr in SWEEP:
                sub = [f for f in scored if f.score >= thr]
                h, f_, m = match(sub, truth.parasites, TOLERANCE_PX)
                at[str(thr)] = [len(sub), h, f_, m]
            rows.append({
                "patient": patient.name, "image": path.name,
                "found": len(found), "truth": len(truth.parasites),
                "hits": hits, "false_alarms": fa, "misses": miss,
                "wbc_found": wbc, "wbc_truth": len(truth.white_cells),
                "at": at,
            })
        if rows and len(rows) % 25 == 0:
            print(f"  {len(rows)} images")

    hits = sum(r["hits"] for r in rows); fa = sum(r["false_alarms"] for r in rows)
    miss = sum(r["misses"] for r in rows)
    precision = hits / (hits + fa) if hits + fa else 0.0
    recall = hits / (hits + miss) if hits + miss else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = np.array([r["found"] - r["truth"] for r in rows], float)
    truths = np.array([r["truth"] for r in rows], float)

    # parasites per white cell: ours against ground truth, per image
    ours = np.array([r["found"] / r["wbc_found"] for r in rows if r["wbc_found"] > 0 and r["wbc_truth"] > 0])
    gt = np.array([r["truth"] / r["wbc_truth"] for r in rows if r["wbc_found"] > 0 and r["wbc_truth"] > 0])
    # log ratio is the natural scale for a density that spans two decades
    keep = (ours > 0) & (gt > 0)
    log_diff = np.log(ours[keep]) - np.log(gt[keep])

    sweep = []
    for thr in SWEEP:
        k = str(thr)
        H = sum(r["at"][k][1] for r in rows); FA = sum(r["at"][k][2] for r in rows); M = sum(r["at"][k][3] for r in rows)
        pr = H / (H + FA) if H + FA else 0.0; rc = H / (H + M) if H + M else 0.0
        sweep.append({"threshold": thr, "precision": round(pr, 4), "recall": round(rc, 4),
                      "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0,
                      "count_bias_pct": round((H + FA - (H + M)) / max(H + M, 1) * 100, 2)})

    return {
        "sweep": sweep,
        "images": len(rows), "patients": len({r["patient"] for r in rows}),
        "parasites_in_truth": int(truths.sum()), "threshold": threshold, "model": model,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "count_bias": round(float(errors.mean()), 3),
        "count_mae": round(float(np.abs(errors).mean()), 3),
        "count_bias_pct": round(float(errors.sum() / max(truths.sum(), 1)) * 100, 2),
        "ratio_images": int(keep.sum()),
        "ratio_bias_pct": round(float((np.exp(log_diff.mean()) - 1) * 100), 2),
        "ratio_loa_pct": [round(float((np.exp(log_diff.mean() - 1.96 * log_diff.std()) - 1) * 100), 1),
                          round(float((np.exp(log_diff.mean() + 1.96 * log_diff.std()) - 1) * 100), 1)],
        "ratio_spearman": round(float(np.corrcoef(np.argsort(np.argsort(ours)), np.argsort(np.argsort(gt)))[0, 1]), 3),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-patient", type=int, default=2)
    ap.add_argument("--tag", default="holdout")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=["cnn", "hybrid", "mlp"], default="cnn")
    args = ap.parse_args()

    report = evaluate(args.per_patient, args.seed, args.holdout, args.threshold, args.max_patients, args.model)
    OUT.mkdir(exist_ok=True)
    (OUT / f"parasite_eval_{args.tag}.json").write_text(json.dumps(report, indent=2))
    print(f"\n{report['images']} images, {report['patients']} patients, "
          f"{report['parasites_in_truth']} annotated parasites, model {args.model}, threshold {args.threshold}\n")
    print(f"  precision      {report['precision']:.3f}")
    print(f"  recall         {report['recall']:.3f}")
    print(f"  F1             {report['f1']:.3f}")
    print(f"  count bias     {report['count_bias']:+.2f} per image ({report['count_bias_pct']:+.1f}%)")
    print(f"  count MAE      {report['count_mae']:.2f} per image")
    print(f"\n  parasites per white cell, ours vs truth on {report['ratio_images']} images:")
    print(f"    bias           {report['ratio_bias_pct']:+.1f}%")
    print(f"    95% limits     {report['ratio_loa_pct'][0]:+.1f}% .. {report['ratio_loa_pct'][1]:+.1f}%")
    print(f"    rank corr.     {report['ratio_spearman']:.3f}")
    print("\n  sensitivity to the threshold (information only, the threshold is chosen on the fit set):")
    print("    thr   precision  recall   F1    count bias")
    for row in report["sweep"]:
        print(f"    {row['threshold']:.2f}   {row['precision']:.3f}      {row['recall']:.3f}   "
              f"{row['f1']:.3f}   {row['count_bias_pct']:+.1f}%")


if __name__ == "__main__":
    main()
