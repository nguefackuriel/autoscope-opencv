"""Count every field of every held-out patient, once.

This is the expensive half of the "does the agent help?" question, and it does
not depend on which policy is being tested: a field returns the same counts
whoever asked for it. So we pay for it once here, and agent_eval.py replays
the policies over the table.

For each image we store what the pipeline found and what the experts annotated.
That gives two counters for the same session: the real one, and an oracle that
returns ground truth. Running the agent with both separates agent mistakes from
vision mistakes, which is the whole reason the counter is injected.
"""

import argparse
import json
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from autoscope import annotations as ann
from autoscope.counter import count_field
from autoscope.parasites import load_scorer
from autoscope.quality import assess_field

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None, help="default: the model's own")
    ap.add_argument("--model", default="cnn", choices=["cnn", "heatmap", "heat-cnn"])
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--out", default="agent_fields.json")
    args = ap.parse_args()

    fitted = set(json.loads((OUT / "wbc_sweep.json").read_text())["fit_patients"])
    patients = [p for p in sorted(RAW.iterdir())
                if p.is_dir() and p.name.startswith("TF") and p.name not in fitted]
    if args.max_patients:
        patients = patients[:args.max_patients]
    print(f"{len(patients)} held-out patients")

    scorer = load_scorer(args.model)
    from autoscope.parasites import threshold_for
    args.threshold = threshold_for(args.model, args.threshold)
    rows, t0 = [], time.perf_counter()
    for i, patient in enumerate(patients, 1):
        for path in sorted(patient.glob("*.jpg")):
            img = cv.imread(str(path))
            if img is None:
                continue
            h, w = img.shape[:2]
            truth = ann.for_image(path, w, h)
            if truth is None:
                continue
            t = time.perf_counter()
            result = count_field(img, model=args.model, threshold=args.threshold, scorer=scorer)
            quality = assess_field(img)
            rows.append({
                "patient": patient.name, "image": path.name,
                "found_parasites": result.parasites, "found_wbc": result.white_cells,
                "truth_parasites": len(truth.parasites), "truth_wbc": len(truth.white_cells),
                "focus": round(quality.focus_score, 2),
                "even": round(quality.illumination_uniformity, 4),
                "debris": round(quality.debris_ratio, 4),
                "usable": quality.why_not() is None,
                "seconds": round(time.perf_counter() - t, 2),
            })
        done = time.perf_counter() - t0
        print(f"  [{i}/{len(patients)}] {patient.name}: {len(rows)} fields, "
              f"{done / 60:.1f} min elapsed", flush=True)

    (OUT / args.out).write_text(json.dumps({
        "threshold": args.threshold, "model": args.model, "patients": len(patients), "fields": len(rows),
        "seconds_per_field": round(float(np.mean([r["seconds"] for r in rows])), 2),
        "rows": rows,
    }, indent=2))
    print(f"\n{len(rows)} fields from {len(patients)} patients, "
          f"{np.mean([r['seconds'] for r in rows]):.2f} s per field")


if __name__ == "__main__":
    main()
