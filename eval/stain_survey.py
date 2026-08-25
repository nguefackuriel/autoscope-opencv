"""Measure the stain reference, and how much normalisation actually narrows it.

Two jobs:

1. Produce the reference statistics that photometry.NLM_REFERENCE holds. They
   have to be measured, not chosen.
2. Report the spread of stain across slides before and after normalising. If
   normalisation does not visibly narrow it, it is not earning its place.
"""

import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from autoscope import fov, photometry

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parent / "results"


def survey(per_patient: int = 1, limit: int | None = None) -> dict:
    before, after = [], []
    patients = sorted(p for p in RAW.iterdir() if p.is_dir() and p.name.startswith("TF"))
    if limit:
        patients = patients[:limit]

    for i, patient in enumerate(patients, 1):
        for path in sorted(patient.glob("*.jpg"))[:per_patient]:
            img = cv.imread(str(path))
            if img is None:
                continue
            cropped, mask = fov.crop(img)
            before.append(photometry.measure(cropped, mask).as_dict())
        if i % 40 == 0:
            print(f"  {i}/{len(patients)} patients")

    keys = ["mean_L", "mean_a", "mean_b", "std_L", "std_a", "std_b"]
    reference = photometry.StainStats(
        mean=tuple(float(np.median([b[f"mean_{c}"] for b in before])) for c in "Lab"),
        std=tuple(float(np.median([b[f"std_{c}"] for b in before])) for c in "Lab"),
    )

    # Second pass: normalise to that reference and re-measure.
    for i, patient in enumerate(patients, 1):
        for path in sorted(patient.glob("*.jpg"))[:per_patient]:
            img = cv.imread(str(path))
            if img is None:
                continue
            cropped, mask = fov.crop(img)
            fixed = photometry.normalise(cropped, reference, mask)
            after.append(photometry.measure(fixed, mask).as_dict())

    def spread(rows, key):
        v = np.array([r[key] for r in rows])
        return {
            "median": round(float(np.median(v)), 2),
            "p5": round(float(np.percentile(v, 5)), 2),
            "p95": round(float(np.percentile(v, 95)), 2),
            "iqr": round(float(np.percentile(v, 75) - np.percentile(v, 25)), 2),
        }

    return {
        "images": len(before),
        "reference": reference.as_dict(),
        "before": {k: spread(before, k) for k in keys},
        "after": {k: spread(after, k) for k in keys},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    report = survey(limit=args.limit)
    OUT.mkdir(exist_ok=True)
    (OUT / "stain_survey.json").write_text(json.dumps(report, indent=2))

    r = report["reference"]
    print(f"\n{report['images']} images\n")
    print("measured reference:")
    print(f"    mean=({r['mean_L']:.1f}, {r['mean_a']:.1f}, {r['mean_b']:.1f}),")
    print(f"    std=({r['std_L']:.1f}, {r['std_a']:.1f}, {r['std_b']:.1f}),")
    print(f"\n{'channel':10} {'IQR before':>12} {'IQR after':>12} {'narrowed by':>13}")
    for k in ("mean_L", "mean_a", "mean_b", "std_L", "std_a", "std_b"):
        b, a = report["before"][k]["iqr"], report["after"][k]["iqr"]
        cut = (1 - a / b) if b > 0 else 0.0
        print(f"{k:10} {b:>12.2f} {a:>12.2f} {cut:>12.0%}")


if __name__ == "__main__":
    main()
