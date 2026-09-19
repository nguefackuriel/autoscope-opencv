"""Measure the real spread of field quality in the NLM thick smear set.

The thresholds in quality.py started as guesses. This replaces them with values
measured on the data, and it answers the question that decides how much we can
claim about refocusing: does the real data contain genuinely unusable fields, or
is every capture already sharp?
"""

import argparse
import json
import random
from pathlib import Path

import cv2 as cv
import numpy as np

# --- repo paths: run from anywhere, installed or not -------------------------
# Azure ML runs the code snapshot without `pip install -e .`, so the editable
# install that normally puts src/ on sys.path is absent and every `from
# autoscope import ...` fails. This depends on __file__ alone, so it survives
# any working directory, any launch style, and a missing install.
import sys as _sys, pathlib as _pl
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d)
                 for _d in ("src", "eval", "train")]
# ----------------------------------------------------------------------------

import splits

from autoscope import fov
from autoscope.quality import assess_field

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parent / "results"


def ground_truth(txt: Path) -> tuple[int, int]:
    """Parasites and white cells from one annotation file."""
    parasites = wbc = 0
    for line in txt.read_text(errors="ignore").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        if parts[1] == "Parasite":
            parasites += 1
        elif parts[1] == "White_Blood_Cell":
            wbc += 1
    return parasites, wbc


def survey(per_patient: int, seed: int = 0) -> list[dict]:
    rows = []
    patients = sorted(p for p in RAW.iterdir() if p.is_dir() and p.name.startswith("TF"))

    for i, patient in enumerate(patients, 1):
        images = splits.annotated_images(patient)
        rng = splits.image_rng(seed, patient)
        if not images:
            continue
        for img_path in rng.sample(images, min(per_patient, len(images))):
            image = cv.imread(str(img_path), cv.IMREAD_COLOR)
            if image is None:
                continue
            txt = RAW / "GT_updated" / patient.name / f"{img_path.stem}.txt"
            parasites, wbc = ground_truth(txt) if txt.exists() else (-1, -1)
            # assess_field crops to the round field of view first, which is the
            # only way these numbers describe the sample and not the surround.
            q = assess_field(image)
            disc = fov.detect(image)
            rows.append({
                "patient": patient.name,
                "image": img_path.name,
                "focus": round(q.focus_score, 1),
                "even": round(q.illumination_uniformity, 3),
                "cells": round(q.cell_density, 3),
                "debris": round(q.debris_ratio, 3),
                "fov_radius": disc.radius if disc else -1,
                "parasites": parasites,
                "wbc": wbc,
            })
        if i % 25 == 0:
            print(f"  {i}/{len(patients)} patients, {len(rows)} images")
    return rows


def percentiles(values: list[float], points=(1, 5, 25, 50, 75, 95, 99)) -> dict:
    a = np.asarray(values, dtype=float)
    return {f"p{p}": round(float(np.percentile(a, p)), 3) for p in points}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-patient", type=int, default=3)
    args = ap.parse_args()

    rows = survey(args.per_patient)
    if not rows:
        raise SystemExit(f"no images found under {RAW}, run `make data` first")

    report = {
        "images": len(rows),
        "patients": len({r["patient"] for r in rows}),
        "focus": percentiles([r["focus"] for r in rows]),
        "even": percentiles([r["even"] for r in rows]),
        "cells": percentiles([r["cells"] for r in rows]),
        "debris": percentiles([r["debris"] for r in rows]),
        "fov_radius": percentiles([r["fov_radius"] for r in rows if r["fov_radius"] > 0]),
        "fov_found": round(sum(1 for r in rows if r["fov_radius"] > 0) / len(rows), 3),
        "wbc_per_image": percentiles([r["wbc"] for r in rows if r["wbc"] >= 0]),
        "parasites_per_image": percentiles([r["parasites"] for r in rows if r["parasites"] >= 0]),
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "focus_survey.json").write_text(json.dumps({"summary": report, "rows": rows}, indent=2))

    print(f"\n{report['images']} images from {report['patients']} patients\n")
    print(f"round field of view found in {report['fov_found']:.0%} of images\n")
    for key in ("focus", "even", "cells", "debris", "fov_radius",
                "wbc_per_image", "parasites_per_image"):
        p = report[key]
        print(f"{key:20} p5 {p['p5']:>9}   median {p['p50']:>9}   p95 {p['p95']:>9}")

    wbc = [r["wbc"] for r in rows if r["wbc"] >= 0]
    if wbc:
        mean_wbc = float(np.mean(wbc))
        print(f"\nmean white cells per image: {mean_wbc:.1f}")
        print(f"images needed for the WHO 200-cell count: {200 / mean_wbc:.1f}")
    print(f"\nwritten to {OUT / 'focus_survey.json'}")


if __name__ == "__main__":
    main()
