"""Dump every candidate's score, per image, with the expert counts.

Threshold experiments then become replays over this table instead of a
40-minute pipeline run each time. Nothing here decides anything: it records
what the classifier said about every candidate, so a calibration can be fitted
on the fit patients and checked on the held-out ones without touching pixels
again.

Scores are stored as float16, which is finer than any threshold we sweep.
"""

import argparse
import json
import time
from pathlib import Path

import sys

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

from autoscope import annotations as ann, parasites as par
from autoscope.segmentation import find_white_cells
from parasite_candidates import MATCH_PX, label_candidates

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["fit", "holdout", "calibration"], required=True,
                    help="calibration = the 20 fit patients the shipped model was not trained on, "
                         "so its scores on them are honest and a calibration fitted there is not "
                         "fitted on its own training data")
    ap.add_argument("--per-patient", type=int, default=0, help="0 means every image")
    args = ap.parse_args()

    fitted = set(json.loads((OUT / "wbc_sweep.json").read_text())["fit_patients"])
    patients = [p for p in sorted(RAW.iterdir()) if p.is_dir() and p.name.startswith("TF")]
    if args.group == "calibration":
        keep = set(json.loads((OUT / "calibration_patients.json").read_text()))
        patients = [p for p in patients if p.name in keep]
    else:
        patients = [p for p in patients if (p.name in fitted) == (args.group == "fit")]
    print(f"{len(patients)} {args.group} patients")

    scorer = par.PatchScorer.load()
    rows, t0 = [], time.perf_counter()
    for i, patient in enumerate(patients, 1):
        images = splits.annotated_images(patient)
        if args.per_patient:
            images = images[:args.per_patient]
        for path in images:
            img = cv.imread(str(path))
            if img is None:
                continue
            h, w = img.shape[:2]
            truth = ann.for_image(path, w, h)
            if truth is None:
                continue
            prepared, mask, ox, oy = par._prepare(img, True, True)
            cands, _ = par.candidates(prepared, mask)
            scores = (scorer.score(par.patches_at(prepared, cands))
                      if cands else np.zeros(0, np.float32))
            # Per-candidate labels, by the same 15 px rule the evaluation uses.
            # Needed to calibrate score -> probability; harmless elsewhere.
            pts = (np.array([[p.x - ox, p.y - oy] for p in truth.parasites])
                   if truth.parasites else np.zeros((0, 2)))
            labels, _ = label_candidates(cands, pts, MATCH_PX)
            wbc = len(find_white_cells(prepared, crop_to_fov=False, normalise_stain=False))
            rows.append({
                "patient": patient.name, "image": path.name,
                "scores": np.asarray(scores, np.float16),
                "labels": np.asarray(labels, np.uint8),
                "candidates": len(cands),
                "truth_parasites": len(truth.parasites),
                "truth_wbc": len(truth.white_cells),
                "found_wbc": wbc,
            })
        print(f"  [{i}/{len(patients)}] {patient.name}: {len(rows)} images, "
              f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    flat = np.concatenate([r["scores"] for r in rows]) if rows else np.zeros(0, np.float16)
    flat_labels = (np.concatenate([r["labels"] for r in rows]) if rows
                   else np.zeros(0, np.uint8))
    offsets = np.cumsum([0] + [len(r["scores"]) for r in rows])
    np.savez_compressed(
        OUT / f"parasite_scores_{args.group}.npz",
        scores=flat, labels=flat_labels, offsets=offsets,
        patient=np.array([r["patient"] for r in rows]),
        image=np.array([r["image"] for r in rows]),
        truth_parasites=np.array([r["truth_parasites"] for r in rows]),
        truth_wbc=np.array([r["truth_wbc"] for r in rows]),
        found_wbc=np.array([r["found_wbc"] for r in rows]),
        candidates=np.array([r["candidates"] for r in rows]),
    )
    print(f"\n{len(rows)} images, {len(flat)} candidates, "
          f"{(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
