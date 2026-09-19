"""Shared loading, so a sweep does not re-read the same JPEGs on every step."""

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

from autoscope import annotations as ann, photometry

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_pairs(limit: int, seed: int = 0, per_patient: int = 1, normalise: bool = False):
    """Load (image, ground truth) pairs once and keep them in memory.

    With normalise=True the stain correction is applied here, once per image,
    because it does not depend on the parameters a sweep varies. Doing it inside
    the loop instead would repeat the same work at every grid point.
    """
    rng = random.Random(seed)
    patients = sorted(p for p in RAW.iterdir() if p.is_dir() and p.name.startswith("TF"))
    rng.shuffle(patients)

    pairs = []
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
            if normalise:
                img = photometry.normalise(img)
            pairs.append((img, truth))
            if len(pairs) >= limit:
                return pairs
    return pairs
