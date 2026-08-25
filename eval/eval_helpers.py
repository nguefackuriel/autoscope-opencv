"""Shared loading, so a sweep does not re-read the same JPEGs on every step."""

import random
from pathlib import Path

import cv2 as cv

from autoscope import annotations as ann

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_pairs(limit: int, seed: int = 0, per_patient: int = 1):
    """Load (image, ground truth) pairs once and keep them in memory."""
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
            pairs.append((img, truth))
            if len(pairs) >= limit:
                return pairs
    return pairs
