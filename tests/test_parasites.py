"""Parasite detection: candidate generation on a synthetic field, and the
cv.dnn scorer round trip. No dataset needed."""

from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from autoscope import parasites as par


def _field(seed=0, n_dots=12, size=600):
    """A pink field with small purple dots at known places, plus one big
    purple nucleus that must be excluded, on a black eyepiece surround."""
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size, 3), np.uint8)
    cx = cy = size // 2
    r = int(size * 0.45)
    # field: light pink (BGR)
    cv.circle(img, (cx, cy), r, (235, 215, 240), -1)
    nucleus = (cx + 80, cy - 90)
    dots = []
    while len(dots) < n_dots:
        x, y = rng.randint(60, size - 60, 2)
        if np.hypot(x - cx, y - cy) > r - 40:
            continue
        if np.hypot(x - nucleus[0], y - nucleus[1]) < 75:
            continue    # inside the white cell exclusion zone, by design not a candidate
        if any(np.hypot(x - dx, y - dy) < 40 for dx, dy in dots):
            continue
        cv.circle(img, (int(x), int(y)), 5, (150, 40, 120), -1)      # dark purple dot
        dots.append((int(x), int(y)))
    cv.circle(img, nucleus, 45, (120, 30, 90), -1)                    # big nucleus
    img = cv.GaussianBlur(img, (0, 0), 1.0)
    return img, dots, nucleus


def test_candidates_find_every_dot_and_skip_the_nucleus():
    img, dots, nucleus = _field()
    prepared, mask, ox, oy = par._prepare(img, crop_to_fov=True, normalise_stain=False)
    cands, wbc = par.candidates(prepared, mask)
    assert cands, "no candidates on a field with twelve obvious dots"
    pts = np.array([[c.x + ox, c.y + oy] for c in cands])
    for dx, dy in dots:
        assert np.min(np.hypot(pts[:, 0] - dx, pts[:, 1] - dy)) <= 4, f"dot at {dx},{dy} missed"
    # nothing inside the nucleus
    assert np.all(np.hypot(pts[:, 0] - nucleus[0], pts[:, 1] - nucleus[1]) > 45)


def test_features_have_the_declared_shape_and_no_nans():
    img, _, _ = _field(seed=1)
    prepared, mask, _, _ = par._prepare(img, True, False)
    cands, _ = par.candidates(prepared, mask)
    X = np.stack([c.features for c in cands])
    assert X.shape[1] == len(par.FEATURE_NAMES)
    assert np.isfinite(X).all()


@pytest.mark.skipif(not par.MODEL_PATH.exists(), reason="model not trained yet")
def test_scorer_runs_through_cv_dnn_and_returns_probabilities():
    scorer = par.Scorer.load()
    img, _, _ = _field(seed=2)
    prepared, mask, _, _ = par._prepare(img, True, False)
    cands, _ = par.candidates(prepared, mask)
    X = np.stack([c.features for c in cands])
    p = scorer.score(X)
    assert p.shape == (len(cands),)
    assert np.all((p >= 0) & (p <= 1))
    assert np.isfinite(p).all()


@pytest.mark.skipif(not par.MODEL_PATH.exists(), reason="model not trained yet")
def test_find_parasites_reports_in_the_input_frame():
    img, dots, _ = _field(seed=3)
    found = par.find_parasites(img, threshold=0.0, model="mlp")   # keep every candidate
    pts = np.array([[f.x, f.y] for f in found])
    for dx, dy in dots:
        assert np.min(np.hypot(pts[:, 0] - dx, pts[:, 1] - dy)) <= 4
