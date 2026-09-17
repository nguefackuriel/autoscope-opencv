"""Site calibration: the scale is measured from the nuclei, the image is
brought to NLM scale, and count_field maps everything back to the
caller's pixels. Synthetic fields, no dataset."""

import cv2 as cv
import numpy as np

from autoscope import photometry, site
from autoscope.counter import count_field


def _field(size=2400, nucleus_d=360, n=3, seed=0):
    """A pale field with n round dark nuclei of a given diameter, well apart."""
    rng = np.random.RandomState(seed)
    img = np.full((size, size, 3), (225, 215, 230), np.uint8)
    centres = []
    while len(centres) < n:
        c = rng.randint(nucleus_d, size - nucleus_d, 2)
        if all(np.hypot(*(c - d)) > 2.2 * nucleus_d for d in centres):
            centres.append(c)
            cv.circle(img, (int(c[0]), int(c[1])), nucleus_d // 2, (120, 25, 60), -1)
    return cv.GaussianBlur(img, (0, 0), 2.0), centres


def test_nucleus_diameters_recover_the_drawn_size():
    img, _ = _field(nucleus_d=360)
    d = site.nucleus_diameters(photometry.normalise(img), expected_px=350)
    assert d, "no nucleus found"
    assert abs(np.median(d) - 360) < 25


def test_calibrate_gives_the_nlm_ratio():
    img, _ = _field(nucleus_d=360)
    img2, _ = _field(nucleus_d=360, seed=3)
    cal = site.calibrate("synthetic", [img, img2], photometry.normalise, expected_px=350)
    assert abs(cal.scale - site.NLM_ROUND_NUCLEUS_PX / 360) < 0.02
    assert cal.images == 2


def test_rescale_is_identity_at_one_and_shrinks_below():
    img = np.zeros((400, 600, 3), np.uint8)
    assert site.rescale(img, 1.0) is img
    assert site.rescale(img, 0.25).shape[:2] == (100, 150)


def test_count_field_reports_the_callers_pixels_at_any_scale():
    """The same synthetic field at NLM scale and at four times the size must
    give the same white cells, reported where they are in each image."""
    small, centres = _field(size=900, nucleus_d=100, n=3, seed=1)
    big = cv.resize(small, None, fx=4, fy=4, interpolation=cv.INTER_CUBIC)
    a = count_field(small, model="mlp", threshold=1.1)
    b = count_field(big, model="mlp", threshold=1.1, scale=0.25)
    assert a.white_cells == b.white_cells >= 1
    pa = sorted((round(c.x), round(c.y)) for c in a.white_cell_points)
    pb = sorted((round(c.x / 4), round(c.y / 4)) for c in b.white_cell_points)
    assert all(abs(x1 - x2) <= 3 and abs(y1 - y2) <= 3 for (x1, y1), (x2, y2) in zip(pa, pb))
