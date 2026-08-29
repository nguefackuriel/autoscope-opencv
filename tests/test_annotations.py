"""The coordinate transform is load bearing, so it is pinned by tests.

If someone "simplifies" the rotation away, these fail loudly rather than
letting every downstream metric quietly measure noise.
"""

import cv2 as cv
import pytest

from autoscope import annotations as ann

pytestmark = pytest.mark.skipif(
    not (__import__("pathlib").Path("data/raw").exists()),
    reason="needs the NLM data, run `make data`",
)


def _first_pair():
    from pathlib import Path

    raw = Path("data/raw")
    for patient in sorted(raw.glob("TF*")):
        for image in sorted(patient.glob("*.jpg")):
            gt = raw / "GT_updated" / patient.name / f"{image.stem}.txt"
            if gt.exists():
                return image, gt
    pytest.skip("no annotated image found")


def test_transform_is_a_clockwise_rotation():
    # A point at the top left of the landscape frame lands top right in portrait.
    assert ann.to_image_frame(0, 0, 3024) == (3024, 0)
    assert ann.to_image_frame(100, 200, 3024) == (2824, 100)


def test_points_land_inside_the_image():
    image_path, _ = _first_pair()
    img = cv.imread(str(image_path))
    h, w = img.shape[:2]
    gt = ann.for_image(image_path, w, h)
    assert gt is not None and gt.items

    outside = [a for a in gt.items if not (0 <= a.x < w and 0 <= a.y < h)]
    assert not outside, f"{len(outside)} of {len(gt.items)} annotations fall outside the image"


def test_white_cells_sit_on_something_dark():
    """The real check: a nucleus is much darker than the field around it."""
    image_path, _ = _first_pair()
    img = cv.imread(str(image_path))
    h, w = img.shape[:2]
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)[:, :, 0]
    gt = ann.for_image(image_path, w, h)

    lit = lab[lab > 40]
    field = float(lit.mean())

    samples = []
    for a in gt.white_cells:
        x, y = int(a.x), int(a.y)
        if 12 <= x < w - 12 and 12 <= y < h - 12:
            samples.append(float(lab[y - 10 : y + 11, x - 10 : x + 11].mean()))

    assert samples, "no white cells far enough inside the frame to test"
    darker_by = field - sum(samples) / len(samples)
    assert darker_by > 40, f"white cells only {darker_by:.0f} L darker than the field"


def test_white_cells_have_no_radius_and_parasites_do():
    image_path, _ = _first_pair()
    img = cv.imread(str(image_path))
    h, w = img.shape[:2]
    gt = ann.for_image(image_path, w, h)
    assert all(a.radius is None for a in gt.white_cells)
    assert all(a.radius and a.radius > 0 for a in gt.parasites)


def test_parasite_centre_is_the_first_point_not_the_midpoint():
    """The Circle's first point is the centre. Reading the midpoint of the two
    points instead moves every parasite about 13 px off a 10 px object."""
    import numpy as np

    image_path, gt_path = _first_pair()
    img = cv.imread(str(image_path))
    h, w = img.shape[:2]
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab).astype(np.float32)
    stain = cv.GaussianBlur(lab[:, :, 1] - lab[:, :, 2], (5, 5), 0)

    win = 30
    on_centre, on_midpoint = [], []
    for line in gt_path.read_text(errors="ignore").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 9 or parts[1] != ann.PARASITE:
            continue
        a1, b1, a2, b2 = (float(v) for v in parts[5:9])
        cx, cy = ann.to_image_frame(a1, b1, w)
        rx, ry = ann.to_image_frame(a2, b2, w)
        mx, my = (cx + rx) / 2, (cy + ry) / 2
        if not (win <= mx < w - win and win <= my < h - win):
            continue
        patch = stain[int(my) - win : int(my) + win + 1, int(mx) - win : int(mx) + win + 1]
        iy, ix = np.unravel_index(np.argmax(patch), patch.shape)
        px, py = int(mx) - win + ix, int(my) - win + iy
        on_centre.append(float(np.hypot(px - cx, py - cy)))
        on_midpoint.append(float(np.hypot(px - mx, py - my)))

    if len(on_centre) < 10:
        pytest.skip("too few parasites in the first annotated image")
    assert np.median(on_centre) < np.median(on_midpoint)
    assert np.median(on_centre) < 12

    gt = ann.for_image(image_path, w, h)
    # and the parser agrees with the raw reading above
    first = gt.parasites[0]
    assert first.radius > 5
