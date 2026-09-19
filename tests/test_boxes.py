"""Boxes must come from the stain, not from the annotator's loose circle.

No dataset is needed to check the logic: a synthetic field with a purple disc
of known size has a known right answer, and these tests pin the properties the
export depends on. What they cannot check is whether real parasites look like
synthetic discs, which is what the drop rate reported by the export is for.
"""

import cv2 as cv
import numpy as np
import pytest

from autoscope import boxes as bx
from autoscope.parasites import tophat

SIZE = 200
CENTRE = SIZE // 2
PURPLE = (200, 60, 180)          # strongly magenta in BGR: high a, low b in Lab
BACKGROUND = 170


def field(dot_radius: int | None, colour=PURPLE, at=(CENTRE, CENTRE)) -> np.ndarray:
    img = np.full((SIZE, SIZE, 3), BACKGROUND, np.uint8)
    if dot_radius:
        cv.circle(img, at, dot_radius, colour, -1)
    return cv.GaussianBlur(img, (3, 3), 0)


def top_of(*args, **kwargs) -> np.ndarray:
    return tophat(field(*args, **kwargs))[0]


def test_box_tightly_bounds_a_chromatin_sized_dot():
    """A 10 px dot, the size annotations.py measures, gets a ~10 px box.

    The whole point of deriving boxes: the annotator's radius here would be
    about 22 px, which would be a box four times too big in area.
    """
    box = bx.box_at(top_of(5), CENTRE, CENTRE)
    assert box is not None
    assert 8 <= box.w <= 18, f"width {box.w} is not close to the 10 px dot"
    assert 8 <= box.h <= 18, f"height {box.h} is not close to the 10 px dot"
    assert abs(box.cx - CENTRE) <= 2
    assert abs(box.cy - CENTRE) <= 2


def test_bigger_dot_gets_a_bigger_box():
    """The box tracks the object. A fixed-size box would fail this."""
    small = bx.box_at(top_of(3), CENTRE, CENTRE)
    large = bx.box_at(top_of(8), CENTRE, CENTRE)
    assert small is not None and large is not None
    assert large.w > small.w


def test_an_off_centre_click_still_finds_the_dot():
    """The annotated centre is a median 7.9 px off the most purple pixel.

    If the search trusted the click exactly, that error would drag every box
    off the parasite and the detector would learn the annotator's jitter.
    """
    top = top_of(5)
    off = bx.box_at(top, CENTRE + 8, CENTRE - 6)
    true = bx.box_at(top, CENTRE, CENTRE)
    assert off is not None and true is not None
    assert abs(off.cx - true.cx) <= 2 and abs(off.cy - true.cy) <= 2


def test_no_stain_means_no_box():
    """An empty field yields nothing. Inventing a box here would be noise."""
    assert bx.box_at(top_of(None), CENTRE, CENTRE) is None


def test_a_click_far_from_any_stain_is_refused():
    top = top_of(5)
    assert bx.box_at(top, 30, 30) is None


def test_a_blob_far_larger_than_a_parasite_is_refused():
    """A stain artefact is not a parasite, and must not become a training box."""
    assert bx.box_at(top_of(25), CENTRE, CENTRE) is None


def test_a_click_outside_the_image_is_refused():
    top = top_of(5)
    assert bx.box_at(top, -5, CENTRE) is None
    assert bx.box_at(top, CENTRE, SIZE + 10) is None


def test_boxes_for_reports_which_annotations_were_dropped():
    """The export needs the drop rate, so it cannot be silently absorbed."""
    img = np.full((SIZE, SIZE, 3), BACKGROUND, np.uint8)
    cv.circle(img, (60, 60), 5, PURPLE, -1)
    cv.circle(img, (140, 140), 5, PURPLE, -1)
    top = tophat(cv.GaussianBlur(img, (3, 3), 0))[0]

    centres = [(60, 60), (100, 30), (140, 140)]   # the middle one has no stain
    found, kept = bx.boxes_for(top, centres)
    assert kept == [0, 2]
    assert len(found) == 2


def test_box_centre_helpers_agree_with_the_corner():
    b = bx.Box(10, 20, 30, 40)
    assert (b.cx, b.cy) == (25.0, 40.0)


def test_the_annotated_circle_bounds_the_box():
    """The ReadMe defines the second stored point as on the circumference, so
    2 x radius is the annotator's own claim about how big the object can be.

    A component that grows past it has bled into a neighbour. Measured on real
    data this rejects a further 0.3% on top of the 1.0% with no stain peak.
    """
    top = top_of(8)                       # ~16 px of stain
    assert bx.box_at(top, CENTRE, CENTRE, radius=20) is not None   # limit 40
    assert bx.box_at(top, CENTRE, CENTRE, radius=4) is None        # limit 8


def test_without_a_radius_the_global_clamp_still_applies():
    """White cells are stored as a bare Point and carry no radius at all."""
    assert bx.box_at(top_of(5), CENTRE, CENTRE, radius=None) is not None
    assert bx.box_at(top_of(25), CENTRE, CENTRE, radius=None) is None


def test_boxes_for_applies_the_radius_per_annotation():
    import cv2 as cv
    import numpy as np
    img = np.full((SIZE, SIZE, 3), BACKGROUND, np.uint8)
    cv.circle(img, (60, 60), 8, PURPLE, -1)
    cv.circle(img, (140, 140), 8, PURPLE, -1)
    top = tophat(cv.GaussianBlur(img, (3, 3), 0))[0]

    centres = [(60, 60), (140, 140)]
    # generous for the first, impossibly tight for the second
    found, kept = bx.boxes_for(top, centres, radii=[20, 3])
    assert kept == [0]
    assert len(found) == 1
