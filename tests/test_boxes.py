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
    found, kept, prov = bx.boxes_for(top, centres, fallback=False)
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
    found, kept, prov = bx.boxes_for(top, centres, radii=[20, 3], fallback=False)
    assert kept == [0]
    assert len(found) == 1


def test_every_annotation_gets_a_box():
    """300 annotated parasites must yield 300 boxes, in order.

    A dropped annotation is not a neutral omission. The parasite is still in
    the tile, so with no box on it the detection loss counts it as background
    and the model is taught that a real parasite is nothing there. One
    slightly wrong box beats one confident false negative label.
    """
    top = top_of(5)
    centres = [(CENTRE, CENTRE), (30, 30), (CENTRE + 40, CENTRE)]  # only the 1st has stain
    found, kept, prov = bx.boxes_for(top, centres, radii=[20, 20, 20])
    assert len(found) == len(centres) == 3
    assert kept == [0, 1, 2]
    assert prov[0] == "stain"
    assert prov[1] == "fallback" and prov[2] == "fallback"


def test_fallback_box_is_sized_from_the_annotator_radius():
    """Measured: the contour radius is 0.23 of the annotated one, so a 20 px
    annotated radius should fall back to roughly a 10 px box, not a 40 px one."""
    b = bx.fallback_box(100, 100, radius=20, w=400, h=400)
    assert b is not None
    assert 8 <= b.w <= 12, f"fallback side {b.w} does not match the measured ratio"


def test_fallback_can_be_turned_off_for_measurement():
    """Measuring the true derivation rate needs the un-padded answer."""
    top = top_of(5)
    centres = [(CENTRE, CENTRE), (30, 30)]
    found, kept, prov = bx.boxes_for(top, centres, radii=[20, 20], fallback=False)
    assert len(found) == 1 and kept == [0] and prov == ["stain"]


def test_white_cell_clicks_get_one_box_each_from_the_mask():
    """White cells carry no radius, so the mask is the only source of extent."""
    import numpy as np
    labels = np.ones((300, 300), np.int32)          # 1 == background
    labels[100:160, 100:170] = 5                    # one nucleus
    found, kept, prov = bx.nucleus_boxes_for(labels, [(130, 130), (250, 250)])
    assert len(found) == 2 and kept == [0, 1]
    assert prov[0] == "mask" and prov[1] == "fallback"
    assert (found[0].x, found[0].y, found[0].w, found[0].h) == (100, 100, 70, 60)


def test_a_click_just_outside_a_nucleus_still_finds_it():
    import numpy as np
    labels = np.ones((300, 300), np.int32)
    labels[100:160, 100:170] = 5
    found, _, prov = bx.nucleus_boxes_for(labels, [(175, 130)])   # 5 px outside
    assert prov == ["mask"]
    assert found[0].w == 70
