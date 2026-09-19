"""Tiling rules the detection export depends on.

The dataset is not needed to check these: tiling is arithmetic, and getting it
wrong is silent. A tile grid that misses the right-hand strip of every field
would simply train on less data than we thought, and nothing would complain.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import export_detection as ex
from autoscope.boxes import Box

TILE = 640
STRIDE = 512


def test_tiles_cover_the_whole_field_to_the_last_pixel():
    """The final tile is flush with the edge, so no strip is left unseen."""
    for size in (3024, 4032, 2000, 1000):
        starts = ex.tile_origins(size, TILE, STRIDE)
        assert starts[0] == 0
        assert starts[-1] + TILE == size, f"{size} px field leaves a gap"


def test_tiles_overlap_by_the_stride_difference():
    starts = ex.tile_origins(3024, TILE, STRIDE)
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(g <= STRIDE for g in gaps)


def test_a_field_smaller_than_a_tile_yields_one_tile():
    assert ex.tile_origins(400, TILE, STRIDE) == [0]


def test_a_field_exactly_one_tile_yields_one_tile():
    assert ex.tile_origins(TILE, TILE, STRIDE) == [0]


def test_box_straddling_a_tile_edge_is_dropped():
    """Half a parasite is a bad target: it teaches the wrong shape."""
    straddling = Box(636, 100, 12, 12)          # runs to x=648, past the edge
    assert ex.boxes_in_tile([straddling], 0, 0, TILE) == []


def test_the_overlap_recovers_a_box_dropped_at_the_previous_edge():
    """Dropping an edge box is only safe if a neighbour holds it whole.

    This is the property that makes the 128 px overlap load-bearing rather
    than decorative.
    """
    straddling = Box(636, 100, 12, 12)
    assert ex.boxes_in_tile([straddling], 0, 0, TILE) == []
    assert ex.boxes_in_tile([straddling], STRIDE, 0, TILE) == [(124, 100, 12, 12)]


def test_box_coordinates_are_rebased_to_the_tile():
    inside = ex.boxes_in_tile([Box(600, 300, 10, 10)], STRIDE, 256, TILE)
    assert inside == [(88, 44, 10, 10)]


def test_a_box_outside_the_tile_entirely_is_not_included():
    assert ex.boxes_in_tile([Box(2000, 2000, 10, 10)], 0, 0, TILE) == []


def test_yolo_normalisation_puts_the_centre_in_range():
    """Every label written must be a valid YOLO line, or training silently
    skips it."""
    inside = ex.boxes_in_tile([Box(600, 300, 10, 10)], STRIDE, 256, TILE)
    for x, y, w, h in inside:
        cx, cy = (x + w / 2) / TILE, (y + h / 2) / TILE
        assert 0.0 < cx < 1.0 and 0.0 < cy < 1.0
        assert 0.0 < w / TILE < 1.0 and 0.0 < h / TILE < 1.0


def test_min_fov_covers_essentially_every_parasite():
    """The FOV threshold decides how much of the slide is looked at.

    At 0.9 only tiles almost entirely inside the eyepiece disc survive: 10.2
    tiles per image out of 29.6, covering 65.8% of annotated parasites. At
    inference that capped recall at 0.652 before the model did anything, and
    the shortfall was reported as geometry rather than as a threshold.
    """
    from autoscope.tiling import MIN_FOV
    assert MIN_FOV <= 0.5, (
        f"MIN_FOV={MIN_FOV} discards tiles near the disc rim, where parasites "
        "genuinely sit. Measured coverage: 0.2 -> 99.9%, 0.5 -> 98.5%, "
        "0.9 -> 65.8%.")


def test_export_and_inference_share_one_fov_threshold():
    """They must tile identically, or the detector is scored on parasites it
    was never shown."""
    import importlib
    from autoscope.tiling import MIN_FOV, TILE_PX, STRIDE_PX
    de = importlib.import_module("detector_eval")
    assert de.MIN_FOV is MIN_FOV
    assert de.TILE_PX is TILE_PX and de.STRIDE_PX is STRIDE_PX


def test_covered_tiles_keeps_rim_tiles():
    import numpy as np
    from autoscope.tiling import covered_tiles
    mask = np.zeros((1400, 1400), np.uint8)
    cv_centre, r = 700, 690
    yy, xx = np.mgrid[:1400, :1400]
    mask[((xx - cv_centre) ** 2 + (yy - cv_centre) ** 2) <= r * r] = 255
    strict = covered_tiles(mask, min_fov=0.9)
    loose = covered_tiles(mask, min_fov=0.2)
    assert len(loose) > len(strict), "the looser threshold must see more of the field"
