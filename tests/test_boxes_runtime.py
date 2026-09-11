"""A box detector through the pipeline, with a fake backend: no torch, no
weights, no dataset. What is checked is the plumbing a real model rides on:
the field is tiled with overlap, a detection in the overlap is reported
once, coordinates come back in the field's frame, and find_parasites /
count_field accept a BoxScorer like any other model."""

import numpy as np
import pytest

from autoscope import boxes as bx, parasites as par


class FakeBackend:
    """Reports a box at fixed FIELD coordinates whenever a tile contains them."""

    def __init__(self, points, score=0.9):
        self.points = points
        self.score = score
        self.calls = 0

    def __call__(self, tiles, conf):
        self.calls += 1
        # detect_field does not tell the backend the tile offsets, so the fake
        # keeps its own copy of the tiling and answers per tile in order.
        out = []
        for ox, oy in self._offsets[self._i:self._i + len(tiles)]:
            rows = [[x - ox - 3, y - oy - 3, x - ox + 3, y - oy + 3, self.score]
                    for x, y in self.points if ox <= x < ox + self.tile and oy <= y < oy + self.tile]
            out.append(np.array(rows, np.float32).reshape(-1, 5))
        self._i += len(tiles)
        return out

    def expect(self, h, w, tile, stride, mask):
        self.tile = tile
        self._offsets = [(ox, oy) for oy in bx.tile_origins(h, tile, stride)
                         for ox in bx.tile_origins(w, tile, stride)
                         if float((mask[oy:oy + tile, ox:ox + tile] > 0).mean()) >= bx.MIN_FOV]
        self._i = 0


def test_tile_origins_cover_the_field_and_overlap():
    starts = bx.tile_origins(1000, 640, 512)
    assert starts[0] == 0 and starts[-1] == 1000 - 640
    assert all(b - a <= 512 for a, b in zip(starts, starts[1:]))
    assert bx.tile_origins(500, 640, 512) == [0]


def test_a_detection_in_the_overlap_is_reported_once():
    h = w = 1152                       # three tiles a side at 640 / 512
    mask = np.full((h, w), 255, np.uint8)
    points = [(600, 600), (100, 100), (1100, 300)]   # the first sits in four tiles' overlap
    fake = FakeBackend(points)
    fake.expect(h, w, 640, 512, mask)
    dets = bx.detect_field(fake, np.zeros((h, w, 3), np.uint8), mask, 640, 512, 0.05)
    got = sorted((round(d.x), round(d.y)) for d in dets)
    assert got == sorted(points)
    assert fake.calls >= 1


def test_dedupe_keeps_the_strongest_of_a_cluster():
    d = [bx.Detection(10, 10, 0.5), bx.Detection(13, 11, 0.9), bx.Detection(100, 100, 0.7)]
    kept = bx.dedupe(d, 12.0)
    assert [(k.x, k.y, k.score) for k in kept] == [(13.0, 11.0, 0.9), (100.0, 100.0, 0.7)]


def test_box_scorer_reports_field_coordinates_and_the_pipeline_accepts_it():
    h = w = 900
    img = np.full((h, w, 3), 200, np.uint8)
    mask = np.full((h, w), 255, np.uint8)
    points = [(200, 250), (700, 640)]
    fake = FakeBackend(points, score=0.8)
    fake.expect(h, w, 640, 512, mask)
    scorer = bx.BoxScorer(fake, 640, 512)
    pts, scores = scorer.detections(img, mask)
    assert sorted(map(tuple, pts.round().astype(int).tolist())) == sorted(points)
    assert np.allclose(scores, 0.8)
    # threshold defaults to the box models' own; above it nothing is found
    fake.expect(h, w, 640, 512, mask)
    found = par.find_parasites(img, crop_to_fov=False, normalise_stain=False, model="yolo",
                               scorer=scorer, threshold=0.5)
    assert sorted((round(f.x), round(f.y)) for f in found) == sorted(points)
    fake.expect(h, w, 640, 512, mask)
    assert par.find_parasites(img, crop_to_fov=False, normalise_stain=False, model="yolo",
                              scorer=scorer, threshold=0.95) == []
    assert par.threshold_for("yolo", None) == par.BOX_THRESHOLD


def test_missing_weights_say_how_to_get_them():
    with pytest.raises(FileNotFoundError, match="AUTOSCOPE_YOLO_WEIGHTS"):
        par._box_scorer("yolo", "/nowhere/best.pt")
