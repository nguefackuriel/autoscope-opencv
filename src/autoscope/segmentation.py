"""Stage 4: find the white blood cells.

White cells come first because they are the denominator. A parasite count is
meaningless on its own; parasites per 200 white cells is what turns into a
density, so an error here moves every number downstream.

WHAT A NUCLEUS LOOKS LIKE, measured on 680 annotated cells across 29 images:

    channel   nucleus        field          gap
    L          75 +- 38      206 +- 11     -131
    a         179 +- 11      138 +-  5      +41
    b          72 +- 12      119 +-  4      -48

    equivalent radius   p10 37   median 51   p90 61 px
    parasite radius     p10  7   median 10   p90 13 px

Giemsa stains nuclei dark and strongly purple, so they are both much darker and
much further toward magenta than the field. We use `a - b` for that: about 107
inside a nucleus against 19 in the field. Brightness alone would work too, but
it confuses a nucleus with any other dark thing, and a thick smear has plenty.

Parasites are five times smaller in radius and about twenty five times smaller
in area, so size alone separates the two classes cleanly at this stage.

MULTI-LOBED NUCLEI

A neutrophil nucleus is two to five lobes joined by thin strands. Left alone,
each lobe is a separate blob and one cell counts as four. A morphological close
before labelling rejoins them. The kernel size is the trade: too small and cells
are over-counted, too big and two neighbouring cells merge into one. It is fitted
against ground truth in eval/wbc_eval.py rather than chosen by eye.
"""

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from autoscope import fov

# Detection runs on a quarter-scale image. Nuclei are 100 px across at full
# resolution, so a quarter still leaves 25 px of blob, and the pipeline gets
# roughly sixteen times cheaper. That matters: this is the stage we benchmark.
SCALE = 0.25

# Fitted against ground truth by eval/wbc_sweep.py, see eval/results/wbc_sweep.json.
#
# The grid had a better F1 available: 0.745 at SEED_FRACTION 0.75 with a 24 px
# close, against 0.696 here. We did not take it, because it over-counts cells by
# 18% and this count is the denominator of the density. An 18% surplus of white
# cells makes every parasitaemia we report 18% too low, for every patient, in the
# same direction. These values cost a little localisation accuracy and bring the
# count bias down to +1.8%.
#
# Optimise the number that reaches the clinician, not the metric that looks best.
STAIN_MIN = 55.0        # a - b inside a nucleus, in Lab units
AREA_MIN_FULL = 1800    # px^2 at full resolution
AREA_MAX_FULL = 30000
CLOSE_KERNEL_FULL = 44  # px at full resolution, to rejoin nucleus lobes
MIN_SOLIDITY = 0.55
SEED_FRACTION = 0.85    # of each blob's own distance maximum, never the image's


@dataclass(frozen=True)
class Cell:
    x: float
    y: float
    area: float
    solidity: float


def stain_map(image_bgr: np.ndarray) -> np.ndarray:
    """How purple each pixel is. High inside a nucleus, low in the field."""
    lab = cv.cvtColor(image_bgr, cv.COLOR_BGR2Lab)
    a = lab[:, :, 1].astype(np.int16)
    b = lab[:, :, 2].astype(np.int16)
    return (a - b).astype(np.float32)


def _nucleus_mask(small_bgr: np.ndarray, mask: np.ndarray | None, close_px: int) -> np.ndarray:
    stain = stain_map(small_bgr)
    binary = (stain > STAIN_MIN).astype(np.uint8) * 255
    if mask is not None:
        binary = cv.bitwise_and(binary, mask)

    # Close first to rejoin the lobes of one nucleus, then open to drop specks.
    close_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_px, close_px))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, close_k)
    open_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    return cv.morphologyEx(binary, cv.MORPH_OPEN, open_k)


def _split_touching(binary: np.ndarray) -> np.ndarray:
    """Watershed, so two cells that touch do not become one.

    Seeds come from the distance transform: the centre of a cell is further from
    the background than the isthmus joining it to its neighbour.

    The threshold is worked out per blob, not once for the whole image. A single
    global cut-off silently deletes every cell smaller than the largest one:
    with one big clump at distance 30 and a small cell at 8, a cut at 0.55 of
    the global maximum leaves the small cell with no seed at all, so it vanishes
    rather than being split. That bug cost half the recall.
    """
    dist = cv.distanceTransform(binary, cv.DIST_L2, 5)
    if dist.max() <= 0:
        return np.zeros_like(binary, np.int32)

    n_blobs, blobs = cv.connectedComponents(binary)
    peaks = np.zeros_like(binary)
    for label in range(1, n_blobs):
        blob = blobs == label
        local_max = dist[blob].max()
        if local_max <= 0:
            continue
        peaks[blob & (dist > SEED_FRACTION * local_max)] = 255

    if not peaks.any():
        return blobs

    _, seeds = cv.connectedComponents(peaks)
    unknown = cv.subtract(binary, peaks)
    seeds = seeds + 1
    seeds[unknown == 255] = 0
    cv.watershed(cv.cvtColor(binary, cv.COLOR_GRAY2BGR), seeds)
    return seeds


def find_white_cells(
    image_bgr: np.ndarray,
    crop_to_fov: bool = True,
    scale: float = SCALE,
) -> list[Cell]:
    """Locate white cell nuclei. Coordinates are in the input image's frame."""
    offset_x = offset_y = 0
    disc_mask = None

    if crop_to_fov:
        disc = fov.detect(image_bgr)
        if disc is not None:
            x, y, side, _ = disc.bounds
            offset_x, offset_y = max(x, 0), max(y, 0)
            image_bgr, disc_mask = fov.crop(image_bgr, disc)

    small = cv.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    small_mask = (
        cv.resize(disc_mask, (small.shape[1], small.shape[0]), interpolation=cv.INTER_NEAREST)
        if disc_mask is not None
        else None
    )

    close_px = max(int(CLOSE_KERNEL_FULL * scale) | 1, 3)
    binary = _nucleus_mask(small, small_mask, close_px)
    labels = _split_touching(binary)

    area_min = AREA_MIN_FULL * scale * scale
    area_max = AREA_MAX_FULL * scale * scale

    cells: list[Cell] = []
    for label in range(2, labels.max() + 1):
        blob = (labels == label).astype(np.uint8)
        area = float(blob.sum())
        if not (area_min <= area <= area_max):
            continue

        contours, _ = cv.findContours(blob, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv.contourArea)

        hull_area = cv.contourArea(cv.convexHull(contour))
        solidity = float(cv.contourArea(contour) / hull_area) if hull_area > 0 else 0.0
        if solidity < MIN_SOLIDITY:
            continue

        moments = cv.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"] / scale + offset_x
        cy = moments["m01"] / moments["m00"] / scale + offset_y
        cells.append(Cell(cx, cy, area / (scale * scale), solidity))

    return cells


def count_white_cells(image_bgr: np.ndarray, **kwargs) -> int:
    return len(find_white_cells(image_bgr, **kwargs))
