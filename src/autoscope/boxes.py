"""Bounding boxes for parasites, measured from the stain instead of assumed.

WHY THIS FILE EXISTS

The NLM ground truth has no boxes. A `Parasite` is a centre plus a point on
the rim, and `annotations.py` is explicit that the annotator's radius (median
22 px) is "an upper bound on the object, not its size" -- the chromatin dot
itself is about 10 px across. So a box drawn from the annotator's radius would
be roughly four times the area of the thing it is supposed to contain, and a
box of some fixed invented size would be worse.

Instead we take the box from the pixels. `parasites.tophat()` already produces
a background-free stain map in which a parasite is a bright compact blob; the
same map the candidate generator reaches 88% recall with. Around each annotated
centre we threshold that map relative to its own local peak and bound the
component that results. The box is then as big as the stain says the parasite
is.

WHAT THIS IS AND IS NOT GOOD FOR

These boxes are training targets, not ground truth. They are an algorithm's
opinion, so scoring a detector by IoU against them would mostly measure how
well the detector imitates this file, and a perfect score would mean we had
reproduced the top-hat generator we were trying to beat. Detections get scored
the way everything else in this repo is scored: box centre against annotated
centre, greedy nearest match, 15 px tolerance (`wbc_eval.match`).

The centre is itself imprecise. Distance from the annotated centre to the most
purple pixel is a median 7.9 px (docs/results-parasites.md), so the search for
the local peak has to allow for that rather than trust the click exactly.
"""

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from autoscope.parasites import PEAK_MIN

# How far from the annotated click to look for the real stain peak. The median
# error is 7.9 px and the annotator's circle is ~22 px, so 12 covers the click
# error without wandering onto a neighbouring parasite.
PEAK_SEARCH_PX = 12

# Window the component is grown in. Just over the median annotator radius, so a
# genuinely large parasite still fits while a distant blob cannot be reached.
WINDOW_PX = 26

# A pixel joins the parasite if the top-hat there is at least this fraction of
# the local peak. Half-maximum is the usual way to size a blob with soft edges.
LEVEL = 0.5

# Sanity clamps. The chromatin dot is about 10 px; anything under 6 px is noise
# and anything over 40 px is a stain artefact or two parasites merged.
MIN_SIDE_PX = 6
MAX_SIDE_PX = 40


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in the frame of the image the top-hat came from."""
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


def _local_peak(top: np.ndarray, cx: int, cy: int, radius: int) -> tuple[int, int, float]:
    """Brightest top-hat pixel within `radius` of the click. Returns (x, y, value)."""
    h, w = top.shape
    x0, x1 = max(cx - radius, 0), min(cx + radius + 1, w)
    y0, y1 = max(cy - radius, 0), min(cy + radius + 1, h)
    if x0 >= x1 or y0 >= y1:
        return cx, cy, float("-inf")
    patch = top[y0:y1, x0:x1]
    iy, ix = np.unravel_index(int(np.argmax(patch)), patch.shape)
    return x0 + int(ix), y0 + int(iy), float(patch[iy, ix])


def box_at(top: np.ndarray, cx: float, cy: float) -> Box | None:
    """Box for the parasite annotated at (cx, cy), or None if the stain has none.

    `top` is the top-hat map from `parasites.tophat()`. Returning None is a real
    answer, not a failure to handle: if there is no stain peak near the click
    then no box drawn here would describe a parasite, and a detector trained on
    an invented one would be learning noise.
    """
    h, w = top.shape
    cxi, cyi = int(round(cx)), int(round(cy))
    if not (0 <= cxi < w and 0 <= cyi < h):
        return None

    px, py, peak = _local_peak(top, cxi, cyi, PEAK_SEARCH_PX)
    if peak <= PEAK_MIN:
        return None

    x0, x1 = max(px - WINDOW_PX, 0), min(px + WINDOW_PX + 1, w)
    y0, y1 = max(py - WINDOW_PX, 0), min(py + WINDOW_PX + 1, h)
    window = top[y0:y1, x0:x1]

    binary = (window >= LEVEL * peak).astype(np.uint8)
    n, labels, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None

    # The component the peak itself sits in. Thresholding relative to the peak
    # guarantees the peak is above the level, so this is never background.
    label = int(labels[py - y0, px - x0])
    if label == 0:
        return None

    bx = int(stats[label, cv.CC_STAT_LEFT]) + x0
    by = int(stats[label, cv.CC_STAT_TOP]) + y0
    bw = int(stats[label, cv.CC_STAT_WIDTH])
    bh = int(stats[label, cv.CC_STAT_HEIGHT])

    if bw > MAX_SIDE_PX or bh > MAX_SIDE_PX:
        return None

    # Grow a box that is merely small up to the floor, keeping it centred, so a
    # tiny component still yields a target a detector can regress to.
    if bw < MIN_SIDE_PX:
        bx -= (MIN_SIDE_PX - bw + 1) // 2
        bw = MIN_SIDE_PX
    if bh < MIN_SIDE_PX:
        by -= (MIN_SIDE_PX - bh + 1) // 2
        bh = MIN_SIDE_PX

    bx, by = max(bx, 0), max(by, 0)
    bw, bh = min(bw, w - bx), min(bh, h - by)
    if bw < MIN_SIDE_PX or bh < MIN_SIDE_PX:
        return None
    return Box(bx, by, bw, bh)


def boxes_for(top: np.ndarray, centres) -> tuple[list[Box], list[int]]:
    """Boxes for every annotated centre. Returns the boxes and the indices kept.

    The indices matter: a caller writing a label file needs to know which
    annotations produced a box and which the stain refused, so the drop rate can
    be reported rather than silently absorbed.
    """
    boxes, kept = [], []
    for i, (cx, cy) in enumerate(centres):
        b = box_at(top, cx, cy)
        if b is not None:
            boxes.append(b)
            kept.append(i)
    return boxes, kept
