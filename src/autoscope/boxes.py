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

WHAT THE DATA SAYS, MEASURED

On 953 annotated parasites across 20 fit patients, with the dataset in hand:

    boxes derived                        99.0%  (10 of 953 refused)
    box side px                          p10 8, median 11, p90 17, max 39
    annotator circle radius px           median 20, p90 25
    derived box side / circle diameter   p50 0.30, p90 0.52, max 1.33

So the annotated circle is about 3.3x the parasite across. Drawing the box
from it would have made every target roughly eleven times too large in area,
and the median derived box lands at 11 px, which is the chromatin dot the
ReadMe and annotations.py both describe. The 1% refusal rate is low enough
that dropping those annotations costs nothing worth recovering.

The circle is still a real bound. The dataset ReadMe states the second point
is "a random point on the circumference", so 2 x radius is the annotator's own
claim about how big the object could be, and a component growing past it has
bled into something else. That rejects a further 0.3%.

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
from autoscope.tiling import MIN_FOV, STRIDE_PX, TILE_PX, covered_tiles, tile_origins

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

# Sanity clamps. Measured median box side is 11 px and p90 is 17, so 6 px is
# below anything real and 40 px is beyond every legitimate box in the sample
# (max 39). MAX_SIDE_PX only applies when the annotation carries no radius;
# when it does, the annotator's own circle is the tighter and better bound.
MIN_SIDE_PX = 6
MAX_SIDE_PX = 40

# Fallback geometry, for an annotation the stain refuses.
#
# WHY A FALLBACK EXISTS AT ALL
#
# Silently dropping an annotation is not neutral in detection training: the
# object is still in the tile, so with no box on it the loss treats it as
# background and the model is actively taught that a real parasite is nothing.
# One slightly wrong box is a smaller error than one confident false negative
# label, so every annotation gets a box and the provenance is recorded instead.
#
# The size is measured, not invented. Across 2209 parasites the contour radius
# is 0.23 of the annotator's radius, so a side of 2 x 0.23 x radius reproduces
# the median 10 px box from the annotator's median 20.8 px circle.
FALLBACK_RADIUS_RATIO = 0.23
FALLBACK_SIDE_PX = 10          # when the annotation carries no radius at all

# Measured median nucleus radius, for a white cell click the mask refuses.
NUCLEUS_FALLBACK_RADIUS_PX = 42


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


def box_at(top: np.ndarray, cx: float, cy: float,
           radius: float | None = None) -> Box | None:
    """Box for the parasite annotated at (cx, cy), or None if the stain has none.

    `top` is the top-hat map from `parasites.tophat()`. Returning None is a real
    answer, not a failure to handle: if there is no stain peak near the click
    then no box drawn here would describe a parasite, and a detector trained on
    an invented one would be learning noise.

    `radius` is the annotator's circle radius when the annotation carries one.
    It is not the size of the parasite -- measured, the circle is about 3.3x
    across what the stain says the object is -- but the ReadMe defines the
    second stored point as lying on the circumference, so 2 x radius is the
    annotator's own upper bound and a component that grows past it has bled
    into a neighbour. Without a radius the global MAX_SIDE_PX applies instead.
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

    limit = 2.0 * radius if radius else MAX_SIDE_PX
    if bw > limit or bh > limit:
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


def _square(cx: float, cy: float, side: float, w: int, h: int) -> Box | None:
    """A square of `side` centred on (cx, cy), clipped to the image."""
    side = max(int(round(side)), MIN_SIDE_PX)
    x = int(round(cx - side / 2.0))
    y = int(round(cy - side / 2.0))
    x, y = max(x, 0), max(y, 0)
    bw, bh = min(side, w - x), min(side, h - y)
    if bw < MIN_SIDE_PX or bh < MIN_SIDE_PX:
        return None
    return Box(x, y, bw, bh)


def fallback_box(cx: float, cy: float, radius: float | None,
                 w: int, h: int) -> Box | None:
    """The box to use when the stain gives nothing, sized from the annotation."""
    side = (2.0 * radius * FALLBACK_RADIUS_RATIO) if radius else FALLBACK_SIDE_PX
    return _square(cx, cy, side, w, h)


def boxes_for(top: np.ndarray, centres, radii=None,
              fallback: bool = True) -> tuple[list[Box], list[int], list[str]]:
    """A box for every annotated centre. Returns (boxes, indices, provenance).

    With `fallback=True` (the default) the result is **one box per annotation**,
    in order, so a set of 300 annotated parasites yields exactly 300 boxes. That
    matters for more than tidiness: an annotation left without a box does not
    vanish from the tile, it becomes background in the detection loss, and the
    model is taught that a real parasite is nothing there.

    `provenance` says where each box came from -- "stain" for a contoured one,
    "fallback" for one sized from the annotator's radius -- so the share that
    is measured rather than inferred stays visible instead of being absorbed
    into a total. Measured on real fit patients, "fallback" is about 0.5%.

    With `fallback=False` the old behaviour returns: only contoured boxes, and
    `indices` tells the caller which annotations they belong to.
    """
    h, w = top.shape
    boxes, kept, provenance = [], [], []
    for i, (cx, cy) in enumerate(centres):
        r = radii[i] if radii is not None else None
        b = box_at(top, cx, cy, r)
        source = "stain"
        if b is None:
            if not fallback:
                continue
            b = fallback_box(cx, cy, r, w, h)
            source = "fallback"
            if b is None:
                # Off the edge of the image: no box can exist here at all.
                continue
        boxes.append(b)
        kept.append(i)
        provenance.append(source)
    return boxes, kept, provenance


def nucleus_box_at(labels: np.ndarray, cx: float, cy: float,
                   search_px: int = 60) -> Box | None:
    """Box for the white cell a click lands on, from a segmented label image.

    White cells are stored as a bare `Point` with no size, so unlike a parasite
    there is no annotated radius to fall back on and the pixels are the only
    source of extent. `labels` is the watershed-split nucleus labelling at full
    resolution, as `segmentation._split_touching` produces.

    Takes the label under the click; if the click landed just outside a
    nucleus, takes the nearest label within `search_px`.
    """
    h, w = labels.shape
    cxi, cyi = int(round(cx)), int(round(cy))
    if not (0 <= cxi < w and 0 <= cyi < h):
        return None

    label = int(labels[cyi, cxi])
    if label <= 1:                       # 0 is unknown, 1 is background
        y0, y1 = max(cyi - search_px, 0), min(cyi + search_px + 1, h)
        x0, x1 = max(cxi - search_px, 0), min(cxi + search_px + 1, w)
        win = labels[y0:y1, x0:x1]
        candidates = np.unique(win[win > 1])
        if candidates.size == 0:
            return None
        ys, xs = np.mgrid[y0:y1, x0:x1]
        best, best_d = None, 1e18
        for c in candidates:
            m = win == c
            d = float(np.min((xs[m] - cxi) ** 2 + (ys[m] - cyi) ** 2))
            if d < best_d:
                best, best_d = int(c), d
        label = best

    ys, xs = np.nonzero(labels == label)
    if ys.size == 0:
        return None
    return Box(int(xs.min()), int(ys.min()),
               int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def nucleus_boxes_for(labels: np.ndarray, centres,
                      fallback: bool = True) -> tuple[list[Box], list[int], list[str]]:
    """One box per white cell click, in order. Same contract as `boxes_for`.

    This is a per-annotation derivation, not a segmentation of the whole field.
    Segmenting independently and counting what comes out is a different
    question with a different answer: on real fields it returns 14 nuclei where
    12 were annotated, or 39 where 55 were, because annotation stops once 200
    white cells have been counted. Anchoring on the click keeps the generated
    set the same size as the ground truth.
    """
    h, w = labels.shape
    boxes, kept, provenance = [], [], []
    for i, (cx, cy) in enumerate(centres):
        b = nucleus_box_at(labels, cx, cy)
        source = "mask"
        if b is None:
            if not fallback:
                continue
            b = _square(cx, cy, 2 * NUCLEUS_FALLBACK_RADIUS_PX, w, h)
            source = "fallback"
            if b is None:
                continue
        boxes.append(b)
        kept.append(i)
        provenance.append(source)
    return boxes, kept, provenance


# ---------------------------------------------------------------------------
# Running a trained box detector on a whole field, 11 September 2026.
#
# Moved here from eval/detector_eval.py so the pipeline can call a detector
# the way it calls the patch network or the heat map: parasites.find_parasites
# (model="yolo" or "frcnn") and counter.count_field take a BoxScorer, and
# SKIP_TRAIN=1 MODEL=yolo WEIGHTS=... bash eval/chain.sh measures every number
# a shipped detector is judged on. torch and ultralytics are imported inside
# the backends, so the runtime venv never pays for them unless a box model is
# asked for.
# ---------------------------------------------------------------------------

# TILE_PX, STRIDE_PX and MIN_FOV come from autoscope.tiling and are NOT
# redefined here. They were duplicated in this file until 19 September, which
# is how the export and the inference pass came to disagree: the export was
# fixed to MIN_FOV 0.2 and this copy stayed at 0.9, so a detector trained on
# 99.9% of the parasites would still have been scored on 65.8% of them. One
# definition, in one module, is the fix.
DEDUPE_PX = 12.0     # two detections closer than this are the same parasite seen in two tiles
DETECT_FLOOR = 0.05  # confidence below which a detection is not even a candidate


class Detection:
    """Carries .x and .y so wbc_eval.match can consume it unchanged."""

    __slots__ = ("x", "y", "score")

    def __init__(self, x, y, score):
        self.x = float(x)
        self.y = float(y)
        self.score = float(score)


def dedupe(dets: list[Detection], min_sep: float = DEDUPE_PX) -> list[Detection]:
    """Greedy suppression by centre distance, strongest detection first.

    Distance rather than IoU because the metric is a centre distance: a
    duplicate 3 px away would otherwise score as one hit plus one false alarm
    and quietly halve precision.
    """
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all((d.x - k.x) ** 2 + (d.y - k.y) ** 2 > min_sep ** 2 for k in kept):
            kept.append(d)
    return kept


def detect_field(backend, prepared: np.ndarray, mask: np.ndarray, tile: int = TILE_PX,
                 stride: int = STRIDE_PX, conf: float = DETECT_FLOOR, batch: int = 8,
                 min_fov: float = MIN_FOV) -> list[Detection]:
    """Run a detector over one whole field, tile by tile, and return
    deduplicated centres in the field's own coordinates.

    `backend(tiles, conf)` takes a list of BGR tiles and returns, per tile, an
    (n, 5) array of x1, y1, x2, y2, score.
    """
    # The same coverage test the export uses, from the same module, so the
    # detector is shown at inference exactly the tiles it was trained on.
    crops, offsets = [], []
    for ox, oy in covered_tiles(mask, tile, stride, min_fov):
        crops.append(prepared[oy:oy + tile, ox:ox + tile])
        offsets.append((ox, oy))

    dets: list[Detection] = []
    for i in range(0, len(crops), batch):
        chunk, chunk_off = crops[i:i + batch], offsets[i:i + batch]
        for arr, (ox, oy) in zip(backend(chunk, conf), chunk_off):
            for x1, y1, x2, y2, score in arr:
                dets.append(Detection((x1 + x2) / 2 + ox, (y1 + y2) / 2 + oy, score))
    return dedupe(dets)


class YoloBackend:
    name = "yolo"

    def __init__(self, weights):
        from ultralytics import YOLO   # only when a box model is asked for
        self.model = YOLO(str(weights))

    def __call__(self, tiles, conf):
        out = self.model.predict(tiles, conf=conf, verbose=False)
        results = []
        for r in out:
            b = r.boxes
            if b is None or len(b) == 0:
                results.append(np.zeros((0, 5), np.float32))
                continue
            xyxy = b.xyxy.cpu().numpy()
            scores = b.conf.cpu().numpy().reshape(-1, 1)
            results.append(np.hstack([xyxy, scores]).astype(np.float32))
        return results


class FrcnnBackend:
    name = "frcnn"

    def __init__(self, weights, device=None):
        import sys
        from pathlib import Path
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train"))
        from frcnn_model import build_model
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(num_classes=2)
        state = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.to(self.device).eval()

    def __call__(self, tiles, conf):
        torch = self.torch
        batch = [torch.from_numpy(cv.cvtColor(t, cv.COLOR_BGR2RGB))
                 .permute(2, 0, 1).float().div(255.0).to(self.device) for t in tiles]
        with torch.no_grad():
            preds = self.model(batch)
        results = []
        for p in preds:
            keep = p["scores"] >= conf
            xyxy = p["boxes"][keep].cpu().numpy()
            scores = p["scores"][keep].cpu().numpy().reshape(-1, 1)
            if len(xyxy) == 0:
                results.append(np.zeros((0, 5), np.float32))
            else:
                results.append(np.hstack([xyxy, scores]).astype(np.float32))
        return results


class BoxScorer:
    """A trained box detector as the pipeline sees it: a field in, parasite
    centres with a confidence out. Same shape as parasites.HeatmapScorer, so
    find_parasites and count_field treat the two alike."""

    BACKENDS = {"yolo": YoloBackend, "frcnn": FrcnnBackend}

    def __init__(self, backend, tile: int = TILE_PX, stride: int = STRIDE_PX):
        self.backend = backend
        self.tile = tile
        self.stride = stride

    @classmethod
    def load(cls, kind: str, weights, tile: int = TILE_PX, stride: int = STRIDE_PX) -> "BoxScorer":
        if kind not in cls.BACKENDS:
            raise ValueError(f"unknown box detector {kind!r}, expected one of {sorted(cls.BACKENDS)}")
        return cls(cls.BACKENDS[kind](weights), tile, stride)

    def detections(self, prepared: np.ndarray, mask: np.ndarray,
                   floor: float = DETECT_FLOOR) -> tuple[np.ndarray, np.ndarray]:
        """(n, 2) xy centres in `prepared`'s coordinates, and their confidences."""
        dets = detect_field(self.backend, prepared, mask, self.tile, self.stride, floor)
        if not dets:
            return np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
        return (np.array([[d.x, d.y] for d in dets], np.float32),
                np.array([d.score for d in dets], np.float32))