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

TILE_PX = 640        # what the detectors were trained on (export_detection.py)
STRIDE_PX = 512      # 128 px of overlap, so a parasite cut by one tile edge is whole in the next
DEDUPE_PX = 12.0     # two detections closer than this are the same parasite seen in two tiles
DETECT_FLOOR = 0.05  # confidence below which a detection is not even a candidate
MIN_FOV = 0.9        # a tile has to be this much inside the eyepiece disc


class Detection:
    """Carries .x and .y so wbc_eval.match can consume it unchanged."""

    __slots__ = ("x", "y", "score")

    def __init__(self, x, y, score):
        self.x = float(x)
        self.y = float(y)
        self.score = float(score)


def tile_origins(size: int, tile: int, stride: int) -> list[int]:
    """Tile starts covering `size`, with the last one flush against the edge."""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


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
    height, width = prepared.shape[:2]
    crops, offsets = [], []
    for oy in tile_origins(height, tile, stride):
        for ox in tile_origins(width, tile, stride):
            if float((mask[oy:oy + tile, ox:ox + tile] > 0).mean()) < min_fov:
                continue
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
