"""Stage 5: find the parasites.

This is the numerator of the density. The white cell count (segmentation.py) is
the denominator; this module gives the count that goes on top.

WHAT A PARASITE LOOKS LIKE in a Giemsa thick smear, measured on 2192 annotated
parasites across 40 fit images after stain normalisation:

    channel      parasite centre    field      look-alike blob
    L             163 (108..200)    200        165
    a             155 (139..178)    137        151
    b             101 ( 83..115)    117        104
    a - b          54 ( 24.. 93)     20         46

    chromatin dot   about 10 px across at full resolution
    annotated radius  median 21 px, an annotator's loose circle, not the dot

The "look-alike" column is the problem. Those are the small purple blobs that
are NOT annotated as anything: platelets, stain precipitate, bits of lysed
cells. In colour they sit on top of the parasites. So colour gets us a
candidate list with high recall, and a second step has to sort candidates by
shape and by context (what surrounds the dot: a parasite carries a ring of
blue cytoplasm, a platelet does not).

TWO STEPS

  1. candidates   threshold the stain map, connected components, keep blobs
                  of about the right area, and drop anything inside a white
                  cell. Tuned for recall; precision is the next step's job.
  2. classify     nineteen measured features per candidate, scored by a small
                  network (19 -> 64 -> 32 -> 1) trained on the fit patients
                  against ground truth and run through cv.dnn from an ONNX
                  file. Held-out patients are never touched during training.

Why cv.dnn and not cv.ml: OpenCV 5 no longer ships the ml module (it moved to
contrib), so there is no RTrees here. The network is trained in plain numpy
by eval/parasite_train.py and written as an ONNX graph (the `onnx` package is
the one training-time dependency, requirements-train.txt). At run time the
classifier is one ONNX file and cv.dnn, so the deployed pipeline stays
opencv + numpy, which is what the Graviton benchmark times. Feature
standardisation is baked into the first layer of the graph, so the caller
passes raw features.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from autoscope import fov, photometry, segmentation

# Candidate generation. Loose on purpose: the classifier tightens it.
#
# Why peaks and not a threshold. The first version thresholded the stain map
# and took connected components. On the fit set it reached 52% candidate
# recall, and the diagnosis was unambiguous: 35% of all parasites sat inside
# blobs of several thousand px^2, because where the field itself is heavily
# stained a global threshold merges everything into one region. Those missed
# parasites were not faint, their stain was stronger than average. So the
# background has to be removed first (a morphological top-hat: stain minus
# its opening with a kernel bigger than any parasite) and a parasite is then a
# local maximum of what remains. Two parasites that touch are two peaks.
TOPHAT_PX = 31            # opening kernel, about three parasite diameters
PEAK_MIN = 14.0           # top-hat value a peak must reach to be a candidate
PEAK_NMS_PX = 7           # two peaks closer than this are one candidate
CORE_R = 4                # px, disc around the peak that is "the parasite"
RING_INNER = 8            # px, annulus around a candidate for context features
RING_OUTER = 16
WBC_DILATE_PX = 6         # keep clear of white cell nuclei by this margin
SMOOTH_SIGMA = 1.2

FEATURE_NAMES = [
    "peak_tophat", "peak_stain", "core_stain", "core_L_min", "core_L", "core_a", "core_b",
    "ring_L", "ring_a", "ring_b", "ring_stain", "contrast_L", "contrast_stain",
    "size_half", "elongation", "solidity", "dist_wbc", "neighbours_20", "local_field_L",
]

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "parasite_mlp.onnx"
CNN_PATH = Path(__file__).resolve().parents[2] / "models" / "parasite_cnn.onnx"
HYBRID_PATH = Path(__file__).resolve().parents[2] / "models" / "parasite_hybrid.onnx"
PATCH_PX = 32             # the patch network's input, centred on the peak


@dataclass(frozen=True)
class Parasite:
    x: float
    y: float
    area: float
    score: float          # classifier probability, 1.0 is certain


@dataclass
class Candidate:
    x: float
    y: float
    area: float
    features: np.ndarray


def _prepare(image_bgr: np.ndarray, crop_to_fov: bool, normalise_stain: bool):
    offset_x = offset_y = 0
    mask = None
    if crop_to_fov:
        disc = fov.detect(image_bgr)
        if disc is not None:
            x, y, _, _ = disc.bounds
            offset_x, offset_y = max(x, 0), max(y, 0)
            image_bgr, mask = fov.crop(image_bgr, disc)
    if normalise_stain:
        image_bgr = photometry.normalise(image_bgr, mask=mask)
    if mask is None:
        mask = np.full(image_bgr.shape[:2], 255, np.uint8)
    return image_bgr, mask, offset_x, offset_y


def _wbc_exclusion(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Everything inside a white cell nucleus, plus a margin, is not a parasite."""
    small = cv.resize(image_bgr, None, fx=segmentation.SCALE, fy=segmentation.SCALE,
                      interpolation=cv.INTER_AREA)
    small_mask = cv.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv.INTER_NEAREST)
    close_px = max(int(segmentation.CLOSE_KERNEL_FULL * segmentation.SCALE) | 1, 3)
    nuclei = segmentation._nucleus_mask(small, small_mask, close_px)
    n, labels, stats, _ = cv.connectedComponentsWithStats(nuclei)
    keep = np.zeros_like(nuclei)
    for i in range(1, n):
        if stats[i, cv.CC_STAT_AREA] >= segmentation.AREA_MIN_FULL * segmentation.SCALE ** 2:
            keep[labels == i] = 255
    full = cv.resize(keep, (mask.shape[1], mask.shape[0]), interpolation=cv.INTER_NEAREST)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * WBC_DILATE_PX + 1,) * 2)
    return cv.dilate(full, k)


def tophat(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Background-free stain map. Returns (tophat, smoothed stain, Lab float)."""
    lab = cv.cvtColor(image_bgr, cv.COLOR_BGR2Lab).astype(np.float32)
    stain = cv.GaussianBlur(lab[:, :, 1] - lab[:, :, 2], (0, 0), SMOOTH_SIGMA)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (TOPHAT_PX, TOPHAT_PX))
    top = cv.morphologyEx(stain, cv.MORPH_TOPHAT, k)
    return top, stain, lab


def peaks(top: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Local maxima of the top-hat above PEAK_MIN, inside `allowed`. Nx2 (x, y)."""
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * PEAK_NMS_PX + 1,) * 2)
    dil = cv.dilate(top, k)
    is_peak = (top >= dil) & (top > PEAK_MIN) & (allowed > 0)
    ys, xs = np.nonzero(is_peak)
    return np.stack([xs, ys], axis=1)


def candidates(image_bgr: np.ndarray, mask: np.ndarray) -> tuple[list[Candidate], np.ndarray]:
    """Purple peaks inside the field and outside white cells.

    Returns the candidates and the white-cell exclusion mask (for callers that
    want to draw it).
    """
    top, stain, lab = tophat(image_bgr)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    h, w = top.shape

    wbc = _wbc_exclusion(image_bgr, mask)
    allowed = cv.bitwise_and(mask, cv.bitwise_not(wbc))
    # keep a margin so every feature window fits
    allowed[:RING_OUTER + 2, :] = 0; allowed[-(RING_OUTER + 2):, :] = 0
    allowed[:, :RING_OUTER + 2] = 0; allowed[:, -(RING_OUTER + 2):] = 0
    pts = peaks(top, allowed)
    if len(pts) == 0:
        return [], wbc

    wbc_dist = cv.distanceTransform(cv.bitwise_not(wbc), cv.DIST_L2, 3)
    field_L = cv.GaussianBlur(L, (0, 0), 25)
    field_stain = cv.GaussianBlur(stain, (0, 0), 25)

    # neighbour count within 20 px, from a density image of the peaks
    dens = np.zeros((h, w), np.float32)
    dens[pts[:, 1], pts[:, 0]] = 1.0
    nk = cv.getStructuringElement(cv.MORPH_ELLIPSE, (41, 41)).astype(np.float32)
    neighbours = cv.filter2D(dens, -1, nk, borderType=cv.BORDER_CONSTANT) - 1.0

    R = RING_OUTER
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(xx, yy)
    core = rr <= CORE_R
    ring = (rr > RING_INNER) & (rr <= RING_OUTER)

    out: list[Candidate] = []
    for x, y in pts:
        x, y = int(x), int(y)
        Lw = L[y - R:y + R + 1, x - R:x + R + 1]
        aw = a[y - R:y + R + 1, x - R:x + R + 1]
        bw = b[y - R:y + R + 1, x - R:x + R + 1]
        sw = stain[y - R:y + R + 1, x - R:x + R + 1]
        tw = top[y - R:y + R + 1, x - R:x + R + 1]

        pk = float(top[y, x])
        # the parasite's own extent: pixels above half the peak, connected to it
        above = (tw >= 0.5 * pk).astype(np.uint8)
        n, lbl = cv.connectedComponents(above, connectivity=8)
        region = (lbl == lbl[R, R]).astype(np.uint8)
        size_half = float(region.sum())
        contours, _ = cv.findContours(region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv.contourArea) if contours else None
        solidity, elong = 1.0, 1.0
        if contour is not None and len(contour) >= 5:
            hull = cv.contourArea(cv.convexHull(contour))
            area_c = cv.contourArea(contour)
            solidity = float(area_c / hull) if hull > 0 else 1.0
            (_, _), (ax1, ax2), _ = cv.fitEllipse(contour)
            elong = float(max(ax1, ax2) / max(min(ax1, ax2), 1e-3))

        core_L = float(Lw[core].mean())
        core_st = float(sw[core].mean())
        feats = np.array([
            pk,
            float(stain[y, x]),
            core_st,
            float(Lw[core].min()),
            core_L,
            float(aw[core].mean()),
            float(bw[core].mean()),
            float(Lw[ring].mean()),
            float(aw[ring].mean()),
            float(bw[ring].mean()),
            float(sw[ring].mean()),
            float(field_L[y, x]) - core_L,
            core_st - float(field_stain[y, x]),
            size_half,
            elong,
            solidity,
            float(wbc_dist[y, x]),
            float(neighbours[y, x]),
            float(field_L[y, x]),
        ], dtype=np.float32)
        out.append(Candidate(float(x), float(y), size_half, feats))
    return out, wbc


class Scorer:
    """The candidate classifier, run by cv.dnn from an ONNX file.

    Input  (N, 19) float32 raw features, in FEATURE_NAMES order.
    Output (N,)    probability that the candidate is a parasite.
    """

    def __init__(self, net: "cv.dnn.Net"):
        self.net = net

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "Scorer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Train it with `python eval/parasite_train.py` "
                "(needs requirements-train.txt) or pull it from the repo.")
        return cls(cv.dnn.readNetFromONNX(str(path)))

    def score(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0, np.float32)
        self.net.setInput(np.ascontiguousarray(X, dtype=np.float32))
        return self.net.forward().reshape(-1).astype(np.float32)


class PatchScorer:
    """The patch classifier, run by cv.dnn from an ONNX file.

    Input  (N, 32, 32, 3) uint8 BGR patches, centred on the candidate peak.
    Output (N,) probability that the candidate is a parasite.
    Pixel normalisation is inside the graph; pass raw 0..255 values.
    """

    def __init__(self, net: "cv.dnn.Net"):
        self.net = net

    @classmethod
    def load(cls, path: str | Path = CNN_PATH) -> "PatchScorer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Train it with `make train-cnn`.")
        return cls(cv.dnn.readNetFromONNX(str(path)))

    def score(self, patches: np.ndarray, features: np.ndarray | None = None,
              batch: int = 1024) -> np.ndarray:
        """`features` is required by the hybrid network (input "features"),
        ignored by the plain patch network."""
        if len(patches) == 0:
            return np.zeros(0, np.float32)
        out = []
        for i in range(0, len(patches), batch):
            blob = np.ascontiguousarray(patches[i:i + batch].transpose(0, 3, 1, 2), dtype=np.float32)
            if features is not None:
                self.net.setInput(blob, "patch")
                self.net.setInput(np.ascontiguousarray(features[i:i + batch], dtype=np.float32), "features")
            else:
                self.net.setInput(blob)
            out.append(self.net.forward().reshape(-1))
        return np.concatenate(out).astype(np.float32)


def patches_at(image_bgr: np.ndarray, cands: list[Candidate], size: int = PATCH_PX) -> np.ndarray:
    """Cut a size x size patch around each candidate; edge candidates are padded."""
    half = size // 2
    padded = cv.copyMakeBorder(image_bgr, half, half, half, half, cv.BORDER_REFLECT_101)
    out = np.empty((len(cands), size, size, 3), np.uint8)
    for i, c in enumerate(cands):
        x, y = int(c.x) + half, int(c.y) + half
        out[i] = padded[y - half:y + half, x - half:x + half]
    return out


_SCORER: Scorer | None = None
_PATCH_SCORER: PatchScorer | None = None
_HYBRID_SCORER: PatchScorer | None = None


def _hybrid_scorer() -> PatchScorer:
    global _HYBRID_SCORER
    if _HYBRID_SCORER is None:
        _HYBRID_SCORER = PatchScorer.load(HYBRID_PATH)
    return _HYBRID_SCORER


def _scorer() -> Scorer:
    global _SCORER
    if _SCORER is None:
        _SCORER = Scorer.load()
    return _SCORER


def _patch_scorer() -> PatchScorer:
    global _PATCH_SCORER
    if _PATCH_SCORER is None:
        _PATCH_SCORER = PatchScorer.load()
    return _PATCH_SCORER


def find_parasites(
    image_bgr: np.ndarray,
    crop_to_fov: bool = True,
    normalise_stain: bool = True,
    threshold: float = 0.5,
    model: str = "cnn",
    scorer: "Scorer | PatchScorer | None" = None,
) -> list[Parasite]:
    """Locate parasites. Coordinates are in the input image's frame.

    `model` is "cnn" (patch network, the default), "hybrid" (patch network
    plus the 19 context features) or "mlp" (the feature network, kept as the
    cheaper baseline and for the ablation in the report).
    """
    img, mask, ox, oy = _prepare(image_bgr, crop_to_fov, normalise_stain)
    cands, _ = candidates(img, mask)
    if not cands:
        return []
    if model == "cnn":
        scorer = scorer or _patch_scorer()
        scores = scorer.score(patches_at(img, cands))
    elif model == "hybrid":
        scorer = scorer or _hybrid_scorer()
        scores = scorer.score(patches_at(img, cands), np.stack([c.features for c in cands]))
    elif model == "mlp":
        scorer = scorer or _scorer()
        scores = scorer.score(np.stack([c.features for c in cands]))
    else:
        raise ValueError(f"unknown model {model!r}, expected 'cnn', 'hybrid' or 'mlp'")
    return [
        Parasite(c.x + ox, c.y + oy, c.area, float(s))
        for c, s in zip(cands, scores) if s >= threshold
    ]


def count_parasites(image_bgr: np.ndarray, **kwargs) -> int:
    return len(find_parasites(image_bgr, **kwargs))
