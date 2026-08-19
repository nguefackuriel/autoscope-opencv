"""Stage 1: decide whether a field is worth counting.

Every number in here feeds `assess_field()`, the first tool the agent calls.
When one of them is out of range the agent refocuses or moves the stage, and
that is the OpenCV-output-changes-a-decision link the rules ask us to show.
"""

from dataclasses import asdict, dataclass

import cv2 as cv
import numpy as np

# Thresholds are starting points measured on the NLM thick smear set. Re-fit
# them in eval/ before the final report rather than trusting these.
FOCUS_MIN = 120.0
UNIFORMITY_MIN = 0.75
DEBRIS_MAX = 0.35


@dataclass(frozen=True)
class FieldQuality:
    focus_score: float
    illumination_uniformity: float
    cell_density: float
    debris_ratio: float

    @property
    def usable(self) -> bool:
        return (
            self.focus_score >= FOCUS_MIN
            and self.illumination_uniformity >= UNIFORMITY_MIN
            and self.debris_ratio <= DEBRIS_MAX
        )

    def why_not(self) -> str | None:
        """One short sentence for the agent log, or None when the field is fine."""
        if self.focus_score < FOCUS_MIN:
            return f"out of focus, sharpness {self.focus_score:.0f} needs {FOCUS_MIN:.0f}"
        if self.illumination_uniformity < UNIFORMITY_MIN:
            return f"uneven lighting, {self.illumination_uniformity:.2f} needs {UNIFORMITY_MIN:.2f}"
        if self.debris_ratio > DEBRIS_MAX:
            return f"too much debris, {self.debris_ratio:.2f} over the {DEBRIS_MAX:.2f} limit"
        return None

    def as_dict(self) -> dict:
        return asdict(self)


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def focus_score(image: np.ndarray) -> float:
    """Variance of the Laplacian. Higher is sharper.

    Absolute values depend on magnification and stain, so this is only
    comparable between fields of the same slide. That is all the agent needs,
    because it only ever compares a field against the same field refocused.
    """
    return float(cv.Laplacian(_gray(image), cv.CV_64F).var())


def illumination_uniformity(image: np.ndarray, grid: int = 8) -> float:
    """1.0 means perfectly even lighting, lower means vignetting or a bad lamp.

    Microscope images taken down a phone camera almost always darken at the
    edges, so this decides whether stage 2 can rescue the field or not.
    """
    gray = _gray(image).astype(np.float64)
    h, w = gray.shape
    tiles = [
        gray[y : y + h // grid, x : x + w // grid].mean()
        for y in range(0, h - h % grid, max(h // grid, 1))
        for x in range(0, w - w % grid, max(w // grid, 1))
    ]
    tiles = np.asarray(tiles, dtype=np.float64)
    if tiles.mean() <= 0:
        return 0.0
    return float(max(0.0, 1.0 - tiles.std() / tiles.mean()))


def _foreground_mask(image: np.ndarray) -> np.ndarray:
    """Everything darker than its neighbourhood: cells, parasites and dirt."""
    gray = _gray(image)
    blurred = cv.medianBlur(gray, 5)
    mask = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, 8
    )
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    return cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)


def cell_density(image: np.ndarray) -> float:
    """Fraction of the field covered by anything solid.

    A field with nothing in it wastes a count, and a field that is completely
    packed cannot be segmented, so the agent uses this at both ends.
    """
    mask = _foreground_mask(image)
    return float(np.count_nonzero(mask) / mask.size)


def debris_ratio(image: np.ndarray, min_cell_area: int = 40) -> float:
    """Share of foreground objects too small or too jagged to be cells.

    Stain precipitate and dust are the main false-positive source in thick
    smears, so we measure them rather than hoping the classifier copes.
    """
    mask = _foreground_mask(image)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    junk = 0
    for c in contours:
        area = cv.contourArea(c)
        if area < min_cell_area:
            junk += 1
            continue
        perimeter = cv.arcLength(c, True)
        if perimeter <= 0:
            junk += 1
            continue
        # 1.0 is a perfect circle. Cells are roundish; precipitate is not.
        circularity = 4 * np.pi * area / (perimeter**2)
        if circularity < 0.35:
            junk += 1

    return float(junk / len(contours))


def assess_field(image: np.ndarray) -> FieldQuality:
    """The agent's first tool. One image in, four numbers out."""
    return FieldQuality(
        focus_score=focus_score(image),
        illumination_uniformity=illumination_uniformity(image),
        cell_density=cell_density(image),
        debris_ratio=debris_ratio(image),
    )


def frame_motion(previous: np.ndarray, current: np.ndarray) -> float:
    """Median pixel movement between two frames, via sparse optical flow.

    Used when the input is eyepiece video rather than stills: a large value
    means the operator is still moving the slide and the frame will be smeared.
    """
    prev_gray, cur_gray = _gray(previous), _gray(current)
    corners = cv.goodFeaturesToTrack(
        prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=8
    )
    if corners is None or len(corners) == 0:
        return 0.0

    moved, status, _ = cv.calcOpticalFlowPyrLK(prev_gray, cur_gray, corners, None)
    tracked = status.ravel() == 1
    if not tracked.any():
        return 0.0

    shift = np.linalg.norm(
        moved[tracked].reshape(-1, 2) - corners[tracked].reshape(-1, 2), axis=1
    )
    return float(np.median(shift))
