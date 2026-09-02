"""Stage 1: decide whether a field is worth counting.

Every number in here feeds `assess_field()`, the first tool the agent calls.
When one of them is out of range the agent refocuses or moves the stage, and
that is the OpenCV-output-changes-a-decision link the rules ask us to show.
"""

from dataclasses import asdict, dataclass

import cv2 as cv
import numpy as np

from autoscope import fov

# Every metric here works inside the round field of view, never on the raw
# frame. See autoscope/fov.py: about 54% of a phone-through-eyepiece capture is
# the black surround, and a statistic taken over the whole frame is mostly a
# statistic about that border.
#
# Measured on 300 images across all 150 patients of the NLM thick smear set,
# see eval/focus_survey.py and eval/results/focus_survey.json. Each threshold
# sits just outside the 5th or 95th percentile, so a normal field passes and
# only the genuinely bad ones get flagged. FOCUS_MIN was re-surveyed on
# 2 September 2026 on the 100 fit patients only, after the sharpness measure
# changed (eval/results/focus_survey_smoothed.json: p5 2.4, median 3.1,
# p95 4.4). On the same scale the uninfected TIFFs read p5 2.25, median 2.97.
#
# DEBRIS_MAX especially is a placeholder with a number behind it rather than a
# validated limit. A thick smear is lysed blood and is full of small particles
# by nature, so the median image already scores 0.69. Before the final report,
# check this against fields the counter actually got wrong, not against the
# distribution.
#
# Do not carry these to another dataset. Laplacian variance scales with
# resolution, magnification and stain, so a threshold fitted on 4032x3024 phone
# captures means nothing on a different rig. Re-run the survey instead.
FOCUS_MIN = 2.2         # p5 of the fit patients is 2.4 on the smoothed scale (was 11 on the raw one)
UNIFORMITY_MIN = 0.90   # p5 is 0.915, median 0.965
DEBRIS_MAX = 0.85       # p95 is 0.80, median 0.685


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
        # Two decimals, not zero. Rounding 10.95 to "11" against a limit of 11
        # produced log lines reading "sharpness 11 needs 11", which read as a
        # bug to anyone looking at the trace and cost more than the digits save.
        if self.focus_score < FOCUS_MIN:
            return f"out of focus, sharpness {self.focus_score:.2f} needs {FOCUS_MIN:.2f}"
        if self.illumination_uniformity < UNIFORMITY_MIN:
            return f"uneven lighting, {self.illumination_uniformity:.3f} needs {UNIFORMITY_MIN:.3f}"
        if self.debris_ratio > DEBRIS_MAX:
            return f"too much debris, {self.debris_ratio:.3f} over the {DEBRIS_MAX:.3f} limit"
        return None

    def as_dict(self) -> dict:
        return asdict(self)


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def _inside(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.full(shape, 255, np.uint8)
    return mask


# Smooth before taking the Laplacian. Measured on 2 September 2026
# (eval/focus_transfer.py): the plain Laplacian variance correlates at 0.97
# with the image's residual noise, so it scores grain, not focus. The NLM
# uninfected TIFFs are smoother than the phone JPEGs and scored 1.35x lower
# without being softer, which made the gate skip 45 of 46 clean fields. A
# sigma of 1.0 removes pixel noise and JPEG grain while a cell edge, 5 to 10
# px wide, survives it; the two sets then sit within 8% of each other.
FOCUS_SMOOTH_SIGMA = 1.0


def focus_score(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Variance of the Laplacian of the lightly smoothed image. Higher is sharper.

    Absolute values still depend on magnification and stain, so the threshold
    below is only valid for this rig, and a different rig needs the survey
    re-run. What the smoothing buys is that two captures of the same rig with
    different compression or sensor noise score alike, which the raw
    Laplacian did not give us.
    """
    smooth = cv.GaussianBlur(_gray(image), (0, 0), FOCUS_SMOOTH_SIGMA)
    lap = cv.Laplacian(smooth, cv.CV_32F)
    inside = _inside(mask, lap.shape).astype(bool)
    values = lap[inside]
    return float(values.var()) if values.size else 0.0


def illumination_uniformity(
    image: np.ndarray, mask: np.ndarray | None = None, grid: int = 8
) -> float:
    """1.0 means even lighting across the field, lower means vignetting.

    Measured on the background, not on the raw pixels. A thick smear is full of
    dark cells, so raw tile means vary enormously between a crowded corner and
    an empty one, and that variation says nothing about the lamp. Estimating
    the background first with a heavy blur separates the illumination from the
    cells, which is the thing stage 2 actually has to correct.

    An earlier version skipped the blur and scored every real image at about
    0.05, which would have rejected the whole dataset.
    """
    gray = _gray(image)

    # Lighting is low frequency, so shrink first. A wide blur on a 12 megapixel
    # frame costs seconds; the same blur on a 256 pixel thumbnail costs nothing
    # and tells us exactly the same thing.
    small = cv.resize(gray, (256, 256), interpolation=cv.INTER_AREA).astype(np.float32)
    background = cv.GaussianBlur(small, (0, 0), 10.0)

    inside = cv.resize(
        _inside(mask, gray.shape), (256, 256), interpolation=cv.INTER_NEAREST
    ).astype(bool)

    h, w = background.shape
    step_y, step_x = max(h // grid, 1), max(w // grid, 1)
    tiles = []
    for y in range(0, h - h % grid, step_y):
        for x in range(0, w - w % grid, step_x):
            cell = background[y : y + step_y, x : x + step_x]
            keep = inside[y : y + step_y, x : x + step_x]
            # Only score tiles that are essentially all inside the disc. A tile
            # straddling the rim is half black and would look like bad lighting.
            if keep.mean() > 0.9:
                tiles.append(cell[keep].mean())

    if not tiles:
        return 0.0
    arr = np.asarray(tiles, dtype=np.float64)
    if arr.mean() <= 0:
        return 0.0
    return float(max(0.0, 1.0 - arr.std() / arr.mean()))


def _foreground_mask(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Everything darker than its neighbourhood: cells, parasites and dirt."""
    gray = _gray(image)
    blurred = cv.medianBlur(gray, 5)
    binary = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, 8
    )
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    fg = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=1)
    return cv.bitwise_and(fg, _inside(mask, fg.shape))


def cell_density(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Fraction of the field covered by anything solid.

    A field with nothing in it wastes a count, and a field that is completely
    packed cannot be segmented, so the agent uses this at both ends.
    """
    fg = _foreground_mask(image, mask)
    inside = _inside(mask, fg.shape)
    area = np.count_nonzero(inside)
    return float(np.count_nonzero(fg) / area) if area else 0.0


def debris_ratio(
    image: np.ndarray, mask: np.ndarray | None = None, min_cell_area: int = 40
) -> float:
    """Share of foreground objects too small or too jagged to be cells.

    Stain precipitate and dust are the main false-positive source in thick
    smears, so we measure them rather than hoping the classifier copes.
    """
    fg = _foreground_mask(image, mask)
    contours, _ = cv.findContours(fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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


def assess_field(image: np.ndarray, crop_to_fov: bool = True) -> FieldQuality:
    """The agent's first tool. One image in, four numbers out.

    Crops to the round field of view first, so the numbers describe the sample
    and not the black surround. Pass crop_to_fov=False for an image that is
    already a tile from inside a field.
    """
    if crop_to_fov:
        image, mask = fov.crop(image)
    else:
        mask = None

    return FieldQuality(
        focus_score=focus_score(image, mask),
        illumination_uniformity=illumination_uniformity(image, mask),
        cell_density=cell_density(image, mask),
        debris_ratio=debris_ratio(image, mask),
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
