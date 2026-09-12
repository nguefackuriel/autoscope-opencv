"""Site calibration: what turns another microscope's images into the scale
the pipeline was fitted on.

Every threshold in this repo was measured on the NLM rig, a phone held to
an eyepiece: a white cell nucleus is about 100 px across, a parasite's
chromatin dot about 10 px. A hospital's own microscope and camera will
differ, and the Tanzania set (4K camera on a compound microscope) is the
first case: its nuclei are about four times larger in pixels and its
Giemsa much bluer. Two measurements, on ten of the site's own slides, are
the calibration:

  scale   the ratio of the NLM nucleus size to the site's, both measured by
          the same function (round single nuclei, L < 130 after stain
          normalisation), so the ratio does not depend on where a soft
          boundary is drawn. The image
          is resized by it before anything else runs, so the top-hat, the
          patch network, the heat map and the white cell gates all see
          objects at the size they were fitted for.
  stain   handled per image by photometry.normalise, which maps each
          image's Lab statistics onto the NLM reference; the calibration
          records what the site's raw statistics were, so a drift can be
          seen, but ships no parameter of its own.

`count_field(image, scale=cal.scale)` is all a deployment needs. The
measurement lives in eval/tanzania_eval.py --calibrate, and the scale it
found is written to eval/results/tanzania_calibration.json.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import cv2 as cv
import numpy as np

# The reference: round single nuclei measured by nucleus_diameters() itself on
# the NLM rig, 30 fit images, 603 nuclei, L < 130 (12 September 2026). Not the
# 102 px of segmentation.py, which is the detector's blob including multi-lobed
# neutrophils: the ratio has to be taken between two numbers measured the same
# way, or the scale inherits the difference between the two definitions (0.28
# against 0.22 on Tanzania).
NLM_ROUND_NUCLEUS_PX = 77.2
L_NUCLEUS_MAX = 130           # the dark path's L_DARK_MAX: where a nucleus boundary is drawn on NLM


@dataclass(frozen=True)
class SiteCalibration:
    name: str
    scale: float              # multiply the site's pixels by this to reach NLM scale
    nucleus_px: float         # what a nucleus measures on the site, in its own pixels
    images: int               # how many calibration images the measurement used
    raw_L: float = 0.0        # the site's raw field brightness and purpleness, for the record
    raw_stain: float = 0.0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SiteCalibration":
        return cls(**json.loads(Path(path).read_text()))


def rescale(image_bgr: np.ndarray, scale: float) -> np.ndarray:
    """The site's image at NLM scale. Area interpolation when shrinking."""
    if scale == 1.0:
        return image_bgr
    interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_CUBIC
    return cv.resize(image_bgr, None, fx=scale, fy=scale, interpolation=interp)


def nucleus_diameters(normalised_bgr: np.ndarray, expected_px: float = NLM_ROUND_NUCLEUS_PX,
                      l_max: int = L_NUCLEUS_MAX, max_aspect: float = 1.6) -> list[float]:
    """Equivalent diameters of whole round nuclei in a stain-normalised image:
    the dark objects, closed to rejoin lobes, kept when their size is within
    0.45 to 2.3 times `expected_px` and their aspect says one round nucleus
    rather than a lobe or a touching pair. The window and the closing kernel
    scale with `expected_px`, so the same function measures any rig once a
    rough size is known."""
    L = cv.cvtColor(normalised_bgr, cv.COLOR_BGR2Lab)[:, :, 0]
    dark = (L < l_max).astype(np.uint8) * 255
    close_px = max(3, int(round(expected_px / 9)) | 1)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_px, close_px))
    dark = cv.morphologyEx(dark, cv.MORPH_CLOSE, k)
    area_min = np.pi * (0.45 * expected_px / 2) ** 2
    area_max = np.pi * (2.3 * expected_px / 2) ** 2
    n, _, stats, _ = cv.connectedComponentsWithStats(dark)
    out = []
    for i in range(1, n):
        a = stats[i, cv.CC_STAT_AREA]
        w, h = stats[i, cv.CC_STAT_WIDTH], stats[i, cv.CC_STAT_HEIGHT]
        if area_min <= a <= area_max and max(w, h) / max(min(w, h), 1) < max_aspect:
            out.append(float(2 * np.sqrt(a / np.pi)))
    return out


def calibrate(name: str, images: list[np.ndarray], normalise, expected_px: float) -> SiteCalibration:
    """Scale from the round nuclei of the site's calibration images, measured
    exactly as NLM_ROUND_NUCLEUS_PX was. `expected_px` is a rough guess of a
    nucleus on the site, in its own pixels (one look at one image); the
    measurement is the median, and it has to land well inside the window the
    guess sets. `normalise` is photometry.normalise."""
    diams, raw_L, raw_stain = [], [], []
    for img in images:
        lab = cv.cvtColor(img, cv.COLOR_BGR2Lab).astype(np.float32)
        raw_L.append(float(lab[:, :, 0].mean())); raw_stain.append(float((lab[:, :, 1] - lab[:, :, 2]).mean()))
        diams += nucleus_diameters(normalise(img), expected_px)
    if len(diams) < 5:
        raise ValueError(f"only {len(diams)} whole nuclei on the calibration images: check expected_px ({expected_px})")
    nucleus = float(np.median(diams))
    if not 0.6 * expected_px <= nucleus <= 1.7 * expected_px:
        raise ValueError(f"measured {nucleus:.0f} px against a guess of {expected_px:.0f}: redo with the measured value as the guess")
    return SiteCalibration(name, round(NLM_ROUND_NUCLEUS_PX / nucleus, 4), round(nucleus, 1), len(images),
                           round(float(np.mean(raw_L)), 1), round(float(np.mean(raw_stain)), 1))
