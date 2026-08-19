"""Find the round field of view inside a phone-through-eyepiece capture.

Every image in the NLM thick smear set is a bright circle on black: the view
down the eyepiece, photographed by a phone held to it. On a 4032x3024 frame,
about 54% of the pixels are the black surround.

Two reasons this module has to run before anything else:

  Correctness. Any statistic taken over the whole frame is mostly a statistic
  about the black border. An earlier version of quality.py measured lighting
  evenness across the full frame and scored every real image at about 0.05,
  which would have rejected the entire dataset.

  Cost. Half of every frame is nothing. Cropping to the disc roughly halves the
  pixels the rest of the pipeline touches, and the rest of the pipeline is what
  we are benchmarking on Graviton.
"""

from dataclasses import dataclass

import cv2 as cv
import numpy as np

# Below this, a pixel is the black surround rather than the lit field.
DARK_LEVEL = 25

# Detection runs on a thumbnail. The disc is a huge, smooth shape, so full
# resolution buys nothing and costs a lot.
WORK_WIDTH = 512


@dataclass(frozen=True)
class FieldOfView:
    cx: int
    cy: int
    radius: int
    covers: float  # fraction of the frame inside the circle

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """x, y, width, height of the square that contains the circle."""
        return (
            self.cx - self.radius,
            self.cy - self.radius,
            2 * self.radius,
            2 * self.radius,
        )


def detect(image: np.ndarray) -> FieldOfView | None:
    """Locate the lit disc. None when the frame is not a circular view."""
    gray = image if image.ndim == 2 else cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape

    scale = WORK_WIDTH / w
    small = cv.resize(gray, (WORK_WIDTH, int(h * scale)), interpolation=cv.INTER_AREA)

    _, lit = cv.threshold(small, DARK_LEVEL, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    lit = cv.morphologyEx(lit, cv.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv.findContours(lit, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    biggest = max(contours, key=cv.contourArea)
    (cx, cy), radius = cv.minEnclosingCircle(biggest)

    # A frame that is already fully lit has no disc to find, and one where the
    # blob is tiny is not a field of view either.
    covers = (np.pi * radius**2) / (small.shape[0] * small.shape[1])
    if covers > 0.95 or covers < 0.05:
        return None

    return FieldOfView(
        cx=int(cx / scale),
        cy=int(cy / scale),
        radius=int(radius / scale),
        covers=float(covers),
    )


def crop(image: np.ndarray, fov: FieldOfView | None = None, inset: float = 0.96):
    """Return the square around the disc, plus a mask of what is inside it.

    `inset` pulls the mask in slightly. The rim of an eyepiece view is dim and
    smeared, and counting cells there costs accuracy for no gain.
    """
    fov = fov or detect(image)
    if fov is None:
        return image, np.full(image.shape[:2], 255, np.uint8)

    h, w = image.shape[:2]
    x, y, side, _ = fov.bounds
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + side, w), min(y + side, h)
    cropped = image[y0:y1, x0:x1]

    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv.circle(
        mask,
        (fov.cx - x0, fov.cy - y0),
        int(fov.radius * inset),
        255,
        -1,
    )
    return cropped, mask
