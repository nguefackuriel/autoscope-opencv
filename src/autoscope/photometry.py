"""Stage 2: make slides stained by different people, in different labs, on
different cameras, look alike enough to be measured the same way.

WHAT THIS DOES NOT FIX

We checked before building it. On the NLM set, stain variation does not explain
the counting bias: the correlation between an image's stain level and its count
error is +0.05, and between the separation of nucleus from field and the error,
-0.06. The nucleus level is already stable across that set, 113 to 129 in a-b.
Crowding explains the bias instead, r = -0.55, and that is fixed in
segmentation.py, not here.

WHAT IT IS FOR

Transfer. Every threshold in the pipeline was measured on 4032x3024 phone
captures through one eyepiece, Giemsa stained by one group of people. The
Tanzania set is a different country, a different microscope and a 4K Sony
IMX334 sensor. Nothing says its stain sits where ours does, and if it does not,
a fixed threshold is arbitrary there.

Normalising to a common reference is what lets us claim the pipeline transfers
instead of hoping. It is also the honest order of work: normalise first, then
re-measure on the held-out set, rather than re-fitting thresholds per dataset
and calling the result generalisation.

METHOD

Reinhard colour transfer in Lab. Match the mean and spread of each channel to a
reference, using only pixels inside the round field of view, because the black
surround would otherwise dominate every statistic.

Lab is the right space for this: L carries how dark the stain went and a and b
carry its hue, so a Giemsa slide stained twice as long moves mostly in L and a
while the shape of the cells stays put. Doing the same in BGR would drag the
channels around together and change hue as a side effect.
"""

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from autoscope import fov


@dataclass(frozen=True)
class StainStats:
    """Mean and spread per Lab channel, inside the field of view."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def as_dict(self) -> dict:
        return {
            "mean_L": self.mean[0], "mean_a": self.mean[1], "mean_b": self.mean[2],
            "std_L": self.std[0], "std_a": self.std[1], "std_b": self.std[2],
        }


# Median statistics over 150 NLM patients, measured by eval/stain_survey.py.
# This is the frame everything else gets mapped into, including Tanzania.
NLM_REFERENCE = StainStats(
    mean=(194.7, 134.6, 118.8),
    std=(24.4, 10.1, 10.0),
)


def field_mask(image_bgr: np.ndarray) -> np.ndarray:
    """A mask of the lit disc, the same size as the image given.

    Works whether the image is a full frame or already cropped to the disc.
    An earlier version called fov.crop() and, when the returned mask did not
    match the image size, fell back to marking everything as inside. On a full
    frame that quietly handed the black surround a vote in every statistic,
    which is the one thing this module exists to prevent.
    """
    disc = fov.detect(image_bgr)
    if disc is None:
        return np.full(image_bgr.shape[:2], 255, np.uint8)

    mask = np.zeros(image_bgr.shape[:2], np.uint8)
    cv.circle(mask, (disc.cx, disc.cy), int(disc.radius * 0.96), 255, -1)
    return mask


def measure(image_bgr: np.ndarray, mask: np.ndarray | None = None) -> StainStats:
    """Lab statistics inside the field of view."""
    lab = cv.cvtColor(image_bgr, cv.COLOR_BGR2Lab).astype(np.float32)
    if mask is None or mask.shape[:2] != lab.shape[:2]:
        mask = field_mask(image_bgr)

    inside = mask > 0
    if not inside.any():
        inside = np.ones(lab.shape[:2], bool)

    pixels = lab[inside]
    return StainStats(
        mean=tuple(float(v) for v in pixels.mean(axis=0)),
        std=tuple(float(max(v, 1e-3)) for v in pixels.std(axis=0)),
    )


def normalise(
    image_bgr: np.ndarray,
    reference: StainStats = NLM_REFERENCE,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Map this image's stain onto the reference, leaving geometry alone.

    Only pixels inside the field of view drive the statistics. The whole image
    is transformed, so the black surround shifts too, which is harmless because
    every later stage masks it out anyway.
    """
    if mask is None or mask.shape[:2] != image_bgr.shape[:2]:
        mask = field_mask(image_bgr)

    source = measure(image_bgr, mask)
    lab = cv.cvtColor(image_bgr, cv.COLOR_BGR2Lab).astype(np.float32)

    for c in range(3):
        gain = reference.std[c] / source.std[c]
        # A slide that is nearly uniform in one channel has a tiny spread, and
        # dividing by it would amplify noise into stripes. Cap the correction.
        gain = float(np.clip(gain, 0.5, 2.0))
        lab[:, :, c] = (lab[:, :, c] - source.mean[c]) * gain + reference.mean[c]

    return cv.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv.COLOR_Lab2BGR)


def normalise_cropped(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop to the field of view and normalise in one step.

    The order matters. Cropping first means the statistics come from sample
    pixels only, and the black surround never gets a vote.
    """
    cropped, mask = fov.crop(image_bgr)
    return normalise(cropped, mask=mask), mask
