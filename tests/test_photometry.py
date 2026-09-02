"""Stain normalisation has to move colour and leave geometry alone."""

import cv2 as cv
import numpy as np

from autoscope import photometry
from autoscope.photometry import StainStats


def slide(tint: tuple[int, int, int], brightness: int, size: int = 240) -> np.ndarray:
    """A fake field: pale background, dark round nuclei, on a black surround."""
    rng = np.random.default_rng(3)
    img = np.zeros((size, size, 3), np.uint8)
    cv.circle(img, (size // 2, size // 2), size // 2 - 6, tint, -1)
    lit = img[:, :, 0] > 0
    noise = rng.integers(-8, 9, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise + brightness * lit[..., None], 0, 255).astype(np.uint8)
    for cx, cy in ((90, 90), (150, 120), (110, 160)):
        cv.circle(img, (cx, cy), 14, (110, 40, 120), -1)
    return img


def test_measure_ignores_the_black_surround():
    img = slide((200, 190, 210), 0)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv.circle(mask, (120, 120), 114, 255, -1)

    inside = photometry.measure(img, mask)
    whole = photometry.measure(img, np.full(img.shape[:2], 255, np.uint8))

    # Counting the surround drags the mean brightness down hard.
    assert inside.mean[0] > whole.mean[0] + 20


def test_two_differently_stained_slides_converge():
    pale = slide((215, 205, 220), 12)
    dark = slide((150, 130, 170), -20)
    mask = np.zeros(pale.shape[:2], np.uint8)
    cv.circle(mask, (120, 120), 114, 255, -1)

    before = abs(photometry.measure(pale, mask).mean[0] - photometry.measure(dark, mask).mean[0])
    after = abs(
        photometry.measure(photometry.normalise(pale, mask=mask), mask).mean[0]
        - photometry.measure(photometry.normalise(dark, mask=mask), mask).mean[0]
    )
    assert after < before / 3, f"gap only closed from {before:.1f} to {after:.1f}"


def test_normalising_does_not_move_the_cells():
    img = slide((200, 190, 210), 0)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv.circle(mask, (120, 120), 114, 255, -1)
    fixed = photometry.normalise(img, mask=mask)

    def nuclei(x):
        stain = cv.cvtColor(x, cv.COLOR_BGR2Lab).astype(np.int16)
        return ((stain[:, :, 1] - stain[:, :, 2]) > 40) & (mask > 0)

    before, after = nuclei(img), nuclei(fixed)
    overlap = (before & after).sum() / max(before.sum(), 1)
    assert overlap > 0.9, f"only {overlap:.0%} of the nucleus pixels survived"


def test_gain_is_capped_on_a_flat_image():
    """A nearly uniform channel has almost no spread. Dividing by it unchecked
    turns sensor noise into banding, so the gain is clamped."""
    flat = np.full((160, 160, 3), 180, np.uint8)
    mask = np.full((160, 160), 255, np.uint8)
    out = photometry.normalise(flat, mask=mask)
    assert out.std() < 30


def test_reference_is_the_measured_one():
    r = photometry.NLM_REFERENCE
    assert isinstance(r, StainStats)
    # Measured over 150 NLM patients by eval/stain_survey.py. If someone edits
    # these, the survey has to be re-run and this test updated with it.
    assert r.mean == (197.8, 139.3, 115.2)
    assert r.std == (24.3, 9.0, 9.7)


def test_full_frame_statistics_ignore_the_surround():
    """normalise() on an uncropped frame must find its own disc.

    It used to fall back to treating the whole frame as sample when the mask it
    fetched was a different size, which let the black border set the statistics.
    """
    size = 300
    frame = np.zeros((size, size, 3), np.uint8)
    cv.circle(frame, (size // 2, size // 2), 100, (200, 190, 210), -1)

    auto = photometry.measure(frame)                       # no mask given
    everything = photometry.measure(frame, np.full((size, size), 255, np.uint8))

    assert auto.mean[0] > everything.mean[0] + 30, (
        f"surround still counted: {auto.mean[0]:.0f} vs {everything.mean[0]:.0f}"
    )


def test_a_wrong_sized_mask_is_not_trusted():
    frame = np.zeros((300, 300, 3), np.uint8)
    cv.circle(frame, (150, 150), 100, (200, 190, 210), -1)
    out = photometry.normalise(frame, mask=np.full((50, 50), 255, np.uint8))
    assert out.shape == frame.shape


def test_statistics_are_exact_on_a_large_field():
    """The first version accumulated 15 million float32 values along a strided
    axis and was off by up to 6.5 Lab units, differently on every image. The
    statistics have to agree with a float64 reduction to well under a level."""
    rng = np.random.default_rng(11)
    big = rng.integers(0, 256, (2000, 2000, 3), dtype=np.uint8)
    mask = np.full((2000, 2000), 255, np.uint8)
    got = photometry.measure(big, mask)
    lab = cv.cvtColor(big, cv.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float64)
    want_mean, want_std = lab.mean(axis=0), lab.std(axis=0)
    assert np.allclose(got.mean, want_mean, atol=0.01), (got.mean, want_mean)
    assert np.allclose(got.std, want_std, atol=0.01), (got.std, want_std)


def test_lookup_table_matches_the_per_pixel_transform():
    """normalise() applies the affine map through a 256-entry table per
    channel. It has to give exactly what applying the float32 expression to
    every pixel gives, or a count could change for a reason nobody can see."""
    img = slide((200, 190, 210), 0)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv.circle(mask, (120, 120), 114, 255, -1)
    src = photometry.measure(img, mask)
    ref = photometry.NLM_REFERENCE
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab).astype(np.float32)
    for c in range(3):
        gain = float(np.clip(ref.std[c] / src.std[c], 0.5, 2.0))
        lab[:, :, c] = (lab[:, :, c] - src.mean[c]) * gain + ref.mean[c]
    want = cv.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv.COLOR_Lab2BGR)
    got = photometry.normalise(img, mask=mask)
    assert np.array_equal(got, want)
