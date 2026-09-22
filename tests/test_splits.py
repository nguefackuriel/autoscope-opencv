"""The reserved inference sample must stay out of everything measurable.

The failure this guards against is silent: an image with no ground truth
contributes no matches, so a script that samples one simply reports a number
computed over fewer images than it claims, and nothing complains.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import splits

RAW = ROOT / "data" / "raw"
needs_data = pytest.mark.skipif(not RAW.exists(), reason="needs the NLM data, run `make data`")


def test_no_eval_script_globs_images_directly():
    """Every sampling site goes through splits.annotated_images().

    A raw `glob("*.jpg")` is how the reserved images get back in. splits.py
    itself is exempt: it is the one place allowed to enumerate them.
    """
    offenders = []
    for path in sorted((ROOT / "eval").glob("*.py")):
        if path.name == "splits.py":
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            # `.glob("*.jpg")` but not `.rglob(...)`, which walks an output
            # directory rather than sampling a patient's images.
            if re.search(r'(?<!r)\bglob\(\s*["\']\*\.jpg["\']\s*\)', line):
                offenders.append(f"{path.name}:{i}")
    assert not offenders, (
        "these sample images without going through splits.annotated_images(), "
        f"so the reserved inference set can leak back in: {offenders}")


@needs_data
def test_annotated_and_reserved_partition_the_patient():
    """Together they account for every image, and they never overlap."""
    for name in splits.all_patients()[:12]:
        a = set(splits.annotated_images(name))
        u = set(splits.unannotated_images(name))
        every = set((splits.patient_root() / name).glob("*.jpg"))
        assert a | u == every
        assert not (a & u)


@needs_data
def test_annotated_images_all_have_ground_truth():
    from autoscope import annotations as ann
    for name in splits.all_patients()[:12]:
        for p in splits.annotated_images(name):
            assert ann.gt_path(p).exists(), p


@needs_data
def test_the_reserved_sample_matches_the_committed_manifest():
    """The set is pinned, so it cannot drift without someone noticing."""
    import json
    manifest = json.loads(
        (ROOT / "eval" / "results" / "inference_sample.json").read_text())
    live = {f"{p.parent.name}/{p.name}" for p in splits.inference_sample()}
    assert live == set(manifest["images"]), (
        "the reserved inference sample has changed; regenerate the manifest "
        "deliberately if that is intended")


@needs_data
def test_assert_measurable_rejects_a_reserved_image():
    reserved = splits.inference_sample()
    if not reserved:
        pytest.skip("no unannotated images in this copy of the data")
    with pytest.raises(AssertionError, match="inference sample"):
        splits.assert_measurable(reserved[0])


@needs_data
def test_assert_measurable_accepts_an_annotated_one():
    name = splits.all_patients()[0]
    splits.assert_measurable(splits.annotated_images(name)[0])


def test_image_rng_is_keyed_on_the_patient_not_the_path():
    """The key must be the directory name.

    A full path would embed the checkout location, so Azure ML and a laptop
    would sample different images from the same seed and the same data.
    """
    from pathlib import Path as P
    a = splits.image_rng(0, "TF100_CS39").random()
    b = splits.image_rng(0, P("/somewhere/else/TF100_CS39")).random()
    assert a == b
    assert a != splits.image_rng(0, "TF102_234").random()
    assert a != splits.image_rng(1, "TF100_CS39").random()


def test_image_rng_makes_sampling_order_independent():
    """The whole point: one patient's draw cannot disturb another's.

    With a single RNG shared across the loop, reversing the iteration order
    changed 49 of 50 held-out samples. Sampling must not depend on the order
    patients happen to be visited in, or the loop cannot safely be
    parallelised and a filtering change ripples into unrelated patients.
    """
    import random
    names = [f"TF{i}_X" for i in range(20)]
    pool = {n: list(range(10)) for n in names}

    def draw(order):
        return {n: splits.image_rng(0, n).sample(pool[n], 2) for n in order}

    forward = draw(names)
    backward = draw(list(reversed(names)))
    shuffled = draw(random.Random(7).sample(names, len(names)))
    assert forward == backward == shuffled


def test_image_rng_isolates_a_changed_population():
    """Shortening one patient's list must not move any other patient."""
    names = [f"TF{i}_X" for i in range(20)]
    full = {n: list(range(10)) for n in names}
    trimmed = dict(full)
    trimmed["TF5_X"] = list(range(9))          # one image removed

    def draw(pool):
        return {n: splits.image_rng(0, n).sample(pool[n], 2) for n in names}

    a, b = draw(full), draw(trimmed)
    moved = {n for n in names if a[n] != b[n]}
    # The trimmed patient MAY move (it need not, if the removed image was not
    # going to be drawn). Nothing else may move at all -- that is the property.
    assert moved <= {"TF5_X"}, f"unrelated patients moved: {sorted(moved - {'TF5_X'})}"
