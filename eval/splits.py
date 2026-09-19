"""The patient split, read from the one artifact that defines it.

The split is not a seed and cannot be recomputed. It is a literal list of 100
patient ids in `eval/results/wbc_sweep.json` under `fit_patients`, written when
the white cell parameters were swept, and every held-out number this project
has published is "the patients that list does not contain". The white cell
work, the parasite work and anything added later must read the same file, or
the numbers stop being comparable and the held-out set quietly stops being
held out.

The 50 held-out patients are never listed anywhere. They are the complement,
by construction, so there is nothing to keep in sync.
"""

import json
import random

# --- repo paths: run from anywhere, installed or not -------------------------
import sys as _sys, pathlib as _pl
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d)
                 for _d in ("src", "eval", "train")]
# ----------------------------------------------------------------------------

from pathlib import Path

from autoscope import annotations as ann

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
SWEEP = Path(__file__).resolve().parent / "results" / "wbc_sweep.json"


def fit_patients() -> set[str]:
    """The 100 patients every parameter in this repo was fitted on."""
    if not SWEEP.exists():
        raise SystemExit(f"{SWEEP} is missing: it is what defines the split")
    ids = json.loads(SWEEP.read_text())["fit_patients"]
    if len(ids) != 100:
        raise SystemExit(f"expected 100 fit patients, found {len(ids)}")
    return set(ids)


def _has_patients(d: Path) -> bool:
    return any(p.is_dir() and p.name.startswith("TF") for p in d.iterdir())


def patient_root() -> Path:
    """The directory that actually holds the TF* patient folders.

    download.py extracts to data/raw/ThickBloodSmears_150/, but the older eval
    scripts scan data/raw/ directly, so where the patients land depends on how
    the archive was unpacked. Resolving it here turns a confusing "0 patients,
    F1 0.000" run into an error that says what is wrong.
    """
    if not RAW.exists():
        raise SystemExit(f"{RAW} is missing: run `make data` first")
    if _has_patients(RAW):
        return RAW
    for child in sorted(p for p in RAW.iterdir() if p.is_dir()):
        if _has_patients(child):
            return child
    raise SystemExit(
        f"no TF* patient directories under {RAW} or one level below it.\n"
        "Run `make data` and read the layout it prints.")


def all_patients() -> list[str]:
    root = patient_root()
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("TF"))


def holdout_patients() -> list[str]:
    """The complement. Never fitted on, never trained on, scored once."""
    fit = fit_patients()
    return [p for p in all_patients() if p not in fit]


def train_val(val_fraction: float = 0.2, seed: int = 0) -> tuple[list[str], list[str]]:
    """Split the FIT patients into train and val, by patient.

    Checkpoint selection has to happen on something, and it must not be the
    held-out 50: choosing a model by its score on those patients is fitting on
    them, and the generalisation claim in the report depends on not doing that.
    So the validation set is carved out of the fit patients instead.

    Split by patient, never by image. Two fields from one slide share a stain,
    a focus and a camera, so putting one in train and one in val would leak.
    """
    fit = sorted(fit_patients() & set(all_patients()))
    rng = random.Random(seed)
    rng.shuffle(fit)
    n_val = max(1, int(round(len(fit) * val_fraction)))
    val = sorted(fit[:n_val])
    train = sorted(fit[n_val:])
    return train, val


# ---------------------------------------------------------------------------
# Which images may be used at all
#
# 53 of the 1883 images carry no ground truth file. That is by design: the
# dataset ReadMe says annotation stops once 200 white cells have been counted,
# because that is enough for a parasitaemia. A detection on one of those images
# cannot be scored right or wrong, so they belong in no set that produces a
# number -- not training, not validation, not test.
#
# They are not worthless. Paired against an annotated field from the same
# patient they are marginally better focused, so they are reserved as the
# INFERENCE SAMPLE: the images a chosen model is finally run over for a
# qualitative look, once evaluation is finished and the model is fixed.
#
# Everything that samples images goes through annotated_images(). Filtering
# before sampling matters: sampling first and skipping afterwards quietly
# returns fewer images than asked for, and the shortfall falls entirely on the
# 37 patients that have unannotated fields.
# ---------------------------------------------------------------------------

def _patient_dir(patient) -> Path:
    """Accept either a patient id or a directory."""
    p = Path(patient)
    return p if p.is_absolute() or p.exists() else patient_root() / str(patient)


def annotated_images(patient) -> list[Path]:
    """Images for `patient` that have ground truth. The only usable ones."""
    return [p for p in sorted(_patient_dir(patient).glob("*.jpg"))
            if ann.gt_path(p).exists()]


def unannotated_images(patient) -> list[Path]:
    """Images for `patient` reserved for final inference. Never measured on."""
    return [p for p in sorted(_patient_dir(patient).glob("*.jpg"))
            if not ann.gt_path(p).exists()]


def inference_sample() -> list[Path]:
    """Every reserved image, across all patients."""
    out = []
    for name in all_patients():
        out.extend(unannotated_images(name))
    return out


def assert_measurable(path) -> None:
    """Guard: refuse an image that cannot be scored.

    Cheap to call and loud when it matters. A number computed over an image
    with no ground truth is not a number, and the failure is otherwise silent.
    """
    path = Path(path)
    if not ann.gt_path(path).exists():
        raise AssertionError(
            f"{path.name} has no ground truth: it is part of the reserved "
            "inference sample and must not be measured on. "
            "Use splits.annotated_images() to pick images.")


def image_rng(seed, patient) -> random.Random:
    """A generator keyed on the patient, for sampling images within it.

    One `random.Random(seed)` shared across a patient loop makes every draw
    depend on every draw before it. Shorten one patient's image list and the
    generator state shifts for every patient after it, so a change that should
    touch 18 patients touches 20, and the sample depends on the order the loop
    happened to run in. Measured on the held-out 50: reversing the iteration
    order changed 49 of 50 samples.

    Keying on the patient instead makes each draw depend only on the seed, that
    patient's name and its own image list. The loop becomes order-independent
    and therefore safe to parallelise, a filtering change affects only the
    patients it actually touches, and one patient's draw can be reproduced on
    its own without replaying the loop to reach it.

    The key is the directory NAME, never the full path: a path would embed the
    checkout location and give different samples on Azure ML than on a laptop.
    """
    return random.Random(f"{seed}:{Path(str(patient)).name}")
