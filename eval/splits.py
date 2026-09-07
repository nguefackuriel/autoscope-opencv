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
from pathlib import Path

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


def all_patients() -> list[str]:
    if not RAW.exists():
        raise SystemExit(f"{RAW} is missing: run `make data` first")
    return sorted(p.name for p in RAW.iterdir() if p.is_dir() and p.name.startswith("TF"))


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
