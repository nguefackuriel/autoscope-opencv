"""Where the time goes in reading one field, stage by stage.

The COOL benchmark times isolated OpenCV calls on a synthetic tile; this times
the real pipeline on real fields, which is the number a lab pays for. Run it
on the laptop for the profile, and on c8g (COOL and stock) for the claim.

    python bench/field_profile.py            # 6 fields spread across the set
    python bench/field_profile.py --fields 20
"""

import argparse
import glob
import time

import cv2 as cv
import numpy as np

from autoscope import parasites as par
from autoscope.counter import count_field
from autoscope.quality import assess_field
from autoscope.segmentation import find_white_cells

STAGES = ["imread", "assess_field", "prepare (disc + stain)", "white cells",
          "parasite candidates", "patches", "patch network (cv.dnn)"]


def profile(path: str, scorer) -> dict:
    t = {}
    t0 = time.perf_counter(); img = cv.imread(path); t["imread"] = time.perf_counter() - t0
    t0 = time.perf_counter(); assess_field(img); t["assess_field"] = time.perf_counter() - t0
    t0 = time.perf_counter(); prepared, mask, _, _ = par._prepare(img, crop_to_fov=True, normalise_stain=True); t["prepare (disc + stain)"] = time.perf_counter() - t0
    t0 = time.perf_counter(); find_white_cells(prepared, crop_to_fov=False, normalise_stain=False); t["white cells"] = time.perf_counter() - t0
    t0 = time.perf_counter(); cands, _ = par.candidates(prepared, mask, features=False); t["parasite candidates"] = time.perf_counter() - t0
    t0 = time.perf_counter(); patches = par.patches_at(prepared, cands) if cands else None; t["patches"] = time.perf_counter() - t0
    t0 = time.perf_counter(); scorer.score(patches) if cands else None; t["patch network (cv.dnn)"] = time.perf_counter() - t0
    t0 = time.perf_counter(); count_field(img, scorer=scorer); t["count_field, end to end"] = time.perf_counter() - t0
    t["candidates"] = len(cands)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=int, default=6)
    ap.add_argument("--glob", default="data/raw/TF*/*.jpg")
    args = ap.parse_args()
    paths = sorted(glob.glob(args.glob))
    paths = paths[:: max(len(paths) // args.fields, 1)][: args.fields]
    scorer = par.PatchScorer.load()
    profile(paths[0], scorer)   # warm up: model load, first-call allocations
    rows = [profile(p, scorer) for p in paths]
    total = np.median([r["count_field, end to end"] for r in rows])
    print(f"{len(rows)} fields, {cv.getNumThreads()} threads, opencv {cv.__version__}\n")
    print(f"{'stage':<26}{'median s':>10}{'share':>8}")
    for k in STAGES:
        v = np.median([r[k] for r in rows])
        print(f"{k:<26}{v:>10.3f}{v / total:>8.0%}")
    print(f"{'count_field, end to end':<26}{total:>10.3f}")
    print(f"\ncandidates per field: median {np.median([r['candidates'] for r in rows]):.0f}")


if __name__ == "__main__":
    main()
