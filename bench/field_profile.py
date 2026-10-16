"""Where the time goes in reading one field, stage by stage.

The COOL benchmark times isolated OpenCV calls on a synthetic tile; this times
the real pipeline on real fields, which is the number a lab pays for. Run it
on the laptop for the profile, and on c8g (COOL and stock) for the claim.

    python bench/field_profile.py            # 6 fields spread across the set
    python bench/field_profile.py --fields 20
"""

import argparse
import glob
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

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
    t0 = time.perf_counter(); counted = count_field(img, scorer=scorer); t["count_field, end to end"] = time.perf_counter() - t0
    t["candidates"] = len(cands)
    # The answer, not only the time. A faster build that counts differently is
    # not the same pipeline, and a speed claim without this is a claim about a
    # benchmark rather than about AutoScope.
    t["field"] = Path(path).name
    t["parasites"] = counted.parasites
    t["white_cells"] = counted.white_cells
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=int, default=6)
    ap.add_argument("--glob", default="data/raw/TF*/*.jpg")
    ap.add_argument("--label", default="local",
                    help="c8g-cool, c8g-stock, c8g-no-kleidicv, c7i-stock")
    ap.add_argument("--fields-per-slide", type=float, default=29.6,
                    help="Median fields the agent reads per slide "
                         "(eval/results/agent_eval.json, decision arm).")
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

    # What a lab actually buys: slides per hour on one core-set, and fields per
    # second. The COOL claim is only interesting if it survives the trip from a
    # microbenchmark to this number.
    threads = cv.getNumThreads()
    fields_per_second = 1.0 / total
    slides_per_hour = 3600.0 / (total * args.fields_per_slide)

    report = {
        "label": args.label,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "opencv_version": cv.__version__,
        "machine": platform.machine(),
        "threads": threads,
        # The line the whole COOL claim rests on. A result file that does not
        # say which build produced it is not evidence.
        "kleidicv_in_build": "kleidicv" in cv.getBuildInformation().lower(),
        "fields": len(rows),
        "fields_per_slide": args.fields_per_slide,
        "seconds_per_field_median": round(float(total), 4),
        "fields_per_second": round(fields_per_second, 4),
        "slides_per_hour": round(slides_per_hour, 2),
        "stages": {k: round(float(np.median([r[k] for r in rows])), 4) for k in STAGES},
        # Per field, so two arms can be diffed directly. The comparison is in
        # bench/compare_counts.py.
        "counts": [{"field": r["field"], "parasites": r["parasites"],
                    "white_cells": r["white_cells"]} for r in rows],
    }

    out = Path(__file__).resolve().parent / "results" / f"profile-{args.label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n{slides_per_hour:.1f} slides/hour, {fields_per_second:.2f} fields/s "
          f"on {threads} threads | kleidicv in build: {report['kleidicv_in_build']}")
    print("counts: " + ", ".join(f"{r['field']} {r['parasites']}p/{r['white_cells']}w"
                                 for r in rows))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
