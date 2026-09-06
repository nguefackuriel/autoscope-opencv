"""Count every field of every UNINFECTED patient, once.

The companion of agent_fields.py for the NLM uninfected thick smear set
(`make data-uninfected`). Every one of the 150 development patients is
infected, so until this ran nothing had measured what the pipeline says about
a clean slide. That number is specificity, and a screening tool without it is
not a screening tool.

Same schema as agent_fields.json so negative_eval.py can replay the real
policies over it with the same CachedStage. Truth parasites is the folder
label: zero. Truth white cells come from the set's own annotations.

LAYOUT AND COORDINATES, measured on 2 September 2026

    NIH-NLM-ThickBloodSmearsU/
      Uninfected Patients/<TFnnn_x>/tiled/<stamp>.tiff   50 patients, 1141 images
      Annotations/<TFnnn_x>/<stamp>.txt                  White_Blood_Cell points only

The TIFFs are 7-page pyramids; page 0 is the full 4032 x 3024 frame, and it
is LANDSCAPE, unlike the infected set's portrait JPEGs. The annotation header
is the same `count,4032,3024`, so here the points map directly (x = coord1,
y = coord2). Checked the same way as for the infected set: mean L under the
points is 66 with the direct mapping against 156 for the infected set's
rotation, on a lit field at 197. Do not reuse annotations.for_image here.

Frozen test material: nothing here is used to fit anything. Run once, after
the pipeline is fixed, and report what it says.
"""

import argparse
import json
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from autoscope.counter import count_field
from autoscope.parasites import DEFAULT_THRESHOLD, load_scorer
from autoscope.quality import assess_field

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
ROOT = RAW / "NIH-NLM-ThickBloodSmearsU"
OUT = Path(__file__).resolve().parent / "results"


IMAGES = ROOT / "Uninfected Patients"
ANNOTATIONS = ROOT / "Annotations"


def patients_under(root: Path) -> list[Path]:
    """One folder per patient."""
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("TF"))


def images_of(patient: Path) -> list[Path]:
    return sorted((patient / "tiled").glob("*.tiff"))


def truth_wbc(patient: Path, image: Path, w: int, h: int) -> int | None:
    """White cells the annotators marked, or None when there is no file."""
    txt = ANNOTATIONS / patient.name / f"{image.stem}.txt"
    if not txt.exists():
        return None
    n = 0
    for line in txt.read_text(errors="ignore").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 7 and parts[1] == "White_Blood_Cell":
            x, y = float(parts[5]), float(parts[6])
            if 0 <= x < w and 0 <= y < h:
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None, help="default: the model's own")
    ap.add_argument("--model", default="cnn", choices=["cnn", "heatmap"])
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--max-images", type=int, default=None, help="per patient, for a quick pass")
    ap.add_argument("--out", default="negative_fields.json")
    args = ap.parse_args()

    if not ROOT.exists():
        raise SystemExit(f"{ROOT} not found: run `make data-uninfected` first")
    patients = patients_under(IMAGES)
    if args.max_patients:
        patients = patients[:args.max_patients]
    print(f"{len(patients)} uninfected patient folders under {ROOT.name}")

    scorer = load_scorer(args.model)
    from autoscope.parasites import threshold_for
    args.threshold = threshold_for(args.model, args.threshold)
    rows, t0 = [], time.perf_counter()
    for i, patient in enumerate(patients, 1):
        images = images_of(patient)
        if args.max_images:
            images = images[:args.max_images]
        before = len(rows)
        for path in images:
            img = cv.imread(str(path), cv.IMREAD_COLOR)   # page 0 of the pyramid
            if img is None:
                continue
            h, w = img.shape[:2]
            t = time.perf_counter()
            result = count_field(img, model=args.model, threshold=args.threshold, scorer=scorer)
            quality = assess_field(img)
            rows.append({
                "patient": patient.name, "image": path.name,
                "found_parasites": result.parasites, "found_wbc": result.white_cells,
                "truth_parasites": 0, "truth_wbc": truth_wbc(patient, path, w, h),
                "focus": round(quality.focus_score, 2),
                "even": round(quality.illumination_uniformity, 4),
                "debris": round(quality.debris_ratio, 4),
                "usable": quality.why_not() is None,
                "seconds": round(time.perf_counter() - t, 2),
            })
        mine = rows[before:]
        fp = sum(r["found_parasites"] for r in mine)
        wbc = sum(r["found_wbc"] for r in mine)
        hit = sum(r["found_parasites"] > 0 for r in mine)
        print(f"  [{i}/{len(patients)}] {patient.name}: {len(mine)} images, "
              f"{fp} false parasites in {wbc} white cells "
              f"({fp / wbc * 200 if wbc else 0:.1f} per 200 WBC), "
              f"{hit}/{len(mine)} images with >= 1, "
              f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        # Save as we go: a 40 minute run should survive an interruption.
        (OUT / args.out).write_text(json.dumps({
            "threshold": args.threshold, "model": args.model, "patients": i, "fields": len(rows),
            "label": "uninfected", "source_set": ROOT.name,
            "seconds_per_field": round(float(np.mean([r["seconds"] for r in rows])), 2),
            "rows": rows,
        }, indent=2))

    print(f"\n{len(rows)} fields from {len(patients)} patients, "
          f"{np.mean([r['seconds'] for r in rows]):.2f} s per field")


if __name__ == "__main__":
    main()
