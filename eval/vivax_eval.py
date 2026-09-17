"""*P. vivax*: does a detector trained only on *P. falciparum* find it?

Plan step 6, and the last frozen set. The 150 vivax patients are the same NLM
rig, same camera, same stain, same annotators as the 150 falciparum patients
everything here was built on. The only thing that changes is the parasite, so
whatever the numbers say is about morphology and nothing else. That is rare
and it is why this set is worth spending.

    python eval/vivax_eval.py --layout        # what the files actually are
    python eval/vivax_eval.py --run           # the frozen run, ONCE

WHAT THIS CAN AND CANNOT ANSWER

It can answer: what fraction of annotated vivax parasites does the shipped
pair find at the shipped threshold, how do their scores compare with
falciparum's, and would the density be right if a vivax slide were put through
the pipeline as if it were falciparum.

It cannot answer: species identification. Telling vivax from falciparum is a
different job needing vivax in training, and it is out of scope for 26 October.
The report has to say so plainly — it is the one ECAMM row we will not have.

WHAT A LOW RECALL WOULD MEAN, AND WHAT A HIGH ONE WOULD

Both outcomes are useful and the report says which happened.

  Low recall   the detector is falciparum-specific. A vivax patient would be
               under-called or missed, which is a safety statement about where
               AutoScope may be deployed, and the honest answer is a deployment
               note, not a fix.
  High recall  the ring stages look alike enough that the detector generalises.
               Then the risk is the opposite one: a vivax infection reported as
               a falciparum density, treated with the wrong drug.

Either way the product needs the same thing, and the score distribution is
what decides its threshold: a "non-falciparum morphology, confirm species"
flag raised when a slide's detections score systematically lower than
falciparum's do.

RUN ONCE

Nothing is fitted here. The threshold is the shipped model's, measured on the
calibration patients months ago; the scale is 1.0 because it is the NLM rig.
The results file is written once and `--again` is the only way to overwrite
it, which makes a second look something a person has to type rather than
something that happens.
"""

import argparse
import json
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from autoscope import annotations as ann
from autoscope import fov
from autoscope import parasites as par
from autoscope.counter import count_field

OUT = Path(__file__).resolve().parent / "results"
ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "NIH-NLM-ThickBloodSmearsPV"
IMAGES = ROOT / "All_PvTk"
LABELS = ROOT / "All_annotations"

# THIS SET IS NOT SHAPED LIKE THE INFECTED SET. Two differences, both found by
# reading the files on 16 September 2026 before anything was run:
#
#   1. The annotation label is "Parasitized", not "Parasite", and the files sit
#      under All_annotations/<patient>/ rather than beside the images in
#      GT_updated/. So annotations.for_image cannot read this set and returns
#      None for every image; using it would have reported a recall of 0/0 and
#      looked like a clean run.
#   2. There are NO white cell annotations. Across all 3 014 images the only
#      label present is Parasitized: 43 101 parasites, 0 white cells.
#   3. THE ROTATION IS THE OTHER WAY. The infected set maps the landscape
#      annotation frame into the portrait image clockwise (x = width - coord2,
#      y = coord1). This set is counter-clockwise (x = coord2,
#      y = height - coord1). Measured on 489 annotations across 30 patients,
#      as the darkest pixel within 6 px of the point:
#
#          reading             L median   share darker than 150
#          counter-clockwise      102            72.8%
#          clockwise              204             5.5%
#          random inside the disc 205             4.1%
#
#      Clockwise is indistinguishable from throwing darts at the disc. Using
#      annotations.to_image_frame here would have scored the detector against
#      noise and reported a recall near zero as a finding about vivax.
#
# (2) is what decides the scope of this run. With no annotated denominator
# there is no true density to compare against, so this set can answer "does the
# detector see vivax" and cannot answer "would the density be right". The
# report says that rather than quoting a density error computed against our own
# white cell count, which would be measuring the detector against itself.
VIVAX_LABEL = "Parasitized"

# The same tolerance parasite_eval.py uses on falciparum. Changing it here
# would make the two recalls incomparable, which is the only thing this run is
# for.
TOLERANCE_PX = 15.0

# How far around an annotation the transform check looks for the dark dot. The
# annotator's own centre is a median 0.8 px from the darkest point on the
# infected set, but the chromatin dot is only about 10 px across, so a window
# of about one dot radius is what turns "is there a parasite here" into a
# question a single number can answer.
PROBE_PX = 6

# A parasite's chromatin dot reads well under this; the lit film sits near 205.
DARK_L = 150


def to_image_frame(coord1: float, coord2: float, height: int) -> tuple[float, float]:
    """Landscape annotation coordinates into this set's portrait pixels.

    Counter-clockwise, unlike the infected set. See the note above: this was
    measured, not assumed, and --layout re-measures it before every run.
    """
    return coord2, height - coord1


def _darkest_near(lab, x: int, y: int, radius: int) -> tuple[float, float]:
    """L and b* of the darkest pixel within `radius` of (x, y).

    Darkest, not mean: a parasite is a small dark dot on a lighter film, and a
    mean over the window is mostly film whichever transform is used.
    """
    h, w = lab.shape[:2]
    x0, x1 = max(x - radius, 0), min(x + radius + 1, w)
    y0, y1 = max(y - radius, 0), min(y + radius + 1, h)
    window = lab[y0:y1, x0:x1]
    if window.size == 0:
        return (255.0, 128.0)
    flat = window.reshape(-1, 3)
    darkest = flat[int(np.argmin(flat[:, 0]))]
    return (float(darkest[0]), float(darkest[2]))


def read_annotations(image_path: Path, height: int) -> list[ann.Annotation]:
    """The vivax ground truth for one image, in the image's own frame.

    Same Circle convention as the infected set — point 1 is the centre, point 2
    is on the rim — but the opposite rotation, which --layout re-measures.
    """
    path = LABELS / image_path.parent.name / f"{image_path.stem}.txt"
    if not path.exists():
        return []

    items: list[ann.Annotation] = []
    for line in path.read_text(errors="ignore").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 9 or parts[1] != VIVAX_LABEL or parts[3] != "Circle":
            continue
        a1, b1, a2, b2 = (float(v) for v in parts[5:9])
        cx, cy = to_image_frame(a1, b1, height)
        rx, ry = to_image_frame(a2, b2, height)
        radius = ((rx - cx) ** 2 + (ry - cy) ** 2) ** 0.5
        items.append(ann.Annotation(ann.PARASITE, cx, cy, radius))
    return items


def images(limit_patients: int = 0, per_patient: int = 0) -> list[Path]:
    patients = sorted(p for p in IMAGES.rglob("*") if p.is_dir() and any(p.glob("*.jpg")))
    if limit_patients:
        patients = patients[:limit_patients]
    out: list[Path] = []
    for patient in patients:
        found = sorted(patient.glob("*.jpg"))
        out.extend(found[:per_patient] if per_patient else found)
    return out


def layout(samples: int = 12) -> None:
    """Print what is on disk, and check the coordinate transform, before
    anything is written against it.

    The uninfected set turned out to be landscape TIFF pyramids when the
    infected set is portrait JPEG, and its annotations needed no rotation. Two
    hours of reading the files saved a run measured against noise. Same rule
    here, and it earned its keep again: this set uses a different label and a
    different directory.

    The transform check is the one from annotations.py: a parasite annotation
    should land on something darker and more purple than the lit field. Both
    readings are printed, rotated and plain, so the winner is visible rather
    than asserted.
    """
    if not ROOT.exists():
        raise SystemExit(f"{ROOT} is missing. Run: make data-vivax")

    files = images()
    patients = sorted({f.parent.name for f in files})
    print(f"{ROOT}")
    print(f"  {len(patients)} patients, {len(files)} images")

    annotated = sum(len(read_annotations(f, 4032)) for f in files[:50])
    print(f"  {annotated} parasites in the first 50 images "
          f"({annotated / 50:.1f} per image)")

    sample = files[0]
    img = cv.imread(str(sample))
    print(f"  first image {sample.relative_to(ROOT)}: {img.shape[1]} x {img.shape[0]}")
    print(f"  label used: {VIVAX_LABEL}; white cell annotations: none in this set")

    # THE CHECK THAT DECIDES WHETHER THIS RUN MEANS ANYTHING.
    #
    # The reference is not the whole frame: these images are a circular eyepiece
    # view on a black surround, so the median over the whole image is the
    # surround, reads L = 1, and makes every annotation look bright. It is not
    # the lit disc either. It is RANDOM POINTS INSIDE THE LIT DISC, because the
    # question is not "is this point darkish" but "is this point darker than
    # somewhere nobody annotated". A wrong transform scatters points over blood
    # film, and blood film is darkish.
    #
    # Both possible rotations are measured. The plain reading is not a
    # candidate: coord1 reaches 3593 on a 3024-wide image, so the annotation
    # frame is the landscape one and some rotation is required.
    #
    # The probe is the darkest pixel within PROBE_PX. A single pixel is the
    # right probe for a 100 px white cell nucleus and the wrong one for a 10 px
    # chromatin dot, which is what is being checked here.
    rng = np.random.default_rng(0)
    clockwise, counter, background = [], [], []
    total = 0

    for path in files[:samples]:
        image = cv.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        lab = cv.cvtColor(image, cv.COLOR_BGR2Lab)
        disc = fov.detect(image)
        if disc is None:
            continue

        label_file = LABELS / path.parent.name / f"{path.stem}.txt"
        if not label_file.exists():
            continue

        points = []
        for line in label_file.read_text(errors="ignore").splitlines()[1:]:
            parts = line.split(",")
            if len(parts) >= 9 and parts[1] == VIVAX_LABEL and parts[3] == "Circle":
                points.append((float(parts[5]), float(parts[6])))

        for c1, c2 in points:
            total += 1
            cx, cy = int(width - c2), int(c1)                 # the infected set's way
            if 0 <= cx < width and 0 <= cy < height:
                clockwise.append(_darkest_near(lab, cx, cy, PROBE_PX)[0])
            ux, uy = to_image_frame(c1, c2, height)           # this set's way
            if 0 <= int(ux) < width and 0 <= int(uy) < height:
                counter.append(_darkest_near(lab, int(ux), int(uy), PROBE_PX)[0])

        for _ in range(len(points)):
            angle = rng.uniform(0, 2 * np.pi)
            radius = disc.radius * 0.85 * np.sqrt(rng.uniform())
            background.append(_darkest_near(
                lab, int(disc.cx + radius * np.cos(angle)),
                int(disc.cy + radius * np.sin(angle)), PROBE_PX)[0])

    def row(name, values):
        v = np.asarray(values, float)
        if v.size == 0:
            return f"    {name:<26}{'no points':>10}"
        return (f"    {name:<26}{np.median(v):>10.0f}{np.percentile(v, 25):>8.0f}"
                f"{np.mean(v < DARK_L):>12.1%}")

    print(f"\n  transform check on {samples} images, {total} annotations")
    print(f"    {'reading':<26}{'L median':>10}{'p25':>8}{'share dark':>12}")
    print(row("counter-clockwise (used)", counter))
    print(row("clockwise (infected set)", clockwise))
    print(row("random inside the disc", background))

    dark_counter = float(np.mean(np.asarray(counter) < DARK_L)) if counter else 0.0
    dark_background = float(np.mean(np.asarray(background) < DARK_L)) if background else 1.0

    if dark_counter > 3 * max(dark_background, 0.01):
        print(f"\n  The counter-clockwise reading lands on dark objects "
              f"{dark_counter / max(dark_background, 0.01):.0f}x more often than chance. "
              "Safe to run.")
    else:
        print("\n  WARNING: the transform does not beat chance. Do not run until this "
              "is explained (src/autoscope/annotations.py).")


def match(found, truth, tolerance: float = TOLERANCE_PX):
    """Greedy nearest match, highest-scoring detection first.

    Returns matched scores, missed truth count, and false-positive scores.
    """
    taken = set()
    matched_scores = []
    false_scores = []

    for detection in sorted(found, key=lambda f: -f.score):
        best, best_d = None, tolerance
        for i, t in enumerate(truth):
            if i in taken:
                continue
            d = ((detection.x - t.x) ** 2 + (detection.y - t.y) ** 2) ** 0.5
            if d < best_d:
                best, best_d = i, d
        if best is None:
            false_scores.append(float(detection.score))
        else:
            taken.add(best)
            matched_scores.append(float(detection.score))

    return matched_scores, len(truth) - len(taken), false_scores


def run(model: str, threshold: float | None, limit_patients: int, per_patient: int) -> dict:
    scorer = par.load_scorer(model)
    threshold = par.threshold_for(model, threshold)
    files = images(limit_patients, per_patient)
    if not files:
        raise SystemExit(f"no images under {ROOT}. Run: make data-vivax")

    print(f"{len(files)} images, model {model} at {threshold}", flush=True)

    rows = []
    started = time.perf_counter()
    for i, path in enumerate(files, 1):
        img = cv.imread(str(path))
        if img is None:
            continue
        truth_parasites = read_annotations(path, img.shape[0])

        result = count_field(img, model=model, threshold=threshold, scorer=scorer)
        matched, missed, false = match(result.parasite_points, truth_parasites)

        rows.append({
            "image": str(path.relative_to(ROOT)),
            "patient": path.parent.name,
            "annotated_parasites": len(truth_parasites),
            "found_parasites": result.parasites,
            "found_wbc": result.white_cells,
            "matched": len(matched),
            "missed": missed,
            "false": len(false),
            "matched_scores": [round(s, 4) for s in matched],
            "false_scores": [round(s, 4) for s in false],
        })
        if i % 100 == 0:
            print(f"  {i}/{len(files)}, {(time.perf_counter() - started) / 60:.0f} min",
                  flush=True)

    return summarise(rows, model, threshold)


def summarise(rows: list[dict], model: str, threshold: float) -> dict:
    """Aggregate, counting only patients that have ground truth.

    60 of the 150 patient folders carry NO annotation file at all (1 210 of the
    3 014 images). Those images still produce detections — 17 028 of them — and
    including them in the "found" total against an annotated total of 25 706
    makes the detector look like it over-counts by 58% when on the annotated
    patients it slightly under-counts. Precision computed over them would be
    dividing by a denominator that has no truth behind it. So every rate here
    is over the 90 annotated patients, and the unannotated ones are reported
    separately as what they are: images we cannot score.
    """
    scored = [r for r in rows if r["annotated_parasites"] > 0
              or _patient_has_truth(rows, r["patient"])]
    unscored = [r for r in rows if r not in scored]

    total_annotated = sum(r["annotated_parasites"] for r in scored)
    total_matched = sum(r["matched"] for r in scored)
    total_found = sum(r["found_parasites"] for r in scored)
    all_scores = [s for r in scored for s in r["matched_scores"]]
    false_scores = [s for r in scored for s in r["false_scores"]]

    by_patient: dict[str, list[int]] = {}
    for r in scored:
        acc = by_patient.setdefault(r["patient"], [0, 0])
        acc[0] += r["matched"]
        acc[1] += r["annotated_parasites"]
    patient_recall = [m / a for m, a in by_patient.values() if a > 0]

    precision = total_matched / total_found if total_found else None
    recall = total_matched / total_annotated if total_annotated else None
    f1 = (2 * total_matched / (total_annotated + total_found)
          if total_annotated + total_found else None)

    return {
        "set": "NIH-NLM-ThickBloodSmearsPV",
        "species": "P. vivax",
        "detector_trained_on": "P. falciparum only",
        "model": model,
        "threshold": threshold,
        "images_scored": len(scored),
        "images_without_ground_truth": len(unscored),
        "patients_scored": len(by_patient),
        "annotated_parasites": total_annotated,
        "matched": total_matched,
        "found": total_found,
        "recall_overall": round(recall, 4) if recall is not None else None,
        "precision_overall": round(precision, 4) if precision is not None else None,
        "f1_overall": round(f1, 4) if f1 is not None else None,
        "recall_per_patient_median": round(float(np.median(patient_recall)), 4) if patient_recall else None,
        "recall_per_patient_p10": round(float(np.percentile(patient_recall, 10)), 4) if patient_recall else None,
        "recall_per_patient_p90": round(float(np.percentile(patient_recall, 90)), 4) if patient_recall else None,
        "patients_under_10pc_recall": int(sum(1 for r in patient_recall if r < 0.10)),
        # The number that kills the planned species flag: if a detection the
        # annotator did not mark scores as high as one they did, no threshold on
        # the score can tell vivax morphology from falciparum morphology.
        "score_median_matched": round(float(np.median(all_scores)), 4) if all_scores else None,
        "score_median_unmatched": round(float(np.median(false_scores)), 4) if false_scores else None,
        "density_error_median": None,
        "density_error_note": "not measurable: no white cell annotations in this set",
        "rows": rows,
    }


def _patient_has_truth(rows: list[dict], patient: str) -> bool:
    return any(r["annotated_parasites"] > 0 for r in rows if r["patient"] == patient)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", action="store_true", help="inspect the files, write nothing")
    ap.add_argument("--layout-samples", type=int, default=12)
    ap.add_argument("--run", action="store_true", help="the frozen run, once")
    ap.add_argument("--again", action="store_true",
                    help="overwrite a previous run; this counts as a second look "
                         "and the report has to say so")
    ap.add_argument("--model", default="cnn")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--patients", type=int, default=0, help="smoke tests only")
    ap.add_argument("--per-patient", type=int, default=0, help="smoke tests only")
    ap.add_argument("--resummarise", action="store_true",
                    help="recompute the summary from the rows already written. "
                         "Not a second look: no image is opened.")
    args = ap.parse_args()

    if args.layout:
        layout(args.layout_samples)
        return

    if not args.run and not args.resummarise:
        ap.error("--layout, --run or --resummarise")

    OUT.mkdir(exist_ok=True)
    prior = OUT / "vivax_eval.json"

    if args.resummarise:
        previous = json.loads(prior.read_text())
        summary = summarise(previous["rows"], previous["model"], previous["threshold"])
        prior.write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
        return
    smoke = args.patients or args.per_patient

    if prior.exists() and not args.again and not smoke:
        raise SystemExit(
            f"{prior.name} exists: P. vivax has been run. It is a frozen set and this "
            "is its one look. --again overwrites it, and any result that came from a "
            "second look has to say so in the report."
        )

    summary = run(args.model, args.threshold, args.patients, args.per_patient)

    if smoke:
        print("\nSMOKE TEST, not written. Drop --patients/--per-patient for the real run.")
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
        return

    prior.write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"\nwritten to {prior}")


if __name__ == "__main__":
    main()
