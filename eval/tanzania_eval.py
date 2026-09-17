"""Tanzania: another microscope, another camera, another stain. The
generalisation number, measured once.

Two modes, and the rule between them is the whole point.

  --calibrate   the first 10 images of each folder (tanzania_split.json),
                which are the site's calibration slides: measure the nucleus
                size and the raw stain, write tanzania_calibration.json, and
                count those 20 images so a montage can be looked at. This is
                design; these 20 images are spent.
  --run         every other image, once, with the pipeline fixed and the
                calibration applied: per image, parasites and white cells
                found, and the density each would give; then the image-level
                AUC infected against uninfected, the share of uninfected
                images at zero parasites, and the share of infected images
                with at least one. There are no per-parasite annotations
                here, so there is no precision or recall to report: the
                folder label is the only truth (docs/data-notes.md).

Nothing in the pipeline is fitted on this set. The threshold is the
shipped model's, the white cell parameters are the shipped ones, and the
only site-specific number is the scale measured on the ten calibration
slides, which is what any deployment would measure on its own.
"""

import argparse
import json
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from autoscope import photometry
from autoscope.counter import count_field
from autoscope.density import WBC_PER_UL_DEFAULT
from autoscope.parasites import load_scorer, threshold_for
from autoscope.site import SiteCalibration, calibrate

OUT = Path(__file__).resolve().parent / "results"
ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "tanzania"
FOLDERS = {"infected": ROOT / "Thick_Infected" / "Thick_Infected",
           "uninfected": ROOT / "Thick_Uninfected" / "Thick_Uninfected"}


def rank_auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def count_images(paths, label, scale, model, threshold, scorer, out_rows, t0):
    for i, path in enumerate(paths, 1):
        img = cv.imread(str(path))
        if img is None:
            continue
        t = time.perf_counter()
        r = count_field(img, model=model, threshold=threshold, scorer=scorer, scale=scale)
        out_rows.append({"label": label, "image": path.name, "parasites": r.parasites, "wbc": r.white_cells,
                         "per_ul": round(r.parasites / max(r.white_cells, 1) * WBC_PER_UL_DEFAULT, 1) if r.white_cells else None,
                         "seconds": round(time.perf_counter() - t, 2)})
        if i % 50 == 0:
            print(f"  {label}: {i}/{len(paths)} images, {(time.perf_counter() - t0) / 60:.0f} min", flush=True)


def montage(cal_paths, scale, model, threshold, scorer, per_group=48, crop=48, zoom=3, cols=12):
    """Detections on the calibration images, cut from the image at NLM scale
    (what the network saw), so a person can check that a Tanzania detection
    looks like an NLM parasite: docs/figures/tanzania-calibration-<label>.png."""
    from autoscope import parasites as par
    from autoscope.site import rescale
    fig = Path(__file__).resolve().parents[1] / "docs" / "figures"
    fig.mkdir(exist_ok=True)
    for label, paths in cal_paths.items():
        tiles = []
        for p in paths:
            img = rescale(cv.imread(str(p)), scale)
            prepared, mask, ox, oy = par._prepare(img, True, True)
            found = par.find_parasites(prepared, crop_to_fov=False, normalise_stain=False, model=model,
                                       threshold=threshold, scorer=scorer)
            h, w = prepared.shape[:2]; half = crop // 2
            for f in sorted(found, key=lambda f: -f.score)[:6]:
                x0 = int(min(max(f.x - half, 0), w - crop)); y0 = int(min(max(f.y - half, 0), h - crop))
                t = cv.resize(prepared[y0:y0 + crop, x0:x0 + crop], None, fx=zoom, fy=zoom, interpolation=cv.INTER_NEAREST)
                cv.putText(t, f"{f.score:.2f}", (2, 12), cv.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2)
                cv.putText(t, f"{f.score:.2f}", (2, 12), cv.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
                tiles.append(t)
        tiles = tiles[:per_group]
        if not tiles:
            print(f"  no detection on the {label} calibration images"); continue
        size = crop * zoom; rows = (len(tiles) + cols - 1) // cols
        sheet = np.full((rows * size + 24, cols * size, 3), 255, np.uint8)
        cv.putText(sheet, f"Tanzania calibration, {label}, scale {scale}, {model} at {threshold}: top detections per image",
                   (4, 17), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        for k, t in enumerate(tiles):
            r, c = divmod(k, cols)
            sheet[24 + r * size:24 + (r + 1) * size, c * size:(c + 1) * size] = t
        out = fig / f"tanzania-calibration-{label}.png"
        cv.imwrite(str(out), sheet); print(f"  wrote {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run", action="store_true", help="the frozen test images, once")
    ap.add_argument("--again", action="store_true", help="overwrite a previous run; counts as a second look")
    ap.add_argument("--model", default="cnn")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--max-images", type=int, default=0, help="per folder, smoke tests only")
    ap.add_argument("--expected-px", type=float, default=350.0, help="a rough nucleus diameter on the site, its own pixels")
    args = ap.parse_args()
    split = json.loads((OUT / "tanzania_split.json").read_text())
    threshold = threshold_for(args.model, args.threshold)
    scorer = load_scorer(args.model)

    if args.calibrate:
        cal_paths = {k: [FOLDERS[k] / n for n in split["calibration"][k]] for k in FOLDERS}
        images = [cv.imread(str(p)) for k in cal_paths for p in cal_paths[k]]
        # One look at one image: a lymphocyte nucleus is about 350 px across on this rig.
        cal = calibrate("tanzania", images, photometry.normalise, expected_px=args.expected_px)
        cal.save(OUT / "tanzania_calibration.json")
        print(f"calibration on {cal.images} images: nucleus {cal.nucleus_px} px -> scale {cal.scale} "
              f"(raw field L {cal.raw_L}, a-b {cal.raw_stain})")
        rows, t0 = [], time.perf_counter()
        for k in FOLDERS:
            count_images(cal_paths[k], k, cal.scale, args.model, threshold, scorer, rows, t0)
        for r in rows:
            print(f"  {r['label']:10s} {r['image']:16s} parasites {r['parasites']:3d}  white cells {r['wbc']:2d}  {r['seconds']:.1f} s")
        montage(cal_paths, cal.scale, args.model, threshold, scorer)
        (OUT / "tanzania_calibration_counts.json").write_text(json.dumps(
            {"model": args.model, "threshold": threshold, "scale": cal.scale, "rows": rows}, indent=2))
        return

    if args.run:
        # Run once. A second look at a frozen set is a different number, and
        # the report has to say so: --again is the only way to overwrite, and
        # the file records how many runs it has seen.
        prior = OUT / "tanzania_eval.json"
        runs = 1
        if prior.exists():
            if not args.again:
                raise SystemExit(f"{prior.name} exists: Tanzania has been run. --again overwrites it and "
                                 "counts a second look, which the report must state.")
            runs = json.loads(prior.read_text()).get("runs", 1) + 1
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).parent).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                    text=True, cwd=Path(__file__).parent).stdout.strip())
        cal = SiteCalibration.load(OUT / "tanzania_calibration.json")
        rows, t0 = [], time.perf_counter()
        for k, folder in FOLDERS.items():
            cal_names = set(split["calibration"][k])
            paths = [p for p in sorted(folder.glob("*.jpg")) if p.name not in cal_names]
            if args.max_images:
                paths = paths[:args.max_images]
            print(f"{k}: {len(paths)} test images", flush=True)
            count_images(paths, k, cal.scale, args.model, threshold, scorer, rows, t0)
        inf = [r for r in rows if r["label"] == "infected"]; uni = [r for r in rows if r["label"] == "uninfected"]
        p_inf = np.array([r["parasites"] for r in inf]); p_uni = np.array([r["parasites"] for r in uni])
        d_inf = np.array([r["per_ul"] or 0 for r in inf]); d_uni = np.array([r["per_ul"] or 0 for r in uni])
        report = {
            "runs": runs, "commit": commit, "tree_dirty": dirty,
            "model": args.model, "threshold": threshold, "scale": cal.scale, "seconds_per_image": round(float(np.mean([r["seconds"] for r in rows])), 2),
            "infected_images": len(inf), "uninfected_images": len(uni),
            "auc_parasites_per_image": round(rank_auc(p_inf, p_uni), 4),
            "auc_density": round(rank_auc(d_inf, d_uni), 4),
            "uninfected_at_zero_pct": round(float(100 * (p_uni == 0).mean()), 1),
            "infected_with_one_or_more_pct": round(float(100 * (p_inf >= 1).mean()), 1),
            "parasites_per_image": {"infected_median": float(np.median(p_inf)), "infected_p90": float(np.percentile(p_inf, 90)),
                                    "uninfected_mean": round(float(p_uni.mean()), 3), "uninfected_p90": float(np.percentile(p_uni, 90))},
            "wbc_per_image": {"infected_median": float(np.median([r["wbc"] for r in inf])), "uninfected_median": float(np.median([r["wbc"] for r in uni]))},
            "cutoffs": [{"parasites_at_least": c, "sensitivity": round(float((p_inf >= c).mean()), 3), "specificity": round(float((p_uni < c).mean()), 3)} for c in (1, 2, 3, 5, 10)],
            "rows": rows,
        }
        (OUT / "tanzania_eval.json").write_text(json.dumps(report, indent=2))
        print(f"\nTanzania, {len(inf)} infected and {len(uni)} uninfected images, model {args.model} at {threshold}, scale {cal.scale}:")
        print(f"  parasites per image: infected median {np.median(p_inf):.0f} (p90 {np.percentile(p_inf, 90):.0f}), uninfected mean {p_uni.mean():.2f} (p90 {np.percentile(p_uni, 90):.0f})")
        print(f"  white cells per image: infected median {report['wbc_per_image']['infected_median']:.0f}, uninfected {report['wbc_per_image']['uninfected_median']:.0f}")
        print(f"  uninfected images at zero parasites: {report['uninfected_at_zero_pct']}%; infected with at least one: {report['infected_with_one_or_more_pct']}%")
        print(f"  image-level AUC, parasites per image: {report['auc_parasites_per_image']:.3f}; per density: {report['auc_density']:.3f}")
        print(f"  {'parasites >=':>13s} {'sensitivity':>12s} {'specificity':>12s}")
        for c in report["cutoffs"]:
            print(f"  {c['parasites_at_least']:>13d} {c['sensitivity']:>12.3f} {c['specificity']:>12.3f}")
        return
    ap.error("--calibrate or --run")


if __name__ == "__main__":
    main()
