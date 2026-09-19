"""Turn the NLM fit patients into a tiled detection dataset, YOLO and COCO.

WHY TILES

A parasite is about 10 px across in a field roughly 3000 px wide. Feeding the
whole field to a detector at 640 px would shrink it to 2 px, below what any box
head can regress. So tiles are cut at NATIVE resolution and the detector sees
the parasite at its true size. At inference the same tiling is applied and the
detections are merged back (see detector_eval.py).

WHY BOTH FORMATS OVER ONE SET OF TILES

YOLO26 wants a directory of images and .txt labels; torchvision's Faster R-CNN
wants COCO json. Both are written against the SAME jpg tiles, so any difference
between the two models is the model and not the data.

WHAT IS AND IS NOT EXPORTED

Fit patients only, split into train and val by patient. The 50 held-out
patients are never read here and never tiled: they are scored once, on whole
images, by detector_eval.py. Cutting them into tiles at all would invite
somebody to point a training run at the directory.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2 as cv
import numpy as np


# --- repo paths: run from anywhere, installed or not -------------------------
# A remote runner executes the code snapshot without `pip install -e .`, so the
# editable install that normally puts src/ on sys.path is absent and every
# `from autoscope import ...` fails. True of AWS Batch and SageMaker as much as
# of the GPU box the detectors are trained on. This depends on __file__ alone,
# so it survives any working directory, any launch style, and a missing install.
import sys as _sys, pathlib as _pl
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d)
                 for _d in ("src", "eval", "train")]
# ----------------------------------------------------------------------------

from autoscope import annotations as ann, boxes as bx, parasites as par
from autoscope.tiling import MIN_FOV, STRIDE_PX, TILE_PX, covered_tiles, tile_origins
import splits

# Patient images: resolved via splits.patient_root(), not assumed to sit
# directly under data/raw -- download.py may nest them one level down.
CATEGORY = {"id": 1, "name": "parasite", "supercategory": "cell"}


def boxes_in_tile(all_boxes, ox, oy, tile, edge=1):
    """Boxes lying wholly inside the tile, as (x, y, w, h) in tile coordinates.

    A box straddling the tile boundary is a bad target: the detector would be
    taught that a parasite looks like half a parasite. Dropping it is safe
    because the overlap between tiles means a neighbour contains it whole.
    """
    kept = []
    for b in all_boxes:
        x0, y0 = b.x - ox, b.y - oy
        x1, y1 = x0 + b.w, y0 + b.h
        if x0 < edge or y0 < edge or x1 > tile - edge or y1 > tile - edge:
            continue
        kept.append((x0, y0, b.w, b.h))
    return kept


def export(images_per_patient, tile, stride, out, seed, negative_ratio, min_fov,
           uninfected_fields=0, clean_tiles=2, max_patients=0):
    root = splits.patient_root()
    train_ids, val_ids = splits.train_val(seed=seed)
    wanted = {p: "train" for p in train_ids}
    wanted.update({p: "val" for p in val_ids})
    holdout = set(splits.holdout_patients())

    if max_patients:
        # Smoke test only. Keeps both subsets represented so the run still
        # exercises the train/val paths rather than silently testing one.
        keep = set(train_ids[:max_patients]) | set(val_ids[:max(1, max_patients // 4)])
        wanted = {p: s for p, s in wanted.items() if p in keep}
        print(f"SMOKE TEST: {len(wanted)} patients, not the full fit set")

    for sub in ("train", "val"):
        (out / "images" / sub).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sub).mkdir(parents=True, exist_ok=True)

    coco = {s: {"images": [], "annotations": [], "categories": [CATEGORY]}
            for s in ("train", "val")}
    img_id = 0
    ann_id = 0
    stats = Counter()
    negatives_kept = Counter()
    sides = []

    for n_pat, patient in enumerate(sorted(wanted), 1):
        if max_patients and n_pat > max_patients:
            break
        # Belt and braces. The split says these are fit patients, but a bug
        # that leaked a held-out patient into training would silently
        # invalidate every number, so it is checked rather than trusted.
        assert patient not in holdout, f"{patient} is held out, never export it"
        subset = wanted[patient]
        # Only images that actually carry ground truth are candidates. 53 of
        # the 1883 images have none, because the dataset ReadMe says annotation
        # stops once 200 white cells are counted. Sampling first and skipping
        # afterwards would quietly return fewer images than asked for, and the
        # shortfall would land on whichever patients were annotated least.
        candidates = splits.annotated_images(root / patient)
        rng = splits.image_rng(seed, patient)
        stats["images_reserved_for_inference"] += len(
            splits.unannotated_images(root / patient))
        if not candidates:
            continue

        for path in rng.sample(candidates, min(images_per_patient, len(candidates))):
            image = cv.imread(str(path))
            if image is None:
                stats["unreadable"] += 1
                continue
            h_full, w_full = image.shape[:2]
            truth = ann.for_image(path, w_full, h_full)
            if truth is None:
                stats["no_annotation"] += 1
                continue

            prepared, mask, ox_fov, oy_fov = par._prepare(image, True, True)
            top, _, _ = par.tophat(prepared)
            centres = [(p.x - ox_fov, p.y - oy_fov) for p in truth.parasites]
            radii = [p.radius for p in truth.parasites]
            # fallback=False: never invent a box. Fabricating one where the
            # stain shows nothing would be altering the ground truth, and a
            # dataset altered that way would need per-image verification we are
            # not in a position to do. Instead the tiles containing an
            # unboxable annotation are dropped below, so nothing is invented
            # and no real parasite is left labelled as background.
            found, kept, provenance = bx.boxes_for(top, centres, radii, fallback=False)
            unboxable = [centres[i] for i in range(len(centres)) if i not in set(kept)]

            stats["images"] += 1
            stats["annotated_parasites"] += len(centres)
            stats["boxes_derived"] += len(found)
            stats["boxes_from_stain"] += sum(1 for s in provenance if s == "stain")
            stats["annotations_unboxable"] += len(unboxable)
            for b in found:
                sides.append(b.w)
                sides.append(b.h)

            height, width = prepared.shape[:2]
            positives = []
            empties = []
            for oy in tile_origins(height, tile, stride):
                for ox in tile_origins(width, tile, stride):
                    sub_mask = mask[oy:oy + tile, ox:ox + tile]
                    if float((sub_mask > 0).mean()) < min_fov:
                        stats["tiles_outside_fov"] += 1
                        continue
                    # A tile holding an annotation we could not box is not
                    # usable either way: keep it and a real parasite is labelled
                    # background, invent a box and we have altered the ground
                    # truth. Declining the tile does neither.
                    if any(ox <= ux < ox + tile and oy <= uy < oy + tile
                           for ux, uy in unboxable):
                        stats["tiles_with_unboxable"] += 1
                        continue
                    inside = boxes_in_tile(found, ox, oy, tile)
                    if inside:
                        positives.append((ox, oy, inside))
                    else:
                        empties.append((ox, oy, inside))

            # How many DISTINCT boxes reached at least one kept tile. Counting
            # label lines instead would double-count: the tiles overlap by
            # 128 px, so a box near a seam is written into several of them and
            # the total can exceed the number of boxes that exist.
            covered = set()
            for ox, oy, _ in positives:
                for i, b in enumerate(found):
                    if (ox + 1 <= b.x and b.y >= oy + 1
                            and b.x + b.w <= ox + tile - 1
                            and b.y + b.h <= oy + tile - 1):
                        covered.add(i)
            stats["boxes_covered"] += len(covered)

            # Background tiles teach the detector what is not a parasite, but a
            # dataset that is mostly background trains a detector that says no.
            n_neg = int(round(len(positives) * negative_ratio))
            chosen = positives + rng.sample(empties, min(n_neg, len(empties)))
            negatives_kept[subset] += len(chosen) - len(positives)

            for ox, oy, inside in chosen:
                crop = prepared[oy:oy + tile, ox:ox + tile]
                name = f"{patient}_{path.stem}_{ox}_{oy}.jpg"
                cv.imwrite(str(out / "images" / subset / name), crop,
                           [cv.IMWRITE_JPEG_QUALITY, 95])

                lines = []
                for x, y, w, h in inside:
                    lines.append(
                        f"0 {(x + w / 2) / tile:.6f} {(y + h / 2) / tile:.6f} "
                        f"{w / tile:.6f} {h / tile:.6f}")
                (out / "labels" / subset / (name[:-4] + ".txt")).write_text(
                    "\n".join(lines))

                coco[subset]["images"].append(
                    {"id": img_id, "file_name": name, "width": tile, "height": tile})
                for x, y, w, h in inside:
                    coco[subset]["annotations"].append({
                        "id": ann_id, "image_id": img_id, "category_id": 1,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "area": int(w * h), "iscrowd": 0,
                    })
                    ann_id += 1
                img_id += 1
                stats["tiles_" + subset] += 1
                stats["boxes_" + subset] += len(inside)

        done = stats["tiles_train"] + stats["tiles_val"]
        print(f"  {patient} ({subset}): {done} tiles so far")

    # Clean fields from the FIT half of the uninfected set, all background.
    # Every negative above came from an infected image, where a platelet next
    # to a parasite is rare and a platelet on its own is the whole population
    # of a clean field. Chain 3 (docs/results-parasites.md section 15) divided
    # the clean-field false alarms by 22 by adding them to the patch network's
    # training set; a box detector needs the same lesson. Train subset only:
    # the other 25 uninfected patients are the frozen specificity test and are
    # never read here.
    if uninfected_fields:
        split = json.loads((Path(__file__).resolve().parent / "results" / "uninfected_split.json").read_text())
        uroot = Path(__file__).resolve().parents[1] / "data" / "raw" / "NIH-NLM-ThickBloodSmearsU" / "Uninfected Patients"
        n_clean = 0
        for pat in split["fit"]:
            for path in sorted((uroot / pat / "tiled").glob("*.tiff")):
                if uninfected_fields > 0 and n_clean >= uninfected_fields:
                    break
                image = cv.imread(str(path), cv.IMREAD_COLOR)
                if image is None:
                    continue
                n_clean += 1
                prepared, mask, _, _ = par._prepare(image, True, True)
                height, width = prepared.shape[:2]
                origins = [(ox, oy) for oy in tile_origins(height, tile, stride)
                           for ox in tile_origins(width, tile, stride)
                           if float((mask[oy:oy + tile, ox:ox + tile] > 0).mean()) >= min_fov]
                for ox, oy in rng.sample(origins, min(clean_tiles, len(origins))):
                    crop = prepared[oy:oy + tile, ox:ox + tile]
                    name = f"{pat}_{path.stem}_{ox}_{oy}.jpg"
                    cv.imwrite(str(out / "images" / "train" / name), crop, [cv.IMWRITE_JPEG_QUALITY, 95])
                    (out / "labels" / "train" / (name[:-4] + ".txt")).write_text("")
                    coco["train"]["images"].append({"id": img_id, "file_name": name, "width": tile, "height": tile})
                    img_id += 1
                    stats["tiles_train"] += 1
                    stats["tiles_clean"] += 1
        stats["clean_fields"] = n_clean
        print(f"  + {n_clean} clean fields from {len(split['fit'])} uninfected fit patients, {stats['tiles_clean']} background tiles")

    (out / "annotations").mkdir(exist_ok=True)
    for sub in ("train", "val"):
        (out / "annotations" / f"instances_{sub}.json").write_text(
            json.dumps(coco[sub]))

    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: parasite\n")

    derived = stats["boxes_derived"]
    annotated = stats["annotated_parasites"]
    manifest = {
        "split_source": "eval/results/wbc_sweep.json:fit_patients",
        "train_patients": len(train_ids),
        "val_patients": len(val_ids),
        "holdout_patients_untouched": len(holdout),
        "images_per_patient": images_per_patient,
        "tile_px": tile,
        "stride_px": stride,
        "seed": seed,
        "annotated_parasites": annotated,
        "boxes_derived": derived,
        # One box per annotation is the contract, so this is 0.0 unless an
        # annotation sits off the edge of the cropped field entirely.
        "box_drop_rate": round(1 - derived / max(annotated, 1), 4),
        "boxes_from_stain": stats["boxes_from_stain"],
        "boxes_invented": 0,
        "annotations_unboxable": stats["annotations_unboxable"],
        "tiles_dropped_unboxable": stats["tiles_with_unboxable"],
        # Derived boxes that reached a tile. The rest fell in tiles skipped for
        # sitting outside the eyepiece disc, or straddled a tile edge in every
        # tile covering them. Rim parasites are excluded from counting
        # elsewhere in the pipeline too, so this is consistent -- but it is a
        # second, larger shrinkage than the drop rate and must not be invisible.
        "box_instances_written": stats["boxes_train"] + stats["boxes_val"],
        "boxes_covered": stats["boxes_covered"],
        "box_tile_loss_rate": round(
            1 - stats["boxes_covered"] / max(derived, 1), 4),
        "box_side_median_px": float(np.median(sides)) if sides else None,
        "box_side_p10_px": float(np.percentile(sides, 10)) if sides else None,
        "box_side_p90_px": float(np.percentile(sides, 90)) if sides else None,
        "negative_tiles_kept": dict(negatives_kept),
        "clean_fields": stats.get("clean_fields", 0),
        "clean_tiles": stats.get("tiles_clean", 0),
        "counts": dict(stats),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-per-patient", type=int, default=3,
                    help="3 is what the patch network's best held-out run used")
    ap.add_argument("--tile", type=int, default=TILE_PX)
    ap.add_argument("--stride", type=int, default=STRIDE_PX,
                    help="640 minus 512 gives 128 px of overlap")
    ap.add_argument("--negative-ratio", type=float, default=0.2)
    ap.add_argument("--min-fov", type=float, default=MIN_FOV,
                    help="skip tiles less than this fraction inside the disc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-patients", type=int, default=0,
                    help="smoke test on a few patients instead of the fit set")
    ap.add_argument("--out", default="data/derived/detect")
    ap.add_argument("--uninfected-fields", type=int, default=-1,
                    help="clean fields from the fit half of the uninfected set, as background tiles; -1 = every field (569), 0 = none")
    ap.add_argument("--clean-tiles", type=int, default=2, help="background tiles cut from each clean field")
    ap.add_argument("--max-patients", type=int, default=0, help="smoke tests")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    m = export(args.images_per_patient, args.tile, args.stride, out,
               args.seed, args.negative_ratio, args.min_fov,
               args.uninfected_fields, args.clean_tiles, args.max_patients)

    print(f"\ntrain {m['counts'].get('tiles_train', 0)} tiles, "
          f"{m['counts'].get('boxes_train', 0)} boxes")
    print(f"val   {m['counts'].get('tiles_val', 0)} tiles, "
          f"{m['counts'].get('boxes_val', 0)} boxes")
    print(f"\n{m['boxes_derived']} boxes from {m['annotated_parasites']} "
          f"annotations ({m['box_drop_rate'] * 100:.2f}% unboxed)")
    print(f"  every box contoured from the stain; none invented")
    print(f"  {m['annotations_unboxable']} annotations had no stain peak; the "
          f"{m['tiles_dropped_unboxable']} tiles containing them were dropped")
    print(f"box side px: p10 {m['box_side_p10_px']}, "
          f"median {m['box_side_median_px']}, p90 {m['box_side_p90_px']}")
    print(f"{m['boxes_covered']} of those reached a tile "
          f"({m['box_tile_loss_rate'] * 100:.1f}% never covered), written as "
          f"{m['box_instances_written']} label lines across overlapping tiles")
    print(f"\nheld-out patients untouched: {m['holdout_patients_untouched']}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
