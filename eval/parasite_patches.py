"""Cut a small image patch around every candidate, with its label, for the
patch classifier.

Same images, same candidate generator, same 15 px matching rule as
parasite_candidates.py, so the two tables describe the same candidates. The
patch is 40 x 40 BGR from the cropped, stain-normalised image, centred on the
peak. Training crops it to 32 x 32 at a random offset, which is the cheapest
augmentation there is and the one that matters most, because the peak is not
always at the chromatin dot's exact centre.

Runs on the fit patients only. Held-out patients are never touched here.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoscope import parasites as par
from eval_helpers import load_pairs
from parasite_candidates import label_candidates

OUT = Path(__file__).resolve().parent / "results"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
PATCH = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--per-patient", type=int, default=1)
    ap.add_argument("--out", default="parasite_patches.npz")
    ap.add_argument("--uninfected-images", type=int, default=0,
                    help="add this many clean fields from the fit half of the uninfected set, all negatives")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fit = set(json.loads((OUT / "wbc_sweep.json").read_text())["fit_patients"])
    half = PATCH // 2
    patches, feats, labels, meta = [], [], [], []
    n_img = total_truth = total_hit = 0
    for img, truth in load_pairs(args.images * 2, per_patient=args.per_patient):
        if truth.path.parent.name not in fit:
            continue
        if n_img >= args.images:
            break
        n_img += 1
        prepared, mask, ox, oy = par._prepare(img, True, True)
        cands, _ = par.candidates(prepared, mask)
        pts = np.array([[p.x - ox, p.y - oy] for p in truth.parasites]) if truth.parasites else np.zeros((0, 2))
        y, hits = label_candidates(cands, pts)
        total_truth += len(pts); total_hit += hits
        h, w = prepared.shape[:2]
        for c, lab in zip(cands, y):
            x, yy = int(c.x), int(c.y)
            if not (half <= x < w - half and half <= yy < h - half):
                continue
            patches.append(prepared[yy - half:yy + half, x - half:x + half].copy())
            feats.append(c.features); labels.append(lab)
            meta.append(truth.path.parent.name)
        if n_img % 20 == 0:
            print(f"  {n_img} images, {len(labels)} patches, candidate recall {total_hit / max(total_truth, 1):.3f}")

    # Clean fields from the FIT half of the uninfected set. Every candidate on
    # them is a negative by definition, and they are the platelets and
    # precipitate the classifier had never been shown: on 3 September 2026
    # it scored them at 0.8 to 0.98 and every clean slide read positive
    # (docs/results-negatives.md). The other 25 patients stay frozen.
    n_clean = 0
    if args.uninfected_images:
        split = json.loads((OUT / "uninfected_split.json").read_text())
        root = RAW / "NIH-NLM-ThickBloodSmearsU" / "Uninfected Patients"
        rng = random.Random(args.seed)
        per = max(1, -(-args.uninfected_images // len(split["fit"])))
        for pat in split["fit"]:
            for path in rng.sample(sorted((root / pat / "tiled").glob("*.tiff")), per):
                if n_clean >= args.uninfected_images:
                    break
                img = cv.imread(str(path), cv.IMREAD_COLOR)
                if img is None:
                    continue
                n_clean += 1
                prepared, mask, _, _ = par._prepare(img, True, True)
                cands, _ = par.candidates(prepared, mask)
                h, w = prepared.shape[:2]
                for c in cands:
                    x, yy = int(c.x), int(c.y)
                    if not (half <= x < w - half and half <= yy < h - half):
                        continue
                    patches.append(prepared[yy - half:yy + half, x - half:x + half].copy())
                    feats.append(c.features); labels.append(0)
                    meta.append(pat)
        print(f"  + {n_clean} clean fields from {len(split['fit'])} uninfected fit patients, {len(labels)} patches now")

    P = np.stack(patches).astype(np.uint8)
    np.savez_compressed(OUT / args.out, patches=P, X=np.stack(feats),
                        y=np.array(labels), patient=np.array(meta))
    print(f"saved {P.shape} patches, {int(np.sum(labels))} positives, {n_img} infected images, {n_clean} clean images")


if __name__ == "__main__":
    main()
