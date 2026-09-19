"""Cut a small image patch around every candidate, with its label, for the
patch classifier.

Same candidate generator and 15 px matching rule as parasite_candidates.py,
so the tables describe the same candidates. The patch is PATCH x PATCH BGR
from the cropped, stain-normalised image, centred on the peak. Training crops
it at a random offset (56 to 48, or 40 to 32), which is the cheapest
augmentation there is and the one that matters most, because the peak is not
always at the chromatin dot's exact centre.

Runs on the fit patients only. Held-out patients are never touched here.

TWO WAYS TO BUILD THE SET

  --images N --per-patient k   the recipe up to chain 2: N fit images, k per
                               patient, every candidate kept.

  --mine                       chain 3, 5 September 2026: every image of every
                               fit patient (1 247 instead of 280), but not
                               every candidate. Nine in ten candidates are easy
                               negatives the network already scores near zero,
                               and the failures are the 3% it scores above 0.2
                               on clean and sparse fields (results-parasites.md
                               section 13). A set that is 90% easy teaches it
                               little about those. So: every positive, every
                               negative the current network scores at or above
                               --hard-min, and a random --easy-keep share of
                               the rest. The set stays near 150 000 patches
                               while the network sees every hard look-alike on
                               four times the images.

The calibration patients (calibration_patients.json) are the trainer's
validation set. They are kept the old way, --per-patient images each and every
candidate, so their table stays comparable across chains and is not shaped by
the mining. The trainer must hold out exactly these patients (--val-file):
in chain 2 it shuffled the 125 patient names in the file and held out a
different twenty, six of them uninfected, so the threshold was chosen on
patients the network had trained on. Fixed on 5 September.

Clean fields: --uninfected-images N adds N fields from the fit half of the
uninfected set, all negatives, mined the same way when --mine is on. -1 means
every field of that half (569). The other half stays frozen.
"""

import argparse
import json
import random
import sys
import time
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

from autoscope import parasites as par
from autoscope import annotations as ann, parasites as par
from eval_helpers import load_pairs
from parasite_candidates import label_candidates

OUT = Path(__file__).resolve().parent / "results"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
PATCH = 40


class Miner:
    """Scores candidates with the previous network(s) so the hard ones can be kept.

    Several models, comma-separated, score every candidate and the maximum
    counts: a miner hides what it already rejects (results-parasites.md
    16e, 16f: chain 5 lost the rods its chain-3 miner rejected, chain 6
    lost the platelets its chain-5 miner rejected), so chain 7 mines with
    all of them at once. All must take the same patch size.
    """

    def __init__(self, models: str | Path, size: int):
        paths = [Path(m) for m in str(models).split(",") if m]
        self.scorers = [par.PatchScorer.load(m) for m in paths]
        self.size = size

    def __call__(self, prepared, cands) -> np.ndarray:
        if not cands:
            return np.zeros(0, np.float32)
        patches = par.patches_at(prepared, cands, size=self.size)
        return np.max(np.stack([s.score(patches) for s in self.scorers]), axis=0)


def keep_mask(y: np.ndarray, scores: np.ndarray, hard_min: float, easy_keep: float,
              rng: random.Random) -> np.ndarray:
    """Every positive, every hard negative, a random share of the easy ones."""
    keep = (y == 1) | (scores >= hard_min)
    easy = np.nonzero(~keep)[0]
    n_easy = int(round(len(easy) * easy_keep))
    if n_easy:
        keep[rng.sample(easy.tolist(), n_easy)] = True
    return keep


def cut(prepared, cands, keep, half):
    """Patches for the kept candidates that fit inside the image."""
    h, w = prepared.shape[:2]
    out, idx = [], []
    for i, c in enumerate(cands):
        if not keep[i]:
            continue
        x, yy = int(c.x), int(c.y)
        if not (half <= x < w - half and half <= yy < h - half):
            continue
        out.append(prepared[yy - half:yy + half, x - half:x + half].copy())
        idx.append(i)
    return out, idx


def fit_images(fit: set[str], calibration: set[str], per_patient_cal: int):
    """Every annotated image of every fit patient; calibration patients get
    `per_patient_cal` images each, chosen the way load_pairs chooses (seed 0),
    so their table is the same kind of sample as before."""
    rng = random.Random(0)
    for patient in sorted(p for p in RAW.iterdir() if p.is_dir() and p.name in fit):
        paths = sorted(patient.glob("*.jpg"))
        if patient.name in calibration:
            paths = rng.sample(paths, min(per_patient_cal, len(paths)))
        for path in paths:
            img = cv.imread(str(path))
            if img is None:
                continue
            h, w = img.shape[:2]
            truth = ann.for_image(path, w, h)
            if truth is None:
                continue
            yield img, truth, patient.name in calibration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--per-patient", type=int, default=1)
    ap.add_argument("--out", default="parasite_patches.npz")
    ap.add_argument("--patch", type=int, default=PATCH, help="patch cut from the image; training crops 8 px off it")
    ap.add_argument("--uninfected-images", type=int, default=0,
                    help="add this many clean fields from the fit half of the uninfected set, all negatives; -1 = every field")
    ap.add_argument("--mine", action="store_true",
                    help="every fit image; keep positives, hard negatives (>= --hard-min under --miner-model) and --easy-keep of the rest")
    ap.add_argument("--hard-min", type=float, default=0.2)
    ap.add_argument("--easy-keep", type=float, default=0.1)
    ap.add_argument("--miner-model", default=str(par.CNN_PATH),
                    help="the network that scores candidates for mining; several, comma-separated, keep the maximum")
    ap.add_argument("--miner-size", type=int, default=32, help="that network's input size")
    ap.add_argument("--features", action="store_true",
                    help="also compute the 19 hand-made features (hybrid path); a second per image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=0, help="smoke tests: stop after this many infected and clean images")
    ap.add_argument("--candidates", default="tophat", choices=["tophat", "heatmap", "both"],
                    help="heatmap: the learned detector's peaks propose (chain 4, 6 September 2026), so the judge learns their "
                         "look-alikes; both: the union of the two proposers (chain 5), so the judge keeps the top-hat set's "
                         "balance and sees the heat map's look-alikes as well")
    ap.add_argument("--easy-keep-clean", type=float, default=None,
                    help="share of easy negatives kept on clean fields (default: --easy-keep). Chain 4 kept 2 608 clean-field "
                         "patches of 72 000 and read 1.0 false parasites per clean field; 1.0 keeps every clean-field candidate")
    args = ap.parse_args()
    if args.easy_keep_clean is None:
        args.easy_keep_clean = args.easy_keep

    fit = set(json.loads((OUT / "wbc_sweep.json").read_text())["fit_patients"])
    calibration = set(json.loads((OUT / "calibration_patients.json").read_text()))
    half = args.patch // 2
    rng = random.Random(args.seed)
    miner = Miner(args.miner_model, args.miner_size) if args.mine else None
    features = (args.features or not args.mine) and args.candidates == "tophat"   # the old recipe always computed them

    def propose(prepared, mask):
        if args.candidates == "heatmap":
            return par.heatmap_candidates(prepared, mask)
        cands = par.candidates(prepared, mask, features=features)[0]
        if args.candidates == "both":
            extra = par.heatmap_candidates(prepared, mask)
            if cands and extra:
                a = np.array([[c.x, c.y] for c in cands]); b = np.array([[c.x, c.y] for c in extra])
                d = np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2).min(axis=1)
                extra = [c for c, dist in zip(extra, d) if dist > 6.0]   # a heat peak on a top-hat peak is the same candidate
            cands = cands + extra
        return cands

    patches, feats, labels, meta, source = [], [], [], [], []
    stats = {"fit": [0, 0, 0, 0], "calibration": [0, 0, 0, 0], "clean": [0, 0, 0, 0]}  # images, candidates, kept, positives
    total_truth = total_hit = 0
    t0 = time.perf_counter()

    def take(prepared, cands, y, scores, src, patient):
        if miner is not None and src != "calibration":
            keep = keep_mask(y, scores, args.hard_min, args.easy_keep_clean if src == "clean" else args.easy_keep, rng)
        else:
            keep = np.ones(len(cands), bool)
        cut_patches, idx = cut(prepared, cands, keep, half)
        patches.extend(cut_patches)
        labels.extend(int(y[i]) for i in idx)
        meta.extend([patient] * len(idx))
        source.extend([src] * len(idx))
        if features:
            feats.extend(cands[i].features for i in idx)
        s = stats[src]
        s[0] += 1; s[1] += len(cands); s[2] += len(idx); s[3] += int(sum(y[i] for i in idx))

    def infected(img, truth, src):
        nonlocal total_truth, total_hit
        prepared, mask, ox, oy = par._prepare(img, True, True)
        cands = propose(prepared, mask)
        pts = (np.array([[p.x - ox, p.y - oy] for p in truth.parasites])
               if truth.parasites else np.zeros((0, 2)))
        y, hits = label_candidates(cands, pts)
        total_truth += len(pts); total_hit += hits
        scores = miner(prepared, cands) if miner is not None else None
        take(prepared, cands, y, scores, src, truth.path.parent.name)

    if args.mine:
        for n, (img, truth, is_cal) in enumerate(fit_images(fit, calibration, args.per_patient), 1):
            if args.max_images and n > args.max_images:
                break
            infected(img, truth, "calibration" if is_cal else "fit")
            if n % 50 == 0:
                print(f"  {n} images, {len(labels)} patches, candidate recall "
                      f"{total_hit / max(total_truth, 1):.3f}, {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
    else:
        n_img = 0
        for img, truth in load_pairs(args.images * 2, per_patient=args.per_patient):
            if truth.path.parent.name not in fit:
                continue
            if n_img >= args.images:
                break
            n_img += 1
            infected(img, truth, "calibration" if truth.path.parent.name in calibration else "fit")
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
        crng = random.Random(args.seed)
        want = args.uninfected_images
        per = None if want < 0 else max(1, -(-want // len(split["fit"])))
        for pat in split["fit"]:
            paths = sorted((root / pat / "tiled").glob("*.tiff"))
            if per is not None:
                paths = crng.sample(paths, min(per, len(paths)))
            for path in paths:
                if (want >= 0 and n_clean >= want) or (args.max_images and n_clean >= args.max_images):
                    break
                img = cv.imread(str(path), cv.IMREAD_COLOR)
                if img is None:
                    continue
                n_clean += 1
                prepared, mask, _, _ = par._prepare(img, True, True)
                cands = propose(prepared, mask)
                y = np.zeros(len(cands), np.int32)
                scores = miner(prepared, cands) if miner is not None else None
                take(prepared, cands, y, scores, "clean", pat)
        print(f"  + {n_clean} clean fields from {len(split['fit'])} uninfected fit patients, {len(labels)} patches now")

    P = np.stack(patches).astype(np.uint8)
    arrays = dict(patches=P, y=np.array(labels), patient=np.array(meta), source=np.array(source))
    if features:
        arrays["X"] = np.stack(feats)
    np.savez_compressed(OUT / args.out, **arrays)
    summary = {
        "out": args.out, "patch": args.patch, "mine": args.mine, "candidates": args.candidates, "easy_keep_clean": args.easy_keep_clean,
        "hard_min": args.hard_min if args.mine else None, "easy_keep": args.easy_keep if args.mine else None,
        "miner_model": ",".join(Path(m).name for m in args.miner_model.split(",")) if args.mine else None,
        "candidate_recall_fit_images": round(total_hit / max(total_truth, 1), 4),
        "patches": int(len(P)), "positives": int(sum(labels)),
        "by_source": {k: {"images": v[0], "candidates": v[1], "kept": v[2], "positives": v[3]} for k, v in stats.items()},
        "minutes": round((time.perf_counter() - t0) / 60, 1),
    }
    (OUT / (Path(args.out).stem + ".json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved {P.shape} patches, {int(np.sum(labels))} positives, "
          f"{stats['fit'][0] + stats['calibration'][0]} infected images, {n_clean} clean images")


if __name__ == "__main__":
    main()
