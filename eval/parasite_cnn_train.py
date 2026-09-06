"""Train the patch classifier and write it as ONNX for cv.dnn.

Runs in the TRAINING venv (.venv-train: torch 2.2.2, numpy 1.x, onnx), not
the runtime one. The two never mix: this script reads a parasite_patches
file, which the runtime venv wrote, and writes models/parasite_cnn.onnx,
which the runtime venv checks against its own cv.dnn (parasite_cnn_check.py).

    make train-cnn

Why a patch network at all. Nineteen hand-made features reached F1 0.657
(cross-validated by patient) and the folds spread from 0.57 to 0.72: the
look-alikes, platelets and stain debris, sit on top of the parasites in every
feature we could write down. A parasite is a shape with a context, a dark dot
with a faint blue body, and a small convolutional net reads that from the
pixels directly.

The input is --crop pixels square: 32 up to chain 2, 48 from chain 3
(5 September 2026), because the rods and platelet pairs that pass at 32 px
are 30 to 40 px long and the ring of cytoplasm that says "parasite" sits
outside a 32 px window. `autoscope.parasites.PATCH_PX` has to match the
exported model; parasite_cnn_check.py refuses to pass if it does not.

Validation is by patient. With --val-file it is exactly the patients listed
there (the calibration patients, so the threshold chosen on them afterwards
is chosen on patients the network never saw). Without it, --val-patients
shuffles the infected patient names and takes that many; --folds does k-fold.
Reported at candidate level here; the end-to-end held-out number comes from
parasite_eval.py --model cnn.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUT = Path(__file__).resolve().parent / "results"
MODEL = Path(__file__).resolve().parents[1] / "models" / "parasite_cnn.onnx"
HYBRID = Path(__file__).resolve().parents[1] / "models" / "parasite_hybrid.onnx"


class Net(nn.Module):
    """Input: BGR uint8-range float [n, 3, crop, crop]. Output: probability [n, 1]."""

    def __init__(self, mean, std, crop: int, feat_mean=None, feat_std=None):
        super().__init__()
        self.crop = crop
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        # hybrid: the 19 context features join the conv embedding before the head
        self.hybrid = feat_mean is not None
        n_feat = 0
        if self.hybrid:
            self.register_buffer("fmean", torch.tensor(feat_mean, dtype=torch.float32).view(1, -1))
            self.register_buffer("fstd", torch.tensor(feat_std, dtype=torch.float32).view(1, -1))
            n_feat = len(feat_mean)

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 24), block(24, 48), block(48, 96))   # crop -> crop/8
        side = crop // 8
        self.embed = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(96 * side * side, 96), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(96 + n_feat, 1))

    def logits(self, x, f=None):
        x = (x - self.mean) / self.std
        e = self.embed(self.features(x))
        if self.hybrid:
            e = torch.cat([e, (f - self.fmean) / self.fstd], dim=1)
        return self.head(e)

    def forward(self, x, f=None):
        return torch.sigmoid(self.logits(x, f))


def augment(P: torch.Tensor, rng: torch.Generator, crop: int) -> torch.Tensor:
    """Random crop from the cut patch, flips and 90 degree turns.
    P is uint8 [n, 3, cut, cut]; returns float [n, 3, crop, crop]."""
    ox = torch.randint(0, P.shape[3] - crop + 1, (1,), generator=rng).item()
    oy = torch.randint(0, P.shape[2] - crop + 1, (1,), generator=rng).item()
    x = P[:, :, oy:oy + crop, ox:ox + crop]
    if torch.rand(1, generator=rng).item() < 0.5:
        x = x.flip(3)
    if torch.rand(1, generator=rng).item() < 0.5:
        x = x.flip(2)
    k = torch.randint(0, 4, (1,), generator=rng).item()
    return torch.rot90(x, k, (2, 3)).contiguous().float()


def centre(P: torch.Tensor, crop: int) -> torch.Tensor:
    o = (P.shape[2] - crop) // 2
    return P[:, :, o:o + crop, o:o + crop].contiguous().float()


def prf(y, s, thr, candidate_recall: float = 1.0):
    """Precision, recall, F1, and the count bias.

    `candidate_recall` matters more than it looks. Without it the bias is
    measured against the candidates only, and the parasites that never became
    candidates are missing from both sides of the ratio. That made 0.80 look
    unbiased while it under-counted real parasites by 12%, and it is how the
    shipped threshold came to be wrong. Pass the measured recall so the bias
    is against every annotated parasite.
    """
    pred = s >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    truth = (tp + fn) / max(candidate_recall, 1e-9)
    return p, r, f, (tp + fp - truth) / max(truth, 1) * 100


def as_tensor(P: np.ndarray) -> torch.Tensor:
    """uint8 [n, 3, cut, cut]. Kept as bytes: 160 000 patches at 56 px are
    1.5 GB as bytes and 6 GB as float, and a batch is converted as it is used."""
    return torch.from_numpy(P).permute(0, 3, 1, 2)


def train(P, y, epochs, seed, mean, std, crop, lr=2e-3, F=None):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    if F is not None:
        net = Net(mean, std, crop, F.mean(axis=0), F.std(axis=0) + 1e-6)
        Ft = torch.tensor(F, dtype=torch.float32)
    else:
        net = Net(mean, std, crop)
        Ft = None
    pos = min(max(float(y.mean()), 1e-3), 1 - 1e-3)   # guarded: a smoke set can have no positives
    pos_weight = torch.tensor((1 - pos) / pos * 0.5)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(y) + 255) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    Pt = as_tensor(P)
    yt = torch.tensor(y).float()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(y), generator=rng)
        for i in range(0, len(y), 256):
            idx = perm[i:i + 256]
            xb = augment(Pt[idx], rng, crop)
            fb = Ft[idx] if Ft is not None else None
            loss = nn.functional.binary_cross_entropy_with_logits(net.logits(xb, fb).squeeze(1), yt[idx], pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        print(f"    epoch {ep + 1}/{epochs} done", flush=True)
    net.eval()
    return net


def predict(net, P, crop, F=None):
    Pt = as_tensor(P)
    Ft = torch.tensor(F, dtype=torch.float32) if F is not None else None
    out = []
    with torch.no_grad():
        for i in range(0, len(Pt), 1024):
            fb = Ft[i:i + 1024] if Ft is not None else None
            out.append(net(centre(Pt[i:i + 1024], crop), fb).squeeze(1).numpy())
    return np.concatenate(out)


def input_stats(P: np.ndarray, n: int = 20000, seed: int = 0):
    """Per-channel mean and std of the pixels, from a sample: the full set
    would need a float copy of every patch just to subtract the mean."""
    rs = np.random.RandomState(seed)
    sample = P[rs.choice(len(P), min(n, len(P)), replace=False)].reshape(-1, 3).astype(np.float32)
    return sample.mean(axis=0), sample.std(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-cv", action="store_true", help="skip cross-validation, just train and export")
    ap.add_argument("--patches", default="parasite_patches.npz")
    ap.add_argument("--crop", type=int, default=32, help="network input, cut from the stored patch; must equal parasites.PATCH_PX")
    ap.add_argument("--val-patients", type=int, default=0,
                    help="instead of k-fold: hold this many fit patients out for the threshold table, "
                         "train once on the rest, and export THAT model (no retrain on everything)")
    ap.add_argument("--val-file", default=None,
                    help="instead of k-fold: hold out exactly the patients in this JSON list (results/), e.g. calibration_patients.json")
    ap.add_argument("--hybrid", action="store_true",
                    help="also feed the 19 context features (neighbours, distance to white cells, local field)")
    ap.add_argument("--model-out", default=None, help="where to write the ONNX (default: models/); for smoke tests")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads() - 1))

    d = np.load(OUT / args.patches)
    P, y, patient = d["patches"], d["y"], d["patient"]
    if P.shape[1] < args.crop:
        raise SystemExit(f"stored patches are {P.shape[1]} px, cannot crop {args.crop}")
    recall = json.loads((OUT / "parasite_candidates.json").read_text())["candidate_recall"]
    print(f"candidate recall {recall:.3f}: the count bias below is against every "
          f"annotated parasite, not just the candidates")
    F = d["X"].astype(np.float32) if args.hybrid else None
    model_path = Path(args.model_out) if args.model_out else (HYBRID if args.hybrid else MODEL)
    print(f"{len(y)} patches {P.shape[1:]} cropped to {args.crop}  {int(y.sum())} parasites  {len(set(patient))} patients"
          + ("  + 19 features (hybrid)" if args.hybrid else ""))
    mean, std = input_stats(P)

    table = []
    val = None
    if args.val_file:
        val = set(json.loads((OUT / args.val_file).read_text()))
        missing = val - set(patient.tolist())
        if missing:
            raise SystemExit(f"{len(missing)} validation patients have no patches: {sorted(missing)[:5]}")
    elif args.val_patients:
        # infected patients only: the file may also hold uninfected patients
        # (names TF2xx), and chain 2 let six of them into this shuffle
        infected = sorted(p for p in set(patient.tolist()) if int(y[patient == p].sum()) > 0)
        rs = np.random.RandomState(0); rs.shuffle(infected)
        val = set(infected[:args.val_patients])

    if val is not None:
        te = np.array([p in val for p in patient]); tr = ~te
        print(f"validation: {len(val)} patients, {int(te.sum())} patches; training on {int(tr.sum())} "
              f"({y[tr].mean():.1%} positive)", flush=True)
        final = train(P[tr], y[tr], args.epochs, 0, mean, std, args.crop, F=None if F is None else F[tr])
        scores = predict(final, P[te], args.crop, None if F is None else F[te])
        print("\nvalidation patients (candidate level):\n  thr   precision  recall   F1    count bias")
        for thr in np.arange(0.2, 0.91, 0.05):
            p_, r, f, b = prf(y[te], scores, thr, recall)
            table.append({"threshold": round(float(thr), 2), "precision": round(p_, 4), "recall": round(r, 4),
                          "f1": round(f, 4), "count_bias_pct": round(b, 2)})
            print(f"  {thr:.2f}   {p_:.3f}      {r:.3f}   {f:.3f}   {b:+.1f}%")
    elif not args.no_cv:
        pats = sorted(set(patient)); rs = np.random.RandomState(0); rs.shuffle(pats)
        fold_of = {p: i % args.folds for i, p in enumerate(pats)}
        folds = np.array([fold_of[p] for p in patient])
        scores = np.zeros(len(y), np.float32)
        for k in range(args.folds):
            tr, te = folds != k, folds == k
            net = train(P[tr], y[tr], args.epochs, k, mean, std, args.crop, F=None if F is None else F[tr])
            scores[te] = predict(net, P[te], args.crop, None if F is None else F[te])
            print(f"  fold {k}: {int(te.sum())} patches, "
                  f"F1@0.5 {prf(y[te], scores[te], 0.5, recall)[2]:.3f}", flush=True)
        print("\ncross-validated by patient (candidate level):\n  thr   precision  recall   F1    count bias")
        for thr in np.arange(0.2, 0.81, 0.05):
            p, r, f, b = prf(y, scores, thr, recall)
            table.append({"threshold": round(float(thr), 2), "precision": round(p, 4), "recall": round(r, 4),
                          "f1": round(f, 4), "count_bias_pct": round(b, 2)})
            print(f"  {thr:.2f}   {p:.3f}      {r:.3f}   {f:.3f}   {b:+.1f}%")

    if val is None:
        final = train(P, y, args.epochs, 0, mean, std, args.crop, F=F)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    crop = args.crop
    if args.hybrid:
        torch.onnx.export(final, (torch.zeros(1, 3, crop, crop), torch.zeros(1, F.shape[1])), str(model_path),
                          input_names=["patch", "features"], output_names=["p"],
                          dynamic_axes={"patch": {0: "n"}, "features": {0: "n"}, "p": {0: "n"}},
                          opset_version=13, do_constant_folding=True)
        check = predict(final, P[:512], crop, F[:512])
    else:
        torch.onnx.export(final, torch.zeros(1, 3, crop, crop), str(model_path), input_names=["patch"], output_names=["p"],
                          dynamic_axes={"patch": {0: "n"}, "p": {0: "n"}}, opset_version=13, do_constant_folding=True)
        check = predict(final, P[:512], crop)
    tag = "hybrid" if args.hybrid else "cnn"
    np.save(OUT / f"parasite_{tag}_check.npy", check)   # the runtime venv compares cv.dnn against this
    (OUT / f"parasite_{tag}_train.json").write_text(json.dumps({
        "patches": int(len(y)), "positives": int(y.sum()), "patients": len(set(patient)),
        "epochs": args.epochs, "folds": args.folds, "val_patients": len(val) if val else 0,
        "val_file": args.val_file, "validation_patient_list": sorted(val) if val else None,
        "crop": crop, "stored_patch": int(P.shape[1]),
        "cv_by_patient" if val is None else "validation_patients": table,
        "input_mean_bgr": mean.tolist(), "input_std_bgr": std.tolist(), "hybrid": args.hybrid,
        "patches_file": args.patches,
    }, indent=2))
    print(f"\nexported {model_path.name} ({crop} px input)")


if __name__ == "__main__":
    main()
