"""Train the patch classifier and write it as ONNX for cv.dnn.

Runs in the TRAINING venv (.venv-train: torch 2.2.2, numpy 1.x, onnx), not
the runtime one. The two never mix: this script reads parasite_patches.npz,
which the runtime venv wrote, and writes models/parasite_cnn.onnx, which the
runtime venv checks against its own cv.dnn (parasite_cnn_check.py).

    make train-cnn

Why a patch network at all. Nineteen hand-made features reached F1 0.657
(cross-validated by patient) and the folds spread from 0.57 to 0.72: the
look-alikes, platelets and stain debris, sit on top of the parasites in every
feature we could write down. A parasite is a shape with a context, a dark dot
with a faint blue body, and a small convolutional net reads that from 32 x 32
pixels directly.

Validation is by patient, five folds, same as everything else. Reported at
candidate level here; the end-to-end held-out number comes from
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
CROP = 32


class Net(nn.Module):
    """Input: BGR uint8-range float [n, 3, 32, 32]. Output: probability [n, 1]."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 24), block(24, 48), block(48, 96))   # 32 -> 4
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(96 * 4 * 4, 96),
                                  nn.ReLU(), nn.Dropout(0.3), nn.Linear(96, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return torch.sigmoid(self.head(self.features(x)))

    def logits(self, x):
        x = (x - self.mean) / self.std
        return self.head(self.features(x))


def augment(P: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Random 32 crop from 40, flips and 90 degree turns. P is [n, 3, 40, 40]."""
    n = P.shape[0]
    ox = torch.randint(0, P.shape[3] - CROP + 1, (1,), generator=rng).item()
    oy = torch.randint(0, P.shape[2] - CROP + 1, (1,), generator=rng).item()
    x = P[:, :, oy:oy + CROP, ox:ox + CROP]
    if torch.rand(1, generator=rng).item() < 0.5:
        x = x.flip(3)
    if torch.rand(1, generator=rng).item() < 0.5:
        x = x.flip(2)
    k = torch.randint(0, 4, (1,), generator=rng).item()
    return torch.rot90(x, k, (2, 3)).contiguous()


def centre(P: torch.Tensor) -> torch.Tensor:
    o = (P.shape[2] - CROP) // 2
    return P[:, :, o:o + CROP, o:o + CROP].contiguous()


def prf(y, s, thr):
    pred = s >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, (tp + fp - (tp + fn)) / max(tp + fn, 1) * 100


def train(P, y, epochs, seed, mean, std, lr=2e-3):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    net = Net(mean, std)
    pos = float(y.mean())
    pos_weight = torch.tensor((1 - pos) / pos * 0.5)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(y) + 255) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    Pt = torch.tensor(P).permute(0, 3, 1, 2).float()
    yt = torch.tensor(y).float()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(y), generator=rng)
        for i in range(0, len(y), 256):
            idx = perm[i:i + 256]
            xb = augment(Pt[idx], rng)
            loss = nn.functional.binary_cross_entropy_with_logits(net.logits(xb).squeeze(1), yt[idx], pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    return net


def predict(net, P):
    Pt = centre(torch.tensor(P).permute(0, 3, 1, 2).float())
    out = []
    with torch.no_grad():
        for i in range(0, len(Pt), 1024):
            out.append(net(Pt[i:i + 1024]).squeeze(1).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-cv", action="store_true", help="skip cross-validation, just train and export")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads() - 1))

    d = np.load(OUT / "parasite_patches.npz")
    P, y, patient = d["patches"], d["y"], d["patient"]
    print(f"{len(y)} patches {P.shape[1:]}  {int(y.sum())} parasites  {len(set(patient))} patients")
    mean = P.reshape(-1, 3).mean(axis=0); std = P.reshape(-1, 3).std(axis=0)

    table = []
    if not args.no_cv:
        pats = sorted(set(patient)); rs = np.random.RandomState(0); rs.shuffle(pats)
        fold_of = {p: i % args.folds for i, p in enumerate(pats)}
        folds = np.array([fold_of[p] for p in patient])
        scores = np.zeros(len(y), np.float32)
        for k in range(args.folds):
            tr, te = folds != k, folds == k
            net = train(P[tr], y[tr], args.epochs, k, mean, std)
            scores[te] = predict(net, P[te])
            print(f"  fold {k}: {int(te.sum())} patches, F1@0.5 {prf(y[te], scores[te], 0.5)[2]:.3f}", flush=True)
        print("\ncross-validated by patient (candidate level):\n  thr   precision  recall   F1    count bias")
        for thr in np.arange(0.2, 0.81, 0.05):
            p, r, f, b = prf(y, scores, thr)
            table.append({"threshold": round(float(thr), 2), "precision": round(p, 4), "recall": round(r, 4),
                          "f1": round(f, 4), "count_bias_pct": round(b, 2)})
            print(f"  {thr:.2f}   {p:.3f}      {r:.3f}   {f:.3f}   {b:+.1f}%")

    final = train(P, y, args.epochs, 0, mean, std)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(final, torch.zeros(1, 3, CROP, CROP), str(MODEL), input_names=["patch"], output_names=["p"],
                      dynamic_axes={"patch": {0: "n"}, "p": {0: "n"}}, opset_version=13, do_constant_folding=True)
    check = predict(final, P[:512])
    np.save(OUT / "parasite_cnn_check.npy", check)   # the runtime venv compares cv.dnn against this
    (OUT / "parasite_cnn_train.json").write_text(json.dumps({
        "patches": int(len(y)), "positives": int(y.sum()), "patients": len(set(patient)),
        "epochs": args.epochs, "folds": args.folds, "crop": CROP, "cv_by_patient": table,
        "input_mean_bgr": mean.tolist(), "input_std_bgr": std.tolist(),
    }, indent=2))
    print(f"\nexported {MODEL.name}")


if __name__ == "__main__":
    main()
