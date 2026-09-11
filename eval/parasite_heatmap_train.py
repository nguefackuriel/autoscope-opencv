"""Train the learned parasite detector: a small U-Net that turns the
half-resolution field into a heat map with a peak on every parasite, and
write it as ONNX for cv.dnn.

Runs in the TRAINING venv (.venv-train). Reads parasite_tiles.npz
(parasite_tiles.py, runtime venv), writes models/parasite_heatmap.onnx and
parasite_heatmap_check.npy, which parasite_heatmap_check.py compares
against cv.dnn.

WHY A HEAT MAP. The two-step detector (top-hat peaks, then a patch
classifier) has a recall ceiling of 0.91 from the candidate step and judges
each candidate without its neighbours; the dense slides under-count by half
and the sparse ones still over-count (results-parasites.md section 15). A
fully convolutional network reads the field with its context, and its own
peaks are the candidates: no merge at 7 px, no white cell exclusion zone,
no rim margin. The point annotations become Gaussian targets (sigma 2.5 px
at half resolution, the chromatin dot is about 5 px there) and the network
regresses them with a binary cross-entropy that weights the peaks up.

VALIDATION is by patient: the calibration patients' tiles, never trained
on, give the precision / recall table per threshold; the threshold that
ships is chosen afterwards on their whole images by choose_threshold.py,
like the patch network's. The held-out patients are never read here.

Export is fully convolutional with dynamic height and width, so the same
graph runs on a 256 px tile here and a 1 336 px field in the pipeline.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUT = Path(__file__).resolve().parent / "results"
MODEL = Path(__file__).resolve().parents[1] / "models" / "parasite_heatmap.onnx"
CROP = 256
SIGMA = 2.5          # px at half resolution
MATCH_PX = 7.5       # half of the pipeline's 15 px rule
NMS_PX = 4


class Block(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                               nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

    def forward(self, x):
        return self.f(x)


class UNet(nn.Module):
    """Input: BGR uint8-range float [n, 3, h, w], h and w multiples of 8.
    Output: heat in 0..1, [n, 1, h, w]."""

    def __init__(self, mean, std, c=(12, 24, 48, 96)):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.e1, self.e2, self.e3, self.e4 = Block(3, c[0]), Block(c[0], c[1]), Block(c[1], c[2]), Block(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.u3, self.d3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2), Block(2 * c[2], c[2])
        self.u2, self.d2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2), Block(2 * c[1], c[1])
        self.u1, self.d1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2), Block(2 * c[0], c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def logits(self, x):
        x = (x - self.mean) / self.std
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)

    def forward(self, x):
        return torch.sigmoid(self.logits(x))


def render(points: np.ndarray, size: int, sigma: float = SIGMA) -> np.ndarray:
    """Gaussian bump at every point, max where they overlap."""
    target = np.zeros((size, size), np.float32)
    r = int(3 * sigma)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    bump = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)).astype(np.float32)
    for x, y in points:
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = max(cx - r, 0), min(cx + r + 1, size)
        y0, y1 = max(cy - r, 0), min(cy + r + 1, size)
        if x0 >= x1 or y0 >= y1:
            continue
        target[y0:y1, x0:x1] = np.maximum(target[y0:y1, x0:x1],
                                          bump[y0 - cy + r:y1 - cy + r, x0 - cx + r:x1 - cx + r])
    return target


def points_by_tile(P: np.ndarray, n: int) -> list[np.ndarray]:
    out = [np.zeros((0, 2), np.float32) for _ in range(n)]
    if len(P):
        order = np.argsort(P[:, 0], kind="stable")
        P = P[order]
        idx = P[:, 0].astype(int)
        starts = np.searchsorted(idx, np.arange(n)); ends = np.searchsorted(idx, np.arange(n), side="right")
        for i in range(n):
            if ends[i] > starts[i]:
                out[i] = P[starts[i]:ends[i], 1:3].astype(np.float32)
    return out


def make_batch(T, pts, idx, rng: np.random.RandomState, train: bool):
    """Random 256 crop (centre crop at validation), flips and turns, with
    the points moved along; returns float tensors x [b,3,256,256], y [b,1,256,256]."""
    xs, ys = [], []
    cut = T.shape[1]
    for i in idx:
        ox, oy = (rng.randint(0, cut - CROP + 1), rng.randint(0, cut - CROP + 1)) if train else ((cut - CROP) // 2,) * 2
        tile = T[i, oy:oy + CROP, ox:ox + CROP]
        p = pts[i] - [ox, oy] if len(pts[i]) else pts[i]
        p = p[(p[:, 0] >= 0) & (p[:, 0] < CROP) & (p[:, 1] >= 0) & (p[:, 1] < CROP)] if len(p) else p
        if train:
            if rng.rand() < 0.5:
                tile = tile[:, ::-1]; p = np.stack([CROP - 1 - p[:, 0], p[:, 1]], 1) if len(p) else p
            if rng.rand() < 0.5:
                tile = tile[::-1]; p = np.stack([p[:, 0], CROP - 1 - p[:, 1]], 1) if len(p) else p
            k = rng.randint(0, 4)
            if k:
                tile = np.rot90(tile, k)   # (x, y) -> rotated k times counter-clockwise
                for _ in range(k):
                    p = np.stack([p[:, 1], CROP - 1 - p[:, 0]], 1) if len(p) else p
        xs.append(np.ascontiguousarray(tile)); ys.append(render(p, CROP))
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float()
    y = torch.from_numpy(np.stack(ys))[:, None]
    return x, y


def peaks(heat: np.ndarray, floor: float, nms: int = NMS_PX) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima above the floor: (n, 2) xy and their heights. A square
    max filter in torch (this venv has no OpenCV); the pipeline's
    HeatmapScorer uses a round one in cv.dnn, same radius."""
    t = torch.from_numpy(heat)[None, None]
    dil = nn.functional.max_pool2d(t, 2 * nms + 1, stride=1, padding=nms)[0, 0].numpy()
    ys, xs = np.nonzero((heat >= dil) & (heat > floor))
    return np.stack([xs, ys], 1).astype(np.float32), heat[ys, xs]


def match(found: np.ndarray, truth: np.ndarray, tol: float = MATCH_PX) -> int:
    if len(found) == 0 or len(truth) == 0:
        return 0
    d = np.linalg.norm(found[:, None, :] - truth[None, :, :], axis=2)
    hits, used_f, used_t = 0, set(), set()
    for i, j in np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]:
        if d[i, j] > tol:
            break
        if i in used_f or j in used_t:
            continue
        used_f.add(int(i)); used_t.add(int(j)); hits += 1
    return hits


def validate(net, T, pts, idx, rng, thresholds) -> list[dict]:
    """Peaks against points on the validation tiles, per threshold."""
    net.eval()
    heats, truths = [], []
    with torch.no_grad():
        for i in range(0, len(idx), 32):
            x, _ = make_batch(T, pts, idx[i:i + 32], rng, train=False)
            heats.extend(net(x)[:, 0].numpy())
            cut = T.shape[1]; o = (cut - CROP) // 2
            for j in idx[i:i + 32]:
                p = pts[j] - [o, o] if len(pts[j]) else pts[j]
                truths.append(p[(p[:, 0] >= 0) & (p[:, 0] < CROP) & (p[:, 1] >= 0) & (p[:, 1] < CROP)] if len(p) else p)
    table = []
    n_truth = sum(len(t) for t in truths)
    for thr in thresholds:
        tp = fp = 0
        for h, t in zip(heats, truths):
            f, _ = peaks(h, thr)
            hit = match(f, t); tp += hit; fp += len(f) - hit
        p = tp / max(tp + fp, 1); r = tp / max(n_truth, 1)
        table.append({"threshold": round(float(thr), 2), "precision": round(p, 4), "recall": round(r, 4),
                      "f1": round(2 * p * r / max(p + r, 1e-9), 4), "count_bias_pct": round((tp + fp - n_truth) / max(n_truth, 1) * 100, 2)})
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="parasite_tiles.npz")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--tiles-per-epoch", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pos-weight", type=float, default=10.0, help="loss weight is 1 + this x target")
    ap.add_argument("--val-file", default="calibration_patients.json")
    ap.add_argument("--val-every", type=int, default=4)
    ap.add_argument("--channels", default="12,24,48,96")
    ap.add_argument("--model-out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from results/parasite_heatmap_last.pt")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads() - 1))
    torch.manual_seed(args.seed); rng = np.random.RandomState(args.seed)

    d = np.load(OUT / args.tiles)
    T, P, patient, source = d["tiles"], d["points"], d["patient"], d["source"]
    pts = points_by_tile(P, len(T))
    val_patients = set(json.loads((OUT / args.val_file).read_text()))
    is_val = np.array([p in val_patients for p in patient])
    tr_idx, va_idx = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
    print(f"{len(T)} tiles {T.shape[1:]}, {len(P)} parasites, {len(set(patient.tolist()))} patients; "
          f"training on {len(tr_idx)} tiles, validating on {len(va_idx)} from {len(val_patients & set(patient.tolist()))} calibration patients", flush=True)
    sample = T[rng.choice(len(T), min(2000, len(T)), replace=False)].reshape(-1, 3).astype(np.float32)
    mean, std = sample.mean(axis=0), sample.std(axis=0)
    channels = tuple(int(c) for c in args.channels.split(","))
    net = UNet(mean, std, channels)
    print(f"U-Net {channels}: {sum(p.numel() for p in net.parameters()) / 1e3:.0f}k parameters", flush=True)

    per_epoch = min(args.tiles_per_epoch, len(tr_idx))
    steps = args.epochs * ((per_epoch + args.batch - 1) // args.batch)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    thresholds = np.arange(0.1, 0.91, 0.1)
    history = []
    t0 = time.perf_counter()
    ckpt = OUT / "parasite_heatmap_last.pt"
    first = 0
    if args.resume and ckpt.exists():
        state = torch.load(ckpt)
        net.load_state_dict(state["net"]); opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        sched.total_steps = steps   # resume with the same --epochs; a different one reshapes the schedule
        rng.set_state(state["rng"]); history = state["history"]; first = state["epoch"]
        print(f"resumed after epoch {first}", flush=True)
    for ep in range(first, args.epochs):
        net.train()
        order = rng.choice(tr_idx, per_epoch, replace=per_epoch > len(tr_idx))
        total = 0.0
        for i in range(0, per_epoch, args.batch):
            x, y = make_batch(T, pts, order[i:i + args.batch], rng, train=True)
            logits = net.logits(x)
            w = 1.0 + args.pos_weight * y
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += float(loss) * len(x)
        line = f"    epoch {ep + 1}/{args.epochs}: loss {total / per_epoch:.4f}, {(time.perf_counter() - t0) / 60:.0f} min"
        if len(va_idx) and ((ep + 1) % args.val_every == 0 or ep + 1 == args.epochs):
            table = validate(net, T, pts, va_idx, rng, thresholds)
            best = max(table, key=lambda r: r["f1"])
            line += f"; validation F1 {best['f1']:.3f} at {best['threshold']:.1f} (P {best['precision']:.3f} R {best['recall']:.3f})"
            history.append({"epoch": ep + 1, "table": table})
        print(line, flush=True)
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": rng.get_state(), "history": history, "epoch": ep + 1}, ckpt)

    table = history[-1]["table"] if history else []
    if table:
        print("\nvalidation tiles, calibration patients (peak level):\n  thr   precision  recall   F1    count bias")
        for r in table:
            print(f"  {r['threshold']:.2f}   {r['precision']:.3f}      {r['recall']:.3f}   {r['f1']:.3f}   {r['count_bias_pct']:+.1f}%")

    net.eval()
    model_path = Path(args.model_out) if args.model_out else MODEL
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, torch.zeros(1, 3, CROP, CROP), str(model_path), input_names=["image"], output_names=["heat"],
                      dynamic_axes={"image": {0: "n", 2: "h", 3: "w"}, "heat": {0: "n", 2: "h", 3: "w"}},
                      opset_version=13, do_constant_folding=True)
    check_idx = (va_idx if len(va_idx) else tr_idx)[:8]
    with torch.no_grad():
        x, _ = make_batch(T, pts, check_idx, rng, train=False)
        check = net(x)[:, 0].numpy()
    np.save(OUT / "parasite_heatmap_check.npy", check)
    (OUT / "parasite_heatmap_train.json").write_text(json.dumps({
        "tiles": int(len(T)), "parasites": int(len(P)), "patients": len(set(patient.tolist())),
        "training_tiles": int(len(tr_idx)), "validation_tiles": int(len(va_idx)), "val_file": args.val_file,
        "epochs": args.epochs, "tiles_per_epoch": per_epoch, "batch": args.batch, "lr": args.lr, "pos_weight": args.pos_weight,
        "channels": channels, "crop": CROP, "sigma": SIGMA, "nms_px": NMS_PX, "match_px": MATCH_PX,
        "input_mean_bgr": mean.tolist(), "input_std_bgr": std.tolist(),
        "validation": table, "history": history, "check_tiles": [int(i) for i in check_idx], "tiles_file": args.tiles,
        "minutes": round((time.perf_counter() - t0) / 60, 1),
    }, indent=2))
    print(f"\nexported {model_path.name} (dynamic size, {(time.perf_counter() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
