"""Train Faster R-CNN on the tiled parasite dataset, tracked in ClearML.

Reads the COCO json written by eval/export_detection.py, which describes the
same jpg tiles the YOLO run trains on, so a difference between the two models
is the model and not the data.

Checkpoints are written every epoch and none is anointed here. The model and
the confidence threshold are both chosen afterwards with

    python eval/detector_eval.py --backend frcnn --weights <ckpt> \
        --split val --sweep

on fit patients held back from training, and only then is the winner scored
once against the held-out 50. Selecting on the held-out set is the one thing
that would invalidate the comparison with the patch network's 0.762.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import shutil
import torch
from torch.utils.data import DataLoader, Dataset

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

from frcnn_model import build_model, describe


class CocoTiles(Dataset):
    """Minimal COCO reader.

    Deliberately not torchvision's CocoDetection, which drags in pycocotools.
    Nothing here needs COCO's mAP: scoring is centre matching in
    eval/detector_eval.py, so the json is only ever a list of boxes.
    """

    def __init__(self, root: Path, split: str, augment: bool = False):
        self.images_dir = root / "images" / split
        data = json.loads((root / "annotations" / f"instances_{split}.json").read_text())
        self.images = data["images"]
        self.augment = augment
        by_image = {}
        for a in data["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a["bbox"])
        self.boxes = by_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        info = self.images[i]
        img = cv.imread(str(self.images_dir / info["file_name"]))
        if img is None:
            raise FileNotFoundError(self.images_dir / info["file_name"])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        raw = self.boxes.get(info["id"], [])
        boxes = [[x, y, x + w, y + h] for x, y, w, h in raw]

        if self.augment:
            # Flips only. Anything that rescales would change the apparent size
            # of a 10 px parasite, which is the one property the tiling exists
            # to preserve. Colour jitter is also skipped: the tiles are already
            # stain-normalised by photometry.normalise, so jittering colour
            # would be undoing stage 2 of the pipeline on purpose.
            h, w = img.shape[:2]
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
                boxes = [[w - x2, y1, w - x1, y2] for x1, y1, x2, y2 in boxes]
            if np.random.rand() < 0.5:
                img = img[::-1]
                boxes = [[x1, h - y2, x2, h - y1] for x1, y1, x2, y2 in boxes]

        tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        if boxes:
            b = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
        else:
            b = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        return tensor, {"boxes": b, "labels": labels, "image_id": torch.tensor([info["id"]])}


def collate(batch):
    return tuple(zip(*batch))


def run_epoch(model, loader, optimiser, device, scaler, epoch, log_every=50):
    model.train()
    running, seen = 0.0, 0
    start = time.time()
    for i, (images, targets) in enumerate(loader):
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            losses = model(images, targets)
            loss = sum(losses.values())

        if not math.isfinite(loss.item()):
            raise SystemExit(f"loss became {loss.item()} at step {i}: "
                             "lower the learning rate rather than pressing on")

        optimiser.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            optimiser.step()

        running += loss.item() * len(images)
        seen += len(images)
        if i % log_every == 0:
            rate = seen / max(time.time() - start, 1e-6)
            print(f"  epoch {epoch} step {i}/{len(loader)} "
                  f"loss {running / max(seen, 1):.4f} ({rate:.1f} img/s)")
    return running / max(seen, 1)


@torch.no_grad()
def val_loss(model, loader, device):
    """Validation loss for monitoring only.

    Faster R-CNN only returns losses in train mode, so the module stays in
    train() while gradients stay off. This number watches for divergence; it
    does not choose the checkpoint. detector_eval.py does that, on centres.
    """
    model.train()
    running, seen = 0.0, 0
    for images, targets in loader:
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses = model(images, targets)
        running += sum(losses.values()).item() * len(images)
        seen += len(images)
    return running / max(seen, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/derived/detect")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="models/frcnn")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--mirror", default=None,
                    help="copy each checkpoint here after the epoch that wrote "
                         "it. Use mounted Drive on Colab; keep --out local.")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the highest epoch*.pt in --out. Point "
                         "--out at mounted Drive on Colab so a session ending "
                         "does not cost the run.")
    ap.add_argument("--no-clearml", action="store_true")
    ap.add_argument("--project", default="AutoScope/parasites")
    ap.add_argument("--task-name", default="faster-rcnn")
    args = ap.parse_args()

    root = Path(args.data)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    manifest = json.loads((root / "manifest.json").read_text())
    print(f"split source: {manifest['split_source']}")
    print(f"train patients {manifest['train_patients']}, val {manifest['val_patients']}, "
          f"held out and untouched {manifest['holdout_patients_untouched']}")

    task = None
    if not args.no_clearml:
        from clearml import Task
        task = Task.init(project_name=args.project, task_name=args.task_name,
                         output_uri=True)
        task.connect(vars(args))
        task.connect(manifest, name="dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no CUDA device. Faster R-CNN on CPU is not a viable "
              "training run, only a smoke test.")

    train_ds = CocoTiles(root, "train", augment=True)
    val_ds = CocoTiles(root, "val", augment=False)
    print(f"{len(train_ds)} train tiles, {len(val_ds)} val tiles")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, collate_fn=collate,
                          pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate)

    model = build_model(num_classes=2, tile=manifest["tile_px"]).to(device)
    print(describe(model))

    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    scaler = None if args.no_amp or device.type != "cuda" else torch.amp.GradScaler()

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    # Stale checkpoints from a previous run are a silent trap. Checkpoints are
    # named epoch001.pt upward and the eval picks sorted(...)[-1], so a shorter
    # run leaves the longer run's higher-numbered files behind and the eval
    # selects one of THOSE -- a model trained on different data.
    start_epoch = 1
    if args.resume:
        found = sorted(out.glob("epoch*.pt")) if out.exists() else []
        if not found:
            raise SystemExit(
                f"--resume was asked for but {out} holds no epoch*.pt. "
                "Drop --resume to start from scratch.")
        ckpt = torch.load(found[-1], map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimiser" in ckpt:
            optimiser.load_state_dict(ckpt["optimiser"])
        if "schedule" in ckpt:
            schedule.load_state_dict(ckpt["schedule"])
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"resuming from {found[-1].name}, continuing at epoch {start_epoch}")
        if start_epoch > args.epochs:
            raise SystemExit(
                f"{found[-1].name} already reached epoch {ckpt['epoch']} and "
                f"--epochs is {args.epochs}. Nothing to do; raise --epochs.")
    elif out.exists():
        stale = sorted(out.glob("epoch*.pt"))
        if stale:
            print(f"removing {len(stale)} checkpoints from a previous run in {out}")
            for f in stale:
                f.unlink()
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from yolo26 import check_writable        # shared probe, same failure mode
    check_writable(out, "run dir")
    mirror = Path(args.mirror) if args.mirror else None
    if mirror:
        check_writable(mirror, "mirror")

    # Stamp the run with what it trained on, so a checkpoint is self-describing
    # and cannot be confused with one trained on the smoke set.
    (out / "dataset.json").write_text(json.dumps({
        "data": str(root),
        "tiles_train": manifest.get("counts", {}).get("tiles_train"),
        "tiles_val": manifest.get("counts", {}).get("tiles_val"),
        "images_per_patient": manifest.get("images_per_patient"),
        "min_fov": manifest.get("min_fov"),
        "export_git_revision": manifest.get("git_revision"),
    }, indent=2))

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        tr = run_epoch(model, train_dl, optimiser, device, scaler, epoch)
        va = val_loss(model, val_dl, device)
        schedule.step()
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        print(f"epoch {epoch}: train {tr:.4f}  val {va:.4f}")
        if task is not None:
            task.get_logger().report_scalar("loss", "train", tr, epoch)
            task.get_logger().report_scalar("loss", "val", va, epoch)
        # Optimiser and schedule go in too: resuming with a fresh optimiser
        # loses the momentum buffers and restarts the cosine schedule, which
        # is a different training run wearing the same checkpoint's name.
        torch.save({"model": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "schedule": schedule.state_dict(),
                    "epoch": epoch, "args": vars(args)},
                   out / f"epoch{epoch:03d}.pt")
        if mirror:
            # Copy straight after writing, so a session that dies mid-epoch
            # costs that epoch and nothing earlier.
            try:
                mirror.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out / f"epoch{epoch:03d}.pt",
                             mirror / f"epoch{epoch:03d}.pt")
            except Exception as exc:
                print(f"  mirror failed ({type(exc).__name__}: {exc}); continuing")

    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\ncheckpoints in {out}")
    print("Now pick the epoch and the threshold on VAL, never on holdout:")
    print(f"  python eval/detector_eval.py --backend frcnn "
          f"--weights {out}/epoch{args.epochs:03d}.pt --split val --sweep")


if __name__ == "__main__":
    main()
