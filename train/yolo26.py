"""Train YOLO26 on the tiled parasite dataset, tracked in ClearML.

Trains on the same jpg tiles Faster R-CNN sees, via the data.yaml written by
eval/export_detection.py.

AUGMENTATION IS TURNED DOWN ON PURPOSE

Ultralytics' defaults are tuned for COCO, where objects are large and scale
augmentation is free. Here a parasite is about 10 px:

  mosaic  stitches four images into one and resizes to imgsz, roughly halving
          every object. A 10 px parasite becomes 5 px. Off.
  scale   defaults to +/- 50%, which at the low end shrinks a parasite below
          what the smallest stride can resolve. Cut right down.
  hsv_*   the tiles are already stain-normalised by photometry.normalise, so
          strong colour jitter deliberately undoes stage 2 of the pipeline.
          Left small rather than zero, since real stain varies between clinics.

Flips are kept: a smear has no canonical orientation.

Nothing here picks a checkpoint or a threshold. Both are chosen afterwards on
VAL patients with eval/detector_eval.py, then the winner is scored once on the
held-out 50.
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def check_writable(directory: Path, label: str) -> None:
    """Prove a directory can hold a checkpoint, before hours are spent filling it.

    Google Drive on Colab is a FUSE mount and its failure mode is not an
    exception: a write can appear to succeed and leave nothing behind, so the
    run finishes with an empty weights directory and no error anywhere. This
    writes real bytes, flushes them to the device, reopens the file and
    compares, which is the only check that catches that.
    """
    probe = directory / ".autoscope_write_probe"
    payload = os.urandom(1 << 20)          # 1 MB: enough to defeat a lazy mount
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with open(probe, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        back = probe.read_bytes()
    except Exception as exc:
        raise SystemExit(f"{label} {directory} is NOT writable: "
                         f"{type(exc).__name__}: {exc}")
    finally:
        try:
            probe.unlink()
        except Exception:
            pass
    if back != payload:
        raise SystemExit(
            f"{label} {directory} accepted a 1 MB write and read back something "
            "different. Training into it would produce no usable checkpoint.")
    print(f"{label:<10} {directory}  writable, 1 MB probe verified")


def mirror_checkpoints(src_run: Path, dest: Path) -> None:
    """Copy this run's weights to `dest`. Never raises: a failed copy must not
    kill a training run that is otherwise fine."""
    try:
        weights = src_run / "weights"
        if not weights.is_dir():
            return
        target = dest / src_run.name / "weights"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("last.pt", "best.pt"):
            f = weights / name
            if f.exists():
                shutil.copy2(f, target / name)
        for extra in ("results.csv", "args.yaml"):
            f = src_run / extra
            if f.exists():
                shutil.copy2(f, dest / src_run.name / extra)
    except Exception as exc:
        print(f"  mirror to {dest} failed ({type(exc).__name__}: {exc}); "
              "training continues")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/derived/detect")
    ap.add_argument("--model", default="yolo26s.pt",
                    help="ultralytics weight name. Check `yolo settings` for "
                         "what this build actually ships before trusting it.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="must equal the export tile size, or the parasites "
                         "get rescaled and the tiling was pointless")
    ap.add_argument("--workers", type=int, default=0,
                    help="0 means one per CPU core, capped at 8. Colab VMs have "
                         "2 cores, and asking for 8 workers there makes them "
                         "fight over the same cores instead of feeding the GPU.")
    ap.add_argument("--cache", default=None, choices=["ram", "disk"],
                    help="cache decoded tiles. 'disk' writes .npy beside each "
                         "tile and skips JPEG decode every epoch; 'ram' needs "
                         "about 17 GB for 13.5k tiles at 640 px, so it is "
                         "usually the wrong choice here.")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--project", default="AutoScope/parasites")
    ap.add_argument("--task-name", default="yolo26")
    ap.add_argument("--out", default=None,
                    help="run directory. Keep this on LOCAL disk: ultralytics "
                         "rewrites the same files every epoch and Drive's FUSE "
                         "mount handles that badly.")
    ap.add_argument("--mirror", default=None,
                    help="copy weights here after every epoch. This is what "
                         "Drive is for on Colab: the run stays local and fast, "
                         "and a session ending costs one epoch, not the run.")
    ap.add_argument("--resume", action="store_true",
                    help="continue from weights/last.pt in the run directory")
    ap.add_argument("--no-clearml", action="store_true")
    args = ap.parse_args()

    root = Path(args.data)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    workers = args.workers or min(os.cpu_count() or 2, 8)
    if args.workers == 0:
        print(f"workers: {workers} (one per core, {os.cpu_count()} available)")

    manifest = json.loads((root / "manifest.json").read_text())
    print(f"split source: {manifest['split_source']}")
    print(f"train patients {manifest['train_patients']}, val {manifest['val_patients']}, "
          f"held out and untouched {manifest['holdout_patients_untouched']}")

    if args.imgsz != manifest["tile_px"]:
        raise SystemExit(
            f"--imgsz {args.imgsz} does not match the exported tile size "
            f"{manifest['tile_px']}. Ultralytics would resize every tile and "
            "shrink the parasites, which is the one thing tiling exists to "
            "prevent.")

    task = None
    if not args.no_clearml:
        from clearml import Task
        task = Task.init(project_name=args.project, task_name=args.task_name,
                         output_uri=True)
        task.connect(vars(args))
        task.connect(manifest, name="dataset")

    project = (Path(args.out) if args.out
               else Path(__file__).resolve().parents[1] / "models" / "yolo26")
    run_dir = project / args.task_name
    last = run_dir / "weights" / "last.pt"
    if args.resume and not last.exists():
        raise SystemExit(f"--resume was asked for but {last} does not exist. "
                         "Drop --resume to start from scratch.")

    if (run_dir / "weights" / "best.pt").exists() and not args.resume:
        # exist_ok=True below means ultralytics writes into this directory. If
        # the new run dies before its first improvement, the OLD best.pt
        # survives and is indistinguishable from a fresh one -- so the
        # evaluation would silently score the previous model, quite possibly
        # one trained on the smoke set.
        print(f"WARNING: {run_dir / 'weights' / 'best.pt'} exists and will be "
              f"overwritten.\n"
              f"         If this run fails early that stale file remains, and "
              f"the evaluation\n"
              f"         would score the PREVIOUS model. Delete {run_dir} to be "
              f"certain.\n")

    # Prove both destinations before training, not after. The previous run
    # produced an empty directory after hours of work and reported nothing.
    check_writable(run_dir, "run dir")
    mirror = Path(args.mirror) if args.mirror else None
    if mirror:
        check_writable(mirror, "mirror")

    from ultralytics import YOLO

    def after_epoch(trainer):
        """Mirror, then assert the run directory actually gained a checkpoint."""
        if mirror:
            mirror_checkpoints(run_dir, mirror)
        last_pt = run_dir / "weights" / "last.pt"
        if int(getattr(trainer, "epoch", 0)) == 0 and not last_pt.exists():
            raise SystemExit(
                f"after the first epoch there is still no {last_pt}.\n"
                "Ultralytics is not writing where we asked, so the rest of this "
                "run would produce nothing. Stopping now rather than in an hour.")

    if args.resume:
        # Ultralytics resumes from the checkpoint itself: it restores the
        # optimiser state, the epoch counter and the original arguments, so a
        # session that died at epoch 40 of 150 continues at 41 rather than
        # starting over. Passing the other arguments again would be ignored at
        # best and conflict at worst, so they are deliberately omitted here.
        print(f"resuming from {last}")
        model = YOLO(str(last))
        model.add_callback("on_fit_epoch_end", after_epoch)
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.add_callback("on_fit_epoch_end", after_epoch)
        model.train(
            data=str(root / "data.yaml"),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=workers,
            cache=args.cache or False,
            patience=args.patience,
            project=str(project),
            name=args.task_name,
            exist_ok=True,
            # See the module docstring: these are the small-object settings.
            mosaic=0.0,
            scale=0.10,
            translate=0.10,
            hsv_h=0.010,
            hsv_s=0.30,
            hsv_v=0.20,
            fliplr=0.5,
            flipud=0.5,
            degrees=180.0,      # a smear has no up
            erasing=0.0,
        )

    out = run_dir
    # Stamp the run with what it trained on, so a checkpoint is self-describing
    # and cannot be mistaken for one trained on the smoke set.
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset.json").write_text(json.dumps({
        "data": str(root),
        "tiles_train": manifest.get("counts", {}).get("tiles_train"),
        "tiles_val": manifest.get("counts", {}).get("tiles_val"),
        "images_per_patient": manifest.get("images_per_patient"),
        "min_fov": manifest.get("min_fov"),
        "export_git_revision": manifest.get("git_revision"),
    }, indent=2))

    # Say plainly whether a usable checkpoint exists. A run that produced
    # nothing must not end by printing a cheerful path to a file that is not
    # there -- which is exactly how the last two runs were lost.
    best = out / "weights" / "best.pt"
    if mirror:
        mirror_checkpoints(run_dir, mirror)
        try:
            shutil.copy2(out / "dataset.json", mirror / run_dir.name / "dataset.json")
        except Exception:
            pass
    if not best.exists():
        raise SystemExit(
            f"training finished but {best} does not exist.\n"
            "No usable model was produced. The real error is in the "
            "ultralytics output above.")
    print(f"\nweights: {best}  ({best.stat().st_size / 1e6:.1f} MB)")
    if mirror:
        print(f"mirrored: {mirror / run_dir.name / 'weights' / 'best.pt'}")
    print(f"trained on {manifest.get('counts', {}).get('tiles_train')} train tiles "
          f"from {root}")
    print("Now pick the checkpoint and the threshold on VAL, never on holdout:")
    print(f"  python eval/detector_eval.py --backend yolo "
          f"--weights {out / 'weights' / 'best.pt'} --split val --sweep")


if __name__ == "__main__":
    main()
