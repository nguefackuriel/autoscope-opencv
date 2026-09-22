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
from pathlib import Path


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
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--project", default="AutoScope/parasites")
    ap.add_argument("--task-name", default="yolo26")
    ap.add_argument("--no-clearml", action="store_true")
    args = ap.parse_args()

    root = Path(args.data)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
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

    run_dir = Path(__file__).resolve().parents[1] / "models" / "yolo26" / args.task_name
    if (run_dir / "weights" / "best.pt").exists():
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

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(root / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        project=str(Path(__file__).resolve().parents[1] / "models" / "yolo26"),
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

    print(f"\nweights in {out / 'weights'}")
    print(f"trained on {manifest.get('counts', {}).get('tiles_train')} train tiles "
          f"from {root}")
    print("Now pick the checkpoint and the threshold on VAL, never on holdout:")
    print(f"  python eval/detector_eval.py --backend yolo "
          f"--weights {out / 'weights' / 'best.pt'} --split val --sweep")


if __name__ == "__main__":
    main()
