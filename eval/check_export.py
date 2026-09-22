"""Did the export finish, and what settings produced it?

A long export killed by a closing terminal leaves a directory that looks
plausible: thousands of tiles, sensible filenames, nothing obviously wrong. It
is not usable. Patients are processed in alphabetical order, so a killed run
holds an arbitrary prefix of the fit set, and training on it means training on
whichever patients happened to come first.

manifest.json is written after the last tile, so its presence is the signal
that the run reached the end.

    python eval/check_export.py                     # data/derived/detect
    python eval/check_export.py --out somewhere/else
"""

import argparse
import json

# --- repo paths: run from anywhere, installed or not -------------------------
import sys as _sys, pathlib as _pl
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d)
                 for _d in ("src", "eval", "train")]
# ----------------------------------------------------------------------------

from pathlib import Path

EXPECTED_FIT_PATIENTS = 100


def patients_in(tiles) -> set[str]:
    """Patient ids recovered from tile filenames, `{patient}_{image}_{x}_{y}.jpg`."""
    out = set()
    for t in tiles:
        parts = t.stem.split("_")
        if len(parts) >= 4:
            out.add("_".join(parts[:-3]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/derived/detect")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out

    if not out.exists():
        raise SystemExit(f"{out} does not exist: the export never wrote anything.")

    tiles = list(out.rglob("*.jpg"))
    seen = patients_in(tiles)
    print(f"directory : {out}")
    print(f"tiles     : {len(tiles)}")
    print(f"patients  : {len(seen)} distinct")

    manifest = out / "manifest.json"
    if not manifest.exists():
        last = max(seen) if seen else "(nothing)"
        print("\nSTATUS: INCOMPLETE -- no manifest.json.")
        print("  The manifest is written only after the last tile, so this run")
        print(f"  was killed. It stopped somewhere after '{last}'.")
        print("  Patients are processed alphabetically, so this is an arbitrary")
        print("  prefix of the fit set. Re-run the export; it clears this first.")
        print("\n  To survive a closing terminal next time:")
        print("    nohup python eval/export_detection.py --images-per-patient 10 "
              "> export.log 2>&1 &")
        print("    tail -f export.log")
        raise SystemExit(1)

    m = json.loads(manifest.read_text())
    fit = (m.get("train_patients") or 0) + (m.get("val_patients") or 0)
    print("\nSTATUS: COMPLETE")
    print(f"  train / val patients : {m.get('train_patients')} / {m.get('val_patients')}")
    print(f"  images per patient   : {m.get('images_per_patient')}")
    print(f"  images processed     : {m.get('counts', {}).get('images')}")
    print(f"  tiles train / val    : {m.get('counts', {}).get('tiles_train')} / "
          f"{m.get('counts', {}).get('tiles_val')}")
    print(f"  boxes derived        : {m.get('boxes_derived')}")
    print(f"  boxes covered        : {m.get('boxes_covered')} "
          f"({(m.get('box_tile_loss_rate') or 0) * 100:.1f}% never covered)")

    print("\n  provenance")
    print(f"    min_fov      : {m.get('min_fov', '(absent)')}")
    print(f"    git revision : {m.get('git_revision', '(absent)')}")
    print(f"    seed         : {m.get('seed')}")

    problems = []
    if fit != EXPECTED_FIT_PATIENTS:
        problems.append(f"{fit} fit patients, expected {EXPECTED_FIT_PATIENTS}")
    if m.get("boxes_invented", 0):
        problems.append(f"{m['boxes_invented']} boxes were invented; must be 0")
    if m.get("min_fov") is None:
        problems.append(
            "min_fov is not recorded, so this was exported before the tiling "
            "fix and very likely used 0.9, which covers only 65.8% of parasites")
    elif m["min_fov"] > 0.5:
        problems.append(
            f"min_fov={m['min_fov']} discards tiles at the disc rim; 0.9 covers "
            "only 65.8% of parasites and caps inference recall at 0.652")
    if m.get("holdout_patients_untouched") != 50:
        problems.append(
            f"holdout_patients_untouched={m.get('holdout_patients_untouched')}, "
            "expected 50")

    if problems:
        print("\n  PROBLEMS")
        for p in problems:
            print(f"    - {p}")
        raise SystemExit(1)
    print("\n  no problems found: this export is usable")


if __name__ == "__main__":
    main()
