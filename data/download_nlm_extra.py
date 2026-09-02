"""Fetch the two other NLM thick smear sets: uninfected patients, and P. vivax.

    python data/download_nlm_extra.py --set uninfected    # 2.45 GB
    python data/download_nlm_extra.py --set vivax         # 4.98 GB

Why. Every patient in the 150-patient set is infected, so nothing so far has
measured specificity: what the pipeline says about a clean slide. The
uninfected set answers that. The P. vivax set is the "other species" failure
case the proposal promised a number for.

Both are frozen test material. Nothing gets fitted on them, and they are run
once, at the end, with the pipeline fixed. Same rule as Tanzania.

Resumable, writes to data/raw/ which is gitignored, prints the real layout
after extracting. Read that output before writing code against the files.

Source: US National Library of Medicine, Lister Hill National Center for
Biomedical Communications, https://lhncbc.nlm.nih.gov/LHC-downloads/dataset.html
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://data.lhncbc.nlm.nih.gov/public/Malaria/"
SETS = {
    "uninfected": ("NIH-NLM-ThickBloodSmearsU", 2_452_470_961),
    "vivax": ("NIH-NLM-ThickBloodSmearsPV", 4_979_641_074),
}
RAW = Path(__file__).resolve().parent / "raw"
# The NLM server answers 403 to some default clients. A named agent works.
UA = "autoscope/0.1 (OpenCV AI Competition 2026; +https://github.com/nguefackuriel/autoscope-opencv)"


def _human(n: int) -> str:
    return f"{n / 1_000_000_000:.2f} GB"


def download(name: str, expected: int) -> Path:
    archive = RAW / f"{name}.zip"
    url = f"{BASE}{name}/{name}.zip"
    RAW.mkdir(parents=True, exist_ok=True)
    have = archive.stat().st_size if archive.exists() else 0

    if have == expected:
        print(f"already downloaded: {archive}")
        return archive
    if have > expected:
        print("local file is larger than expected, starting over")
        archive.unlink()
        have = 0

    request = urllib.request.Request(url, headers={"User-Agent": UA})
    if have:
        print(f"resuming from {_human(have)}")
        request.add_header("Range", f"bytes={have}-")

    mode = "ab" if have else "wb"
    with urllib.request.urlopen(request, timeout=120) as response, open(archive, mode) as out:
        done = have
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            pct = 100 * done / expected
            print(f"\r{_human(done)} / {_human(expected)}  {pct:5.1f}%", end="", flush=True)
    print()

    size = archive.stat().st_size
    if size != expected:
        raise SystemExit(f"size mismatch: got {size}, expected {expected}. Run again to resume.")
    return archive


def extract(archive: Path, name: str) -> Path:
    target = RAW / name
    if target.exists() and any(target.iterdir()):
        print(f"already extracted: {target}")
        return target
    print("extracting, this takes a few minutes")
    with zipfile.ZipFile(archive) as z:
        # Archives from this server sometimes carry their own top folder and
        # sometimes not. Extract into a folder named after the set either way.
        names = z.namelist()
        top = {n.split("/")[0] for n in names if "/" in n}
        if top == {name}:
            z.extractall(RAW)
        else:
            z.extractall(target)
    return target


def summarise(root: Path) -> None:
    """Print what we actually got, so nobody codes against a guessed layout."""
    print(f"\nlayout under {root}:")
    for path in sorted(root.rglob("*"))[:25]:
        marker = "/" if path.is_dir() else ""
        print(f"  {path.relative_to(root)}{marker}")
    images = [p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    others = [p for p in root.rglob("*") if p.is_file() and p not in set(images)]
    patients = {p.parent for p in images}
    print(f"\n{len(images)} images in {len(patients)} folders, {len(others)} other files")
    if others:
        kinds = sorted({p.suffix.lower() or "(none)" for p in others})
        print(f"non-image extensions present: {', '.join(kinds)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", choices=sorted(SETS), required=True)
    args = ap.parse_args()
    name, expected = SETS[args.set]

    free = shutil.disk_usage(RAW.parent).free
    if free < 3 * expected:
        sys.exit(f"need about {_human(3 * expected)} free for download plus extraction, have {_human(free)}")
    summarise(extract(download(name, expected), name))
