"""Fetch the NLM thick blood smear set.

3.9 GB, so the download resumes if it drops. Run it once and leave it; it
writes to data/raw/ which is gitignored.

Source: US National Library of Medicine, Lister Hill National Center for
Biomedical Communications. De-identified, publicly released, annotated by
expert readers at the Mahidol-Oxford Tropical Medicine Research Unit.
"""

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://data.lhncbc.nlm.nih.gov/public/Malaria/"
    "Thick_Smears_150/ThickBloodSmears_150.zip"
)
EXPECTED_BYTES = 3_923_792_228
RAW = Path(__file__).resolve().parent / "raw"
ARCHIVE = RAW / "ThickBloodSmears_150.zip"


def _human(n: int) -> str:
    return f"{n / 1_000_000_000:.2f} GB"


def download() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    have = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0

    if have == EXPECTED_BYTES:
        print(f"already downloaded: {ARCHIVE}")
        return ARCHIVE
    if have > EXPECTED_BYTES:
        print("local file is larger than expected, starting over")
        ARCHIVE.unlink()
        have = 0

    request = urllib.request.Request(URL)
    if have:
        print(f"resuming from {_human(have)}")
        request.add_header("Range", f"bytes={have}-")

    mode = "ab" if have else "wb"
    with urllib.request.urlopen(request) as response, open(ARCHIVE, mode) as out:
        done = have
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            pct = 100 * done / EXPECTED_BYTES
            print(f"\r{_human(done)} / {_human(EXPECTED_BYTES)}  {pct:5.1f}%", end="")
    print()

    size = ARCHIVE.stat().st_size
    if size != EXPECTED_BYTES:
        raise SystemExit(
            f"size mismatch: got {size}, expected {EXPECTED_BYTES}. Run again to resume."
        )
    return ARCHIVE


def extract(archive: Path) -> Path:
    target = RAW / "ThickBloodSmears_150"
    if target.exists() and any(target.iterdir()):
        print(f"already extracted: {target}")
        return target
    print("extracting, this takes a few minutes")
    with zipfile.ZipFile(archive) as z:
        z.extractall(RAW)
    return target


def summarise(root: Path) -> None:
    """Print what we actually got, so nobody codes against a guessed layout."""
    print(f"\nlayout under {root}:")
    for path in sorted(root.rglob("*"))[:25]:
        marker = "/" if path.is_dir() else ""
        print(f"  {path.relative_to(root)}{marker}")
    images = [p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    others = [p for p in root.rglob("*") if p.is_file() and p not in set(images)]
    print(f"\n{len(images)} images, {len(others)} other files")
    if others:
        kinds = sorted({p.suffix.lower() or "(none)" for p in others})
        print(f"non-image extensions present: {', '.join(kinds)}")


if __name__ == "__main__":
    free = shutil.disk_usage(RAW.parent).free
    if free < 12_000_000_000:
        sys.exit(f"need about 12 GB free for download plus extraction, have {_human(free)}")
    summarise(extract(download()))
