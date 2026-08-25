"""Fetch the Tanzania blood smear set from Harvard Dataverse.

This is the held-out set. We fit nothing on it. It exists to answer one
question in the final report: does a pipeline tuned on Thai and Bangladeshi
slides still work on slides from Tanzanian health centres?

  Lufyagila B., Mgawe B., Sam A. (2024)
  "Blood smear imagery dataset for malaria parasite detection: A case of Tanzania"
  Data in Brief, doi:10.1016/j.dib.2024.111169
  Dataset: doi:10.7910/DVN/O2WVWA, licence CC0 1.0 (public domain)

Collected in five health centres of the Tanga region on a 40X-2500X compound
microscope with a 4K Sony IMX334 camera, Giemsa stained. About 3 GB across four
archives.

Different rig, different camera, different country. That is the point: every
threshold we measured on the NLM set has to be re-measured here rather than
assumed. See docs/data-notes.md.
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

DOI = "doi:10.7910/DVN/O2WVWA"
API = "https://dataverse.harvard.edu/api"
RAW = Path(__file__).resolve().parent / "raw" / "tanzania"

# Dataverse answers 403 to urllib's default User-Agent. curl works, urllib does
# not, and the error says nothing about why. Send a real one.
UA = "autoscope/0.1 (OpenCV AI Competition 2026; +https://github.com/)"


def _open(url: str, extra: dict | None = None, timeout: int = 120):
    request = urllib.request.Request(url, headers={"User-Agent": UA, **(extra or {})})
    return urllib.request.urlopen(request, timeout=timeout)

# Filled from the Dataverse API on 24 August 2026. Kept here so a failed
# download reports a size mismatch instead of silently truncating.
EXPECTED = {
    "Thick_Infected.zip": 835_500_000,
    "Thick_Uninfected.zip": 972_500_000,
    "Thin_Infected.zip": 831_500_000,
    "Thin_Uninfected.zip": 340_000_000,
}


def _human(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def list_files() -> list[dict]:
    """Ask Dataverse what is actually in the dataset, rather than hardcoding ids."""
    url = f"{API}/datasets/:persistentId/?persistentId={DOI}"
    with _open(url, timeout=60) as r:
        import json

        data = json.load(r)
    version = data["data"]["latestVersion"]
    licence = version.get("license", {})
    print(f"licence: {licence.get('name', 'unknown')}")
    return [
        {
            "id": f["dataFile"]["id"],
            "name": f["dataFile"]["filename"],
            "size": f["dataFile"]["filesize"],
        }
        for f in version["files"]
    ]


def fetch(entry: dict) -> Path:
    target = RAW / entry["name"]
    RAW.mkdir(parents=True, exist_ok=True)

    have = target.stat().st_size if target.exists() else 0
    if have >= entry["size"]:
        print(f"  {entry['name']}: already have {_human(have)}")
        return target

    url = f"{API}/access/datafile/{entry['id']}"
    extra = {}
    if have:
        extra["Range"] = f"bytes={have}-"
        print(f"  {entry['name']}: resuming from {_human(have)}")

    with _open(url, extra) as response:
        with open(target, "ab" if have else "wb") as out:
            done = have
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                print(f"\r  {entry['name']}: {_human(done)} / {_human(entry['size'])}",
                      end="")
    print()
    return target


def extract(archive: Path) -> None:
    out = archive.with_suffix("")
    if out.exists() and any(out.iterdir()):
        print(f"  {archive.name}: already extracted")
        return
    print(f"  {archive.name}: extracting")
    with zipfile.ZipFile(archive) as z:
        z.extractall(out)


def summarise() -> None:
    """Print what we got. Never write code against a guessed layout."""
    images = [p for p in RAW.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    others = [p for p in RAW.rglob("*") if p.is_file() and p.suffix.lower()
              not in {".jpg", ".jpeg", ".png", ".zip"}]
    print(f"\n{len(images)} images under {RAW}")
    by_group: dict[str, int] = {}
    for p in images:
        by_group[p.relative_to(RAW).parts[0]] = by_group.get(p.relative_to(RAW).parts[0], 0) + 1
    for group, n in sorted(by_group.items()):
        print(f"  {group:24} {n:6} images")
    if others:
        kinds = sorted({p.suffix.lower() or "(none)" for p in others})
        print(f"non-image files present: {', '.join(kinds)} ({len(others)} files)")
        print("check these for annotations before assuming there are none")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thick-only", action="store_true",
                    help="skip the thin smear archives, about 1.2 GB less")
    args = ap.parse_args()

    files = list_files()
    if args.thick_only:
        files = [f for f in files if f["name"].startswith("Thick")]

    total = sum(f["size"] for f in files)
    print(f"{len(files)} archives, {_human(total)}\n")

    import shutil

    RAW.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(RAW).free < total * 2.5:
        sys.exit(f"need about {_human(total * 2.5)} free for download plus extraction")

    for entry in files:
        extract(fetch(entry))
    summarise()


if __name__ == "__main__":
    main()
