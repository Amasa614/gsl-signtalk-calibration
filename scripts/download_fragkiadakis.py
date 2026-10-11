"""Download the Fragkiadakis et al. Ghanaian Sign Language lexicon from Zenodo.

Fragkiadakis, Nyst & Nyarko (2021). Ghanaian Sign Language Lexicon.
https://doi.org/10.5281/zenodo.4533753  (~1.2 GB zip, CC BY 4.0)

Used only as unlabeled motion for SSL pretraining (RQ1). Not in the
signer-independent SignTalk-Gh fine-tune split.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PATHS
from src.corpora import FRAGKIADAKIS_URL, FRAGKIADAKIS_ZIP_NAME, load_fragkiadakis_index


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    print(f"  -> {dest}")
    with urlopen(url) as resp, dest.open("wb") as out:
        total = resp.headers.get("Content-Length")
        total_n = int(total) if total and total.isdigit() else None
        copied = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            copied += len(chunk)
            if total_n:
                pct = 100.0 * copied / total_n
                print(f"\r  {copied / 1e9:.2f} / {total_n / 1e9:.2f} GB ({pct:5.1f}%)", end="", flush=True)
            else:
                print(f"\r  {copied / 1e9:.2f} GB", end="", flush=True)
    print()


def _unzip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"unzipping -> {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-download", action="store_true", help="only unzip an existing zip")
    args = p.parse_args()

    PATHS.external.mkdir(parents=True, exist_ok=True)
    existing = PATHS.repo_root / "GSL_openpose_data"
    if existing.exists() and any(existing.glob("*/*_keypoints.json")):
        index = load_fragkiadakis_index(existing)
        print(f"using existing OpenPose dump: {existing}")
        print(f"fragkiadakis clips: {len(index)}")
        if not index.empty:
            print(index.head(5).to_string(index=False))
        print("Next:")
        print("  python scripts/01_extract_landmarks.py --source fragkiadakis")
        print("  python scripts/02_build_windows.py --test-signers 5 --val-signers 4")
        print("  python scripts/03_pretrain_ssl.py")
        return
    zip_path = PATHS.external / FRAGKIADAKIS_ZIP_NAME
    if not args.skip_download:
        if zip_path.exists() and zip_path.stat().st_size > 1_000_000:
            print(f"zip already present: {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")
        else:
            _download(FRAGKIADAKIS_URL, zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    marker = PATHS.fragkiadakis / ".extracted"
    if marker.exists() and any(PATHS.fragkiadakis.rglob("*.mp4")):
        print(f"already unzipped: {PATHS.fragkiadakis}")
    else:
        _unzip(zip_path, PATHS.fragkiadakis)
        marker.write_text("ok", encoding="utf-8")

    index = load_fragkiadakis_index()
    print(f"fragkiadakis videos: {len(index)}")
    if not index.empty:
        print(index.head(5).to_string(index=False))
    print("Next (after SignTalk extraction is idle):")
    print("  python scripts/01_extract_landmarks.py --source fragkiadakis --cpu --workers 10")


if __name__ == "__main__":
    main()
