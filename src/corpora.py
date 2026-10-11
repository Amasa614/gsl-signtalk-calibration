"""Extra unlabeled corpora for SSL pretraining (RQ1).

These clips are never used for signer-independent fine-tune/eval. Labels, if
any, are ignored. SignTalk-Gh remains the only labeled train/val/test source.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import PATHS
from src.metadata import iter_videos

FRAGKIADAKIS_ZENODO_ID = "4533753"
FRAGKIADAKIS_ZIP_NAME = "GSL_openpose_data.zip"
FRAGKIADAKIS_URL = (
    f"https://zenodo.org/api/records/{FRAGKIADAKIS_ZENODO_ID}/files/"
    f"{FRAGKIADAKIS_ZIP_NAME}/content"
)
FRAGKIADAKIS_DOI = "10.5281/zenodo.4533753"


def _video_id(source: str, path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root).with_suffix("")
        slug = str(rel).replace("\\", "/").replace("/", "__")
    except ValueError:
        slug = path.stem
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", slug)
    return f"{source}__{slug}"


def merge_labeled_and_ssl(labeled: pd.DataFrame) -> pd.DataFrame:
    """Concat SignTalk index with extra SSL corpora. SignTalk wins on video_id."""
    extra = load_ssl_index_csv()
    if extra.empty:
        extra = load_ssl_extra_index()
    if extra.empty:
        return labeled
    return pd.concat([labeled, extra], ignore_index=True).drop_duplicates(
        subset=["video_id"], keep="first"
    )


def load_fragkiadakis_index(root: Path | None = None) -> pd.DataFrame:
    """One row per isolated-sign clip in the Fragkiadakis / Leiden GSL lexicon.

    Recorded by Marco Nyarko (single signer). Prefer OpenPose JSON dumps when
    present (`GSL_openpose_data`); otherwise index mp4s for MediaPipe extract.
    Either way these clips are SSL-only and never enter the SignTalk label map.
    """
    from src.openpose import fragkiadakis_root, load_openpose_index

    root = fragkiadakis_root(root)
    if not root.exists():
        return pd.DataFrame()
    openpose = load_openpose_index(root)
    if not openpose.empty:
        return openpose
    videos = iter_videos(root)
    rows = []
    for path in videos:
        gloss = path.parent.name if path.parent != root else path.stem
        rows.append(
            {
                "video_id": _video_id("fragkiadakis", path, root),
                "path": str(path.resolve()),
                "signer_id": "nyarko",
                "sentence_id": gloss,
                "sentence_text": gloss.replace("_", " "),
                "category": "lexicon",
                "source": "fragkiadakis",
                "split": "ssl",
                "pose_source": "video",
            }
        )
    return pd.DataFrame(rows)


def load_afrisign_index(root: Path | None = None) -> pd.DataFrame:
    """Optional: drop Ghanaian Sign Language clips under data/external/afrisign/.

    AfriSign is not redistributed as a zip; the paper scrapes JW Bible videos.
    If you place GSL-only mp4s here they are treated as unlabeled SSL data.
    """
    root = root or PATHS.afrisign
    if not root.exists():
        return pd.DataFrame()
    videos = iter_videos(root)
    rows = []
    for path in videos:
        rows.append(
            {
                "video_id": _video_id("afrisign", path, root),
                "path": str(path.resolve()),
                "signer_id": "afrisign",
                "sentence_id": path.stem,
                "sentence_text": "",
                "category": "afrisign",
                "source": "afrisign",
                "split": "ssl",
            }
        )
    return pd.DataFrame(rows)


def load_ssl_extra_index() -> pd.DataFrame:
    parts = [load_fragkiadakis_index(), load_afrisign_index()]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_ssl_index_csv() -> pd.DataFrame:
    if PATHS.ssl_index.exists():
        return pd.read_csv(PATHS.ssl_index)
    return pd.DataFrame()
