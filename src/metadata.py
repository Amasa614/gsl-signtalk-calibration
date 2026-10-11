"""Discover SignTalk-Gh videos + metadata and resolve signer IDs.

Signer ID is required for every clip: RQ1/RQ2 evaluation is signer-independent.
The Kaggle Metadata.csv listing only advertises sentence id / text / category,
so this module tries several fallbacks and fails loudly if signer cannot be
resolved.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import PATHS

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".MP4"}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "sentence_id": (
        "sentence_id",
        "sentenceid",
        "sentence id",
        "sent_id",
        "id",
        "video_id",
        "videoid",
    ),
    "sentence_text": (
        "sentence",
        "sentence_text",
        "text",
        "english",
        "gloss",
        "translation",
    ),
    "category": (
        "category",
        "thematic_category",
        "thematic category",
        "hospital department",
        "department",
        "theme",
    ),
    "signer_id": (
        "signer_id",
        "signer",
        "signerid",
        "signer id",
        "participant",
        "participant_id",
    ),
    "filename": (
        "filename",
        "file",
        "file_name",
        "video",
        "video_path",
        "path",
        "folder name",
        "folder_name",
        "clip",
    ),
}

SIGNER_IN_PATH = re.compile(
    r"(?:^|[\\/_.\-])(?:signer|s)[_-]?(\d+)(?:$|[\\/_.\-])",
    re.IGNORECASE,
)
SIGNER_FOLDER = re.compile(r"signer[_-]?(\d+)", re.IGNORECASE)
# SignTalk-GH files are "{sentenceId}{letter}.mp4", e.g. 1A.mp4, 1001D.mp4.
# Metadata.xlsx has no signer column. Letter counts on this release match
# paper Table 3 (Ahene et al. 2026) almost exactly:
#   A=2017 ~ Signer 1 (2017), B=2000 ~ Signer 2 (2000), C=2012 ~ Signer 3 (2012),
#   E=2049 ~ Signer 4 (2050), D=1798 ~ Signer 5 (1800).
SIGNTALK_STEM = re.compile(r"^(\d+)([A-Za-z])$")
SIGNTALK_LETTER_TO_SIGNER = {
    "A": "1",
    "B": "2",
    "C": "3",
    "E": "4",
    "D": "5",
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def resolve_column(df: pd.DataFrame, key: str) -> str | None:
    aliases = {_norm(a) for a in COLUMN_ALIASES[key]}
    for col in df.columns:
        if _norm(str(col)) in aliases:
            return col
    return None


def find_metadata_csv(raw_root: Path | None = None) -> Path | None:
    if PATHS.metadata_path is not None and PATHS.metadata_path.exists():
        return PATHS.metadata_path
    root = raw_root or PATHS.raw_videos
    if not root.exists():
        return None
    files = [p for p in root.rglob("*") if p.suffix.lower() in {".csv", ".xlsx", ".xls"}]
    preferred = [p for p in files if "metadata" in p.name.lower()]
    if preferred:
        return sorted(preferred)[0]
    return sorted(files)[0] if files else None


def load_metadata_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def iter_videos(raw_root: Path | None = None) -> list[Path]:
    root = raw_root or PATHS.raw_videos
    if not root.exists():
        return []
    videos = [p for p in root.rglob("*") if p.is_file() and p.suffix in VIDEO_EXTS]
    return sorted(videos)


def _stem_key(path: Path) -> str:
    return path.stem.lower().strip()


def infer_signer_from_path(path: Path) -> str | None:
    m = SIGNTALK_STEM.fullmatch(path.stem)
    if m:
        letter = m.group(2).upper()
        if letter in SIGNTALK_LETTER_TO_SIGNER:
            return SIGNTALK_LETTER_TO_SIGNER[letter]
    text = str(path)
    m = SIGNER_IN_PATH.search(text.replace("\\", "/"))
    if m:
        return m.group(1)
    for part in path.parts:
        m = SIGNER_FOLDER.search(part)
        if m:
            return m.group(1)
    return None


def load_signer_map(path: Path | None = None) -> dict[str, str]:
    path = path or PATHS.signer_map
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    filename_col = resolve_column(df, "filename") or df.columns[0]
    signer_col = resolve_column(df, "signer_id") or df.columns[1]
    mapping = {}
    for _, row in df.iterrows():
        key = str(row[filename_col]).strip()
        mapping[Path(key).stem.lower()] = str(row[signer_col]).strip()
        mapping[Path(key).name.lower()] = str(row[signer_col]).strip()
    return mapping


def load_index(raw_root: Path | None = None) -> pd.DataFrame:
    """Return one row per video: video_id, path, signer_id, sentence_id, text, category."""
    videos = iter_videos(raw_root)
    if not videos:
        raise FileNotFoundError(
            f"No videos found under {raw_root or PATHS.raw_videos}. "
            "Download SignTalk-Gh first (see scripts/00_get_data.md)."
        )

    meta_path = find_metadata_csv(raw_root)
    meta = load_metadata_table(meta_path) if meta_path else pd.DataFrame()
    signer_map = load_signer_map()

    sent_col = resolve_column(meta, "sentence_id") if not meta.empty else None
    text_col = resolve_column(meta, "sentence_text") if not meta.empty else None
    cat_col = resolve_column(meta, "category") if not meta.empty else None
    signer_col = resolve_column(meta, "signer_id") if not meta.empty else None
    file_col = resolve_column(meta, "filename") if not meta.empty else None

    meta_by_stem: dict[str, pd.Series] = {}
    meta_by_sent: dict[str, pd.Series] = {}
    if not meta.empty:
        if file_col:
            for _, row in meta.iterrows():
                stem = Path(str(row[file_col])).stem.lower()
                meta_by_stem[stem] = row
                meta_by_stem[Path(str(row[file_col])).name.lower()] = row
        if sent_col:
            for _, row in meta.iterrows():
                meta_by_sent[str(row[sent_col]).strip().lower()] = row

    rows = []
    missing_signer = []
    for path in videos:
        stem = _stem_key(path)
        row = meta_by_stem.get(stem) or meta_by_stem.get(path.name.lower())
        if row is None:
            # Filenames like "1A.mp4" / "sentence_id_12.mp4" — match on leading id.
            leading = re.match(r"^(?:sentence[_-]?id[_-]?)?(\d+)", stem)
            if leading:
                row = meta_by_sent.get(leading.group(1))
                if row is None:
                    row = meta_by_sent.get(f"sentence_id_{leading.group(1)}")

        sentence_id = ""
        sentence_text = ""
        category = ""
        signer_id = None
        if row is not None:
            if sent_col:
                sentence_id = str(row[sent_col]).strip()
            if text_col:
                sentence_text = str(row[text_col]).strip()
            if cat_col:
                category = str(row[cat_col]).strip()
            if signer_col:
                signer_id = str(row[signer_col]).strip()

        if not sentence_id:
            sentence_id = stem

        if signer_id in (None, "", "nan"):
            signer_id = signer_map.get(stem) or signer_map.get(path.name.lower())
        if signer_id in (None, "", "nan"):
            signer_id = infer_signer_from_path(path)

        if signer_id in (None, "", "nan"):
            missing_signer.append(path.name)
            continue

        rows.append(
            {
                "video_id": stem,
                "path": str(path.resolve()),
                "signer_id": str(signer_id),
                "sentence_id": sentence_id,
                "sentence_text": sentence_text,
                "category": category,
                "source": "signtalk",
            }
        )

    index = pd.DataFrame(rows)
    if missing_signer:
        print(
            f"[metadata] skipped {len(missing_signer)} video(s) with no signer_id "
            f"(e.g. {missing_signer[:5]})"
        )
    if index.empty:
        raise RuntimeError(
            "No videos with resolvable signer_id. "
            "Run: python scripts/00_inspect_metadata.py"
        )
    return index


def summarize_index(index: pd.DataFrame) -> str:
    lines = [
        f"videos          : {len(index)}",
        f"signers         : {sorted(index['signer_id'].astype(str).unique().tolist())}",
        f"unique sentences: {index['sentence_id'].nunique()}",
        f"categories      : {index['category'].nunique()}",
    ]
    counts = index.groupby("signer_id").size().to_dict()
    lines.append(f"videos/signer   : {counts}")
    if "source" in index.columns:
        lines.append(f"sources         : {index.groupby('source').size().to_dict()}")
    return "\n".join(lines)
