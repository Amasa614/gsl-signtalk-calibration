"""Inspect SignTalk-Gh metadata and whether signer IDs can be resolved."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PATHS
from src.metadata import (
    find_metadata_csv,
    infer_signer_from_path,
    iter_videos,
    load_index,
    load_metadata_table,
    load_signer_map,
    resolve_column,
    summarize_index,
)


def main() -> None:
    print(f"raw_videos : {PATHS.raw_videos}")
    print(f"exists     : {PATHS.raw_videos.exists()}")
    print(f"metadata   : {PATHS.metadata_path}")
    videos = iter_videos()
    print(f"videos     : {len(videos)}")
    for p in videos[:8]:
        rel = p.relative_to(PATHS.raw_videos) if p.is_relative_to(PATHS.raw_videos) else p
        print(f"  {rel}")
        print(f"    inferred signer: {infer_signer_from_path(p)}")
    if len(videos) > 8:
        print(f"  ... ({len(videos) - 8} more)")

    meta_path = find_metadata_csv()
    print(f"\nmetadata   : {meta_path}")
    if meta_path is None:
        print("No metadata CSV/XLSX found. Download the dataset (scripts/00_get_data.md).")
        return

    df = load_metadata_table(meta_path)
    print(f"rows       : {len(df)}")
    print(f"columns    : {list(df.columns)}")
    print("resolved   :")
    for key in ("sentence_id", "sentence_text", "category", "signer_id", "filename"):
        print(f"  {key:16s} -> {resolve_column(df, key)}")
    print("\nhead:")
    print(df.head(3).to_string(index=False))

    signer_map = load_signer_map()
    print(f"\nsigner_map.csv entries: {len(signer_map)}")
    n_inferred = sum(infer_signer_from_path(p) is not None for p in videos)
    print(f"inferred signer from path: {n_inferred}/{len(videos)}")
    print()
    print(summarize_index(load_index()))


if __name__ == "__main__":
    main()
