"""Chunk landmark sequences into overlapping windows and build signer-independent splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import CFG, PATHS
from src.corpora import merge_labeled_and_ssl
from src.metadata import load_index, summarize_index
from src.splits import assign_signer_splits, build_label_map, build_windows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--window-size", type=int, default=CFG.window.window_size, help="L in {64, 128} (§3.2)")
    p.add_argument("--stride", type=int, default=CFG.window.stride, help="σ < L")
    p.add_argument("--test-signers", type=str, default="", help="comma-separated; default = last signer")
    p.add_argument("--val-signers", type=str, default="", help="comma-separated; default = second-last")
    args = p.parse_args()

    index_path = PATHS.splits / "video_index.csv"
    if index_path.exists():
        index = pd.read_csv(index_path)
    else:
        index = load_index()
    index = merge_labeled_and_ssl(index)
    print(summarize_index(index))
    if "source" in index.columns:
        print(index.groupby("source").size().to_string())

    def _ids(s: str) -> tuple[str, ...]:
        return tuple(x.strip() for x in s.split(",") if x.strip())

    split_index = assign_signer_splits(
        index,
        test_signers=_ids(args.test_signers) or None,
        val_signers=_ids(args.val_signers) or None,
    )
    # Keep SignTalk extract index separate from SSL-only corpora.
    if "source" in split_index.columns:
        st = split_index[split_index["source"].astype(str).str.lower().eq("signtalk")]
        extra = split_index[split_index["source"].astype(str).str.lower().ne("signtalk")]
        st.to_csv(PATHS.splits / "video_index.csv", index=False)
        if not extra.empty:
            extra.to_csv(PATHS.ssl_index, index=False)
    else:
        split_index.to_csv(PATHS.splits / "video_index.csv", index=False)

    windows = build_windows(split_index, window_size=args.window_size, stride=args.stride)
    label_map = build_label_map(windows)
    print(f"windows: {len(windows)}  classes (train sentences): {len(label_map)}")
    print(windows.groupby("split").size().to_string() if "split" in windows else "")
    print(f"wrote splits under {PATHS.splits}")


if __name__ == "__main__":
    main()
