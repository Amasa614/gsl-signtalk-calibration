"""Extract landmarks from SignTalk-Gh and optional SSL-only corpora.

Fragkiadakis `GSL_openpose_data` is converted from OpenPose JSON (fast).
SignTalk stays on the existing MediaPipe `.npz` files unless you pass `--no-skip`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import PATHS
from src.corpora import load_afrisign_index, load_fragkiadakis_index, load_ssl_extra_index
from src.landmark_extraction import extract_index
from src.metadata import load_index, summarize_index
from src.openpose import convert_openpose_index


def _write_index(done: pd.DataFrame, dest: Path, *, merge: bool) -> None:
    if merge and dest.exists() and not done.empty:
        prev = pd.read_csv(dest)
        done = pd.concat([prev, done], ignore_index=True).drop_duplicates("video_id")
    dest.parent.mkdir(parents=True, exist_ok=True)
    done.to_csv(dest, index=False)
    print(f"index -> {dest}  ({len(done)} rows)")


def _extract_or_convert(index: pd.DataFrame, args) -> list[Path]:
    """OpenPose JSON clips -> npz; everything else -> MediaPipe video extract."""
    if args.max_videos is not None:
        index = index.head(args.max_videos)
    if index.empty:
        return []
    pose = index["pose_source"].astype(str).str.lower() if "pose_source" in index.columns else None
    if pose is not None and not args.from_video:
        op = index[pose.eq("openpose")]
        vid = index[~pose.eq("openpose")]
    else:
        op, vid = pd.DataFrame(), index
    written: list[Path] = []
    if not op.empty:
        written.extend(convert_openpose_index(op, skip_existing=not args.no_skip))
    if not vid.empty:
        written.extend(
            extract_index(
                vid,
                skip_existing=not args.no_skip,
                use_separate=True if args.separate_models else None,
                use_gpu=False if args.cpu else True,
                workers=args.workers,
            )
        )
    return written


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-videos", type=int, default=None, help="debug subset; start here")
    p.add_argument("--per-signer", type=int, default=None, help="take N videos from each signer")
    p.add_argument("--signers", type=str, default=None, help="comma-separated signer IDs")
    p.add_argument(
        "--source",
        choices=("signtalk", "fragkiadakis", "afrisign", "ssl", "all"),
        default="signtalk",
        help="signtalk writes video_index.csv; extra corpora write ssl_index.csv",
    )
    p.add_argument("--no-skip", action="store_true", help="re-extract even if .npz exists")
    p.add_argument("--separate-models", action="store_true", help="Pose+Hands+Face instead of Holistic")
    p.add_argument("--gpu", action="store_true", default=True, help="MediaPipe GPU delegate (default)")
    p.add_argument("--cpu", action="store_true", help="force CPU delegate")
    p.add_argument("--workers", type=int, default=None, help="parallel CPU processes (default: min(10, cores-2))")
    p.add_argument(
        "--from-video",
        action="store_true",
        help="re-extract Fragkiadakis with MediaPipe from mp4s instead of OpenPose JSON",
    )
    args = p.parse_args()

    if args.source == "signtalk":
        index = load_index()
    elif args.source == "fragkiadakis":
        index = load_fragkiadakis_index()
        if index.empty:
            raise FileNotFoundError(
                "No Fragkiadakis OpenPose dump or videos. "
                "Place GSL_openpose_data at the repo root, or run: "
                "python scripts/download_fragkiadakis.py"
            )
    elif args.source == "afrisign":
        index = load_afrisign_index()
        if index.empty:
            raise FileNotFoundError(
                "No AfriSign videos. Drop GSL mp4s under data/external/afrisign/"
            )
    elif args.source == "ssl":
        index = load_ssl_extra_index()
        if index.empty:
            raise FileNotFoundError("No extra SSL videos under data/external/")
    else:
        st = load_index()
        extra = load_ssl_extra_index()
        index = pd.concat([st, extra], ignore_index=True) if not extra.empty else st
        print(
            "[extract] --source all: SignTalk rows go to video_index.csv; "
            "extra corpora go to ssl_index.csv. Do not run this while a "
            "SignTalk extract is still writing video_index.csv."
        )

    print(summarize_index(index))
    if "source" in index.columns:
        print(index.groupby("source").size().to_string())
    if args.signers:
        keep = {s.strip() for s in args.signers.split(",") if s.strip()}
        index = index[index["signer_id"].astype(str).isin(keep)].reset_index(drop=True)
        print(f"filtered to signers {sorted(keep)}: {len(index)} videos")
    if args.per_signer:
        index = (
            index.groupby("signer_id", group_keys=False)
            .head(args.per_signer)
            .reset_index(drop=True)
        )
        print(f"per-signer={args.per_signer}: {len(index)} videos")

    PATHS.landmarks.mkdir(parents=True, exist_ok=True)
    PATHS.splits.mkdir(parents=True, exist_ok=True)
    written = _extract_or_convert(index, args)
    done_ids = {path.stem for path in written}
    done = index.loc[index["video_id"].isin(done_ids)].copy()

    extra_sources = {"fragkiadakis", "afrisign", "ssl"}
    if args.source in extra_sources:
        _write_index(done, PATHS.ssl_index, merge=True)
    elif args.source == "all":
        if "source" in done.columns:
            st = done[done["source"].astype(str).str.lower().eq("signtalk")]
            extra = done[done["source"].astype(str).str.lower().ne("signtalk")]
        else:
            st, extra = done, pd.DataFrame()
        _write_index(st, PATHS.splits / "video_index.csv", merge=False)
        if not extra.empty:
            _write_index(extra, PATHS.ssl_index, merge=True)
    else:
        _write_index(done, PATHS.splits / "video_index.csv", merge=False)
    print(f"wrote {len(written)} landmark files -> {PATHS.landmarks}")


if __name__ == "__main__":
    main()
