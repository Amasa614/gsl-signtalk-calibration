"""Fine-tune the encoder for sentence ID (RQ1) or category ID (diagnostic).

    python scripts/04_finetune.py --pretrained checkpoints/ssl_encoder.pt
    python scripts/04_finetune.py --pretrained none
    python scripts/04_finetune.py --task category --pretrained checkpoints/ssl_encoder.pt
    python scripts/04_finetune.py --task category --pretrained none
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import CFG, PATHS
from src.finetune_loop import run_finetune
from src.splits import build_label_map, load_label_map
from src.train_utils import get_device, set_seed, write_json


def _ids(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _category_map() -> dict[str, int]:
    if PATHS.category_map.exists():
        return load_label_map(PATHS.category_map)
    windows_path = PATHS.splits / "windows.csv"
    if not windows_path.exists():
        raise FileNotFoundError("Need data/splits/windows.csv to build a category map.")
    windows = pd.read_csv(windows_path)
    mapping = build_label_map(windows, column="category", dest=PATHS.category_map)
    print(f"wrote category map ({len(mapping)} classes) -> {PATHS.category_map}")
    return mapping


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", type=str, required=True, help="path to ssl_encoder.pt, or 'none'")
    p.add_argument("--task", choices=("sentence", "category"), default="sentence")
    p.add_argument("--epochs", type=int, default=CFG.finetune.epochs)
    p.add_argument("--batch-size", type=int, default=CFG.finetune.batch_size)
    p.add_argument("--lr", type=float, default=CFG.finetune.lr)
    p.add_argument("--num-workers", type=int, default=CFG.finetune.num_workers)
    p.add_argument("--seed", type=int, default=CFG.finetune.seed)
    p.add_argument(
        "--arch",
        choices=("transformer", "bigru"),
        default="transformer",
        help="supervised backbone (bigru = cheap alt; SSL init only for transformer)",
    )
    p.add_argument("--train-signers", type=str, default="", help="optional subset of train signers")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    use_pretrained = str(args.pretrained).lower() not in {"none", "scratch", ""}
    if use_pretrained and args.arch != "transformer":
        print(f"[04] arch={args.arch} cannot load SSL weights; forcing scratch")
        use_pretrained = False
    label_col = "category" if args.task == "category" else "sentence_id"
    if args.out:
        out = Path(args.out)
    elif args.task == "category":
        stem = "finetune_pretrained_category" if use_pretrained else "finetune_scratch_category"
        out = PATHS.checkpoints / f"{stem}_{args.arch}.pt" if args.arch != "transformer" else (
            PATHS.finetune_pretrained_category if use_pretrained else PATHS.finetune_scratch_category
        )
    else:
        if args.arch != "transformer":
            stem = "finetune_pretrained" if use_pretrained else "finetune_scratch"
            out = PATHS.checkpoints / f"{stem}_{args.arch}.pt"
        else:
            out = PATHS.finetune_pretrained if use_pretrained else PATHS.finetune_scratch

    label_map = _category_map() if args.task == "category" else load_label_map()
    train = pd.read_csv(PATHS.splits / "train.csv")
    if args.train_signers:
        keep = set(_ids(args.train_signers))
        train = train[train["signer_id"].astype(str).isin(keep)].reset_index(drop=True)
        print(f"filtered train signers {sorted(keep)}: {len(train)} windows")
    val_path = PATHS.splits / "val.csv"
    val = pd.read_csv(val_path) if val_path.exists() else None

    tag = ("pretrained" if use_pretrained else "scratch") + (
        "_category" if args.task == "category" else ""
    )
    if args.arch != "transformer":
        tag = f"{tag}_{args.arch}"
    result = run_finetune(
        train,
        label_map,
        device,
        val_table=val,
        pretrained=Path(args.pretrained) if use_pretrained else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        out=out,
        tag=tag,
        label_col=label_col,
        seed=args.seed,
        arch=args.arch,
    )
    hist_name = f"finetune_{tag}_history.json"
    write_json(PATHS.results / hist_name, result["history"])


if __name__ == "__main__":
    main()
