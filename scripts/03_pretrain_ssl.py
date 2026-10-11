"""Self-supervised pretraining: masked pose reconstruction (+ optional contrastive)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CFG, FEATURE_DIM, PATHS
from src.dataset import SSLWindowDataset
from src.model import (
    MaskedPoseTransformer,
    make_frame_mask,
    make_joint_mask,
    nt_xent,
    reconstruction_loss,
)
from src.train_utils import get_device, save_checkpoint, set_seed, write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=CFG.ssl.epochs)
    p.add_argument("--batch-size", type=int, default=CFG.ssl.batch_size)
    p.add_argument("--lr", type=float, default=CFG.ssl.lr)
    p.add_argument(
        "--mask-ratio",
        type=float,
        default=None,
        help="fixed masking ratio; default samples ρ ~ U[0.15, 0.40] per window (§3.3.1)",
    )
    p.add_argument("--mask-mode", choices=("frame", "joint"), default=CFG.ssl.mask_mode)
    p.add_argument("--contrastive", action="store_true", help="add InfoNCE ablation objective (§3.3.2)")
    p.add_argument("--num-workers", type=int, default=CFG.ssl.num_workers)
    p.add_argument("--seed", type=int, default=CFG.ssl.seed)
    p.add_argument("--out", type=str, default=str(PATHS.ssl_encoder))
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    windows_path = PATHS.splits / "windows.csv"
    if not windows_path.exists():
        raise FileNotFoundError("Run scripts/02_build_windows.py first.")
    table = pd.read_csv(windows_path)
    # §3.1: pool all unlabeled sources (SignTalk labels discarded, plus extra corpora).
    # §3.6: the evaluation signer never appears in any training split of any arm,
    # including SSL, otherwise the pretrained arm would have seen test-signer motion.
    if "split" in table.columns:
        table = table[table["split"] != "test"].reset_index(drop=True)
    if "split" in table.columns:
        counts = table.groupby("split").size().to_dict()
        print(f"SSL split mix: {counts}  (test signer held out per §3.6)")
    if "source" in table.columns:
        print("SSL sources:", table.groupby("source").size().to_dict())
    else:
        print("SSL sources: signtalk (extra corpora not merged yet)")
    ds = SSLWindowDataset(table, augment=True, seed=args.seed)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=len(ds) >= args.batch_size,
    )
    print(f"SSL windows: {len(ds)}  device: {device}  contrastive: {args.contrastive}")

    model = MaskedPoseTransformer(feature_dim=FEATURE_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=CFG.ssl.weight_decay)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for batch in tqdm(loader, desc=f"ssl {epoch}/{args.epochs}", leave=False):
            x = batch["x"].to(device)
            ratio_range = (CFG.ssl.mask_ratio_min, CFG.ssl.mask_ratio_max)
            if args.mask_mode == "joint":
                mask = make_joint_mask(x, args.mask_ratio, ratio_range=ratio_range)
                x_in = model.replace_masked_inputs(x, mask)
                pred = model.reconstruct(x_in, frame_mask=None)
                loss = reconstruction_loss(pred, x, mask, kind=CFG.ssl.loss)
            else:
                mask = make_frame_mask(x, args.mask_ratio, ratio_range=ratio_range)
                pred = model.reconstruct(x, frame_mask=mask)
                loss = reconstruction_loss(pred, x, mask, kind=CFG.ssl.loss)
            if args.contrastive:
                za = model.project(batch["view_a"].to(device))
                zb = model.project(batch["view_b"].to(device))
                loss = loss + CFG.ssl.contrastive_weight * nt_xent(za, zb, CFG.ssl.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * x.size(0)
            n += x.size(0)
        avg = running / max(n, 1)
        history.append({"epoch": epoch, "loss": avg})
        print(f"epoch {epoch:03d}  ssl_loss={avg:.4f}")

    out = Path(args.out)
    save_checkpoint(
        out,
        {
            "model": model.state_dict(),
            "feature_dim": FEATURE_DIM,
            "d_model": model.d_model,
            "mask_ratio": args.mask_ratio,
            "mask_ratio_range": [CFG.ssl.mask_ratio_min, CFG.ssl.mask_ratio_max],
            "mask_mode": args.mask_mode,
            "contrastive": args.contrastive,
            "history": history,
        },
    )
    write_json(PATHS.results / "ssl_history.json", history)
    print(f"saved encoder -> {out}")


if __name__ == "__main__":
    main()
