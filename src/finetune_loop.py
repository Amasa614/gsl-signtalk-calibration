"""Shared fine-tuning loop for RQ1, RQ2 simulation, and RQ3 scaling.

Sentence classification with cross-entropy over a pooled encoder (§3.4).
The evaluation signer is never used here; callers pass already-filtered tables.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CFG, FEATURE_DIM
from src.dataset import LabeledWindowDataset
from src.metrics import predict_loader, summarize_predictions
from src.model import build_pose_model
from src.train_utils import save_checkpoint


def load_encoder_weights(model, path: Path, device) -> None:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    own = model.state_dict()
    filtered = {
        k: v
        for k, v in state.items()
        if k in own and own[k].shape == v.shape and not k.startswith("cls_head")
    }
    missing = model.load_state_dict(filtered, strict=False)
    print(f"loaded {len(filtered)} tensors from {path}")
    print(f"  unexpected={list(missing.unexpected_keys)[:6]}  missing={list(missing.missing_keys)[:6]}")


@torch.no_grad()
def eval_loader(model, loader, device) -> dict:
    if loader is None or loader.dataset is None or len(loader.dataset) == 0:
        return {}
    return summarize_predictions(predict_loader(model, loader, device))


def run_finetune(
    train_table: pd.DataFrame | Path,
    label_map: dict[str, int],
    device,
    *,
    val_table: pd.DataFrame | Path | None = None,
    pretrained: Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    num_workers: int | None = None,
    out: Path | None = None,
    tag: str = "finetune",
    label_col: str = "sentence_id",
    train_dataset=None,
    frame_replace_prob: float = 0.0,
    frame_replace_frac: float = 0.25,
    mask_sub_prob: float = 0.0,
    mask_sub_mode: str = "zero",
    seed: int = 42,
    arch: str = "transformer",
) -> dict:
    """Train a classification head (+ encoder) on labeled SignTalk windows.

    If val_table is empty/None, the last checkpoint is kept (no test-set early
    stopping — the held-out signer must not be used for model selection).
    Callers should pass a non-empty val table carved from training signers'
    own clips so "best val" is comparable across k, including k=1.

    RQ6 hooks: ``frame_replace_prob`` / ``mask_sub_prob`` enable fine-tune-time
    augmentation. Pass ``train_dataset`` to supply a fully custom train set
    (e.g. RQ5 synthetic concat); otherwise a standard or augmented dataset is built.

    ``arch``: ``transformer`` (default) or ``bigru`` (cheap alt supervised backbone).
    SSL pretrained weights only apply to the Transformer.
    """
    epochs = epochs if epochs is not None else CFG.finetune.epochs
    batch_size = batch_size if batch_size is not None else CFG.finetune.batch_size
    lr = lr if lr is not None else CFG.finetune.lr
    num_workers = num_workers if num_workers is not None else CFG.finetune.num_workers
    n_classes = len(label_map)
    arch = str(arch).lower().strip()

    if train_dataset is not None:
        train_ds = train_dataset
    elif frame_replace_prob > 0:
        from src.augment_finetune import AugmentedLabeledWindowDataset

        train_ds = AugmentedLabeledWindowDataset(
            train_table,
            label_map=label_map,
            label_col=label_col,
            frame_replace_prob=frame_replace_prob,
            frame_replace_frac=frame_replace_frac,
            seed=seed,
        )
    else:
        train_ds = LabeledWindowDataset(train_table, label_map=label_map, label_col=label_col)
    val_ds = (
        LabeledWindowDataset(val_table, label_map=label_map, label_col=label_col)
        if val_table is not None
        else None
    )
    if val_ds is not None and len(val_ds) == 0:
        val_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if val_ds is not None
        else None
    )
    print(
        f"[{tag}] arch={arch}  train={len(train_ds)}  "
        f"val={len(val_ds) if val_ds is not None else 0}  "
        f"classes={n_classes}  pretrained={pretrained is not None}  "
        f"frame_replace={frame_replace_prob}  mask_sub={mask_sub_prob}"
    )

    model = build_pose_model(arch, feature_dim=FEATURE_DIM, n_classes=n_classes).to(device)
    if pretrained is not None:
        if arch != "transformer":
            print(
                f"[{tag}] warning: SSL checkpoint ignored for arch={arch} "
                f"(pretrained weights are Transformer-only); training from scratch"
            )
        else:
            load_encoder_weights(model, pretrained, device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CFG.finetune.weight_decay)
    history = []
    best_val = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    use_val = val_loader is not None

    for epoch in range(1, epochs + 1):
        model.train()
        freeze = epoch <= CFG.finetune.freeze_encoder_epochs
        for p_ in model.encoder.parameters():
            p_.requires_grad = not freeze
        running = 0.0
        n = 0
        correct = 0
        for batch in tqdm(train_loader, desc=f"{tag} {epoch}/{epochs}", leave=False):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            if mask_sub_prob > 0:
                from src.augment_finetune import apply_mask_substitution

                x = apply_mask_substitution(
                    x,
                    prob=mask_sub_prob,
                    mode=mask_sub_mode,
                    mask_token=model.input_mask_token,
                )
            logits = model.classify(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * x.size(0)
            n += x.size(0)
            correct += int((logits.argmax(-1) == y).sum().item())
        train_acc = correct / max(n, 1)
        val_stats = eval_loader(model, val_loader, device) if use_val else {}
        val_acc = val_stats.get("top1", train_acc)
        row = {"epoch": epoch, "train_loss": running / max(n, 1), "train_acc": train_acc, "val_acc": val_acc}
        history.append(row)
        print(
            f"epoch {epoch:03d}  loss={row['train_loss']:.4f}  "
            f"train_acc={100 * train_acc:.2f}%  val_acc={100 * val_acc:.2f}%"
        )
        if (not use_val) or val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    payload = {
        "model": model.state_dict(),
        "n_classes": n_classes,
        "feature_dim": FEATURE_DIM,
        "arch": arch,
        "pretrained": pretrained is not None and arch == "transformer",
        "pretrained_path": str(pretrained) if pretrained and arch == "transformer" else None,
        "history": history,
        "best_val_acc": best_val,
        "tag": tag,
        "frame_replace_prob": frame_replace_prob,
        "mask_sub_prob": mask_sub_prob,
    }
    if out is not None:
        save_checkpoint(out, payload)
        print(f"saved -> {out}  best_val_acc={100 * best_val:.2f}%")
    return {"model": model, "history": history, "best_val_acc": best_val, "n_train": len(train_ds)}
