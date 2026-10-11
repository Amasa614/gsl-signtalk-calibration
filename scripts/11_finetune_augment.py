"""RQ6: frame-level augmentation during supervised fine-tuning.

Compares four conditions at selected k values, crossed with scratch vs SSL:
  none | frame_replace | mask_sub | both

Reuses the RQ3 LOSO category-clip setup but does *not* run the full exhaustive
grid by default (k in {1,3,4}, sample or exhaustive via flags).

    python -u scripts/11_finetune_augment.py --dry-run
    python -u scripts/11_finetune_augment.py --quick --arms scratch
    python -u scripts/11_finetune_augment.py --ks 1,3,4 --mode sample --draws 2
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import CFG, PATHS
from src.dataset import LabeledWindowDataset
from src.finetune_loop import run_finetune
from src.metrics import predict_loader, score_clip_category
from src.splits import holdout_clips_per_signer
from src.train_utils import get_device, set_seed, write_json

PROTOCOL = "rq6_finetune_augment"
AUG_PRESETS = {
    "none": {"frame_replace_prob": 0.0, "mask_sub_prob": 0.0},
    "frame_replace": {"frame_replace_prob": 0.5, "mask_sub_prob": 0.0},
    "mask_sub": {"frame_replace_prob": 0.0, "mask_sub_prob": 0.5},
    "both": {"frame_replace_prob": 0.5, "mask_sub_prob": 0.5},
}


def _ids(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _category_map(windows: pd.DataFrame) -> dict[str, int]:
    labels = sorted(windows["category"].astype(str).unique().tolist())
    return {lab: i for i, lab in enumerate(labels)}


def _all_subsets(pool: list[str], k: int) -> list[tuple[str, ...]]:
    pool = sorted(pool)
    if k >= len(pool):
        return [tuple(pool)]
    return [tuple(c) for c in itertools.combinations(pool, k)]


def _run_key(test_s, k, train, arm, aug, *, arch: str = "transformer", seed: int = 42) -> tuple:
    return (str(test_s), int(k), tuple(train), arm, aug, str(arch), int(seed))


def _summarize(runs: list[dict]) -> dict:
    out: dict = {}
    for aug in sorted({r["aug"] for r in runs}):
        out[aug] = {}
        for arm in ("scratch", "pretrained"):
            out[aug][arm] = {}
            for k in sorted({int(r["k"]) for r in runs if r["aug"] == aug and r["arm"] == arm}):
                pts = [
                    r
                    for r in runs
                    if r["aug"] == aug and r["arm"] == arm and int(r["k"]) == k
                ]
                cov = [float(r["clip_coverage_aware_top1"]) for r in pts]
                top = [float(r["clip_top1"]) for r in pts]
                out[aug][arm][str(k)] = {
                    "n_runs": len(pts),
                    "clip_coverage_aware_top1": {
                        "mean": float(np.mean(cov)),
                        "std": float(np.std(cov, ddof=1)) if len(cov) > 1 else 0.0,
                    },
                    "clip_top1": {
                        "mean": float(np.mean(top)),
                        "std": float(np.std(top, ddof=1)) if len(top) > 1 else 0.0,
                    },
                }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", type=str, default=str(PATHS.ssl_encoder))
    p.add_argument("--signers", type=str, default="1,2,3,4,5")
    p.add_argument("--ssl-clean-test", type=str, default="5")
    p.add_argument("--arms", type=str, default="scratch,pretrained")
    p.add_argument("--augs", type=str, default="none,frame_replace,mask_sub,both")
    p.add_argument("--ks", type=str, default="1,3,4")
    p.add_argument("--mode", choices=("exhaustive", "sample"), default="sample")
    p.add_argument("--draws", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=CFG.finetune.batch_size)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.20)
    p.add_argument(
        "--arch",
        choices=("transformer", "bigru"),
        default="transformer",
        help="supervised backbone; bigru is scratch-only",
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", type=str, default=str(PATHS.results / "rq6_finetune_augment.json"))
    args = p.parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 5)
        args.ks = "1,4"
        args.augs = "none,both"
        args.draws = 1

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device()
    signers = _ids(args.signers)
    ssl_clean = set(_ids(args.ssl_clean_test))
    want_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    augs = [a.strip() for a in args.augs.split(",") if a.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    arch = str(args.arch).lower().strip()
    if arch != "transformer" and "pretrained" in want_arms:
        print(f"[rq6] arch={arch}: dropping pretrained arm")
        want_arms = [a for a in want_arms if a != "pretrained"]
    pretrained_path = Path(args.pretrained)
    if "pretrained" in want_arms and not pretrained_path.exists():
        want_arms = [a for a in want_arms if a != "pretrained"]

    windows = pd.read_csv(PATHS.splits / "windows.csv")
    if "source" in windows.columns:
        windows = windows[
            windows["source"].astype(str).str.lower().eq("signtalk")
        ].reset_index(drop=True)
    label_map = _category_map(windows)
    id_to_cat = {int(v): str(k) for k, v in label_map.items()}

    jobs = []
    for test_s in signers:
        pool = [s for s in signers if s != test_s]
        for k in ks:
            if k > len(pool):
                continue
            combos = _all_subsets(pool, k)
            if args.mode == "sample" and args.draws < len(combos):
                rng.shuffle(combos)
                combos = combos[: args.draws]
            for subset in combos:
                for arm in want_arms:
                    if arm == "pretrained" and test_s not in ssl_clean:
                        continue
                    for aug in augs:
                        jobs.append(
                            {
                                "test": test_s,
                                "k": k,
                                "train": subset,
                                "arm": arm,
                                "aug": aug,
                            }
                        )

    print(f"[rq6] arch={arch}  jobs={len(jobs)}  augs={augs}  ks={ks}  device={device}")
    if args.dry_run:
        for j in jobs[:12]:
            print(
                f"  test={j['test']} k={j['k']} {j['arm']} {j['aug']} "
                f"train={list(j['train'])}"
            )
        if len(jobs) > 12:
            print(f"  ... ({len(jobs) - 12} more)")
        return

    out_path = Path(args.out)
    payload = {
        "protocol": PROTOCOL,
        "augs": augs,
        "ks": ks,
        "mode": args.mode,
        "epochs": args.epochs,
        "arch": arch,
        "seed": int(args.seed),
        "ssl_clean_test_signers": sorted(ssl_clean),
        "runs": [],
    }
    done = set()
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        if prev.get("protocol") == PROTOCOL:
            for r in prev.get("runs") or []:
                r_arch = str(r.get("arch", prev.get("arch", "transformer")))
                r_seed = int(r.get("seed", prev.get("seed", args.seed)))
                if r_arch != arch or r_seed != int(args.seed):
                    continue
                key = _run_key(
                    r["test_signer"],
                    r["k"],
                    r["train_signers"],
                    r["arm"],
                    r["aug"],
                    arch=r_arch,
                    seed=r_seed,
                )
                payload["runs"].append(r)
                done.add(key)
            print(f"[rq6] resumed {len(done)} runs")

    for i, job in enumerate(jobs, start=1):
        key = _run_key(
            job["test"], job["k"], job["train"], job["arm"], job["aug"], arch=arch, seed=args.seed
        )
        tag = f"rq6-{arch}-t{job['test']}-k{job['k']}-{job['arm']}-{job['aug']}"
        if key in done:
            print(f"[{i}/{len(jobs)}] skip {tag}")
            continue
        preset = AUG_PRESETS[job["aug"]]
        pool = windows[windows["signer_id"].astype(str).isin(job["train"])].reset_index(
            drop=True
        )
        test = windows[windows["signer_id"].astype(str).eq(job["test"])].reset_index(
            drop=True
        )
        split_seed = (
            args.seed
            + 17 * int(job["test"])
            + 31 * int(job["k"])
            + sum(int(s) for s in job["train"])
        )
        train, val = holdout_clips_per_signer(
            pool, val_frac=args.val_frac, seed=split_seed
        )
        print(f"[{i}/{len(jobs)}] {tag}  train={len(train)} val={len(val)}")
        trained = run_finetune(
            train,
            label_map,
            device,
            val_table=val,
            pretrained=pretrained_path if job["arm"] == "pretrained" else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            out=None,
            tag=tag,
            label_col="category",
            frame_replace_prob=preset["frame_replace_prob"],
            mask_sub_prob=preset["mask_sub_prob"],
            seed=args.seed + i,
            arch=arch,
        )
        ds = LabeledWindowDataset(test, label_map=label_map, label_col="category")
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        raw = predict_loader(trained["model"], loader, device)
        scores = score_clip_category(raw, id_to_cat)
        row = {
            "test_signer": job["test"],
            "k": job["k"],
            "train_signers": list(job["train"]),
            "arm": job["arm"],
            "aug": job["aug"],
            "arch": arch,
            "seed": int(args.seed),
            "n_train_windows": int(len(train)),
            "best_val_acc": float(trained.get("best_val_acc", 0.0)),
            **{k: v for k, v in scores.items() if k != "covered_categories"},
        }
        payload["runs"].append(row)
        payload["summary"] = _summarize(payload["runs"])
        write_json(out_path, payload)
        done.add(key)
        print(
            f"  clip={100 * scores['clip_top1']:.2f}%  "
            f"cov={100 * scores['clip_coverage_aware_top1']:.2f}%"
        )
        del trained
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload["summary"] = _summarize(payload["runs"])
    write_json(out_path, payload)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
