"""RQ5: synthetic signer diversity via landmark-space perturbation.

For each RQ3-style LOSO job, compare:
  - real_only: train on k real signers (matched to RQ3)
  - +pseudo: each real train signer spawns N pseudo-signers with a fixed
    perturbation profile; the combined pool is subsampled to the *same*
    window count as real_only so volume is matched

Axes: proportion | speed | jitter | combined

    python -u scripts/12_synthetic_signer_scaling.py --dry-run
    python -u scripts/12_synthetic_signer_scaling.py --axis combined --quick
    python -u scripts/12_synthetic_signer_scaling.py --axis proportion,speed,jitter,combined
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
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.augment_finetune import SyntheticProfileDataset
from src.config import CFG, PATHS
from src.dataset import LabeledWindowDataset
from src.finetune_loop import run_finetune
from src.metrics import predict_loader, score_clip_category
from src.splits import holdout_clips_per_signer
from src.synthetic_signers import default_profiles
from src.train_utils import get_device, set_seed, write_json

PROTOCOL = "rq5_synthetic_signers"


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


def _run_key(test_s, k, train, arm, axis, condition) -> tuple:
    return (str(test_s), int(k), tuple(train), arm, axis, condition)


def _matched_synth_dataset(
    train_table: pd.DataFrame,
    label_map: dict[str, int],
    *,
    axis: str,
    n_pseudo: int,
    seed: int,
    target_n: int,
):
    """Real windows + pseudo variants, subsampled to ``target_n`` examples."""
    real_ds = LabeledWindowDataset(train_table, label_map=label_map, label_col="category")
    profiles = default_profiles(n=n_pseudo, axis=axis, base_seed=seed)
    parts = [real_ds]
    for signer in sorted(train_table["signer_id"].astype(str).unique().tolist()):
        sub = train_table[train_table["signer_id"].astype(str).eq(signer)].reset_index(
            drop=True
        )
        for prof in profiles:
            parts.append(
                SyntheticProfileDataset(
                    sub,
                    label_map,
                    prof,
                    label_col="category",
                    signer_id_override=f"{signer}__{prof.name}",
                )
            )
    full = ConcatDataset(parts)
    rng = np.random.default_rng(seed)
    n = len(full)
    take = min(target_n, n)
    idx = rng.choice(n, size=take, replace=False).tolist()
    return Subset(full, idx), take, len(real_ds)


def _summarize(runs: list[dict]) -> dict:
    out: dict = {}
    for axis in sorted({r["axis"] for r in runs}):
        out[axis] = {}
        for cond in sorted({r["condition"] for r in runs if r["axis"] == axis}):
            out[axis][cond] = {}
            for arm in ("scratch", "pretrained"):
                out[axis][cond][arm] = {}
                for k in sorted(
                    {
                        int(r["k"])
                        for r in runs
                        if r["axis"] == axis
                        and r["condition"] == cond
                        and r["arm"] == arm
                    }
                ):
                    pts = [
                        r
                        for r in runs
                        if r["axis"] == axis
                        and r["condition"] == cond
                        and r["arm"] == arm
                        and int(r["k"]) == k
                    ]
                    cov = [float(r["clip_coverage_aware_top1"]) for r in pts]
                    top = [float(r["clip_top1"]) for r in pts]
                    out[axis][cond][arm][str(k)] = {
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
    p.add_argument("--arms", type=str, default="scratch")
    p.add_argument(
        "--axis",
        type=str,
        default="combined",
        help="comma list: proportion,speed,jitter,combined",
    )
    p.add_argument("--n-pseudo", type=int, default=3)
    p.add_argument("--ks", type=str, default="1,2,3,4")
    p.add_argument("--mode", choices=("exhaustive", "sample"), default="sample")
    p.add_argument("--draws", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=CFG.finetune.batch_size)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.20)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--out",
        type=str,
        default=str(PATHS.results / "rq5_synthetic_signers.json"),
    )
    args = p.parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 5)
        args.ks = "1,4"
        args.draws = 1
        args.n_pseudo = 2

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device()
    signers = _ids(args.signers)
    ssl_clean = set(_ids(args.ssl_clean_test))
    want_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    axes = [a.strip() for a in args.axis.split(",") if a.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
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
                    for axis in axes:
                        for cond in ("real_only", "pseudo"):
                            jobs.append(
                                {
                                    "test": test_s,
                                    "k": k,
                                    "train": subset,
                                    "arm": arm,
                                    "axis": axis,
                                    "condition": cond,
                                }
                            )

    print(
        f"[rq5] jobs={len(jobs)}  axes={axes}  n_pseudo={args.n_pseudo}  "
        f"ks={ks}  device={device}"
    )
    if args.dry_run:
        for j in jobs[:14]:
            print(
                f"  test={j['test']} k={j['k']} {j['arm']} {j['axis']} "
                f"{j['condition']} train={list(j['train'])}"
            )
        if len(jobs) > 14:
            print(f"  ... ({len(jobs) - 14} more)")
        return

    out_path = Path(args.out)
    payload = {
        "protocol": PROTOCOL,
        "axes": axes,
        "n_pseudo": args.n_pseudo,
        "ks": ks,
        "mode": args.mode,
        "epochs": args.epochs,
        "matched_volume": True,
        "ssl_clean_test_signers": sorted(ssl_clean),
        "runs": [],
    }
    done = set()
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        if prev.get("protocol") == PROTOCOL:
            for r in prev.get("runs") or []:
                key = _run_key(
                    r["test_signer"],
                    r["k"],
                    r["train_signers"],
                    r["arm"],
                    r["axis"],
                    r["condition"],
                )
                payload["runs"].append(r)
                done.add(key)
            print(f"[rq5] resumed {len(done)} runs")

    for i, job in enumerate(jobs, start=1):
        key = _run_key(
            job["test"], job["k"], job["train"], job["arm"], job["axis"], job["condition"]
        )
        tag = (
            f"rq5-t{job['test']}-k{job['k']}-{job['arm']}-"
            f"{job['axis']}-{job['condition']}"
        )
        if key in done:
            print(f"[{i}/{len(jobs)}] skip {tag}")
            continue

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

        if job["condition"] == "real_only":
            train_ds = LabeledWindowDataset(
                train, label_map=label_map, label_col="category"
            )
            n_train = len(train_ds)
        else:
            train_ds, n_train, n_real = _matched_synth_dataset(
                train,
                label_map,
                axis=job["axis"],
                n_pseudo=args.n_pseudo,
                seed=split_seed + 99,
                target_n=len(train),
            )
            print(f"  matched volume: real={n_real} -> train_used={n_train}")

        print(f"[{i}/{len(jobs)}] {tag}  train={n_train} val={len(val)}")
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
            train_dataset=train_ds,
            seed=args.seed + i,
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
            "axis": job["axis"],
            "condition": job["condition"],
            "n_pseudo": args.n_pseudo if job["condition"] == "pseudo" else 0,
            "n_train_windows": int(n_train),
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
