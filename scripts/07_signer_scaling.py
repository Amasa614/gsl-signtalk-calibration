"""RQ3: leave-one-signer-out signer-diversity scaling (exhaustive by default).

Goal
----
Measure how signer-independent *category* accuracy changes as you add more
training signers (k = 1..4), with vs without SSL init — without the noise of
randomly sampling which signers make up each k.

What changed vs the previous RQ3 run
------------------------------------
Before: for each held-out test signer, draw `--draws` (default 2) random
subsets of size k from the other four. At k=2 that meant 2 of 6 possible
pairs — a noisy sample, which made a ~1 pp "dip" hard to interpret.

Now (default `--mode exhaustive`): run *every* train-signer combination at
each k. For 5 signers that is:
  k=1: C(4,1)=4   k=2: C(4,2)=6   k=3: C(4,3)=4   k=4: C(4,4)=1
  per test fold → 15 scratch jobs × 5 folds = 75 scratch
  SSL arm only when test signer was held out of SSL (default: 5) → +15
  Total ≈ 90 jobs (resume skips any already in the JSON).

Primary task: 26-way category ID. Primary unit: clip (mean-pooled logits).
Secondary: coverage-aware clip top-1. Val = 20% clip holdout per train signer.

    python -u scripts/07_signer_scaling.py --dry-run
    python -u scripts/07_signer_scaling.py --mode exhaustive --epochs 10 --arms scratch
    python -u scripts/07_signer_scaling.py --mode exhaustive --epochs 10
    python -u scripts/07_signer_scaling.py --mode sample --draws 2   # old behaviour
    python scripts/08_make_figures.py --results-only
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
from src.splits import holdout_clips_per_signer, load_label_map
from src.train_utils import get_device, set_seed, write_json

PROTOCOL = "loso_category_clip"
VAL_MODE = "per_signer_clip_holdout"


def _ids(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _category_map(windows: pd.DataFrame) -> dict[str, int]:
    labels = sorted(windows["category"].astype(str).unique().tolist())
    mapping = {lab: i for i, lab in enumerate(labels)}
    dest = PATHS.splits / "category_map_loso.json"
    dest.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    if PATHS.category_map.exists():
        existing = load_label_map(PATHS.category_map)
        if set(existing) == set(mapping):
            return existing
    return mapping


def _all_subsets(pool: list[str], k: int) -> list[tuple[str, ...]]:
    pool = sorted(pool)
    if k >= len(pool):
        return [tuple(pool)]
    return [tuple(c) for c in itertools.combinations(pool, k)]


def _sample_subsets(
    pool: list[str], k: int, n_draws: int, rng: np.random.Generator
) -> list[tuple[str, ...]]:
    combos = _all_subsets(pool, k)
    if n_draws >= len(combos):
        return combos
    rng.shuffle(combos)
    return combos[:n_draws]


def _subsets_for_mode(
    pool: list[str],
    k: int,
    mode: str,
    n_draws: int,
    rng: np.random.Generator,
) -> list[tuple[str, ...]]:
    if mode == "exhaustive":
        return _all_subsets(pool, k)
    return _sample_subsets(pool, k, n_draws, rng)


def _run_key(
    test_signer: str,
    k: int,
    train_signers: tuple[str, ...],
    arm: str,
    *,
    arch: str = "transformer",
    seed: int = 42,
) -> tuple:
    return (str(test_signer), int(k), tuple(train_signers), arm, str(arch), int(seed))


def _metric_slot(vals: list[float]) -> dict:
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "n": len(vals),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _summarize(runs: list[dict]) -> dict:
    """Overall arm×k means, plus per held-out test signer (for confound checks)."""
    out: dict = {"by_arm": {}, "by_test_signer": {}}
    for arm in ("scratch", "pretrained"):
        out["by_arm"][arm] = {}
        arm_runs = [r for r in runs if r.get("arm") == arm]
        for k in sorted({int(r["k"]) for r in arm_runs}):
            pts = [r for r in arm_runs if int(r["k"]) == k]
            slot: dict = {"n_runs": len(pts)}
            for metric in (
                "clip_top1",
                "clip_coverage_aware_top1",
                "clip_majority",
                "mass_on_covered",
            ):
                vals = [float(r[metric]) for r in pts if metric in r]
                if vals:
                    slot[metric] = _metric_slot(vals)
            out["by_arm"][arm][str(k)] = slot

    for test_s in sorted({str(r["test_signer"]) for r in runs}):
        out["by_test_signer"][test_s] = {}
        for arm in ("scratch", "pretrained"):
            out["by_test_signer"][test_s][arm] = {}
            for k in sorted(
                {
                    int(r["k"])
                    for r in runs
                    if str(r["test_signer"]) == test_s and r.get("arm") == arm
                }
            ):
                pts = [
                    r
                    for r in runs
                    if str(r["test_signer"]) == test_s
                    and r.get("arm") == arm
                    and int(r["k"]) == k
                ]
                slot = {"n_runs": len(pts)}
                for metric in ("clip_top1", "clip_coverage_aware_top1", "clip_majority"):
                    vals = [float(r[metric]) for r in pts if metric in r]
                    if vals:
                        slot[metric] = _metric_slot(vals)
                out["by_test_signer"][test_s][arm][str(k)] = slot

    # Flat summary kept for fig7 (expects summary[arm][k][metric])
    flat = {arm: out["by_arm"][arm] for arm in ("scratch", "pretrained")}
    out["flat"] = flat
    return out


def _print_summary(summary: dict) -> None:
    flat = summary.get("flat") or summary.get("by_arm") or summary
    print("\nRQ3  category / clip  mean ± std  (exhaustive grid when mode=exhaustive)")
    print(f"{'k':>3}  {'arm':<12}  {'clip top-1':>18}  {'coverage-aware':>18}  {'majority':>12}  n")
    for k in ("1", "2", "3", "4"):
        for arm in ("scratch", "pretrained"):
            slot = (flat.get(arm) or {}).get(k)
            if not slot or "clip_top1" not in slot:
                continue

            def _fmt(m: str) -> str:
                d = slot.get(m) or {}
                return f"{100 * d.get('mean', 0):5.2f} ± {100 * d.get('std', 0):4.2f}%"

            print(
                f"{k:>3}  {arm:<12}  {_fmt('clip_top1'):>18}  "
                f"{_fmt('clip_coverage_aware_top1'):>18}  "
                f"{_fmt('clip_majority'):>12}  {slot.get('n_runs', 0)}"
            )

    by_test = summary.get("by_test_signer") or {}
    if by_test:
        print("\nPer held-out test signer (scratch clip top-1 %):")
        for test_s, arms in by_test.items():
            parts = []
            for k in ("1", "2", "3", "4"):
                slot = ((arms.get("scratch") or {}).get(k) or {}).get("clip_top1")
                if slot:
                    parts.append(f"k{k}={100 * slot['mean']:.1f}")
            if parts:
                print(f"  test={test_s}:  " + "  ".join(parts))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", type=str, default=str(PATHS.ssl_encoder))
    p.add_argument("--signers", type=str, default="1,2,3,4,5")
    p.add_argument(
        "--ssl-clean-test",
        type=str,
        default="5",
        help="test signers allowed for the SSL arm (held out of SSL)",
    )
    p.add_argument(
        "--mode",
        choices=("exhaustive", "sample"),
        default="exhaustive",
        help="exhaustive = all train-signer combos at each k (recommended)",
    )
    p.add_argument(
        "--draws",
        type=int,
        default=2,
        help="only used with --mode sample: random subsets per (test, k)",
    )
    p.add_argument("--k-min", type=int, default=1)
    p.add_argument("--k-max", type=int, default=4)
    p.add_argument(
        "--ks",
        type=str,
        default="",
        help="optional comma list (e.g. 1,4); overrides --k-min/--k-max when set",
    )
    p.add_argument(
        "--test-signers",
        type=str,
        default="",
        help="optional subset of --signers to hold out as test (default: all)",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=CFG.finetune.batch_size)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arms", type=str, default="scratch,pretrained")
    p.add_argument(
        "--arch",
        choices=("transformer", "bigru"),
        default="transformer",
        help="supervised backbone; bigru is scratch-only (no SSL init)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="exhaustive but 8 epochs (faster overnight)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.20,
        help="fraction of each train signer's clips held out for val",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(PATHS.results / "rq3_signer_scaling_exhaustive.json"),
    )
    p.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="optional prior JSON to import matching jobs from (same protocol/val)",
    )
    args = p.parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 8)

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device()
    signers = _ids(args.signers)
    test_signers = _ids(args.test_signers) if args.test_signers else list(signers)
    for t in test_signers:
        if t not in signers:
            raise SystemExit(f"--test-signers {t} not in --signers {signers}")
    ssl_clean = set(_ids(args.ssl_clean_test))
    want_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    arch = str(args.arch).lower().strip()
    if arch != "transformer":
        if "pretrained" in want_arms:
            print(f"[rq3] arch={arch}: dropping pretrained arm (SSL is Transformer-only)")
            want_arms = [a for a in want_arms if a != "pretrained"]
    pretrained_path = Path(args.pretrained)
    if "pretrained" in want_arms and not pretrained_path.exists():
        print(f"[rq3] no SSL checkpoint at {pretrained_path}; dropping pretrained arm")
        want_arms = [a for a in want_arms if a != "pretrained"]

    if args.ks.strip():
        ks = sorted({int(x) for x in args.ks.split(",") if x.strip()})
    else:
        ks = list(range(args.k_min, args.k_max + 1))

    windows = pd.read_csv(PATHS.splits / "windows.csv")
    if "source" in windows.columns:
        windows = windows[
            windows["source"].astype(str).str.lower().eq("signtalk")
        ].reset_index(drop=True)
    label_map = _category_map(windows)
    id_to_cat = {int(v): str(k) for k, v in label_map.items()}
    print(
        f"[rq3] mode={args.mode}  arch={arch}  category classes={len(label_map)}  "
        f"signers={signers}  test={test_signers}  ks={ks}  device={device}"
    )

    jobs = []
    for test_s in test_signers:
        pool = [s for s in signers if s != test_s]
        for k in ks:
            if k < 1 or k > len(pool):
                continue
            for subset in _subsets_for_mode(pool, k, args.mode, args.draws, rng):
                for arm in want_arms:
                    if arm == "pretrained" and test_s not in ssl_clean:
                        continue
                    jobs.append({"test": test_s, "k": k, "train": subset, "arm": arm})

    n_scratch = sum(1 for j in jobs if j["arm"] == "scratch")
    n_pre = sum(1 for j in jobs if j["arm"] == "pretrained")
    print(
        f"[rq3] planned jobs={len(jobs)}  (scratch={n_scratch}, pretrained={n_pre})  "
        f"epochs={args.epochs}  mode={args.mode}  arch={arch}"
    )
    if args.mode == "exhaustive":
        print(
            "[rq3] exhaustive grid: for each test signer, all C(4,k) train sets — "
            "no random pair sampling"
        )
    if args.dry_run:
        by_k: dict[int, int] = {}
        for j in jobs:
            by_k[j["k"]] = by_k.get(j["k"], 0) + 1
        print("[rq3] jobs per k:", dict(sorted(by_k.items())))
        for j in jobs[:12]:
            print(f"  test={j['test']}  k={j['k']}  train={list(j['train'])}  {j['arm']}")
        if len(jobs) > 12:
            print(f"  ... ({len(jobs) - 12} more)")
        return

    out_path = Path(args.out)
    payload = {
        "protocol": PROTOCOL,
        "val_mode": VAL_MODE,
        "val_frac": float(args.val_frac),
        "sampling": args.mode,
        "arch": arch,
        "seed": int(args.seed),
        "ssl_clean_test_signers": sorted(ssl_clean),
        "epochs": args.epochs,
        "draws": None if args.mode == "exhaustive" else args.draws,
        "runs": [],
    }
    done = set()

    def _ingest(path: Path, label: str) -> None:
        nonlocal done
        if not path.exists():
            return
        prev = json.loads(path.read_text(encoding="utf-8"))
        same = prev.get("protocol") == PROTOCOL and prev.get("val_mode") == VAL_MODE
        if not same:
            print(f"[rq3] skip ingest {path}: different protocol/val_mode")
            return
        n_before = len(done)
        for r in prev.get("runs") or []:
            r_arch = str(r.get("arch", prev.get("arch", "transformer")))
            r_seed = int(r.get("seed", prev.get("seed", args.seed)))
            key = _run_key(
                r["test_signer"],
                r["k"],
                tuple(r["train_signers"]),
                r["arm"],
                arch=r_arch,
                seed=r_seed,
            )
            if key in done:
                continue
            # Only keep runs matching this invocation's arch/seed.
            if r_arch != arch or r_seed != int(args.seed):
                continue
            payload["runs"].append(r)
            done.add(key)
        print(f"[rq3] ingested {len(done) - n_before} jobs from {label} ({path.name})")

    _ingest(out_path, "out")
    if args.resume_from:
        _ingest(Path(args.resume_from), "resume-from")

    for i, job in enumerate(jobs, start=1):
        key = _run_key(
            job["test"], job["k"], job["train"], job["arm"], arch=arch, seed=args.seed
        )
        tag = f"rq3-{arch}-t{job['test']}-k{job['k']}-{job['arm']}-{'+'.join(job['train'])}"
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
        if len(val) == 0 or len(train) == 0:
            raise RuntimeError(
                f"{tag}: clip holdout produced train={len(train)} val={len(val)} "
                f"from {len(pool)} windows; cannot checkpoint on an empty val set"
            )
        overlap = set(train["video_id"].astype(str)) & set(val["video_id"].astype(str))
        if overlap:
            raise RuntimeError(f"{tag}: train/val share video_ids: {sorted(overlap)[:8]}")
        print(
            f"[{i}/{len(jobs)}] {tag}  train_windows={len(train)}  "
            f"val_windows={len(val)}  test_windows={len(test)}"
        )
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
            seed=args.seed,
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
            "arch": arch,
            "seed": int(args.seed),
            "ssl_leaky": job["arm"] == "pretrained" and job["test"] not in ssl_clean,
            "n_train_windows": int(len(train)),
            "n_val_windows": int(len(val)),
            "n_pool_windows": int(len(pool)),
            "best_val_acc": float(trained.get("best_val_acc", 0.0)),
            **{k: v for k, v in scores.items() if k != "covered_categories"},
            "n_gold_categories": scores["n_gold_categories"],
        }
        payload["runs"].append(row)
        full_summary = _summarize(payload["runs"])
        payload["summary"] = full_summary["flat"]  # fig7-compatible
        payload["summary_detail"] = full_summary
        write_json(out_path, payload)
        done.add(key)
        print(
            f"  clip top-1={100 * scores['clip_top1']:.2f}%  "
            f"coverage-aware={100 * scores['clip_coverage_aware_top1']:.2f}%  "
            f"majority={100 * scores['clip_majority']:.2f}%  "
            f"covered_mass={100 * scores['mass_on_covered']:.1f}%"
        )
        del trained
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_summary = _summarize(payload["runs"])
    payload["summary"] = full_summary["flat"]
    payload["summary_detail"] = full_summary
    write_json(out_path, payload)
    _print_summary(full_summary)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
