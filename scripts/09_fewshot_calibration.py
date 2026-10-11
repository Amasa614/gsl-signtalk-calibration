"""RQ4: signer-adaptive few-shot calibration.

Builds on RQ1-RQ3 nulls: zero-shot SI transfer fails, so ask how little
calibration data from a *new* signer closes the gap.

Protocol
--------
Reuse LOSO. For each held-out test signer:
  1. Train a k=4 category classifier on the other four (scratch and, when
     the test signer was held out of SSL, pretrained).
  2. Split the test signer's clips 80/20 into calibration vs final test
     (by video_id; distinct from the train-signer val holdout).
  3. For n_per_category in {0,1,3,5,10,20}:
       - prototype: nearest-centroid in embedding space (no gradients)
       - partial_ft: short FT of cls_head + last encoder layers on calib

Primary metric: coverage-aware clip top-1 on the final test pool.

    python -u scripts/09_fewshot_calibration.py --dry-run
    python -u scripts/09_fewshot_calibration.py --methods prototype --quick
    python -u scripts/09_fewshot_calibration.py --methods prototype,partial_ft
"""

from __future__ import annotations

import argparse
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

from src.adaptation import (
    build_classifier,
    build_prototypes,
    eval_category_clips,
    extract_embeddings,
    partial_finetune,
    predict_prototypical,
    sample_calib_clips,
    split_calib_test,
)
from src.config import CFG, PATHS
from src.dataset import LabeledWindowDataset
from src.finetune_loop import run_finetune
from src.splits import holdout_clips_per_signer
from src.train_utils import get_device, set_seed, write_json

PROTOCOL = "rq4_fewshot_calibration"


def _ids(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _category_map(windows: pd.DataFrame) -> dict[str, int]:
    labels = sorted(windows["category"].astype(str).unique().tolist())
    mapping = {lab: i for i, lab in enumerate(labels)}
    dest = PATHS.splits / "category_map_loso.json"
    dest.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return mapping


def _run_key(
    test_signer: str, arm: str, method: str, n_per_cat: int, calib_draw: int = 0
) -> tuple:
    return (str(test_signer), arm, method, int(n_per_cat), int(calib_draw))


def _slot_stats(vals: list[float]) -> dict:
    return {
        "mean": float(np.mean(vals)) if vals else 0.0,
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
    }


def _summarize(runs: list[dict]) -> dict:
    """Aggregate across signers and calib draws for each (arm, method, n)."""
    out: dict = {"by_arm_method": {}}
    for arm in sorted({r["arm"] for r in runs}):
        out["by_arm_method"][arm] = {}
        for method in sorted({r["method"] for r in runs if r["arm"] == arm}):
            slots = {}
            for n in sorted(
                {
                    int(r["n_per_category"])
                    for r in runs
                    if r["arm"] == arm and r["method"] == method
                }
            ):
                pts = [
                    r
                    for r in runs
                    if r["arm"] == arm
                    and r["method"] == method
                    and int(r["n_per_category"]) == n
                ]
                slot = {"n_runs": len(pts)}
                for metric in (
                    "clip_top1",
                    "clip_macro_f1",
                    "clip_restricted_top1",
                    "clip_coverage_aware_top1",
                    "clip_oracle_coverage_aware_top1",
                ):
                    vals = [float(r[metric]) for r in pts if metric in r and r[metric] is not None]
                    if vals:
                        slot[metric] = _slot_stats(vals)
                slots[str(n)] = slot
            out["by_arm_method"][arm][method] = slots
    return out


def _print_summary(summary: dict) -> None:
    print("\nRQ4  calibration curves  (unrestricted clip top-1 %)")
    print(f"{'arm':<12} {'method':<12} {'n/cat':>6}  {'clip top-1':>14}  {'macro-F1':>14}  n")
    for arm, methods in (summary.get("by_arm_method") or {}).items():
        for method, slots in methods.items():
            for n, slot in slots.items():
                t = slot.get("clip_top1") or {"mean": 0.0, "std": 0.0}
                m = slot.get("clip_macro_f1") or {"mean": 0.0, "std": 0.0}
                print(
                    f"{arm:<12} {method:<12} {n:>6}  "
                    f"{100 * t['mean']:5.2f} ± {100 * t['std']:4.2f}%  "
                    f"{100 * m['mean']:5.2f} ± {100 * m['std']:4.2f}%  "
                    f"{slot['n_runs']}"
                )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", type=str, default=str(PATHS.ssl_encoder))
    p.add_argument("--signers", type=str, default="1,2,3,4,5")
    p.add_argument("--ssl-clean-test", type=str, default="5")
    p.add_argument("--arms", type=str, default="scratch,pretrained")
    p.add_argument(
        "--methods",
        type=str,
        default="prototype,partial_ft",
        help="comma list: prototype, partial_ft",
    )
    p.add_argument(
        "--n-per-category",
        type=str,
        default="0,1,3,5,10,20",
        help="calibration clips per category",
    )
    p.add_argument(
        "--calib-draws",
        type=int,
        default=1,
        help="independent calibration-set resamples per (signer, method, n); "
        "reuses frozen base checkpoints (no base retrain)",
    )
    p.add_argument("--calib-frac", type=float, default=0.80)
    p.add_argument("--base-epochs", type=int, default=10)
    p.add_argument("--adapt-epochs", type=int, default=8)
    p.add_argument("--unfreeze-layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=CFG.finetune.batch_size)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.20)
    p.add_argument("--quick", action="store_true", help="fewer epochs / sizes")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--out",
        type=str,
        default=str(PATHS.results / "rq4_fewshot_calibration.json"),
    )
    p.add_argument("--ckpt-dir", type=str, default=str(PATHS.checkpoints / "rq4"))
    args = p.parse_args()

    if args.quick:
        args.base_epochs = min(args.base_epochs, 5)
        args.adapt_epochs = min(args.adapt_epochs, 4)
        args.n_per_category = "0,1,5"

    set_seed(args.seed)
    device = get_device()
    signers = _ids(args.signers)
    ssl_clean = set(_ids(args.ssl_clean_test))
    want_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    n_list = [int(x) for x in args.n_per_category.split(",") if x.strip()]
    pretrained_path = Path(args.pretrained)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if "pretrained" in want_arms and not pretrained_path.exists():
        print(f"[rq4] no SSL checkpoint at {pretrained_path}; dropping pretrained arm")
        want_arms = [a for a in want_arms if a != "pretrained"]

    windows = pd.read_csv(PATHS.splits / "windows.csv")
    if "source" in windows.columns:
        windows = windows[
            windows["source"].astype(str).str.lower().eq("signtalk")
        ].reset_index(drop=True)
    label_map = _category_map(windows)
    id_to_cat = {int(v): str(k) for k, v in label_map.items()}
    n_classes = len(label_map)

    jobs = []
    n_draws = max(1, int(args.calib_draws))
    for test_s in signers:
        train_signers = tuple(s for s in signers if s != test_s)
        for arm in want_arms:
            if arm == "pretrained" and test_s not in ssl_clean:
                continue
            for method in methods:
                for n in n_list:
                    # n=0 has no calib sample; one draw is enough
                    draws = 1 if n == 0 else n_draws
                    for draw in range(draws):
                        jobs.append(
                            {
                                "test": test_s,
                                "train": train_signers,
                                "arm": arm,
                                "method": method,
                                "n": n,
                                "calib_draw": draw,
                            }
                        )

    print(
        f"[rq4] jobs={len(jobs)}  arms={want_arms}  methods={methods}  "
        f"n_per_cat={n_list}  calib_draws={n_draws}  device={device}"
    )
    if args.dry_run:
        for j in jobs[:16]:
            print(
                f"  test={j['test']}  arm={j['arm']}  method={j['method']}  "
                f"n={j['n']}  draw={j['calib_draw']}  train={list(j['train'])}"
            )
        if len(jobs) > 16:
            print(f"  ... ({len(jobs) - 16} more)")
        return

    out_path = Path(args.out)
    payload = {
        "protocol": PROTOCOL,
        "calib_frac": float(args.calib_frac),
        "calib_draws": n_draws,
        "base_epochs": args.base_epochs,
        "adapt_epochs": args.adapt_epochs,
        "unfreeze_layers": args.unfreeze_layers,
        "n_per_category": n_list,
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
                    r["arm"],
                    r["method"],
                    r["n_per_category"],
                    int(r.get("calib_draw", 0)),
                )
                payload["runs"].append(r)
                done.add(key)
            print(f"[rq4] resumed {len(done)} runs from {out_path.name}")

    # Cache base models per (test, arm) so we don't retrain for every n/method.
    base_cache: dict[tuple, Path] = {}

    def _ensure_base(test_s: str, train_signers: tuple, arm: str) -> Path:
        key = (test_s, arm)
        if key in base_cache and base_cache[key].exists():
            return base_cache[key]
        ckpt = ckpt_dir / f"base_t{test_s}_k4_{arm}.pt"
        if ckpt.exists():
            base_cache[key] = ckpt
            return ckpt
        pool = windows[windows["signer_id"].astype(str).isin(train_signers)].reset_index(
            drop=True
        )
        split_seed = args.seed + 17 * int(test_s) + sum(int(s) for s in train_signers)
        train, val = holdout_clips_per_signer(pool, val_frac=args.val_frac, seed=split_seed)
        tag = f"rq4-base-t{test_s}-{arm}"
        print(f"[rq4] training base {tag}  train={len(train)} val={len(val)}")
        run_finetune(
            train,
            label_map,
            device,
            val_table=val,
            pretrained=pretrained_path if arm == "pretrained" else None,
            epochs=args.base_epochs,
            batch_size=args.batch_size,
            out=ckpt,
            tag=tag,
            label_col="category",
        )
        base_cache[key] = ckpt
        return ckpt

    for i, job in enumerate(jobs, start=1):
        key = _run_key(
            job["test"], job["arm"], job["method"], job["n"], job["calib_draw"]
        )
        tag = (
            f"rq4-t{job['test']}-{job['arm']}-{job['method']}"
            f"-n{job['n']}-d{job['calib_draw']}"
        )
        if key in done:
            print(f"[{i}/{len(jobs)}] skip {tag}")
            continue

        base_path = _ensure_base(job["test"], job["train"], job["arm"])
        test_all = windows[windows["signer_id"].astype(str).eq(job["test"])].reset_index(
            drop=True
        )
        calib_pool, final_test = split_calib_test(
            test_all,
            calib_frac=args.calib_frac,
            seed=args.seed + 100 * int(job["test"]),
        )
        calib = sample_calib_clips(
            calib_pool,
            n_per_category=job["n"],
            seed=args.seed + 1000 * int(job["test"]) + 17 * job["n"] + 997 * job["calib_draw"],
        )
        n_calib_clips = int(calib["video_id"].nunique()) if len(calib) else 0
        print(
            f"[{i}/{len(jobs)}] {tag}  calib_clips={n_calib_clips}  "
            f"final_test_windows={len(final_test)}"
        )

        model = build_classifier(
            n_classes, device, full_checkpoint=base_path
        )

        if job["n"] == 0:
            allowed = set()
            scores = eval_category_clips(
                model,
                final_test,
                label_map,
                id_to_cat,
                device,
                batch_size=args.batch_size,
                allowed_categories=None,
                oracle_test_coverage=True,
            )
        elif job["method"] == "prototype":
            allowed = set(calib["category"].astype(str).unique().tolist())
            calib_loader = DataLoader(
                LabeledWindowDataset(calib, label_map=label_map, label_col="category"),
                batch_size=args.batch_size,
                shuffle=False,
            )
            calib_emb = extract_embeddings(model, calib_loader, device)
            protos = build_prototypes(calib_emb, n_classes)

            def _pred(m, loader, dev, _p=protos):
                return predict_prototypical(m, loader, _p, dev)

            scores = eval_category_clips(
                model,
                final_test,
                label_map,
                id_to_cat,
                device,
                batch_size=args.batch_size,
                predicter=_pred,
                allowed_categories=allowed,
                oracle_test_coverage=True,
            )
        elif job["method"] == "partial_ft":
            allowed = set(calib["category"].astype(str).unique().tolist())
            model = build_classifier(n_classes, device, full_checkpoint=base_path)
            partial_finetune(
                model,
                calib,
                label_map,
                device,
                epochs=args.adapt_epochs,
                batch_size=args.batch_size,
                n_unfreeze_layers=args.unfreeze_layers,
                tag=tag,
            )
            scores = eval_category_clips(
                model,
                final_test,
                label_map,
                id_to_cat,
                device,
                batch_size=args.batch_size,
                allowed_categories=allowed,
                oracle_test_coverage=True,
            )
        else:
            raise ValueError(f"unknown method {job['method']}")

        row = {
            "test_signer": job["test"],
            "train_signers": list(job["train"]),
            "arm": job["arm"],
            "method": job["method"],
            "n_per_category": job["n"],
            "calib_draw": int(job["calib_draw"]),
            "n_calib_clips": n_calib_clips,
            "n_calib_windows": int(len(calib)),
            "n_final_test_windows": int(len(final_test)),
            "n_allowed_categories": int(len(allowed)) if job["n"] else 0,
            "base_checkpoint": str(base_path),
            **scores,
        }
        payload["runs"].append(row)
        payload["summary"] = _summarize(payload["runs"])
        write_json(out_path, payload)
        done.add(key)
        print(
            f"  top1={100 * scores['clip_top1']:.2f}%  "
            f"macroF1={100 * scores.get('clip_macro_f1', 0):.2f}%  "
            f"calib-restr={100 * scores.get('clip_restricted_top1', 0):.2f}%  "
            f"oracle-cov={100 * scores.get('clip_oracle_coverage_aware_top1', 0):.2f}%"
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload["summary"] = _summarize(payload["runs"])
    write_json(out_path, payload)
    _print_summary(payload["summary"])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
