"""Re-score RQ4 with reviewer-safe metrics (no test-oracle coverage-aware).

Uses existing k=4 base checkpoints under checkpoints/rq4/. Replays the same
calib/test splits and adaptation methods as scripts/09_fewshot_calibration.py,
but reports:
  - unrestricted clip top-1 (primary)
  - macro-F1
  - calib-restricted top-1 (argmax limited to categories seen in the calib set)
  - optional legacy oracle coverage-aware (for comparison only)
  - n_calib_clips, n_allowed_categories, rough recording minutes

    python -u scripts/14_rescore_rq4_metrics.py --quick
    python -u scripts/14_rescore_rq4_metrics.py
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
from src.train_utils import get_device, set_seed, write_json

PROTOCOL = "rq4_fewshot_calibration_v2"


def _ids(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _category_map(windows: pd.DataFrame) -> dict[str, int]:
    labels = sorted(windows["category"].astype(str).unique().tolist())
    return {lab: i for i, lab in enumerate(labels)}


def _run_key(test_signer: str, arm: str, method: str, n_per_cat: int) -> tuple:
    return (str(test_signer), arm, method, int(n_per_cat))


def _estimate_minutes(table: pd.DataFrame, fps: float = 25.0) -> float:
    """Rough recording minutes from n_frames if present, else 4s/clip heuristic."""
    if table is None or len(table) == 0:
        return 0.0
    clips = table.drop_duplicates("video_id")
    if "n_frames" in clips.columns:
        secs = float(clips["n_frames"].astype(float).sum() / max(fps, 1.0))
    else:
        secs = float(len(clips) * 4.0)
    return secs / 60.0


def _summarize(runs: list[dict]) -> dict:
    out: dict = {"by_arm_method": {}}
    metrics = (
        "clip_top1",
        "clip_macro_f1",
        "clip_restricted_top1",
        "clip_oracle_coverage_aware_top1",
        "clip_majority",
        "n_calib_clips",
        "n_allowed_categories",
        "approx_calib_minutes",
    )
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
                for m in metrics:
                    vals = [float(r[m]) for r in pts if m in r and r[m] is not None]
                    if not vals:
                        continue
                    slot[m] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    }
                slots[str(n)] = slot
            out["by_arm_method"][arm][method] = slots
    return out


def _print_summary(summary: dict) -> None:
    print("\nRQ4 v2  (primary = unrestricted clip top-1 %)")
    print(
        f"{'arm':<12} {'method':<12} {'n':>4}  "
        f"{'top1':>14}  {'macroF1':>14}  {'calib-restr':>14}  {'mins':>8}  n"
    )
    for arm, methods in (summary.get("by_arm_method") or {}).items():
        for method, slots in methods.items():
            for n, slot in slots.items():

                def _fmt(m: str) -> str:
                    d = slot.get(m) or {}
                    if not d:
                        return "   —"
                    return f"{100 * d['mean']:5.2f} ± {100 * d.get('std', 0):4.2f}"

                mins = slot.get("approx_calib_minutes") or {}
                mins_s = f"{mins.get('mean', 0):5.1f}" if mins else "   —"
                print(
                    f"{arm:<12} {method:<12} {n:>4}  "
                    f"{_fmt('clip_top1'):>14}  {_fmt('clip_macro_f1'):>14}  "
                    f"{_fmt('clip_restricted_top1'):>14}  {mins_s:>8}  "
                    f"{slot.get('n_runs', 0)}"
                )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", type=str, default=str(PATHS.ssl_encoder))
    p.add_argument("--signers", type=str, default="1,2,3,4,5")
    p.add_argument("--ssl-clean-test", type=str, default="5")
    p.add_argument("--arms", type=str, default="scratch,pretrained")
    p.add_argument("--methods", type=str, default="prototype,partial_ft")
    p.add_argument("--n-per-category", type=str, default="0,1,3,5,10,20")
    p.add_argument("--calib-frac", type=float, default=0.80)
    p.add_argument("--adapt-epochs", type=int, default=8)
    p.add_argument("--unfreeze-layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=CFG.finetune.batch_size)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ckpt-dir", type=str, default=str(PATHS.checkpoints / "rq4"))
    p.add_argument(
        "--out",
        type=str,
        default=str(PATHS.results / "rq4_fewshot_calibration_v2.json"),
    )
    args = p.parse_args()

    if args.quick:
        args.adapt_epochs = min(args.adapt_epochs, 4)
        args.n_per_category = "0,1,5"
        args.methods = "prototype"

    set_seed(args.seed)
    device = get_device()
    signers = _ids(args.signers)
    ssl_clean = set(_ids(args.ssl_clean_test))
    want_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    n_list = [int(x) for x in args.n_per_category.split(",") if x.strip()]
    ckpt_dir = Path(args.ckpt_dir)

    windows = pd.read_csv(PATHS.splits / "windows.csv")
    if "source" in windows.columns:
        windows = windows[
            windows["source"].astype(str).str.lower().eq("signtalk")
        ].reset_index(drop=True)
    label_map = _category_map(windows)
    id_to_cat = {int(v): str(k) for k, v in label_map.items()}
    n_classes = len(label_map)

    jobs = []
    for test_s in signers:
        train_signers = tuple(s for s in signers if s != test_s)
        for arm in want_arms:
            if arm == "pretrained" and test_s not in ssl_clean:
                continue
            for method in methods:
                for n in n_list:
                    jobs.append(
                        {
                            "test": test_s,
                            "train": train_signers,
                            "arm": arm,
                            "method": method,
                            "n": n,
                        }
                    )

    print(f"[rq4v2] jobs={len(jobs)}  device={device}  ckpt_dir={ckpt_dir}")
    if args.dry_run:
        for j in jobs[:12]:
            print(j)
        return

    out_path = Path(args.out)
    payload = {
        "protocol": PROTOCOL,
        "primary_metric": "clip_top1",
        "restricted_rule": "categories present in calibration set only",
        "calib_frac": float(args.calib_frac),
        "adapt_epochs": args.adapt_epochs,
        "fps_for_minutes": args.fps,
        "n_per_category": n_list,
        "ssl_clean_test_signers": sorted(ssl_clean),
        "runs": [],
    }
    done = set()
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        if prev.get("protocol") == PROTOCOL:
            for r in prev.get("runs") or []:
                key = _run_key(r["test_signer"], r["arm"], r["method"], r["n_per_category"])
                payload["runs"].append(r)
                done.add(key)
            print(f"[rq4v2] resumed {len(done)} runs")

    for i, job in enumerate(jobs, start=1):
        key = _run_key(job["test"], job["arm"], job["method"], job["n"])
        tag = f"rq4v2-t{job['test']}-{job['arm']}-{job['method']}-n{job['n']}"
        if key in done:
            print(f"[{i}/{len(jobs)}] skip {tag}")
            continue

        base_path = ckpt_dir / f"base_t{job['test']}_k4_{job['arm']}.pt"
        if not base_path.exists():
            print(f"[{i}/{len(jobs)}] MISSING base {base_path}; skip")
            continue

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
            seed=args.seed + 1000 * int(job["test"]) + job["n"],
        )
        allowed = (
            set(calib["category"].astype(str).unique().tolist()) if len(calib) else set()
        )
        n_calib_clips = int(calib["video_id"].nunique()) if len(calib) else 0
        approx_mins = _estimate_minutes(calib, fps=args.fps)

        print(
            f"[{i}/{len(jobs)}] {tag}  calib_clips={n_calib_clips}  "
            f"allowed_cats={len(allowed)}  ~{approx_mins:.1f} min  "
            f"test_windows={len(final_test)}"
        )

        model = build_classifier(n_classes, device, full_checkpoint=base_path)
        predicter = None

        if job["n"] > 0 and job["method"] == "prototype":
            calib_loader = DataLoader(
                LabeledWindowDataset(calib, label_map=label_map, label_col="category"),
                batch_size=args.batch_size,
                shuffle=False,
            )
            calib_emb = extract_embeddings(model, calib_loader, device)
            protos = build_prototypes(calib_emb, n_classes)

            def _pred(m, loader, dev, _p=protos):
                return predict_prototypical(m, loader, _p, dev)

            predicter = _pred
        elif job["n"] > 0 and job["method"] == "partial_ft":
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
        elif job["n"] > 0:
            raise ValueError(job["method"])

        scores = eval_category_clips(
            model,
            final_test,
            label_map,
            id_to_cat,
            device,
            batch_size=args.batch_size,
            predicter=predicter,
            allowed_categories=allowed if allowed else None,
            oracle_test_coverage=True,  # log legacy metric for comparison only
        )
        row = {
            "test_signer": job["test"],
            "train_signers": list(job["train"]),
            "arm": job["arm"],
            "method": job["method"],
            "n_per_category": job["n"],
            "n_calib_clips": n_calib_clips,
            "n_calib_windows": int(len(calib)),
            "n_final_test_windows": int(len(final_test)),
            "n_allowed_categories": int(len(allowed)),
            "approx_calib_minutes": float(approx_mins),
            "base_checkpoint": str(base_path),
            **scores,
        }
        payload["runs"].append(row)
        payload["summary"] = _summarize(payload["runs"])
        write_json(out_path, payload)
        done.add(key)
        print(
            f"  top1={100 * scores['clip_top1']:.2f}%  "
            f"macroF1={100 * scores['clip_macro_f1']:.2f}%  "
            f"calib-restr={100 * scores['clip_restricted_top1']:.2f}%  "
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
