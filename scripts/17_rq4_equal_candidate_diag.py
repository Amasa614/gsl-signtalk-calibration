"""RQ4 equal-candidate diagnostic (reviewer ablation).

Separates category-coverage information from target-specific prototypes:

  A. Unadapted base, all 26 outputs          (existing zero-shot)
  B. Unadapted base, restricted to calib cats (coverage alone)
  C. Prototype over those same categories     (from existing JSON or rebuild)
  D. Partial FT unrestricted (all 26)         (existing primary)
  E. Partial FT restricted to same cats       (existing clip_restricted_top1)

Only B requires new forward passes; A/C/D/E are joined from prior RQ4 JSONs
when present. Bases and calib/test splits are replayed with the same seeds as
the source protocol (v2 single-draw or multi-draw).

    python -u scripts/17_rq4_equal_candidate_diag.py --source v2
    python -u scripts/17_rq4_equal_candidate_diag.py --source draws --n-per-category 1,5,20
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

from src.adaptation import (
    build_classifier,
    build_prototypes,
    eval_category_clips,
    extract_embeddings,
    predict_prototypical,
    sample_calib_clips,
    split_calib_test,
)
from src.config import CFG, PATHS
from src.dataset import LabeledWindowDataset
from src.train_utils import get_device, set_seed, write_json

PROTOCOL = "rq4_equal_candidate_diag"


def _ids(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _category_map(windows: pd.DataFrame) -> dict[str, int]:
    labels = sorted(windows["category"].astype(str).unique().tolist())
    return {lab: i for i, lab in enumerate(labels)}


def _slot_stats(vals: list[float]) -> dict:
    return {
        "mean": float(np.mean(vals)) if vals else 0.0,
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def _load_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("runs") or [])


def _index_runs(runs: list[dict]) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for r in runs:
        key = (
            str(r["test_signer"]),
            str(r["arm"]),
            str(r["method"]),
            int(r["n_per_category"]),
            int(r.get("calib_draw", 0)),
        )
        out[key] = r
    return out


def _calib_seed(source: str, seed: int, test_s: str, n: int, draw: int) -> int:
    """Match scripts/14 (v2) or scripts/09 (multi-draw) sampling."""
    if source == "v2":
        return seed + 1000 * int(test_s) + int(n)
    return seed + 1000 * int(test_s) + 17 * int(n) + 997 * int(draw)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=("v2", "draws"), default="v2")
    p.add_argument("--signers", type=str, default="1,2,3,4,5")
    p.add_argument("--arms", type=str, default="scratch")
    p.add_argument("--n-per-category", type=str, default="1,5,20")
    p.add_argument("--calib-draws", type=str, default="")
    p.add_argument("--calib-frac", type=float, default=0.80)
    p.add_argument("--batch-size", type=int, default=CFG.finetune.batch_size)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-dir", type=str, default=str(PATHS.checkpoints / "rq4"))
    p.add_argument(
        "--prior-v2",
        type=str,
        default=str(PATHS.results / "rq4_fewshot_calibration_v2.json"),
    )
    p.add_argument(
        "--prior-draws",
        type=str,
        default=str(PATHS.results / "rq4_calib_draws.json"),
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
    )
    p.add_argument("--rebuild-prototype", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    signers = _ids(args.signers)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    n_list = [int(x) for x in args.n_per_category.split(",") if x.strip()]
    ckpt_dir = Path(args.ckpt_dir)

    if args.source == "v2":
        draws = [0]
        prior_path = Path(args.prior_v2)
        default_out = PATHS.results / "rq4_equal_candidate_v2.json"
    else:
        if args.calib_draws.strip():
            draws = [int(x) for x in args.calib_draws.split(",") if x.strip()]
        else:
            draws = list(range(5))
        prior_path = Path(args.prior_draws)
        default_out = PATHS.results / "rq4_equal_candidate_draws.json"
    out_path = Path(args.out) if args.out else default_out

    windows = pd.read_csv(PATHS.splits / "windows.csv")
    if "source" in windows.columns:
        windows = windows[
            windows["source"].astype(str).str.lower().eq("signtalk")
        ].reset_index(drop=True)
    label_map = _category_map(windows)
    id_to_cat = {int(v): str(k) for k, v in label_map.items()}
    n_classes = len(label_map)
    prior = _index_runs(_load_runs(prior_path))

    jobs = []
    for test_s in signers:
        for arm in arms:
            if arm == "pretrained" and test_s != "5":
                continue
            for n in n_list:
                for draw in draws:
                    jobs.append(
                        {"test": test_s, "arm": arm, "n": n, "calib_draw": draw}
                    )

    print(
        f"[eqcand] source={args.source}  jobs={len(jobs)}  "
        f"prior={prior_path.name} ({len(prior)} runs)  device={device}"
    )
    if args.dry_run:
        for j in jobs[:8]:
            print(" ", j)
        return

    rows: list[dict] = []
    for i, job in enumerate(jobs, start=1):
        test_s = job["test"]
        arm = job["arm"]
        n = job["n"]
        draw = job["calib_draw"]
        base_path = ckpt_dir / f"base_t{test_s}_k4_{arm}.pt"
        tag = f"t{test_s}-{arm}-n{n}-d{draw}"
        if not base_path.exists():
            print(f"[{i}/{len(jobs)}] MISSING {base_path.name}; skip")
            continue

        test_all = windows[windows["signer_id"].astype(str).eq(test_s)].reset_index(
            drop=True
        )
        calib_pool, final_test = split_calib_test(
            test_all,
            calib_frac=args.calib_frac,
            seed=args.seed + 100 * int(test_s),
        )
        calib = sample_calib_clips(
            calib_pool,
            n_per_category=n,
            seed=_calib_seed(args.source, args.seed, test_s, n, draw),
        )
        allowed = set(calib["category"].astype(str).unique().tolist())
        print(
            f"[{i}/{len(jobs)}] {tag}  allowed={len(allowed)}  "
            f"test_windows={len(final_test)}"
        )

        model = build_classifier(n_classes, device, full_checkpoint=base_path)
        base_full = eval_category_clips(
            model,
            final_test,
            label_map,
            id_to_cat,
            device,
            batch_size=args.batch_size,
            allowed_categories=None,
        )
        base_restr = eval_category_clips(
            model,
            final_test,
            label_map,
            id_to_cat,
            device,
            batch_size=args.batch_size,
            allowed_categories=allowed,
        )

        proto_key = (test_s, arm, "prototype", n, draw if args.source == "draws" else 0)
        ft_key = (test_s, arm, "partial_ft", n, draw if args.source == "draws" else 0)
        # v2 JSON has no calib_draw field -> indexed as 0
        if args.source == "v2":
            proto_key = (test_s, arm, "prototype", n, 0)
            ft_key = (test_s, arm, "partial_ft", n, 0)

        proto_row = prior.get(proto_key)
        ft_row = prior.get(ft_key)

        if args.rebuild_prototype or proto_row is None:
            from torch.utils.data import DataLoader

            calib_loader = DataLoader(
                LabeledWindowDataset(calib, label_map=label_map, label_col="category"),
                batch_size=args.batch_size,
                shuffle=False,
            )
            calib_emb = extract_embeddings(model, calib_loader, device)
            protos = build_prototypes(calib_emb, n_classes)

            def _pred(m, loader, dev, _p=protos):
                return predict_prototypical(m, loader, _p, dev)

            proto_scores = eval_category_clips(
                model,
                final_test,
                label_map,
                id_to_cat,
                device,
                batch_size=args.batch_size,
                predicter=_pred,
                allowed_categories=allowed,
            )
            proto_top1 = float(proto_scores["clip_top1"])
            proto_restr = float(proto_scores["clip_restricted_top1"])
            proto_src = "rebuilt"
        else:
            proto_top1 = float(proto_row["clip_top1"])
            proto_restr = float(proto_row.get("clip_restricted_top1", proto_top1))
            proto_src = "prior"

        if ft_row is None:
            ft_top1 = None
            ft_restr = None
            ft_src = "missing"
        else:
            ft_top1 = float(ft_row["clip_top1"])
            ft_restr = float(ft_row.get("clip_restricted_top1", ft_top1))
            ft_src = "prior"

        row = {
            "test_signer": test_s,
            "arm": arm,
            "n_per_category": n,
            "calib_draw": draw,
            "n_allowed_categories": len(allowed),
            "base_unrestricted_top1": float(base_full["clip_top1"]),
            "base_calib_restricted_top1": float(base_restr["clip_restricted_top1"]),
            "prototype_top1": proto_top1,
            "prototype_calib_restricted_top1": proto_restr,
            "partial_ft_unrestricted_top1": ft_top1,
            "partial_ft_calib_restricted_top1": ft_restr,
            "delta_coverage_pp": 100.0
            * (float(base_restr["clip_restricted_top1"]) - float(base_full["clip_top1"])),
            "delta_prototype_vs_base_restr_pp": 100.0
            * (proto_top1 - float(base_restr["clip_restricted_top1"])),
            "sources": {
                "prototype": proto_src,
                "partial_ft": ft_src,
                "base": "rescored",
            },
            "base_checkpoint": str(base_path),
        }
        rows.append(row)
        print(
            f"  base26={100 * row['base_unrestricted_top1']:.2f}%  "
            f"base|calib={100 * row['base_calib_restricted_top1']:.2f}%  "
            f"proto={100 * row['prototype_top1']:.2f}%  "
            f"FT26={None if ft_top1 is None else round(100 * ft_top1, 2)}%  "
            f"FT|calib={None if ft_restr is None else round(100 * ft_restr, 2)}%  "
            f"d_cov={row['delta_coverage_pp']:+.2f}pp  "
            f"d_proto={row['delta_prototype_vs_base_restr_pp']:+.2f}pp"
        )
        del model

    # Summaries by n
    summary: dict = {"by_arm_n": {}}
    for arm in sorted({r["arm"] for r in rows}):
        summary["by_arm_n"][arm] = {}
        for n in sorted({int(r["n_per_category"]) for r in rows if r["arm"] == arm}):
            pts = [r for r in rows if r["arm"] == arm and int(r["n_per_category"]) == n]
            slot = {"n_runs": len(pts)}
            for key in (
                "base_unrestricted_top1",
                "base_calib_restricted_top1",
                "prototype_top1",
                "partial_ft_unrestricted_top1",
                "partial_ft_calib_restricted_top1",
                "delta_coverage_pp",
                "delta_prototype_vs_base_restr_pp",
            ):
                vals = [float(r[key]) for r in pts if r.get(key) is not None]
                if vals:
                    slot[key] = _slot_stats(vals)
            summary["by_arm_n"][arm][str(n)] = slot

    payload = {
        "protocol": PROTOCOL,
        "source": args.source,
        "prior_json": str(prior_path),
        "note": (
            "Equal-candidate diagnostic. base_calib_restricted uses the frozen "
            "base head with argmax limited to categories present in the calib "
            "sample. prototype/partial_ft joined from prior JSON when available."
        ),
        "n_per_category": n_list,
        "calib_draws": draws,
        "runs": rows,
        "summary": summary,
    }
    write_json(out_path, payload)

    print("\n=== Equal-candidate summary (scratch means, %) ===")
    print(
        f"{'n':>4}  {'base26':>8}  {'base|C':>8}  {'proto':>8}  "
        f"{'FT26':>8}  {'FT|C':>8}  {'d_cov':>7}  {'d_proto':>8}"
    )
    scratch = summary.get("by_arm_n", {}).get("scratch", {})
    for n, slot in scratch.items():
        def g(k: str) -> str:
            d = slot.get(k)
            return f"{100 * d['mean']:7.2f}" if d and "top1" in k else (
                f"{d['mean']:+6.2f}" if d else "     —"
            )

        def gpp(k: str) -> str:
            d = slot.get(k)
            return f"{d['mean']:+6.2f}" if d else "     —"

        print(
            f"{n:>4}  {g('base_unrestricted_top1')}  {g('base_calib_restricted_top1')}  "
            f"{g('prototype_top1')}  {g('partial_ft_unrestricted_top1')}  "
            f"{g('partial_ft_calib_restricted_top1')}  "
            f"{gpp('delta_coverage_pp')}  {gpp('delta_prototype_vs_base_restr_pp')}"
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
