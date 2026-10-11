"""RQ2 simulation: guided recording vs matched-size random (§3.5).

Leave-signer-out on SignTalk-Gh *before* recruiting anyone. Trains on 1–2 seed
signers, treats non-test leftover signers as a simulated recording pool,
retrains after each addition, and evaluates on a held-out test signer.

Protocols:

  --protocol session   (revised default recommendation)
      Simulate one donor signer's recording session. Rank units (categories or
      sentences) by seed-signer *error* (or uncertainty), add ``budget`` units
      per round from that donor only. Random arm fills the same budget from a
      shuffled list. Train/eval default to 26-way category ID so Δ_AL is
      measurable under signer-independence.

  --protocol category  (legacy)
      Rank categories across all donors; each round adds ``budget`` categories.

  --protocol signer    (legacy)
      Rank remaining full signers; each round adds one signer.

Neither arm uses the test signer for training, scoring, or model selection.

Resume (session protocol, default on)
------------------------------------
Existing ``checkpoints/rq2-session-*.pt`` rounds are loaded instead of re-trained.
Progress is also written to ``*.partial.json`` after each round so a killed job
can continue. Pass ``--fresh`` to ignore checkpoints and retrain everything.
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
from torch.utils.data import DataLoader

from src.active_learning import aggregate_priority, order_units, score_loader
from src.config import CFG, FEATURE_DIM, PATHS
from src.dataset import LabeledWindowDataset
from src.finetune_loop import run_finetune
from src.metrics import predict_loader, score_clip_category, summarize_predictions
from src.model import build_pose_model
from src.splits import load_label_map
from src.train_utils import get_device, load_checkpoint, set_seed, write_json


def _ids(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _labeled_windows() -> pd.DataFrame:
    path = PATHS.splits / "windows.csv"
    if not path.exists():
        raise FileNotFoundError("Run scripts/02_build_windows.py first.")
    df = pd.read_csv(path)
    if "source" in df.columns:
        df = df[df["source"].astype(str).str.lower().eq("signtalk")]
    if "split" in df.columns:
        df = df[df["split"] != "ssl"]
    return df.reset_index(drop=True)


def _split_seed_diag(
    seed_table: pd.DataFrame,
    frac: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a fraction of seed *videos* for error-guided ranking."""
    if frac <= 0 or len(seed_table) == 0:
        return seed_table.reset_index(drop=True), seed_table.iloc[0:0].copy()
    vids = seed_table["video_id"].astype(str).unique().tolist()
    rng.shuffle(vids)
    n_diag = max(1, int(round(len(vids) * frac)))
    if n_diag >= len(vids):
        n_diag = max(1, len(vids) // 5)
    diag_vids = set(vids[:n_diag])
    train = seed_table[~seed_table["video_id"].astype(str).isin(diag_vids)].reset_index(drop=True)
    diag = seed_table[seed_table["video_id"].astype(str).isin(diag_vids)].reset_index(drop=True)
    if len(train) == 0:
        return seed_table.reset_index(drop=True), seed_table.iloc[0:0].copy()
    return train, diag


def _slim(stats: dict) -> dict:
    keep = (
        "n",
        "top1",
        "top5",
        "mean_entropy",
        "majority",
        "per_category_top1",
        "per_category_error",
        "per_signer_top1",
        "clip",
    )
    return {k: stats[k] for k in keep if k in stats}


def _score_table(
    model,
    table: pd.DataFrame,
    label_map,
    device,
    method: str,
    batch_size: int,
    label_col: str,
) -> pd.DataFrame:
    ds = LabeledWindowDataset(table, label_map=label_map, label_col=label_col)
    if len(ds) == 0:
        return pd.DataFrame()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    # score_loader always needs an uncertainty column; for pure error ranking we
    # still run entropy so the DataFrame schema stays consistent.
    score_method = method if method in {"entropy", "margin", "mc_dropout"} else "entropy"
    return score_loader(model, loader, device, method=score_method)


def _eval_table(
    model,
    table: pd.DataFrame,
    label_map,
    device,
    batch_size: int,
    label_col: str,
) -> dict:
    ds = LabeledWindowDataset(table, label_map=label_map, label_col=label_col)
    if len(ds) == 0:
        return {"n": 0, "top1": 0.0, "top5": 0.0}
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = predict_loader(model, loader, device)
    stats = summarize_predictions(out)
    if label_col == "category":
        id_to_cat = {int(v): str(k) for k, v in label_map.items()}
        stats["clip"] = score_clip_category(out, id_to_cat)
        gold = [str(c) for c in out["category"]]
        from src.metrics import majority_accuracy

        stats["majority"] = majority_accuracy(gold)
    return stats


def _pick_label_map(task: str) -> tuple[dict[str, int], str]:
    if task == "category":
        path = PATHS.category_map
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Rebuild windows or write category_map.json "
                "(26 thematic categories)."
            )
        return load_label_map(path), "category"
    return load_label_map(), "sentence_id"


def _session_config_key(args, *, seed_signers, test_signers, donor_list, pretrained) -> dict:
    return {
        "protocol": "session",
        "task": args.task,
        "unit": args.unit,
        "rank_by": args.rank_by,
        "seed_signers": sorted(seed_signers),
        "test_signers": sorted(test_signers),
        "donor_signers": list(donor_list),
        "budget": int(args.budget),
        "max_rounds": int(args.max_rounds or 0),
        "diag_frac": float(args.diag_frac),
        "n_random_orders": int(args.n_random_orders),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "pretrained": str(pretrained) if pretrained else None,
    }


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[rq2] warning: could not read {path}: {exc}")
        return None


def _load_finetuned_model(path: Path, n_classes: int, device):
    model = build_pose_model("transformer", feature_dim=FEATURE_DIM, n_classes=n_classes).to(device)
    ckpt = load_checkpoint(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    return model


def _train_or_resume_round(
    *,
    train: pd.DataFrame,
    label_map: dict,
    device,
    pretrained,
    args,
    out: Path,
    tag: str,
    label_col: str,
    resume: bool,
) -> dict:
    """Train a round, or reload an existing checkpoint when ``resume`` is on."""
    if resume and out.exists():
        print(f"  [resume] skip train -> load {out.name}")
        model = _load_finetuned_model(out, len(label_map), device)
        return {"model": model, "n_train": int(len(train)), "best_val_acc": 0.0, "resumed": True}
    trained = run_finetune(
        train,
        label_map,
        device,
        val_table=None,
        pretrained=pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        out=out,
        tag=tag,
        label_col=label_col,
    )
    trained["resumed"] = False
    return trained


def _run_legacy_arms(
    *,
    protocol: str,
    pool: list[str],
    random_order: list[str],
    seed_table: pd.DataFrame,
    donor_table: pd.DataFrame,
    windows: pd.DataFrame,
    seed_signers: set[str],
    test_table: pd.DataFrame,
    label_map: dict,
    label_col: str,
    pretrained,
    device,
    args,
) -> dict:
    results = {"al": [], "random": []}
    for arm, is_al in (("al", True), ("random", False)):
        included_groups: list[str] = []
        remaining = list(pool) if is_al else list(random_order)
        round_idx = 0
        while True:
            if protocol == "category":
                train = pd.concat(
                    [seed_table, donor_table[donor_table["category"].astype(str).isin(included_groups)]],
                    ignore_index=True,
                )
            else:
                train_signers = seed_signers | set(included_groups)
                train = windows[windows["signer_id"].astype(str).isin(train_signers)].reset_index(drop=True)

            tag = f"rq2-{protocol}-{arm}-r{round_idx}"
            out = PATHS.checkpoints / f"{tag}.pt"
            trained = run_finetune(
                train,
                label_map,
                device,
                val_table=None,
                pretrained=pretrained,
                epochs=args.epochs,
                batch_size=args.batch_size,
                out=out,
                tag=tag,
                label_col=label_col,
            )
            model = trained["model"]
            stats = _eval_table(model, test_table, label_map, device, args.batch_size, label_col)
            results[arm].append(
                {
                    "round": round_idx,
                    "arm": arm,
                    "included": list(included_groups),
                    "n_train_windows": int(trained["n_train"]),
                    "test": _slim(stats),
                }
            )
            print(
                f"[{arm} r{round_idx}] included={included_groups or 'S0 only'}  "
                f"test top-1={100 * stats.get('top1', 0):.2f}%  n={stats.get('n')}"
            )

            leftover = [g for g in remaining if g not in included_groups]
            if not leftover:
                break
            n_add = 1 if protocol == "signer" else min(args.budget, len(leftover))
            if is_al:
                cand = donor_table if protocol == "category" else windows
                if protocol == "category":
                    cand = cand[cand["category"].astype(str).isin(leftover)]
                else:
                    cand = cand[cand["signer_id"].astype(str).isin(leftover)]
                scores = _score_table(
                    model, cand, label_map, device, args.rank_by, args.batch_size, label_col
                )
                by = "category" if protocol == "category" else "signer_id"
                ranked = (
                    aggregate_priority(scores, by=by, rank_by="uncertainty")
                    if len(scores)
                    else pd.DataFrame()
                )
                pick = ranked[by].astype(str).tolist()[:n_add] if len(ranked) else leftover[:n_add]
            else:
                pick = leftover[:n_add]
            if not pick:
                break
            included_groups.extend([g for g in pick if g not in included_groups])
            remaining = leftover
            round_idx += 1
            if args.max_rounds and round_idx >= args.max_rounds:
                break
    return results


def _run_session_arms(
    *,
    donor_id: str,
    donor_table: pd.DataFrame,
    seed_train: pd.DataFrame,
    seed_diag: pd.DataFrame,
    test_table: pd.DataFrame,
    label_map: dict,
    label_col: str,
    unit_col: str,
    rank_by: str,
    pretrained,
    device,
    args,
    rng: np.random.Generator,
    resume: bool = True,
    on_progress=None,
    cached: dict | None = None,
) -> dict:
    """Within one donor: fill a fixed unit budget by guided vs random order.

    Round 0 (seed-only) is trained **once** and shared by both arms so Δ_AL at
    round 0 is identically zero. The random arm may use several shuffled
    recording orders (``args.n_random_orders``); reported random metrics are
    the mean across those orders at each round.

    With ``resume=True``, existing ``checkpoints/rq2-session-*.pt`` files are
    loaded instead of re-trained. Optional ``cached`` may supply a finished
    donor block from a partial JSON.
    """
    pool = sorted(donor_table[unit_col].astype(str).unique().tolist())
    if not pool:
        raise RuntimeError(f"donor {donor_id} has no {unit_col} units")
    n_orders = max(1, int(getattr(args, "n_random_orders", 1)))

    # Always draw the same number of shuffles so RNG stays aligned across donors
    # even when a prior donor is loaded from cache.
    if cached and cached.get("status") == "done" and cached.get("arms"):
        for _ in range(n_orders):
            order = list(pool)
            rng.shuffle(order)
        print(f"[session donor={donor_id}] resume: using cached completed donor")
        return cached["arms"]

    if cached and cached.get("random_orders") and len(cached["random_orders"]) == n_orders:
        random_orders = [list(o) for o in cached["random_orders"]]
        # Consume RNG the same way a fresh draw would.
        for _ in range(n_orders):
            order = list(pool)
            rng.shuffle(order)
        print(f"[session donor={donor_id}] resume: reusing saved random_orders")
    else:
        random_orders = []
        for _ in range(n_orders):
            order = list(pool)
            rng.shuffle(order)
            random_orders.append(order)

    def _emit(partial: dict) -> None:
        if on_progress is not None:
            on_progress(donor_id, partial)

    progress_state: dict = {
        "status": "in_progress",
        "random_orders": random_orders,
        "shared_seed_round0": None,
        "al": [],
        "random_runs": [[] for _ in range(n_orders)],
    }

    # --- Round 0: single shared seed checkpoint ---
    tag0 = f"rq2-session-{donor_id}-seed-r0"
    out0 = PATHS.checkpoints / f"{tag0}.pt"
    if resume and out0.exists():
        print(f"[session donor={donor_id}] resume: load shared seed model (round 0)...")
    else:
        print(f"[session donor={donor_id}] training shared seed model (round 0)...")
    trained0 = _train_or_resume_round(
        train=seed_train,
        label_map=label_map,
        device=device,
        pretrained=pretrained,
        args=args,
        out=out0,
        tag=tag0,
        label_col=label_col,
        resume=resume,
    )
    model0 = trained0["model"]
    stats0 = _eval_table(model0, test_table, label_map, device, args.batch_size, label_col)
    clip0 = (stats0.get("clip") or {}).get("clip_top1")
    maj0 = stats0.get("majority")
    print(
        f"[session donor={donor_id} SHARED r0] units=0  "
        f"top-1={100 * stats0.get('top1', 0):.2f}%"
        + (f"  clip={100 * clip0:.2f}%" if clip0 is not None else "")
        + (f"  majority={100 * maj0:.2f}%" if maj0 is not None else "")
    )
    row0 = {
        "round": 0,
        "arm": "shared_seed",
        "donor_signer": donor_id,
        "included": [],
        "n_train_windows": int(trained0["n_train"]),
        "test": _slim(stats0),
        "shared_seed_checkpoint": str(out0),
    }
    progress_state["shared_seed_round0"] = row0
    _emit(progress_state)

    def _run_arm_from_round1(arm: str, is_al: bool, random_order: list[str] | None) -> list[dict]:
        rows = [
            {
                **row0,
                "arm": arm,
                "random_order_id": None if is_al else getattr(args, "_current_order_id", 0),
            }
        ]
        included: list[str] = []
        leftover_random = list(random_order) if random_order else []
        model = model0
        round_idx = 0
        while True:
            leftover = [u for u in pool if u not in included]
            if not leftover:
                break
            n_add = min(args.budget, len(leftover))
            if is_al:
                if rank_by == "error":
                    diag = seed_diag if len(seed_diag) else seed_train
                    scores = _score_table(
                        model, diag, label_map, device, "entropy", args.batch_size, label_col
                    )
                    ranked = (
                        aggregate_priority(scores, by=unit_col, rank_by="error")
                        if len(scores)
                        else pd.DataFrame()
                    )
                else:
                    cand = donor_table[donor_table[unit_col].astype(str).isin(leftover)]
                    scores = _score_table(
                        model, cand, label_map, device, rank_by, args.batch_size, label_col
                    )
                    ranked = (
                        aggregate_priority(scores, by=unit_col, rank_by="uncertainty")
                        if len(scores)
                        else pd.DataFrame()
                    )
                ordered = order_units(leftover, ranked, unit_col) if len(ranked) else leftover
                pick = ordered[:n_add]
            else:
                pick = [u for u in leftover_random if u in leftover][:n_add]
                if len(pick) < n_add:
                    pick = leftover[:n_add]
            if not pick:
                break
            included.extend(pick)
            round_idx += 1

            add = donor_table[donor_table[unit_col].astype(str).isin(included)]
            train = pd.concat([seed_train, add], ignore_index=True)
            tag = f"rq2-session-{donor_id}-{arm}-r{round_idx}"
            if not is_al and random_order is not None:
                oid = getattr(args, "_current_order_id", 0)
                tag = f"{tag}-o{oid}"
            out = PATHS.checkpoints / f"{tag}.pt"
            trained = _train_or_resume_round(
                train=train,
                label_map=label_map,
                device=device,
                pretrained=pretrained,
                args=args,
                out=out,
                tag=tag,
                label_col=label_col,
                resume=resume,
            )
            model = trained["model"]
            stats = _eval_table(model, test_table, label_map, device, args.batch_size, label_col)
            clip = (stats.get("clip") or {}).get("clip_top1")
            maj = stats.get("majority")
            print(
                f"[session donor={donor_id} {arm} r{round_idx}] "
                f"units={len(included)}  "
                f"top-1={100 * stats.get('top1', 0):.2f}%"
                + (f"  clip={100 * clip:.2f}%" if clip is not None else "")
                + (f"  majority={100 * maj:.2f}%" if maj is not None else "")
            )
            rows.append(
                {
                    "round": round_idx,
                    "arm": arm,
                    "donor_signer": donor_id,
                    "included": list(included),
                    "n_train_windows": int(trained["n_train"]),
                    "test": _slim(stats),
                    "shared_seed_checkpoint": str(out0),
                    "checkpoint": str(out),
                    "resumed": bool(trained.get("resumed")),
                }
            )
            if is_al:
                progress_state["al"] = list(rows)
            else:
                oid = int(getattr(args, "_current_order_id", 0))
                progress_state["random_runs"][oid] = list(rows)
            _emit(progress_state)
            if args.max_rounds and round_idx >= args.max_rounds:
                break
        return rows

    al_rows: list[dict] = []
    random_runs: list[list[dict]] = [[] for _ in range(n_orders)]

    cached_al = (cached or {}).get("al") if cached else None
    if resume and cached_al and args.max_rounds and len(cached_al) >= int(args.max_rounds) + 1:
        al_rows = list(cached_al)
        progress_state["al"] = al_rows
        print(f"[session donor={donor_id}] resume: cached AL arm ({len(al_rows)} rows)")
    else:
        al_rows = _run_arm_from_round1("al", True, None)
        progress_state["al"] = al_rows

    cached_rr = (cached or {}).get("random_runs") if cached else None
    for oid, order in enumerate(random_orders):
        args._current_order_id = oid
        if (
            resume
            and cached_rr
            and oid < len(cached_rr)
            and len(cached_rr[oid]) >= len(al_rows)
            and len(cached_rr[oid]) > 0
        ):
            print(
                f"[session donor={donor_id}] resume: cached random order "
                f"{oid + 1}/{n_orders}"
            )
            random_runs[oid] = list(cached_rr[oid])
            progress_state["random_runs"][oid] = random_runs[oid]
            continue
        print(f"[session donor={donor_id}] random order {oid + 1}/{n_orders}")
        random_runs[oid] = _run_arm_from_round1("random", False, order)
        progress_state["random_runs"][oid] = random_runs[oid]

    random_avg = _average_random_runs(random_runs, n_orders=n_orders, out0=str(out0))
    arms = {"al": al_rows, "random": random_avg, "shared_seed_round0": row0}
    _emit(
        {
            "status": "done",
            "random_orders": random_orders,
            "shared_seed_round0": row0,
            "al": al_rows,
            "random_runs": random_runs,
            "arms": arms,
        }
    )
    return arms


def _mean_numeric_test(tests: list[dict]) -> dict:
    """Average scalar (and nested-clip scalar) fields across random-order runs."""
    out: dict = {}
    scalar_keys = ("n", "top1", "top5", "mean_entropy", "majority")
    for k in scalar_keys:
        vals = [float(t[k]) for t in tests if isinstance(t.get(k), (int, float))]
        if vals:
            out[k] = float(np.mean(vals))
    clip_dicts = [t["clip"] for t in tests if isinstance(t.get("clip"), dict)]
    if clip_dicts:
        clip_out = {}
        keys = set().union(*(c.keys() for c in clip_dicts))
        for ck in keys:
            vals = [float(c[ck]) for c in clip_dicts if isinstance(c.get(ck), (int, float))]
            if vals:
                clip_out[ck] = float(np.mean(vals))
        if clip_out:
            out["clip"] = clip_out
    return out


def _average_random_runs(random_runs: list[list[dict]], *, n_orders: int, out0: str) -> list[dict]:
    max_r = max((len(r) for r in random_runs), default=0)
    random_avg = []
    for i in range(max_r):
        pts = [run[i] for run in random_runs if i < len(run)]
        if not pts:
            continue
        if i == 0:
            random_avg.append({**pts[0], "arm": "random", "n_random_orders": n_orders})
            continue
        random_avg.append(
            {
                "round": i,
                "arm": "random",
                "donor_signer": pts[0].get("donor_signer"),
                "included": pts[0].get("included"),
                "n_train_windows": int(np.mean([p["n_train_windows"] for p in pts])),
                "test": _mean_numeric_test([p.get("test") or {} for p in pts]),
                "n_random_orders": n_orders,
                "shared_seed_checkpoint": out0,
            }
        )
    return random_avg


def _delta_rows(results: dict) -> list[dict]:
    n_rounds = min(len(results["al"]), len(results["random"]))
    delta = []
    for i in range(n_rounds):
        al = results["al"][i]["test"]
        rnd = results["random"][i]["test"]
        row = {
            "round": i,
            "delta_al": al.get("top1", 0.0) - rnd.get("top1", 0.0),
            "al": al.get("top1", 0.0),
            "random": rnd.get("top1", 0.0),
            "delta_al_top5": al.get("top5", 0.0) - rnd.get("top5", 0.0),
            "al_top5": al.get("top5", 0.0),
            "random_top5": rnd.get("top5", 0.0),
        }
        al_clip = (al.get("clip") or {}).get("clip_top1")
        rnd_clip = (rnd.get("clip") or {}).get("clip_top1")
        if al_clip is not None and rnd_clip is not None:
            row["delta_al_clip"] = al_clip - rnd_clip
            row["al_clip"] = al_clip
            row["random_clip"] = rnd_clip
        delta.append(row)
    return delta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", type=str, default=str(PATHS.ssl_encoder), help="ssl_encoder.pt or 'none'")
    p.add_argument(
        "--protocol",
        choices=("session", "category", "signer"),
        default="session",
        help="session = within-donor recording budget (revised RQ2)",
    )
    p.add_argument(
        "--task",
        choices=("category", "sentence"),
        default="category",
        help="classification grain for train/eval (category recommended)",
    )
    p.add_argument(
        "--unit",
        choices=("category", "sentence"),
        default="category",
        help="recording unit selected by the budget (session/category protocols)",
    )
    p.add_argument(
        "--rank-by",
        choices=("error", "entropy", "margin", "mc_dropout"),
        default="error",
        help="error = seed-signer diagnostic error (preferred); else uncertainty on candidates",
    )
    # Back-compat alias used by older commands.
    p.add_argument("--method", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--seed-signers", type=str, default="1,2", help="S0, 1–2 signers (§3.5.1)")
    p.add_argument("--test-signers", type=str, default="5")
    p.add_argument(
        "--donor-signer",
        type=str,
        default="",
        help="session protocol: single donor id (default = run each leftover donor)",
    )
    p.add_argument("--budget", type=int, default=4, help="units added per round")
    p.add_argument("--max-rounds", type=int, default=0, help="0 = until donor units exhausted")
    p.add_argument(
        "--n-random-orders",
        type=int,
        default=3,
        help="session protocol: number of random recording permutations to average",
    )
    p.add_argument(
        "--diag-frac",
        type=float,
        default=0.2,
        help="fraction of seed videos held out for error ranking (session/error)",
    )
    p.add_argument("--epochs", type=int, default=15, help="retrain epochs per round (paper RQ2 = 15)")
    p.add_argument("--batch-size", type=int, default=CFG.finetune.batch_size)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="skip rounds whose checkpoints already exist (default on)",
    )
    p.add_argument(
        "--fresh",
        dest="resume",
        action="store_false",
        help="ignore existing checkpoints / partial JSON and retrain everything",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="final results JSON (default: results/rq2_session_{task}_{unit}_{rank_by}.json)",
    )
    args = p.parse_args()

    if args.method:
        args.rank_by = args.method

    set_seed(args.seed)
    device = get_device()
    use_pretrained = str(args.pretrained).lower() not in {"none", "scratch", ""}
    pretrained = Path(args.pretrained) if use_pretrained else None
    if use_pretrained and not pretrained.exists():
        print(f"[rq2] no SSL checkpoint at {pretrained}; running from scratch")
        pretrained = None

    label_map, label_col = _pick_label_map(args.task)
    unit_col = "category" if args.unit == "category" else "sentence_id"
    windows = _labeled_windows()
    seed_signers = set(_ids(args.seed_signers))
    test_signers = set(_ids(args.test_signers))
    if seed_signers & test_signers:
        raise ValueError("seed and test signers overlap")

    all_signers = set(windows["signer_id"].astype(str).unique())
    leftover_donors = sorted(all_signers - seed_signers - test_signers)
    if not leftover_donors:
        raise RuntimeError("need at least one non-seed, non-test signer to simulate recording")

    if args.donor_signer:
        donor_list = _ids(args.donor_signer)
        bad = [d for d in donor_list if d not in leftover_donors]
        if bad:
            raise ValueError(f"--donor-signer {bad} not in available donors {leftover_donors}")
    else:
        donor_list = leftover_donors

    seed_table = windows[windows["signer_id"].astype(str).isin(seed_signers)].reset_index(drop=True)
    test_table = windows[windows["signer_id"].astype(str).isin(test_signers)].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)

    print(
        f"[rq2] protocol={args.protocol}  task={args.task}  unit={args.unit}  "
        f"rank_by={args.rank_by}  S0={sorted(seed_signers)}  "
        f"donors={donor_list}  test={sorted(test_signers)}  "
        f"classes={len(label_map)}  pretrained={pretrained is not None}  "
        f"resume={args.resume}"
    )

    if args.protocol == "session":
        seed_train, seed_diag = _split_seed_diag(seed_table, args.diag_frac, rng)
        print(
            f"[rq2] seed train windows={len(seed_train)}  "
            f"diag windows={len(seed_diag)}  (diag_frac={args.diag_frac})"
        )
        cfg_key = _session_config_key(
            args,
            seed_signers=seed_signers,
            test_signers=test_signers,
            donor_list=donor_list,
            pretrained=pretrained,
        )
        out_json = (
            Path(args.out)
            if args.out
            else PATHS.results / f"rq2_session_{args.task}_{args.unit}_{args.rank_by}.json"
        )
        partial_json = out_json.with_suffix(".partial.json")
        progress: dict = {
            "config": cfg_key,
            "seed_diag_video_ids": sorted(seed_diag["video_id"].astype(str).unique().tolist()),
            "per_donor": {},
        }
        if args.resume:
            prev = _load_json(partial_json) or _load_json(out_json)
            if prev and prev.get("config") == cfg_key and isinstance(prev.get("per_donor"), dict):
                progress["per_donor"] = prev["per_donor"]
                print(
                    f"[rq2] resume from {partial_json.name if partial_json.exists() else out_json.name} "
                    f"({sum(1 for d in progress['per_donor'].values() if d.get('status')=='done')} donors done)"
                )
            elif prev and prev.get("config") != cfg_key:
                print("[rq2] partial/final JSON config mismatch; keeping checkpoint resume only")

        def _save_progress(donor_id: str, partial: dict) -> None:
            progress["per_donor"][str(donor_id)] = partial
            # Rebuild a best-effort delta snapshot for finished donors.
            all_delta = []
            per_out = {}
            for did, block in progress["per_donor"].items():
                if block.get("status") == "done" and block.get("arms"):
                    delta = _delta_rows(block["arms"])
                    per_out[did] = {"arms": block["arms"], "delta_al": delta}
                    all_delta.extend([{**d, "donor_signer": did} for d in delta])
            snapshot = {
                **cfg_key,
                "n_classes": len(label_map),
                "shared_seed_round0": True,
                "config": cfg_key,
                "seed_diag_video_ids": progress["seed_diag_video_ids"],
                "per_donor": progress["per_donor"],
                "delta_al": all_delta,
                "partial": True,
            }
            write_json(partial_json, snapshot)

        per_donor = {}
        all_delta = []
        for donor_id in donor_list:
            donor_table = windows[windows["signer_id"].astype(str).eq(donor_id)].reset_index(drop=True)
            print(f"\n=== session donor={donor_id}  units={donor_table[unit_col].nunique()} ===")
            cached = progress["per_donor"].get(str(donor_id)) if args.resume else None
            results = _run_session_arms(
                donor_id=donor_id,
                donor_table=donor_table,
                seed_train=seed_train,
                seed_diag=seed_diag,
                test_table=test_table,
                label_map=label_map,
                label_col=label_col,
                unit_col=unit_col,
                rank_by=args.rank_by,
                pretrained=pretrained,
                device=device,
                args=args,
                rng=rng,
                resume=args.resume,
                on_progress=_save_progress,
                cached=cached,
            )
            delta = _delta_rows(results)
            per_donor[donor_id] = {"arms": results, "delta_al": delta}
            progress["per_donor"][str(donor_id)] = {
                "status": "done",
                "arms": results,
                "al": results.get("al"),
                "random_orders": (cached or {}).get("random_orders")
                or progress["per_donor"].get(str(donor_id), {}).get("random_orders"),
                "delta_al": delta,
            }
            all_delta.extend([{**d, "donor_signer": donor_id} for d in delta])
            _save_progress(str(donor_id), progress["per_donor"][str(donor_id)])
            print(f"\nDelta_AL donor={donor_id} (guided - random, matched budget):")
            for d in delta:
                extra = ""
                if "delta_al_clip" in d:
                    extra = f"  clip {100 * d['delta_al_clip']:+.2f} pp"
                print(
                    f"  round {d['round']}: {100 * d['delta_al']:+.2f} pp  "
                    f"(AL {100 * d['al']:.2f}% vs rand {100 * d['random']:.2f}%)"
                    f"{extra}"
                )

        payload = {
            "protocol": "session",
            "task": args.task,
            "unit": args.unit,
            "rank_by": args.rank_by,
            "seed_signers": sorted(seed_signers),
            "test_signers": sorted(test_signers),
            "donor_signers": donor_list,
            "budget": args.budget,
            "diag_frac": args.diag_frac,
            "n_random_orders": args.n_random_orders,
            "shared_seed_round0": True,
            "epochs": args.epochs,
            "n_classes": len(label_map),
            "pretrained": str(pretrained) if pretrained else None,
            "seed": args.seed,
            "max_rounds": int(args.max_rounds or 0),
            "per_donor": per_donor,
            "delta_al": all_delta,
            "config": cfg_key,
        }
    else:
        # Legacy multi-donor category / signer protocols.
        donor_table = windows[windows["signer_id"].astype(str).isin(donor_list)].reset_index(drop=True)
        if args.protocol == "category":
            pool = sorted(donor_table[unit_col].astype(str).unique().tolist())
        else:
            pool = list(donor_list)
            unit_col = "signer_id"
        random_order = list(pool)
        rng.shuffle(random_order)
        # Legacy used --method for uncertainty; map error -> entropy on candidates.
        if args.rank_by == "error":
            print("[rq2] legacy category/signer protocol: rank-by=error falls back to entropy on candidates")
            args.rank_by = "entropy"
        results = _run_legacy_arms(
            protocol=args.protocol,
            pool=pool,
            random_order=random_order,
            seed_table=seed_table,
            donor_table=donor_table,
            windows=windows,
            seed_signers=seed_signers,
            test_table=test_table,
            label_map=label_map,
            label_col=label_col,
            pretrained=pretrained,
            device=device,
            args=args,
        )
        delta = _delta_rows(results)
        payload = {
            "protocol": args.protocol,
            "task": args.task,
            "unit": args.unit if args.protocol == "category" else "signer",
            "rank_by": args.rank_by,
            "seed_signers": sorted(seed_signers),
            "test_signers": sorted(test_signers),
            "donor_signers": donor_list,
            "budget": args.budget,
            "epochs": args.epochs,
            "n_classes": len(label_map),
            "pretrained": str(pretrained) if pretrained else None,
            "arms": results,
            "delta_al": delta,
        }
        out_json = PATHS.results / f"rq2_simulation_{args.protocol}_{args.rank_by}.json"
        print(f"\nDelta_AL by round (guided - random, matched cost):")
        for d in delta:
            print(
                f"  round {d['round']}: {100 * d['delta_al']:+.2f} pp  "
                f"(AL {100 * d['al']:.2f}% vs rand {100 * d['random']:.2f}%)"
            )

    write_json(out_json, payload)
    print(f"wrote {out_json}")
    if args.protocol == "session":
        partial_path = out_json.with_suffix(".partial.json")
        if partial_path.exists():
            try:
                partial_path.unlink()
                print(f"removed {partial_path}")
            except OSError as exc:
                print(f"[rq2] warning: could not remove partial file: {exc}")


if __name__ == "__main__":
    main()
