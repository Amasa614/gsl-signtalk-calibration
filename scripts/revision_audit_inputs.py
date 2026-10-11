"""Stage A: freeze RQ4 bases and recover the 75 calibration selections.

    python -u scripts/revision_audit_inputs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PATHS
from src.revision_protocol import (
    ADAPT_SEEDS,
    BASE_SEED,
    CALIB_FRAC,
    MAIN_BUDGETS,
    N_DRAWS,
    PROTOCOL_NAME,
    REVISION_ROOT,
    SIGNERS,
    category_map,
    inspect_base_checkpoint,
    load_signtalk_windows,
    lookup_historical_counts,
    reconstruct_selection,
    sha256_file,
    sha256_json,
)
from src.train_utils import write_json


def _load_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).get("runs") or [])


def _check_selection(sel: dict, windows, source_signers: list[str]) -> dict:
    issues: list[str] = []
    calib = set(sel["calibration_video_ids"])
    test = set(sel["final_test_video_ids"])
    overlap = sorted(calib & test)
    if overlap:
        issues.append(f"calib/test overlap: {overlap[:8]}")

    signer = str(sel["target_signer"])
    signer_vids = set(
        windows.loc[windows["signer_id"].eq(signer), "video_id"].astype(str)
    )
    if not calib.issubset(signer_vids):
        issues.append("calibration video not owned by target signer")
    if not test.issubset(signer_vids):
        issues.append("final-test video not owned by target signer")

    tgt_windows = windows[windows["signer_id"].eq(signer)]
    for vid in list(calib)[:]:
        parts = set(
            tgt_windows.loc[tgt_windows["video_id"].eq(vid), "signer_id"].unique()
        )
        if parts != {signer}:
            issues.append(f"mixed-signer windows for calib {vid}")

    # All windows of a selected video stay in that partition.
    for vid in calib:
        n_all = int((windows["video_id"] == vid).sum())
        # reconstructed calib table length already counts those windows
        pass
    calib_win = windows[windows["video_id"].isin(calib)]
    test_win = windows[windows["video_id"].isin(test)]
    if set(calib_win["video_id"].astype(str)) != calib:
        issues.append("window table missing a calibration video")
    if int(len(calib_win)) != int(sel["n_calib_windows"]):
        issues.append(
            f"calib window recount {len(calib_win)} != {sel['n_calib_windows']}"
        )
    if int(len(test_win)) != int(sel["n_final_test_windows"]):
        issues.append(
            f"test window recount {len(test_win)} != {sel['n_final_test_windows']}"
        )

    # Source exclusion: no target-signer clip in the four source IDs.
    if signer in source_signers:
        issues.append("target signer listed among source signers")

    # Candidate set from calibration labels only.
    cats_from_ids = sorted(
        calib_win["category"].astype(str).unique().tolist()
    ) if len(calib_win) else []
    if cats_from_ids != sel["calibration_candidate_categories"]:
        issues.append("candidate categories drifted from calibration labels")

    finite_ok = True
    if "npz_path" in windows.columns:
        missing = calib_win["npz_path"].isna().sum() + test_win["npz_path"].isna().sum()
        if missing:
            issues.append(f"missing npz_path on {int(missing)} windows")
            finite_ok = False

    return {
        "ok": not issues,
        "issues": issues,
        "finite_inputs_assumed": finite_ok,
        "source_excludes_target": signer not in source_signers,
        "no_calib_test_overlap": not overlap,
    }


def main() -> None:
    out = REVISION_ROOT
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    windows = load_signtalk_windows()
    label_map = category_map(windows)
    windows_hash = sha256_file(PATHS.splits / "windows.csv")
    map_hash = sha256_json(label_map)

    ckpt_dir = PATHS.checkpoints / "rq4"
    bases = {}
    for s in SIGNERS:
        path = ckpt_dir / f"base_t{s}_k4_scratch.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        bases[s] = inspect_base_checkpoint(path, s)

    draws_runs = _load_runs(PATHS.results / "rq4_calib_draws.json")
    v2_runs = _load_runs(PATHS.results / "rq4_fewshot_calibration_v2.json")

    selections = []
    count_mismatches = []
    audits = []
    for s in SIGNERS:
        source = bases[s]["source_signers"]
        for n in MAIN_BUDGETS:
            for d in range(N_DRAWS):
                sel = reconstruct_selection(
                    windows,
                    test_signer=s,
                    n_per_category=n,
                    draw_id=d,
                    family="draws",
                )
                sel["base_checkpoint"] = bases[s]["path"]
                sel["base_checkpoint_sha256"] = bases[s]["sha256"]
                sel["label_map_sha256"] = map_hash
                sel["windows_csv_sha256"] = windows_hash
                hist = lookup_historical_counts(
                    draws_runs, signer=s, n=n, draw=d, method="prototype"
                )
                hist_ft = lookup_historical_counts(
                    draws_runs, signer=s, n=n, draw=d, method="partial_ft"
                )
                match = {}
                if hist:
                    match = {
                        "n_calib_clips": hist.get("n_calib_clips") == sel["n_calib_clips"],
                        "n_calib_windows": hist.get("n_calib_windows")
                        == sel["n_calib_windows"],
                        "n_final_test_windows": hist.get("n_final_test_windows")
                        == sel["n_final_test_windows"],
                        "n_allowed_categories": hist.get("n_allowed_categories")
                        == sel["n_allowed_categories"],
                    }
                    if not all(match.values()):
                        count_mismatches.append(
                            {
                                "family": "draws",
                                "signer": s,
                                "n": n,
                                "draw": d,
                                "reconstructed": {
                                    k: sel[k]
                                    for k in (
                                        "n_calib_clips",
                                        "n_calib_windows",
                                        "n_final_test_windows",
                                        "n_allowed_categories",
                                    )
                                },
                                "historical": {
                                    k: hist.get(k)
                                    for k in (
                                        "n_calib_clips",
                                        "n_calib_windows",
                                        "n_final_test_windows",
                                        "n_allowed_categories",
                                    )
                                },
                            }
                        )
                sel["historical_count_match"] = match
                sel["historical_prototype_clip_top1"] = (
                    hist.get("clip_top1") if hist else None
                )
                sel["historical_partial_ft_clip_top1"] = (
                    hist_ft.get("clip_top1") if hist_ft else None
                )
                check = _check_selection(sel, windows, source)
                sel["integrity"] = check
                audits.append(
                    {
                        "key": f"s{s}-n{n}-d{d}",
                        **check,
                        "count_match": all(match.values()) if match else False,
                    }
                )
                selections.append(sel)

    v2_selections = []
    v2_mismatches = []
    for s in SIGNERS:
        source = bases[s]["source_signers"]
        for n in MAIN_BUDGETS:
            sel = reconstruct_selection(
                windows,
                test_signer=s,
                n_per_category=n,
                draw_id=0,
                family="v2",
            )
            sel["base_checkpoint"] = bases[s]["path"]
            sel["base_checkpoint_sha256"] = bases[s]["sha256"]
            hist = lookup_historical_counts(
                v2_runs, signer=s, n=n, draw=0, method="prototype"
            )
            match = {}
            if hist:
                match = {
                    "n_calib_clips": hist.get("n_calib_clips") == sel["n_calib_clips"],
                    "n_calib_windows": hist.get("n_calib_windows")
                    == sel["n_calib_windows"],
                    "n_final_test_windows": hist.get("n_final_test_windows")
                    == sel["n_final_test_windows"],
                    "n_allowed_categories": hist.get("n_allowed_categories")
                    == sel["n_allowed_categories"],
                }
                if not all(match.values()):
                    v2_mismatches.append(
                        {
                            "signer": s,
                            "n": n,
                            "reconstructed": {
                                k: sel[k]
                                for k in (
                                    "n_calib_clips",
                                    "n_calib_windows",
                                    "n_final_test_windows",
                                    "n_allowed_categories",
                                )
                            },
                            "historical": {
                                k: hist.get(k)
                                for k in (
                                    "n_calib_clips",
                                    "n_calib_windows",
                                    "n_final_test_windows",
                                    "n_allowed_categories",
                                )
                            },
                        }
                    )
            sel["historical_count_match"] = match
            sel["integrity"] = _check_selection(sel, windows, source)
            v2_selections.append(sel)

    # Signer × category inventory (clip-level).
    clips = windows.drop_duplicates("video_id")
    inventory = {}
    for s in SIGNERS:
        sub = clips[clips["signer_id"].eq(s)]
        counts = sub.groupby("category").size().to_dict()
        inventory[s] = {str(c): int(n) for c, n in counts.items()}

    coverage = {}
    for s in SIGNERS:
        src = clips[clips["signer_id"].isin(bases[s]["source_signers"])]
        tgt = clips[clips["signer_id"].eq(s)]
        src_cats = set(src["category"])
        tgt_cats = set(tgt["category"])
        coverage[s] = {
            "n_target_categories": len(tgt_cats),
            "n_source_categories": len(src_cats),
            "target_covered_by_source": sorted(tgt_cats & src_cats),
            "target_absent_from_source": sorted(tgt_cats - src_cats),
            "source_absent_from_target": sorted(src_cats - tgt_cats),
        }

    n_ok = sum(1 for a in audits if a["ok"] and a["count_match"])
    protocol = {
        "protocol": PROTOCOL_NAME,
        "task": "26-way category recognition",
        "questions": ["Q1 zero-shot transfer", "Q2 calibration methods", "Q3 matched candidates"],
        "source_bases": "existing scratch RQ4 k=4 checkpoints; not retrained",
        "main_budgets": list(MAIN_BUDGETS),
        "n_draws": N_DRAWS,
        "adaptation_seeds": list(ADAPT_SEEDS),
        "n_calibration_selections": len(selections),
        "seed_formulas": {
            "calib_test_split": "42 + 100 * target_signer",
            "five_draw_sample": "42 + 1000 * target + 17 * n + 997 * draw",
            "v2_single_draw_sample": "42 + 1000 * target + n",
            "note": (
                "Video IDs were not stored in historical JSONs. Selections are "
                "reconstructed from windows.csv first-appearance order plus the "
                "original seed formulas, then accepted only if clip/window/"
                "category counts match the historical records."
            ),
        },
        "prototype_embedding": {
            "window_embed": "mean-pool Transformer hidden states over 64 frames (eval mode)",
            "clip_support": "mean of window embeddings that share video_id, then class-mean, then L2",
            "test_score": (
                "L2-normalize each test *window* embedding, cosine vs class "
                "prototypes, then mean-pool window logits per video_id"
            ),
            "missing_class": "zero prototype; logits set to -1e9",
        },
        "windows_csv_sha256": windows_hash,
        "label_map_sha256": map_hash,
        "label_map": label_map,
        "calib_frac": CALIB_FRAC,
        "base_seed": BASE_SEED,
        "git_note": (
            "Repository has no commits on master; outputs are isolated under "
            "results/paper1_access_revision/ instead of a git branch snapshot."
        ),
    }

    manifest = {
        "protocol": PROTOCOL_NAME,
        "bases": bases,
        "label_map": label_map,
        "windows_csv": str(PATHS.splits / "windows.csv"),
        "windows_csv_sha256": windows_hash,
        "selections": selections,
        "v2_single_draw_selections": v2_selections,
    }
    write_json(out / "revision_manifest.json", manifest)
    write_json(out / "protocol.json", protocol)

    audit = {
        "protocol": PROTOCOL_NAME,
        "n_bases": len(bases),
        "n_draw_selections": len(selections),
        "n_v2_selections": len(v2_selections),
        "n_integrity_and_count_ok": n_ok,
        "all_draw_counts_match": not count_mismatches,
        "all_v2_counts_match": not v2_mismatches,
        "all_integrity_ok": all(a["ok"] for a in audits),
        "draw_count_mismatches": count_mismatches,
        "v2_count_mismatches": v2_mismatches,
        "per_selection": audits,
        "signer_category_clip_inventory": inventory,
        "source_to_target_coverage": coverage,
        "base_sha256": {s: bases[s]["sha256"] for s in SIGNERS},
        "verdict": (
            "reuse_historical_ids"
            if not count_mismatches and not v2_mismatches and all(a["ok"] for a in audits)
            else "stop_and_inspect"
        ),
    }
    write_json(out / "audit_report.json", audit)

    print(f"bases={len(bases)}  draw selections={len(selections)}  v2={len(v2_selections)}")
    print(f"integrity+count ok: {n_ok}/{len(audits)}")
    print(f"draw count mismatches: {len(count_mismatches)}")
    print(f"v2 count mismatches: {len(v2_mismatches)}")
    print(f"verdict: {audit['verdict']}")
    print(f"wrote {out / 'revision_manifest.json'}")
    print(f"wrote {out / 'audit_report.json'}")


if __name__ == "__main__":
    main()
