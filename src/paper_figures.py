"""Publication figures and tables for the methodology and results sections.

Methodology schematics (Figs 1–6) do not need experimental outputs.
Results figures (Figs 7–8 plus training curves) are drawn from JSON/CSV
under results/ when those files exist; missing inputs are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.config import (
    CFG,
    FACE_NMM_INDICES,
    FEATURE_DIM,
    LEFT_SHOULDER,
    N_FACE,
    N_HAND,
    N_POSE,
    PATHS,
    RIGHT_SHOULDER,
)

NAVY = "#1f4e79"
RUST = "#b3261e"
GRAY = "#5f6368"
FILL = "#f1f3f4"
FILL2 = "#e8f0fe"
FILL3 = "#fce8e6"
GOLD = "#f9ab00"

POSE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, stem: str) -> list[Path]:
    PATHS.figures.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        path = PATHS.figures / f"{stem}.{ext}"
        fig.savefig(path)
        out.append(path)
    plt.close(fig)
    return out


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_table(df: pd.DataFrame, stem: str, caption: str, label: str) -> list[Path]:
    PATHS.tables.mkdir(parents=True, exist_ok=True)
    csv_path = PATHS.tables / f"{stem}.csv"
    tex_path = PATHS.tables / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    cols = df.columns.tolist()
    header = " & ".join(cols) + r" \\"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{" + "l" * len(cols) + "}",
        r"\hline",
        header,
        r"\hline",
    ]
    for _, row in df.iterrows():
        cells = [str(v).replace("_", r"\_").replace("%", r"\%") for v in row.tolist()]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return [csv_path, tex_path]


def _box(ax, x, y, w, h, text, fc=FILL, ec=NAVY, fontsize=8, lw=1.1):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color="#202124", wrap=True)


def _arrow(ax, x1, y1, x2, y2, color=NAVY):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            color=color,
        )
    )


# ---------------------------------------------------------------------------
# Methodology schematics (§4 diagrams 1–6)
# ---------------------------------------------------------------------------

def fig_pipeline_overview() -> list[Path]:
    """Fig 1: shared representation, two purposes."""
    _style()
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("End-to-end pipeline: shared pose representation, two purposes", loc="left")

    _box(ax, 1.3, 4.7, 2.0, 0.9, "GSL video\nSignTalk-Gh +\nFragkiadakis / AfriSign")
    _box(ax, 3.7, 4.7, 2.0, 0.9, "MediaPipe Holistic\npose + hands +\nface NMMs")
    _box(ax, 6.1, 4.7, 2.0, 0.9, "Normalize\nshoulder center\n/ shoulder width")
    _box(ax, 8.5, 4.7, 2.0, 0.9, "Windows\nL=64 (or 128)\nstride σ < L")
    for a, b in [(2.3, 3.7), (4.7, 5.1), (7.1, 7.5)]:
        _arrow(ax, a, 4.7, b, 4.7)

    _arrow(ax, 8.5, 4.25, 8.5, 3.35)
    _box(ax, 8.5, 2.85, 3.4, 0.95, "300-D landmark window\nshared encoder input", fc=FILL2)

    _arrow(ax, 6.7, 2.85, 4.4, 1.55)
    _arrow(ax, 10.3, 2.85, 10.3, 1.55)
    _box(ax, 3.2, 1.15, 2.6, 1.15, "(a) SSL pretraining\nmasked pose prediction\nlabels discarded", fc=FILL2)
    _box(ax, 10.3, 1.15, 2.8, 1.15, "(b) Fine-tune / AL\nSignTalk labels only\nsigner-independent split", fc=FILL3)
    ax.text(6.75, 0.35, "RQ1 compares (b) with vs without initializing from (a).  RQ2 uses (b) to rank what to record next.", fontsize=8, color=GRAY)
    return _save(fig, "fig1_pipeline_overview")


def _standing_xy(offset=(0.0, 0.0), scale=1.0) -> np.ndarray:
    xy = np.zeros((N_POSE, 2), dtype=np.float32)
    xy[0] = [0.00, 0.62]
    xy[1], xy[2], xy[3], xy[7] = [0.02, 0.56], [0.08, 0.54], [0.12, 0.52], [0.16, 0.53]
    xy[4], xy[5], xy[6], xy[8] = [-0.02, 0.56], [-0.08, 0.54], [-0.12, 0.52], [-0.16, 0.53]
    xy[9], xy[10] = [-0.06, 0.48], [0.06, 0.48]
    xy[11], xy[12] = [-0.22, 0.38], [0.22, 0.38]
    xy[13], xy[14] = [-0.38, 0.18], [0.38, 0.18]
    xy[15], xy[16] = [-0.42, -0.02], [0.50, 0.08]
    xy[17], xy[19], xy[21] = [-0.46, -0.06], [-0.44, -0.08], [-0.40, -0.04]
    xy[18], xy[20], xy[22] = [0.54, 0.04], [0.56, 0.06], [0.52, 0.10]
    xy[23], xy[24] = [-0.12, -0.05], [0.12, -0.05]
    xy[25], xy[26] = [-0.14, -0.38], [0.14, -0.38]
    xy[27], xy[28] = [-0.14, -0.68], [0.14, -0.68]
    xy[29], xy[31] = [-0.18, -0.74], [-0.10, -0.72]
    xy[30], xy[32] = [0.18, -0.74], [0.10, -0.72]
    xy = xy * scale + np.array(offset, dtype=np.float32)
    return xy


def _draw_pose(ax, xy, color=NAVY, lw=1.4, alpha=1.0, mark_shoulders=False):
    pts = xy.copy()
    pts[:, 1] *= -1
    for a, b in POSE_EDGES:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color=color, lw=lw, alpha=alpha, zorder=2)
    ax.scatter(pts[:, 0], pts[:, 1], s=10, c=color, alpha=alpha, zorder=3)
    if mark_shoulders:
        mid = 0.5 * (pts[LEFT_SHOULDER] + pts[RIGHT_SHOULDER])
        ax.plot(
            [pts[LEFT_SHOULDER, 0], pts[RIGHT_SHOULDER, 0]],
            [pts[LEFT_SHOULDER, 1], pts[RIGHT_SHOULDER, 1]],
            color=RUST,
            lw=2.0,
            zorder=4,
        )
        ax.scatter([mid[0]], [mid[1]], s=36, c=RUST, zorder=5, marker="x")
        ax.annotate(
            r"$c_t$  shoulder midpoint",
            xy=mid,
            xytext=(mid[0] + 0.35, mid[1] + 0.18),
            fontsize=8,
            color=RUST,
            arrowprops=dict(arrowstyle="->", color=RUST, lw=0.8),
        )
        ax.annotate(
            r"$s_t$  shoulder width",
            xy=pts[RIGHT_SHOULDER],
            xytext=(0.55, 0.22),
            fontsize=8,
            color=RUST,
            arrowprops=dict(arrowstyle="->", color=RUST, lw=0.8),
        )
    ax.set_aspect("equal")
    ax.axis("off")


def fig_landmark_normalization() -> list[Path]:
    """Fig 2: retained landmarks + shoulder centering/scaling."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6))
    raw = _standing_xy(offset=(0.35, 0.15), scale=1.15)
    _draw_pose(axes[0], raw, mark_shoulders=True)
    axes[0].set_title("Before: camera coordinates")
    axes[0].text(
        0.02,
        0.02,
        "Kept: 33 pose + 21 L hand + 21 R hand\n+ 25 face NMMs (brows, eyes, mouth, chin)",
        transform=axes[0].transAxes,
        fontsize=7.5,
        color=GRAY,
        va="bottom",
    )

    mid = 0.5 * (raw[LEFT_SHOULDER] + raw[RIGHT_SHOULDER])
    width = np.linalg.norm(raw[LEFT_SHOULDER] - raw[RIGHT_SHOULDER])
    normed = (raw - mid) / max(width, 1e-6)
    _draw_pose(axes[1], normed, color=NAVY, mark_shoulders=True)
    axes[1].set_title(r"After: $\hat{x}_t = (x_t - c_t) / s_t$")
    axes[1].axhline(0, color="#dadce0", lw=0.8)
    axes[1].axvline(0, color="#dadce0", lw=0.8)
    fig.suptitle("Pose representation and torso-relative normalization (§3.2)", fontsize=12)
    fig.tight_layout()
    return _save(fig, "fig2_landmark_normalization")


def fig_masked_pose() -> list[Path]:
    """Fig 3: BERT-style masking on a pose window."""
    _style()
    fig = plt.figure(figsize=(11.0, 4.8))
    gs = fig.add_gridspec(2, 8, height_ratios=[1.15, 1.0], hspace=0.55, wspace=0.25)
    masked = {1, 3, 4, 6}
    for i in range(8):
        ax = fig.add_subplot(gs[0, i])
        pose = _standing_xy(offset=(0.04 * np.sin(i), 0.0), scale=0.95)
        pose[:, 0] += 0.03 * i
        if i in masked:
            _draw_pose(ax, pose, color="#9aa0a6", lw=1.0, alpha=0.25)
            ax.set_title(f"t={i+1}\nmasked", fontsize=7, color=RUST)
        else:
            _draw_pose(ax, pose, color=NAVY, lw=1.0)
            ax.set_title(f"t={i+1}", fontsize=7)
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(-1.0, 0.9)

    ax_e = fig.add_subplot(gs[1, 0:3])
    ax_e.axis("off")
    _box(ax_e, 0.5, 0.5, 0.84, 0.7, "Transformer encoder $f_\\theta$\nlearned mask token at\nmasked frames", fc=FILL2)
    ax_e.set_xlim(0, 1)
    ax_e.set_ylim(0, 1)

    ax_d = fig.add_subplot(gs[1, 3:6])
    ax_d.axis("off")
    _box(ax_d, 0.5, 0.5, 0.84, 0.7, "Reconstruct $\\hat{x}'_i$\nonly at masked $i$\n$\\mathcal{L}_{mask}$ = Smooth L1 / MSE", fc=FILL3)
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)

    ax_n = fig.add_subplot(gs[1, 6:8])
    ax_n.axis("off")
    ax_n.text(
        0.5,
        0.5,
        "$\\rho \\sim U[0.15, 0.40]$\nper window\n\nframe- or\njoint-level mask",
        ha="center",
        va="center",
        fontsize=8,
        color=GRAY,
    )
    fig.suptitle("Masked pose prediction pretext (§3.3.1). Loss is computed only on masked positions.", fontsize=11)
    # arrows between encoder and recon in figure coordinates
    return _save(fig, "fig3_masked_pose_prediction")


def fig_contrastive() -> list[Path]:
    """Fig 4: two views, no left-right mirroring."""
    _style()
    fig, ax = plt.subplots(figsize=(10.4, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    ax.set_title("Contrastive ablation: two views of the same window, mirroring excluded (§3.3.2)", loc="left")
    _box(ax, 1.4, 2.2, 2.0, 1.1, "Window $W$\nunaugmented $x$")
    _box(ax, 4.2, 3.2, 2.2, 1.0, "View $W^{(1)}$\ncrop, dropout, jitter", fc=FILL2)
    _box(ax, 4.2, 1.2, 2.2, 1.0, "View $W^{(2)}$\ncrop, dropout, jitter", fc=FILL2)
    _box(ax, 7.0, 3.2, 1.8, 1.0, r"$z^{(1)}=g_\phi(f_\theta(W^{(1)}))$", fc=FILL2)
    _box(ax, 7.0, 1.2, 1.8, 1.0, r"$z^{(2)}=g_\phi(f_\theta(W^{(2)}))$", fc=FILL2)
    _box(ax, 9.6, 2.2, 2.2, 1.3, "InfoNCE\npull same-clip pair\npush other clips\n$\\lambda\\,\\mathcal{L}_{contrast}$", fc=FILL3)
    _arrow(ax, 2.4, 2.55, 3.1, 3.2)
    _arrow(ax, 2.4, 1.85, 3.1, 1.2)
    _arrow(ax, 5.3, 3.2, 6.1, 3.2)
    _arrow(ax, 5.3, 1.2, 6.1, 1.2)
    _arrow(ax, 7.9, 3.2, 8.5, 2.55)
    _arrow(ax, 7.9, 1.2, 8.5, 1.85)
    ax.text(4.2, 0.35, "Augmentations used: temporal crop, frame dropout, spatial jitter.  Not used: left–right mirroring.", fontsize=8, color=RUST, ha="center")
    return _save(fig, "fig4_contrastive_augmentation")


def fig_al_loop() -> list[Path]:
    """Fig 5: active recording loop, simulation vs real-world."""
    _style()
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Active learning for recording prioritization (§3.5)", loc="left")
    steps = [
        (5.0, 5.15, "Train $M_k$\non included signers $S_k$"),
        (8.3, 3.6, "Evaluate held-out\nsigner-independent test"),
        (8.3, 1.7, "Uncertainty\nentropy / margin / MC-dropout"),
        (5.0, 0.7, "Aggregate $U_j$, $U_g$\nsentence / category rank"),
        (1.7, 1.7, "Record next\n(sim: add donor data;\nreal: new signer)"),
        (1.7, 3.6, "Retrain\nlog acc, per-sentence,\nper-category"),
    ]
    for x, y, t in steps:
        _box(ax, x, y, 2.5, 1.05, t, fc=FILL2)
    pairs = [(5.0, 4.6, 8.3, 4.15), (8.3, 3.05, 8.3, 2.25), (8.3, 1.15, 5.0, 1.2), (5.0, 0.7, 1.7, 1.15), (1.7, 2.25, 1.7, 3.05), (1.7, 4.15, 5.0, 4.6)]
    for x1, y1, x2, y2 in pairs:
        _arrow(ax, x1, y1, x2, y2)
    ax.add_patch(FancyBboxPatch((0.25, 5.35), 3.3, 0.5, boxstyle="round,pad=0.02", facecolor=FILL, edgecolor=GRAY, lw=0.8))
    ax.text(1.9, 5.6, "Simulation: existing 5 signers", ha="center", va="center", fontsize=8, color=GRAY)
    ax.add_patch(FancyBboxPatch((6.45, 5.35), 3.3, 0.5, boxstyle="round,pad=0.02", facecolor=FILL3, edgecolor=RUST, lw=0.8))
    ax.text(8.1, 5.6, "Real-world: new signers later", ha="center", va="center", fontsize=8, color=RUST)
    return _save(fig, "fig5_active_learning_loop")


def fig_experimental_arms() -> list[Path]:
    """Fig 6: matched-arm Δ_AL design."""
    _style()
    fig, ax = plt.subplots(figsize=(10.2, 4.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("RQ2 matched-arm comparison: uncertainty vs random at equal cost", loc="left")
    _box(ax, 1.6, 2.1, 2.4, 1.2, "Same start\n$S_0$ = 1–2 signers\nsame encoder init")
    _box(ax, 5.2, 3.2, 2.6, 1.15, "AL arm\nadd by $U_j$ / $U_g$\n$S^{AL}_{new}$", fc=FILL2)
    _box(ax, 5.2, 1.0, 2.6, 1.15, "Random arm\nuniform sentences\n$S^{rand}_{new}$", fc=FILL)
    _box(ax, 9.1, 2.1, 2.8, 1.35, r"$\Delta_{AL} =$" + "\nPerf($S_k \\cup S^{AL}$)\n" + r"$-$ Perf($S_k \cup S^{rand}$)" + "\n$|S^{AL}|=|S^{rand}|$", fc=FILL3)
    _arrow(ax, 2.8, 2.5, 3.9, 3.2)
    _arrow(ax, 2.8, 1.7, 3.9, 1.0)
    _arrow(ax, 6.5, 3.2, 7.7, 2.5)
    _arrow(ax, 6.5, 1.0, 7.7, 1.7)
    ax.text(5.2, 0.22, "Held constant: signer count, sentence count, recording conditions. Only the selection rule differs.", fontsize=8, color=GRAY, ha="center")
    return _save(fig, "fig6_experimental_arms")


# ---------------------------------------------------------------------------
# Results figures
# ---------------------------------------------------------------------------

def _pilot_note(ax, n: int | None) -> None:
    """Legacy helper; no longer stamps SSL figures (full 50-epoch history is retained)."""
    del ax, n


def fig_ssl_curve() -> list[Path]:
    hist = _load_json(PATHS.results / "ssl_history.json")
    if not hist:
        return []
    _style()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ep = [r["epoch"] for r in hist]
    ax.plot(ep, [r["loss"] for r in hist], color=NAVY, lw=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked reconstruction loss (Smooth L1)")
    ax.set_title("SSL pretraining loss (full SignTalk run, signer 5 held out)")
    fig.tight_layout()
    return _save(fig, "fig_ssl_pretrain_loss")


def fig_finetune_curves() -> list[Path]:
    pre = _load_json(PATHS.results / "finetune_pretrained_history.json")
    scr = _load_json(PATHS.results / "finetune_scratch_history.json")
    if not pre and not scr:
        return []
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for hist, name, color in ((pre, "SSL-pretrained", NAVY), (scr, "From scratch", RUST)):
        if not hist:
            continue
        ep = [r["epoch"] for r in hist]
        axes[0].plot(ep, [100 * r["train_acc"] for r in hist], color=color, lw=1.7, label=name)
        axes[1].plot(ep, [100 * r["val_acc"] for r in hist], color=color, lw=1.7, label=name)
    axes[0].set_title("Fine-tune train accuracy")
    axes[1].set_title("Fine-tune val accuracy (held-out signer)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.legend(frameon=False)
        _pilot_note(ax, (pre or scr or [{}])[0].get("epoch") if (pre or scr) else None)
    n = None
    cmp_ = _load_json(PATHS.results / "rq1_comparison.json")
    if cmp_ and cmp_.get("scratch"):
        n = cmp_["scratch"].get("n")
        _pilot_note(axes[1], n)
    fig.tight_layout()
    return _save(fig, "fig_finetune_curves")


def fig_rq1_bars() -> list[Path]:
    payload = _load_json(PATHS.results / "rq1_comparison.json")
    if not payload or not payload.get("scratch") or not payload.get("pretrained"):
        return []
    _style()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    labels = ["Top-1", "Top-5"]
    x = np.arange(len(labels))
    w = 0.35
    s = payload["scratch"]
    p = payload["pretrained"]
    ax.bar(x - w / 2, [100 * s["top1"], 100 * s["top5"]], w, label="From scratch", color=RUST)
    ax.bar(x + w / 2, [100 * p["top1"], 100 * p["top5"]], w, label="SSL-pretrained", color=NAVY)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Held-out signer accuracy (%)")
    ax.set_title("RQ1: signer-independent sign-to-text")
    ax.legend(frameon=False)
    _pilot_note(ax, s.get("n"))
    fig.tight_layout()
    return _save(fig, "fig_rq1_accuracy")


def fig_rq1_diagnostics() -> list[Path]:
    payload = _load_json(PATHS.results / "rq1_diagnostics.json")
    if not payload or not payload.get("sentence"):
        return []
    sent = payload["sentence"]
    win_s = (sent.get("window") or {}).get("scratch") or {}
    win_p = (sent.get("window") or {}).get("pretrained") or {}
    clip_s = (sent.get("clip") or {}).get("scratch") or {}
    clip_p = (sent.get("clip") or {}).get("pretrained") or {}
    if not win_s or not clip_s:
        return []
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    labels = ["Sentence\nwindow", "Sentence\nclip"]
    ys_s = [100 * win_s["top1"], 100 * clip_s.get("top1", 0.0)]
    ys_p = [100 * win_p.get("top1", 0.0), 100 * clip_p.get("top1", 0.0)]
    trained = payload.get("category_trained") or {}
    if trained.get("window") and trained["window"].get("scratch"):
        labels += ["Category\nwindow", "Category\nclip"]
        ys_s += [
            100 * trained["window"]["scratch"]["top1"],
            100 * ((trained.get("clip") or {}).get("scratch") or {}).get("top1", 0.0),
        ]
        ys_p += [
            100 * ((trained.get("window") or {}).get("pretrained") or {}).get("top1", 0.0),
            100 * ((trained.get("clip") or {}).get("pretrained") or {}).get("top1", 0.0),
        ]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, ys_s, w, label="From scratch", color=RUST)
    ax.bar(x + w / 2, ys_p, w, label="SSL-pretrained", color=NAVY)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Held-out signer top-1 (%)")
    ax.set_title("RQ1 diagnostics: window vs clip, sentence vs category")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, "fig_rq1_diagnostics")


def fig_category_heatmap() -> list[Path]:
    payload = _load_json(PATHS.results / "rq1_comparison.json")
    if not payload:
        return []
    rows = []
    for arm in ("scratch", "pretrained"):
        stats = payload.get(arm) or {}
        err = stats.get("per_category_error")
        if not err:
            acc = stats.get("per_category_top1") or {}
            err = {k: 1.0 - v for k, v in acc.items()}
        for cat, e in err.items():
            rows.append({"category": cat, "condition": arm, "error": e})
    # Optional AL column if RQ2 last round logged per-category
    if not rows:
        return []
    df = pd.DataFrame(rows)
    mat = df.pivot_table(index="category", columns="condition", values="error", aggfunc="mean")
    _style()
    fig, ax = plt.subplots(figsize=(6.4, max(2.8, 0.38 * len(mat))))
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(mat.columns)), list(mat.columns))
    ax.set_yticks(range(len(mat.index)), list(mat.index))
    ax.set_title("Per-category error rate (1 − top-1)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="Error rate")
    n = (payload.get("scratch") or {}).get("n")
    _pilot_note(ax, n)
    fig.tight_layout()
    return _save(fig, "fig8_per_category_error")


def fig_rq2_curves() -> list[Path]:
    paths = sorted(
        set(PATHS.results.glob("rq2_simulation_*.json"))
        | set(PATHS.results.glob("rq2_session_*.json"))
    )
    if not paths:
        return []
    written = []
    for path in paths:
        payload = _load_json(path)
        if not payload or "delta_al" not in payload:
            continue
        rank = payload.get("rank_by") or payload.get("method") or "guided"
        protocol = payload.get("protocol", "rq2")
        # Session JSON may stack multiple donors; plot each separately when present.
        series = []
        per_donor = payload.get("per_donor") or {}
        if per_donor:
            for donor_id, block in per_donor.items():
                series.append((f"donor {donor_id}", block.get("delta_al") or []))
        else:
            series.append(("", payload["delta_al"]))
        for suffix, delta in series:
            if not delta:
                continue
            _style()
            fig, ax = plt.subplots(figsize=(6.0, 3.6))
            rounds = [d["round"] for d in delta]
            ax.plot(
                rounds,
                [100 * d["al"] for d in delta],
                color=NAVY,
                lw=1.8,
                marker="o",
                label="Guided",
            )
            ax.plot(
                rounds,
                [100 * d["random"] for d in delta],
                color=GRAY,
                lw=1.8,
                marker="o",
                label="Random (matched size)",
            )
            ax.set_xlabel("Round (matched additions)")
            ylab = "Held-out signer top-1 (%)"
            if payload.get("task") == "category":
                ylab = "Held-out signer category top-1 (%)"
            ax.set_ylabel(ylab)
            title = f"RQ2 $\\Delta_{{AL}}$  ({protocol}, {rank}"
            if payload.get("task"):
                title += f", {payload.get('task')}"
            if suffix:
                title += f", {suffix}"
            title += ")"
            ax.set_title(title)
            ax.legend(frameon=False)
            fig.tight_layout()
            stem = f"fig_rq2_{protocol}_{rank}"
            if payload.get("task"):
                stem += f"_{payload.get('task')}"
            if suffix:
                stem += f"_{suffix.replace(' ', '')}"
            written.extend(_save(fig, stem))
    return written


def fig_rq3_scaling() -> list[Path]:
    # Prefer exhaustive grid; fall back to the older sampled run.
    payload = _load_json(PATHS.results / "rq3_signer_scaling_exhaustive.json")
    if not payload:
        payload = _load_json(PATHS.results / "rq3_signer_scaling.json")
    if not payload:
        return []
    _style()
    if payload.get("protocol") == "loso_category_clip" and payload.get("summary"):
        summary = payload["summary"]
        all_k = sorted({int(k) for arm in summary.values() for k in arm.keys()})
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), sharey=True)
        for ax, metric, title in (
            (axes[0], "clip_top1", "Clip top-1"),
            (axes[1], "clip_coverage_aware_top1", "Coverage-aware clip top-1"),
        ):
            for arm, color in (("scratch", RUST), ("pretrained", NAVY)):
                ks, means, stds = [], [], []
                for k in sorted((summary.get(arm) or {}).keys(), key=int):
                    slot = summary[arm][k].get(metric)
                    if not slot:
                        continue
                    ks.append(int(k))
                    means.append(100 * slot["mean"])
                    stds.append(100 * slot["std"])
                if not ks:
                    continue
                ax.errorbar(
                    ks,
                    means,
                    yerr=stds,
                    color=color,
                    lw=1.8,
                    marker="o",
                    capsize=3,
                    label="From scratch" if arm == "scratch" else "SSL-pretrained",
                )
            maj = []
            maj_k = []
            for k in sorted((summary.get("scratch") or {}).keys(), key=int):
                slot = (summary["scratch"][k].get("clip_majority") or {})
                if slot:
                    maj_k.append(int(k))
                    maj.append(100 * slot["mean"])
            if maj:
                ax.plot(maj_k, maj, color=GRAY, ls="--", lw=1.2, label="Test majority")
            ax.set_xlabel("Training signers (k)")
            ax.set_ylabel("Held-out signer accuracy (%)")
            ax.set_title(title)
            ax.set_xticks(all_k or [1, 2, 3, 4])
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(
            "RQ3: LOSO category scaling (mean ± std"
            + (", exhaustive grid)" if payload.get("sampling") == "exhaustive" else ")")
        )
        fig.tight_layout()
        return _save(fig, "fig7_signer_diversity_scaling")

    if not payload.get("curve"):
        return []
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ks, pre, scr = [], [], []
    for pt in payload["curve"]:
        ks.append(pt["n_signers"])
        pre.append(100 * pt["arms"]["pretrained"]["top1"])
        scr.append(100 * pt["arms"]["scratch"]["top1"])
    ax.plot(ks, pre, color=NAVY, lw=1.8, marker="o", label="SSL-pretrained")
    ax.plot(ks, scr, color=RUST, lw=1.8, marker="o", label="From scratch")
    ax.set_xlabel("Number of training signers")
    ax.set_ylabel("Held-out signer top-1 (%)")
    ax.set_title("RQ3: signer-diversity scaling")
    ax.set_xticks(ks)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, "fig7_signer_diversity_scaling")


def fig_rq4_calibration() -> list[Path]:
    """RQ4: calibration curves. Prefer v2 (unrestricted primary) if present."""
    payload = _load_json(PATHS.results / "rq4_fewshot_calibration_v2.json")
    metric_primary = "clip_top1"
    metric_secondary = "clip_restricted_top1"
    title_note = "primary = unrestricted clip top-1; dashed = calib-restricted"
    if not payload:
        payload = _load_json(PATHS.results / "rq4_fewshot_calibration.json")
        metric_primary = "clip_coverage_aware_top1"
        metric_secondary = "clip_top1"
        title_note = "legacy oracle coverage-aware (v1)"
    if not payload:
        return []
    summary = (payload.get("summary") or {}).get("by_arm_method") or {}
    if not summary:
        return []
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    styles = {
        ("scratch", "prototype"): (RUST, "o", "-"),
        ("scratch", "partial_ft"): (RUST, "s", "--"),
        ("pretrained", "prototype"): (NAVY, "o", "-"),
        ("pretrained", "partial_ft"): (NAVY, "s", "--"),
    }
    for ax, metric, title in (
        (axes[0], metric_primary, "Unrestricted clip top-1" if "v2" in title_note or metric_primary == "clip_top1" else "Coverage-aware clip top-1"),
        (axes[1], metric_secondary, "Calib-restricted top-1" if metric_secondary == "clip_restricted_top1" else "Clip top-1"),
    ):
        for (arm, method), (color, marker, ls) in styles.items():
            slots = (summary.get(arm) or {}).get(method) or {}
            if not slots:
                continue
            xs, ys, es = [], [], []
            for n in sorted(slots.keys(), key=int):
                slot = slots[n].get(metric) or {}
                if not slot:
                    continue
                xs.append(int(n))
                ys.append(100 * slot["mean"])
                es.append(100 * slot.get("std", 0.0))
            if not xs:
                continue
            label = f"{'Scratch' if arm == 'scratch' else 'SSL'} · {method}"
            ax.errorbar(
                xs,
                ys,
                yerr=es if any(e > 0 for e in es) else None,
                color=color,
                marker=marker,
                ls=ls,
                lw=1.7,
                capsize=3,
                label=label,
            )
        ax.axhline(11.9, color=GRAY, ls=":", lw=1.1, label="~majority (signer 5)")
        ax.set_xlabel("Calibration clips per category")
        ax.set_ylabel("Held-out final-test accuracy (%)")
        ax.set_title(title)
        ax.set_xticks([0, 1, 3, 5, 10, 20])
        ax.legend(frameon=False, fontsize=7)
        ax.set_ylim(0, 80)
    fig.suptitle(f"RQ4: few-shot signer calibration ({title_note})")
    fig.tight_layout()
    return _save(fig, "fig_rq4_fewshot_calibration")


def fig_rq5_synthetic() -> list[Path]:
    """RQ5: matched-volume pseudo-signers vs real-only (coverage-aware)."""
    payload = _load_json(PATHS.results / "rq5_synthetic_signers.json")
    if not payload or not payload.get("runs"):
        return []
    runs = payload["runs"]
    axes = sorted({str(r["axis"]) for r in runs})
    ks = sorted({int(r["k"]) for r in runs})
    _style()
    fig, axes_plot = plt.subplots(1, 2, figsize=(9.4, 3.8))

    # Left: cov% vs k for real_only vs pseudo, pooled across axes
    ax = axes_plot[0]
    for cond, color, marker, ls, label in (
        ("real_only", RUST, "o", "-", "Real only"),
        ("pseudo", NAVY, "s", "--", "Real + pseudo (matched vol.)"),
    ):
        means, stds = [], []
        for k in ks:
            vals = [
                100 * float(r["clip_coverage_aware_top1"])
                for r in runs
                if int(r["k"]) == k and r["condition"] == cond
            ]
            means.append(float(np.mean(vals)) if vals else np.nan)
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        ax.errorbar(
            ks, means, yerr=stds, color=color, marker=marker, ls=ls, lw=1.7, capsize=3, label=label
        )
    ax.axhline(10.2, color=GRAY, ls=":", lw=1.1, label="~majority")
    ax.set_xlabel("Training signers (k)")
    ax.set_ylabel("Coverage-aware clip top-1 (%)")
    ax.set_title("Pooled across perturbation axes")
    ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=7)
    ax.set_ylim(0, 20)

    # Right: matched mean Δ (pseudo − real) by axis
    ax = axes_plot[1]
    real = {
        (r["test_signer"], r["k"], tuple(r["train_signers"]), r["axis"]): r
        for r in runs
        if r["condition"] == "real_only"
    }
    labels, deltas, errs = [], [], []
    for axis in axes:
        dvals = []
        for r in runs:
            if r["condition"] != "pseudo" or r["axis"] != axis:
                continue
            key = (r["test_signer"], r["k"], tuple(r["train_signers"]), r["axis"])
            if key in real:
                dvals.append(
                    100
                    * (
                        float(r["clip_coverage_aware_top1"])
                        - float(real[key]["clip_coverage_aware_top1"])
                    )
                )
        if not dvals:
            continue
        labels.append(axis)
        deltas.append(float(np.mean(dvals)))
        errs.append(float(np.std(dvals, ddof=1)) if len(dvals) > 1 else 0.0)
    x = np.arange(len(labels))
    colors = [NAVY if d >= 0 else RUST for d in deltas]
    ax.bar(x, deltas, yerr=errs, color=colors, ecolor=GRAY, capsize=3, width=0.65)
    ax.axhline(0.0, color=GRAY, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Δ coverage-aware (pp)")
    ax.set_title("Matched: pseudo − real_only")
    ax.set_ylim(-4, 3)

    fig.suptitle("RQ5: landmark-space synthetic signers (matched window volume)")
    fig.tight_layout()
    return _save(fig, "fig_rq5_synthetic_signers")


def fig_rq6_augment() -> list[Path]:
    """RQ6: fine-tune augmentation vs none (scratch; coverage-aware)."""
    payload = _load_json(PATHS.results / "rq6_finetune_augment.json")
    if not payload or not payload.get("runs"):
        return []
    runs = [r for r in payload["runs"] if r.get("arm") == "scratch"]
    if not runs:
        return []
    augs = ["none", "frame_replace", "mask_sub", "both"]
    aug_labels = {
        "none": "none",
        "frame_replace": "frame",
        "mask_sub": "mask",
        "both": "both",
    }
    colors = {
        "none": GRAY,
        "frame_replace": NAVY,
        "mask_sub": GOLD,
        "both": RUST,
    }
    ks = sorted({int(r["k"]) for r in runs})
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    # Left: grouped bars of coverage-aware by k × aug
    ax = axes[0]
    width = 0.18
    x = np.arange(len(ks))
    for i, aug in enumerate(augs):
        means, stds = [], []
        for k in ks:
            vals = [
                100 * float(r["clip_coverage_aware_top1"])
                for r in runs
                if int(r["k"]) == k and r.get("aug") == aug
            ]
            means.append(float(np.mean(vals)) if vals else np.nan)
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        ax.bar(
            x + (i - 1.5) * width,
            means,
            width,
            yerr=stds,
            color=colors[aug],
            ecolor="#888888",
            capsize=2,
            label=aug_labels[aug],
        )
    ax.axhline(10.2, color=GRAY, ls=":", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Training signers (k)")
    ax.set_ylabel("Coverage-aware clip top-1 (%)")
    ax.set_title("Scratch accuracy by augmentation")
    ax.set_ylim(0, 20)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    # Right: matched Δ vs none
    ax = axes[1]
    none_map = {
        (r["test_signer"], int(r["k"]), tuple(r["train_signers"])): r
        for r in runs
        if r.get("aug") == "none"
    }
    delta_augs = ["frame_replace", "mask_sub", "both"]
    means, stds = [], []
    for aug in delta_augs:
        dvals = []
        for r in runs:
            if r.get("aug") != aug:
                continue
            key = (r["test_signer"], int(r["k"]), tuple(r["train_signers"]))
            base = none_map.get(key)
            if base is None:
                continue
            dvals.append(
                100
                * (
                    float(r["clip_coverage_aware_top1"])
                    - float(base["clip_coverage_aware_top1"])
                )
            )
        means.append(float(np.mean(dvals)) if dvals else 0.0)
        stds.append(float(np.std(dvals, ddof=1)) if len(dvals) > 1 else 0.0)
    xpos = np.arange(len(delta_augs))
    bar_colors = [NAVY if m >= 0 else RUST for m in means]
    ax.bar(xpos, means, yerr=stds, color=bar_colors, ecolor=GRAY, capsize=3, width=0.6)
    ax.axhline(0.0, color=GRAY, lw=1.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels([aug_labels[a] for a in delta_augs])
    ax.set_ylabel("Δ coverage-aware vs none (pp)")
    ax.set_title("Matched: aug − none")
    ax.set_ylim(-3, 3)

    fig.suptitle("RQ6: fine-tune-time frame / mask augmentation (scratch)")
    fig.tight_layout()
    return _save(fig, "fig_rq6_finetune_augment")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_dataset() -> list[Path]:
    df = pd.DataFrame(
        [
            {
                "Source": "SignTalk-Gh",
                "Role": "SSL (labels discarded) + labeled fine-tune/eval",
                "Approx. clips": "9,877 usable",
                "Signers": "5",
                "Domain": "Healthcare sentences",
            },
            {
                "Source": "Fragkiadakis lexicon",
                "Role": "SSL only",
                "Approx. clips": "~1,200 isolated signs",
                "Signers": "1 (Nyarko)",
                "Domain": "General lexicon",
            },
            {
                "Source": "AfriSign (GSL subset)",
                "Role": "SSL only (optional)",
                "Approx. clips": "if placed under data/external/afrisign/",
                "Signers": "unspecified",
                "Domain": "JW Bible scrape; not redistributed here",
            },
        ]
    )
    return _write_table(df, "table_data_sources", "GSL video sources used in this work (§3.1).", "tab:data-sources")


def table_split() -> list[Path]:
    df = pd.DataFrame(
        [
            {"Split": "train", "Signers": "1, 2, 3", "Used for": "Labeled fine-tune; unlabeled SSL"},
            {"Split": "val", "Signers": "4", "Used for": "Model selection; unlabeled SSL"},
            {"Split": "test", "Signers": "5", "Used for": "Signer-independent evaluation only (never in SSL or FT)"},
            {"Split": "ssl extra", "Signers": "nyarko / afrisign", "Used for": "Unlabeled SSL only; never in label map"},
        ]
    )
    return _write_table(df, "table_signer_split", "Signer-independent split protocol (§3.6).", "tab:split")


def table_hyperparams() -> list[Path]:
    m, s, f, w = CFG.model, CFG.ssl, CFG.finetune, CFG.window
    df = pd.DataFrame(
        [
            {"Group": "Landmarks", "Setting": "Feature dim", "Value": str(FEATURE_DIM)},
            {"Group": "Landmarks", "Setting": "Joints (pose/L-hand/R-hand/face NMM)", "Value": f"{N_POSE}/{N_HAND}/{N_HAND}/{N_FACE}"},
            {"Group": "Landmarks", "Setting": "Face NMM indices", "Value": str(len(FACE_NMM_INDICES))},
            {"Group": "Windows", "Setting": "L / stride", "Value": f"{w.window_size} / {w.stride}"},
            {"Group": "Encoder", "Setting": "layers / d_model / heads / FFN", "Value": f"{m.num_layers} / {m.d_model} / {m.nhead} / {m.dim_feedforward}"},
            {"Group": "SSL", "Setting": "Mask ratio", "Value": f"U[{s.mask_ratio_min}, {s.mask_ratio_max}] per window"},
            {"Group": "SSL", "Setting": "Mask mode / loss", "Value": f"{s.mask_mode} / {s.loss}"},
            {"Group": "SSL", "Setting": "epochs / batch / lr", "Value": f"{s.epochs} / {s.batch_size} / {s.lr}"},
            {"Group": "SSL", "Setting": "Contrastive λ / τ / proj dim", "Value": f"{s.contrastive_weight} / {s.temperature} / {m.proj_dim}"},
            {"Group": "Fine-tune", "Setting": "epochs / batch / lr", "Value": f"{f.epochs} / {f.batch_size} / {f.lr}"},
            {"Group": "Fine-tune", "Setting": "Head / loss", "Value": "linear over mean-pooled encoder / cross-entropy"},
        ]
    )
    return _write_table(df, "table_hyperparams", "Default hyperparameters used in the reported pipeline.", "tab:hyperparams")


def table_rq1() -> list[Path]:
    payload = _load_json(PATHS.results / "rq1_comparison.json")
    if not payload or not payload.get("scratch") or not payload.get("pretrained"):
        return []
    s, p = payload["scratch"], payload["pretrained"]
    delta = 100 * (p["top1"] - s["top1"])
    df = pd.DataFrame(
        [
            {
                "Arm": "From scratch",
                "n windows": s["n"],
                "Top-1 (%)": f"{100 * s['top1']:.2f}",
                "Top-5 (%)": f"{100 * s['top5']:.2f}",
                "Delta vs scratch (pp)": "—",
            },
            {
                "Arm": "SSL-pretrained",
                "n windows": p["n"],
                "Top-1 (%)": f"{100 * p['top1']:.2f}",
                "Top-5 (%)": f"{100 * p['top5']:.2f}",
                "Delta vs scratch (pp)": f"{delta:+.2f}",
            },
        ]
    )
    return _write_table(
        df,
        "table_rq1",
        "RQ1 signer-independent sentence classification on the held-out signer (window-level).",
        "tab:rq1",
    )


def table_rq6() -> list[Path]:
    """Scratch coverage-aware means by k × aug, plus matched Δ vs none."""
    payload = _load_json(PATHS.results / "rq6_finetune_augment.json")
    if not payload or not payload.get("runs"):
        return []
    runs = [r for r in payload["runs"] if r.get("arm") == "scratch"]
    if not runs:
        return []
    augs = ["none", "frame_replace", "mask_sub", "both"]
    ks = sorted({int(r["k"]) for r in runs})
    none_map = {
        (r["test_signer"], int(r["k"]), tuple(r["train_signers"])): r
        for r in runs
        if r.get("aug") == "none"
    }
    rows = []
    for k in ks:
        for aug in augs:
            vals = [
                float(r["clip_coverage_aware_top1"])
                for r in runs
                if int(r["k"]) == k and r.get("aug") == aug
            ]
            clips = [
                float(r["clip_top1"])
                for r in runs
                if int(r["k"]) == k and r.get("aug") == aug
            ]
            if not vals:
                continue
            mean_cov = float(np.mean(vals))
            std_cov = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            mean_clip = float(np.mean(clips))
            if aug == "none":
                delta = "-"
            else:
                dvals = []
                for r in runs:
                    if int(r["k"]) != k or r.get("aug") != aug:
                        continue
                    key = (r["test_signer"], int(r["k"]), tuple(r["train_signers"]))
                    base = none_map.get(key)
                    if base is None:
                        continue
                    dvals.append(
                        100
                        * (
                            float(r["clip_coverage_aware_top1"])
                            - float(base["clip_coverage_aware_top1"])
                        )
                    )
                delta = f"{float(np.mean(dvals)):+.2f}" if dvals else "—"
            rows.append(
                {
                    "k": k,
                    "Augmentation": aug,
                    "n": len(vals),
                    "Cov-aware (%)": f"{100 * mean_cov:.2f} +/- {100 * std_cov:.2f}",
                    "Clip top-1 (%)": f"{100 * mean_clip:.2f}",
                    "Delta vs none (pp)": delta,
                }
            )
    # Overall matched deltas row block
    for aug in ["frame_replace", "mask_sub", "both"]:
        dvals = []
        for r in runs:
            if r.get("aug") != aug:
                continue
            key = (r["test_signer"], int(r["k"]), tuple(r["train_signers"]))
            base = none_map.get(key)
            if base is None:
                continue
            dvals.append(
                100
                * (
                    float(r["clip_coverage_aware_top1"])
                    - float(base["clip_coverage_aware_top1"])
                )
            )
        if dvals:
            rows.append(
                {
                    "k": "all",
                    "Augmentation": aug,
                    "n": len(dvals),
                    "Cov-aware (%)": "-",
                    "Clip top-1 (%)": "-",
                    "Delta vs none (pp)": f"{float(np.mean(dvals)):+.2f}",
                }
            )
    df = pd.DataFrame(rows)
    return _write_table(
        df,
        "table_rq6",
        "RQ6 fine-tune augmentation on held-out signers (scratch). "
        "Coverage-aware clip top-1; matched Delta vs the unaugmented arm.",
        "tab:rq6",
    )


def table_rq1_diagnostics() -> list[Path]:
    payload = _load_json(PATHS.results / "rq1_diagnostics.json")
    if not payload:
        return []
    rows = []

    def _row(setting, unit, arm, stats, extra=""):
        if not stats:
            return
        rows.append(
            {
                "Setting": setting,
                "Unit": unit,
                "Arm": arm,
                "n": stats.get("n", ""),
                "Top-1 (%)": f"{100 * stats['top1']:.2f}",
                "Top-5 (%)": f"{100 * stats.get('top5', 0):.2f}",
                "Note": extra,
            }
        )

    sent = payload.get("sentence") or {}
    for unit in ("window", "clip"):
        block = sent.get(unit) or {}
        _row("sentence ID", unit, "scratch", block.get("scratch"))
        _row("sentence ID", unit, "pretrained", block.get("pretrained"))
    post = payload.get("category_posthoc") or {}
    for unit, block in post.items():
        maj = (block.get("scratch") or {}).get("majority")
        note = f"majority {100 * maj:.1f}%" if maj is not None else "sentence argmax → category"
        _row("category post-hoc", unit, "scratch", block.get("scratch"), note)
        _row("category post-hoc", unit, "pretrained", block.get("pretrained"), note)
    trained = payload.get("category_trained") or {}
    if trained:
        for unit in ("window", "clip"):
            block = trained.get(unit) or {}
            maj = trained.get("majority_window") if unit == "window" else trained.get("majority_clip")
            note = f"{trained.get('n_classes', '?')}-way; majority {100 * (maj or 0):.1f}%"
            _row("category trained", unit, "scratch", block.get("scratch"), note)
            _row("category trained", unit, "pretrained", block.get("pretrained"), note)
    if not rows:
        return []
    df = pd.DataFrame(rows)
    return _write_table(
        df,
        "table_rq1_diagnostics",
        "RQ1 diagnostics: clip-pooled sentence ID and coarser category label space.",
        "tab:rq1diag",
    )


def table_per_category() -> list[Path]:
    payload = _load_json(PATHS.results / "rq1_comparison.json")
    if not payload:
        return []
    cats = set()
    for arm in ("scratch", "pretrained"):
        cats.update((payload.get(arm) or {}).get("per_category_top1", {}).keys())
    if not cats:
        return []
    rows = []
    for cat in sorted(cats):
        row = {"Category": cat}
        for arm, label in (("scratch", "Scratch top-1"), ("pretrained", "Pretrained top-1")):
            acc = (payload.get(arm) or {}).get("per_category_top1", {}).get(cat)
            row[label] = f"{100 * acc:.1f}%" if acc is not None else "—"
            if acc is not None:
                row[label.replace("top-1", "error")] = f"{100 * (1 - acc):.1f}%"
        rows.append(row)
    return _write_table(pd.DataFrame(rows), "table_per_category", "Per-category accuracy and error on the held-out signer (§3.6).", "tab:per-category")


def table_per_sentence() -> list[Path]:
    paths = list(PATHS.results.glob("rq1_per_sentence_*.csv"))
    written = []
    for path in paths:
        df = pd.read_csv(path)
        if "error" in df.columns:
            df = df.sort_values("error", ascending=False).head(25)
        stem = path.stem.replace("rq1_", "table_")
        written.extend(
            _write_table(df, stem, "Highest-error sentences on the held-out signer (top 25).", f"tab:{stem}")
        )
    return written


def table_eval_protocol() -> list[Path]:
    df = pd.DataFrame(
        [
            {"Quantity": "Overall top-1 / top-5", "Where reported": "RQ1 table + bar chart", "Notes": "Signer-independent test signer"},
            {"Quantity": "Per-sentence accuracy/error $e_j$", "Where reported": "table_per_sentence_*.csv", "Notes": "§3.5.2 / §3.6"},
            {"Quantity": "Per-category error $e_g$", "Where reported": "heatmap + per-category table", "Notes": "Mean of sentence errors in the category"},
            {"Quantity": "Signer-diversity curve", "Where reported": "Fig 7 (after 07_signer_scaling.py)", "Notes": "RQ3; pretrained vs scratch"},
            {"Quantity": r"$\Delta_{AL}$", "Where reported": "Fig RQ2 (after 06_simulate_active_learning.py)", "Notes": "Matched signer/sentence budget"},
        ]
    )
    return _write_table(df, "table_eval_protocol", "Evaluation quantities required by §3.6.", "tab:eval")


def write_manifest(written: list[Path]) -> Path:
    PATHS.figures.mkdir(parents=True, exist_ok=True)
    man = PATHS.figures / "MANIFEST.txt"
    lines = [
        "Paper figures and tables",
        "Re-run: python scripts/08_make_figures.py",
        "Methodology figs 1–6 always generate. Results figs appear when JSON/CSV exist.",
        "",
    ]
    for p in written:
        lines.append(str(p.relative_to(PATHS.repo_root)))
    man.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return man


def generate_all(*, methods: bool = True, results: bool = True) -> list[Path]:
    written: list[Path] = []
    if methods:
        written += fig_pipeline_overview()
        written += fig_landmark_normalization()
        written += fig_masked_pose()
        written += fig_contrastive()
        written += fig_al_loop()
        written += fig_experimental_arms()
        written += table_dataset()
        written += table_split()
        written += table_hyperparams()
        written += table_eval_protocol()
    if results:
        written += fig_ssl_curve()
        written += fig_finetune_curves()
        written += fig_rq1_bars()
        written += fig_rq1_diagnostics()
        written += fig_category_heatmap()
        written += fig_rq2_curves()
        written += fig_rq3_scaling()
        written += fig_rq4_calibration()
        written += fig_rq5_synthetic()
        written += fig_rq6_augment()
        written += table_rq1()
        written += table_rq6()
        written += table_rq1_diagnostics()
        written += table_per_category()
        written += table_per_sentence()
    write_manifest(written)
    return written
