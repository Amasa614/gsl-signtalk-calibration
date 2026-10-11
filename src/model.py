"""Modest Transformer encoder over pose-landmark windows.

SSL pretext: masked pose reconstruction (BERT-style on skeleton sequences).
Optional: InfoNCE contrastive loss on two augmented views (no L/R mirroring).
Downstream: sentence classification head for sign-to-text (RQ1 comparison).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CFG, FEATURE_DIM, N_COORDS, N_JOINTS


class LearnedPositionalEncoding(nn.Module):
    """Learned temporal positional encodings (methodology §3.3.3)."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MaskedPoseTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        d_model: int | None = None,
        nhead: int | None = None,
        num_layers: int | None = None,
        dim_feedforward: int | None = None,
        dropout: float | None = None,
        max_len: int | None = None,
        n_classes: int | None = None,
    ):
        super().__init__()
        cfg = CFG.model
        self.feature_dim = feature_dim
        self.d_model = d_model or cfg.d_model
        self.input_proj = nn.Linear(feature_dim, self.d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.input_mask_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.pos = LearnedPositionalEncoding(self.d_model, max_len or cfg.max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead or cfg.nhead,
            dim_feedforward=dim_feedforward or cfg.dim_feedforward,
            dropout=dropout if dropout is not None else cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers or cfg.num_layers,
            enable_nested_tensor=False,
        )
        self.recon_head = nn.Linear(self.d_model, feature_dim)
        proj_dim = cfg.proj_dim
        self.proj_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, proj_dim),
        )
        self.cls_head = nn.Linear(self.d_model, n_classes) if n_classes else None
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.input_mask_token, std=0.02)

    def encode(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, D). frame_mask True = replace frame with the mask token."""
        h = self.input_proj(x)
        if frame_mask is not None:
            h = torch.where(frame_mask.unsqueeze(-1), self.mask_token.expand_as(h), h)
        h = self.pos(h)
        return self.encoder(h)

    def reconstruct(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.recon_head(self.encode(x, frame_mask))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, frame_mask=None)
        return h.mean(dim=1)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Contrastive projection g_φ(f_θ(W)) from methodology §3.3.2."""
        return self.proj_head(self.embed(x))

    def replace_masked_inputs(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Replace masked joints/coords with a learned input-space mask token."""
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1).expand_as(x)
        return torch.where(mask, self.input_mask_token.expand_as(x), x)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        if self.cls_head is None:
            raise RuntimeError("n_classes was not set; cannot classify")
        return self.cls_head(self.embed(x))

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.cls_head is not None and frame_mask is None:
            return self.classify(x)
        return self.reconstruct(x, frame_mask)

    def set_num_classes(self, n_classes: int) -> None:
        self.cls_head = nn.Linear(self.d_model, n_classes)


class PoseBiGRU(nn.Module):
    """Cheap alternate supervised backbone: BiGRU over pose windows.

    Same window interface as ``MaskedPoseTransformer`` (embed / classify) so
    RQ3/RQ6/eval can swap architectures without rewriting the pipeline.
    Not used for masked-pose SSL; scratch supervised only.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_size: int | None = None,
        num_layers: int | None = None,
        dropout: float | None = None,
        bidirectional: bool = True,
        n_classes: int | None = None,
    ):
        super().__init__()
        cfg = CFG.model
        self.feature_dim = feature_dim
        h = hidden_size if hidden_size is not None else getattr(cfg, "bigru_hidden", 64)
        n_layers = num_layers if num_layers is not None else getattr(cfg, "bigru_layers", 2)
        drop = dropout if dropout is not None else cfg.dropout
        self.bidirectional = bidirectional
        self.d_model = h * (2 if bidirectional else 1)
        self.input_proj = nn.Linear(feature_dim, h)
        self.input_mask_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.gru = nn.GRU(
            input_size=h,
            hidden_size=h,
            num_layers=n_layers,
            batch_first=True,
            dropout=drop if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        # Plain view (not nn.Module) so GRU params are not registered twice.
        self.encoder = _GRUEncoderView(self.gru, n_layers=n_layers, bidirectional=bidirectional)
        self.cls_head = nn.Linear(self.d_model, n_classes) if n_classes else None
        nn.init.normal_(self.input_mask_token, std=0.02)

    def encode(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        del frame_mask  # SSL-style frame masks are Transformer-only
        h = self.input_proj(x)
        out, _ = self.gru(h)
        return out

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x).mean(dim=1)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        if self.cls_head is None:
            raise RuntimeError("n_classes was not set; cannot classify")
        return self.cls_head(self.embed(x))

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        del frame_mask
        return self.classify(x)

    def set_num_classes(self, n_classes: int) -> None:
        self.cls_head = nn.Linear(self.d_model, n_classes)


class _GRUEncoderView:
    """Non-module view so freeze / partial-FT can use ``model.encoder`` like Transformer."""

    def __init__(self, gru: nn.GRU, *, n_layers: int, bidirectional: bool):
        self.gru = gru
        self.layers = [_GRULayerView(gru, i, bidirectional) for i in range(n_layers)]

    def parameters(self, recurse: bool = True):
        del recurse
        return self.gru.parameters()


class _GRULayerView:
    """Parameter view of one stacked GRU layer (for requires_grad toggling)."""

    def __init__(self, gru: nn.GRU, layer_idx: int, bidirectional: bool):
        self.gru = gru
        self.layer_idx = layer_idx
        self.bidirectional = bidirectional

    def parameters(self, recurse: bool = True):
        del recurse
        names = [
            f"weight_ih_l{self.layer_idx}",
            f"weight_hh_l{self.layer_idx}",
            f"bias_ih_l{self.layer_idx}",
            f"bias_hh_l{self.layer_idx}",
        ]
        if self.bidirectional:
            names += [
                f"weight_ih_l{self.layer_idx}_reverse",
                f"weight_hh_l{self.layer_idx}_reverse",
                f"bias_ih_l{self.layer_idx}_reverse",
                f"bias_hh_l{self.layer_idx}_reverse",
            ]
        for name in names:
            if hasattr(self.gru, name):
                yield getattr(self.gru, name)


def build_pose_model(
    arch: str = "transformer",
    *,
    feature_dim: int = FEATURE_DIM,
    n_classes: int | None = None,
    **kwargs,
) -> nn.Module:
    """Factory for supervised pose backbones used by fine-tune / RQ scripts."""
    key = str(arch).lower().strip().replace("-", "")
    if key in {"transformer", "maskedpose", "mp", "t"}:
        return MaskedPoseTransformer(feature_dim=feature_dim, n_classes=n_classes, **kwargs)
    if key in {"bigru", "bi_gru", "gru"}:
        return PoseBiGRU(feature_dim=feature_dim, n_classes=n_classes, **kwargs)
    raise ValueError(f"unknown arch={arch!r}; expected 'transformer' or 'bigru'")


def _sample_ratios(
    n: int,
    device: torch.device,
    mask_ratio: float | None,
    ratio_range: tuple[float, float],
    generator: torch.Generator | None,
) -> torch.Tensor:
    if mask_ratio is not None:
        return torch.full((n, 1), float(mask_ratio), device=device)
    lo, hi = ratio_range
    u = torch.rand(n, 1, device=device, generator=generator)
    return lo + (hi - lo) * u


def make_frame_mask(
    batch: torch.Tensor,
    mask_ratio: float | None = None,
    ratio_range: tuple[float, float] = (0.15, 0.40),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Boolean mask over frames. True = masked.

    If mask_ratio is None, sample ρ ~ Uniform[0.15, 0.40] per window (§3.3.1).
    """
    b, t, _ = batch.shape
    rho = _sample_ratios(b, batch.device, mask_ratio, ratio_range, generator)
    noise = torch.rand(b, t, device=batch.device, generator=generator)
    mask = noise < rho
    empty = ~mask.any(dim=1)
    if empty.any():
        rows = empty.nonzero(as_tuple=True)[0]
        mask[rows, noise[rows].argmin(dim=1)] = True
    return mask


def make_joint_mask(
    batch: torch.Tensor,
    mask_ratio: float | None = None,
    ratio_range: tuple[float, float] = (0.15, 0.40),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a (B, T, D) boolean mask over joints (all 3 coords of a joint together)."""
    b, t, d = batch.shape
    if d != N_JOINTS * N_COORDS:
        raise ValueError("joint masking expects packed (N_JOINTS * 3) features")
    rho = _sample_ratios(b, batch.device, mask_ratio, ratio_range, generator).unsqueeze(-1)
    noise = torch.rand(b, t, N_JOINTS, device=batch.device, generator=generator)
    joint_mask = noise < rho
    empty = ~joint_mask.any(dim=-1)
    if empty.any():
        b_idx, t_idx = empty.nonzero(as_tuple=True)
        j_idx = noise[b_idx, t_idx].argmin(dim=-1)
        joint_mask[b_idx, t_idx, j_idx] = True
    return joint_mask.unsqueeze(-1).expand(-1, -1, -1, N_COORDS).reshape(b, t, d)


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    kind: str = "smooth_l1",
) -> torch.Tensor:
    """Compute loss only on masked frames (or masked joints if mask is D-dimensional)."""
    if kind == "mse":
        per = F.mse_loss(pred, target, reduction="none")
    else:
        per = F.smooth_l1_loss(pred, target, reduction="none")
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1).expand_as(per)
    denom = mask.float().sum().clamp_min(1.0)
    return (per * mask.float()).sum() / denom


def info_nce(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """InfoNCE as in methodology §3.3.2: positives are the paired views, negatives are other-view items."""
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = z_a @ z_b.T / temperature
    labels = torch.arange(z_a.size(0), device=z_a.device)
    return F.cross_entropy(logits, labels)


def nt_xent(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Alias kept so older scripts/imports keep working."""
    return info_nce(z_a, z_b, temperature)
