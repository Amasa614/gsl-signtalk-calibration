"""Central configuration for the GSL RQ1 / RQ2 pipeline.

Paths are relative to the repository root. Override any field from the CLI
of the corresponding script; this module is the single source of defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Landmark feature layout
# ---------------------------------------------------------------------------
# MediaPipe Pose: 33 body landmarks.
# MediaPipe Hands: 21 landmarks per hand.
# Face: a reduced non-manual-marker (NMM) subset of Face Mesh (468).
N_POSE = 33
N_HAND = 21
# Face Mesh indices used as non-manual markers (eyebrows, eyes, nose, mouth, chin).
FACE_NMM_INDICES: tuple[int, ...] = (
    1, 4,  # nose tip / lower nose
    33, 133, 159, 145,  # right eye: outer, inner, top, bottom
    362, 263, 386, 374,  # left eye
    70, 105, 107,  # right eyebrow
    336, 334, 296,  # left eyebrow
    61, 291,  # mouth corners
    13, 14,  # inner upper / lower lip
    78, 308,  # inner mouth corners
    0, 17,  # lips
    152,  # chin
)
N_FACE = len(FACE_NMM_INDICES)
N_COORDS = 3  # x, y, z
N_JOINTS = N_POSE + 2 * N_HAND + N_FACE
FEATURE_DIM = N_JOINTS * N_COORDS  # 300

# Pose landmark indices used for torso centering / shoulder-width scaling.
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def _default_raw_videos() -> Path:
    signtalk = REPO_ROOT / "SignTalk-GH"
    if signtalk.exists():
        return signtalk
    return REPO_ROOT / "data" / "raw_videos"


def _default_fragkiadakis() -> Path:
    """Prefer a repo-root GSL_openpose_data dump when present."""
    root_dump = REPO_ROOT / "GSL_openpose_data"
    if root_dump.exists():
        return root_dump
    nested = REPO_ROOT / "data" / "external" / "fragkiadakis" / "GSL_openpose_data"
    if nested.exists():
        return nested
    return REPO_ROOT / "data" / "external" / "fragkiadakis"


def _default_metadata_path() -> Path | None:
    for candidate in (
        REPO_ROOT / "SignTalk-GH" / "Metadata.xlsx",
        REPO_ROOT / "SignTalk-GH" / "Metadata.csv",
        REPO_ROOT / "data" / "raw_videos" / "Metadata.xlsx",
        REPO_ROOT / "data" / "raw_videos" / "Metadata.csv",
    ):
        if candidate.exists():
            return candidate
    return None


@dataclass
class Paths:
    repo_root: Path = REPO_ROOT
    raw_videos: Path = field(default_factory=_default_raw_videos)
    landmarks: Path = REPO_ROOT / "data" / "landmarks"
    splits: Path = REPO_ROOT / "data" / "splits"
    checkpoints: Path = REPO_ROOT / "checkpoints"
    mp_models: Path = REPO_ROOT / "mp_models"
    results: Path = REPO_ROOT / "results"
    figures: Path = REPO_ROOT / "results" / "figures"
    tables: Path = REPO_ROOT / "results" / "tables"

    # Auto-detected SignTalk-GH Metadata.xlsx when present.
    metadata_path: Path | None = field(default_factory=_default_metadata_path)

    external: Path = REPO_ROOT / "data" / "external"
    fragkiadakis: Path = field(default_factory=_default_fragkiadakis)
    afrisign: Path = REPO_ROOT / "data" / "external" / "afrisign"
    ssl_index: Path = REPO_ROOT / "data" / "splits" / "ssl_index.csv"

    holistic_model: Path = REPO_ROOT / "mp_models" / "holistic_landmarker.task"
    pose_model: Path = REPO_ROOT / "mp_models" / "pose_landmarker_full.task"
    hand_model: Path = REPO_ROOT / "mp_models" / "hand_landmarker.task"
    face_model: Path = REPO_ROOT / "mp_models" / "face_landmarker.task"

    ssl_encoder: Path = REPO_ROOT / "checkpoints" / "ssl_encoder.pt"
    finetune_pretrained: Path = REPO_ROOT / "checkpoints" / "finetune_pretrained.pt"
    finetune_scratch: Path = REPO_ROOT / "checkpoints" / "finetune_scratch.pt"
    finetune_pretrained_category: Path = REPO_ROOT / "checkpoints" / "finetune_pretrained_category.pt"
    finetune_scratch_category: Path = REPO_ROOT / "checkpoints" / "finetune_scratch_category.pt"
    label_map: Path = REPO_ROOT / "data" / "splits" / "label_map.json"
    category_map: Path = REPO_ROOT / "data" / "splits" / "category_map.json"
    signer_map: Path = REPO_ROOT / "data" / "splits" / "signer_map.csv"


@dataclass
class ExtractConfig:
    """MediaPipe extraction. CPU delegate is the default on purpose.

    HolisticLandmarker (Tasks API) can produce worse pose landmarks than the
    standalone PoseLandmarker, and GPU delegation can throw DEQUANTIZE /
    STRIDED_SLICE errors. If either happens, set use_separate_models=True.
    """

    use_separate_models: bool = False
    use_gpu: bool = True
    # 1080p Holistic is the main bottleneck. Landmarks are normalized, so
    # shrinking the frame does not change the (x, y) coordinate system.
    max_frame_width: int = 640
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class WindowConfig:
    # Methodology §3.2: L ∈ {64, 128}, stride σ < L.
    window_size: int = 64
    stride: int = 32
    min_frames: int = 16


@dataclass
class SplitConfig:
    """Signer-independent split: never shuffle clips across signers.

    Default: last signer = test, second-last = val, the rest = train.
    Override with explicit signer IDs after inspecting the metadata.
    """

    test_signers: tuple[str, ...] = ()
    val_signers: tuple[str, ...] = ()
    seed: int = 42


@dataclass
class ModelConfig:
    feature_dim: int = FEATURE_DIM
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_len: int = 256
    proj_dim: int = 64  # contrastive projection g_φ output dim (§3.3.2)
    # PoseBiGRU (alt supervised backbone): hidden per direction → d_model = 2 * hidden
    bigru_hidden: int = 64
    bigru_layers: int = 2


@dataclass
class SSLConfig:
    # None => sample ρ ~ Uniform[mask_ratio_min, mask_ratio_max] per window (§3.3.1).
    mask_ratio: float | None = None
    mask_ratio_min: float = 0.15
    mask_ratio_max: float = 0.40
    mask_mode: str = "frame"  # "frame" or "joint"
    loss: str = "smooth_l1"  # "smooth_l1" or "mse"
    contrastive: bool = False
    contrastive_weight: float = 0.5
    temperature: float = 0.1
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-2
    num_workers: int = 0
    seed: int = 42


@dataclass
class FinetuneConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-2
    num_workers: int = 0
    seed: int = 42
    freeze_encoder_epochs: int = 0


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)


CFG = Config()
PATHS = CFG.paths
