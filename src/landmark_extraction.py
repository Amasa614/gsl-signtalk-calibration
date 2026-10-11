"""MediaPipe pose / hand / face landmark extraction.

Default: HolisticLandmarker on CPU.
Fallback: PoseLandmarker + HandLandmarker + FaceLandmarker (USE_SEPARATE_MODELS).

See README "Known MediaPipe gotchas".
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.config import CFG, FEATURE_DIM, PATHS
from src.normalize import flatten_frame, normalize_sequence, pack_landmarks

# Flip this to True if HolisticLandmarker landmarks look wrong or GPU delegate
# throws DEQUANTIZE / STRIDED_SLICE. Equivalent to CFG.extract.use_separate_models.
USE_SEPARATE_MODELS = False


def _delegate(use_gpu: bool):
    from mediapipe.tasks.python.core.base_options import BaseOptions

    if use_gpu:
        return BaseOptions.Delegate.GPU
    return BaseOptions.Delegate.CPU


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {what} model at {path}. Run: python scripts/download_mp_models.py"
        )
    return path


class LandmarkExtractor:
    def __init__(self, use_separate: bool | None = None, use_gpu: bool | None = None):
        self.use_separate = (
            USE_SEPARATE_MODELS if use_separate is None else use_separate
        ) or CFG.extract.use_separate_models
        self.use_gpu = CFG.extract.use_gpu if use_gpu is None else use_gpu
        self._holistic = None
        self._pose = None
        self._hands = None
        self._face = None
        self._init_models()

    def _init_models(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            HandLandmarker,
            HandLandmarkerOptions,
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )

        self._mp = mp
        delegate = _delegate(self.use_gpu)
        print(f"[extract] delegate={'GPU' if self.use_gpu else 'CPU'}  max_frame_width={CFG.extract.max_frame_width}")
        conf = CFG.extract.min_detection_confidence

        if not self.use_separate:
            try:
                from mediapipe.tasks.python.vision import (
                    HolisticLandmarker,
                    HolisticLandmarkerOptions,
                )

                options = HolisticLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(_require(PATHS.holistic_model, "holistic")),
                        delegate=delegate,
                    ),
                    running_mode=RunningMode.IMAGE,
                )
                try:
                    self._holistic = HolisticLandmarker.create_from_options(options)
                    return
                except Exception as exc:
                    if self.use_gpu:
                        print(f"[extract] Holistic GPU init failed ({exc}); falling back to CPU.")
                        self.use_gpu = False
                        options = HolisticLandmarkerOptions(
                            base_options=BaseOptions(
                                model_asset_path=str(_require(PATHS.holistic_model, "holistic")),
                                delegate=_delegate(False),
                            ),
                            running_mode=RunningMode.IMAGE,
                        )
                        self._holistic = HolisticLandmarker.create_from_options(options)
                        return
                    raise
            except (ImportError, FileNotFoundError, OSError) as exc:
                print(f"[extract] HolisticLandmarker unavailable ({exc}); using separate models.")
                self.use_separate = True

        self._pose = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(_require(PATHS.pose_model, "pose")),
                    delegate=delegate,
                ),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=conf,
                min_pose_presence_confidence=conf,
                min_tracking_confidence=CFG.extract.min_tracking_confidence,
            )
        )
        self._hands = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(_require(PATHS.hand_model, "hand")),
                    delegate=delegate,
                ),
                running_mode=RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=conf,
                min_hand_presence_confidence=conf,
                min_tracking_confidence=CFG.extract.min_tracking_confidence,
            )
        )
        try:
            self._face = FaceLandmarker.create_from_options(
                FaceLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(_require(PATHS.face_model, "face")),
                        delegate=delegate,
                    ),
                    running_mode=RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=conf,
                    min_face_presence_confidence=conf,
                    min_tracking_confidence=CFG.extract.min_tracking_confidence,
                )
            )
        except FileNotFoundError:
            print("[extract] face_landmarker.task missing; continuing without face NMMs.")
            self._face = None

    def close(self) -> None:
        for obj in (self._holistic, self._pose, self._hands, self._face):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _mp_image(self, rgb: np.ndarray):
        return self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )

    def detect_frame(self, rgb: np.ndarray, timestamp_ms: int) -> np.ndarray:
        image = self._mp_image(rgb)
        if self._holistic is not None:
            result = self._holistic.detect(image)
            pose = _first_person(result.pose_landmarks)
            left = _first_person(result.left_hand_landmarks)
            right = _first_person(result.right_hand_landmarks)
            face = _first_person(result.face_landmarks)
            return pack_landmarks(pose, left, right, face)

        pose = hands_l = hands_r = face = None
        if self._pose is not None:
            pres = self._pose.detect(image)
            if pres.pose_landmarks:
                pose = pres.pose_landmarks[0]
        if self._hands is not None:
            hres = self._hands.detect(image)
            hands_l, hands_r = _split_hands(hres)
        if self._face is not None:
            fres = self._face.detect(image)
            if fres.face_landmarks:
                face = fres.face_landmarks[0]
        return pack_landmarks(pose, hands_l, hands_r, face)

    def extract_video(self, video_path: Path) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_ms = max(int(round(1000.0 / fps)), 1)
        max_w = CFG.extract.max_frame_width
        frames = []
        timestamp_ms = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if max_w and w > max_w:
                nh = int(round(h * (max_w / w)))
                rgb = cv2.resize(rgb, (max_w, nh), interpolation=cv2.INTER_AREA)
            packed = self.detect_frame(rgb, timestamp_ms)
            frames.append(flatten_frame(packed))
            timestamp_ms += frame_ms
        cap.release()
        if not frames:
            return np.zeros((0, FEATURE_DIM), dtype=np.float32)
        seq = np.stack(frames, axis=0)
        return normalize_sequence(seq)


def _first_person(landmarks):
    """Return a sequence of joints for one person from either MP API shape."""
    if landmarks is None:
        return None
    if hasattr(landmarks, "x"):
        return None
    try:
        if len(landmarks) == 0:
            return None
    except TypeError:
        return None
    first = landmarks[0]
    if hasattr(first, "x"):
        return landmarks
    return first


def _split_hands(hres):
    left = right = None
    if not hres.hand_landmarks:
        return left, right
    handedness = getattr(hres, "handedness", None) or []
    for i, hand in enumerate(hres.hand_landmarks):
        label = "Unknown"
        if i < len(handedness) and handedness[i]:
            cat = handedness[i][0] if isinstance(handedness[i], (list, tuple)) else handedness[i]
            label = getattr(cat, "category_name", None) or getattr(cat, "display_name", "") or ""
        if str(label).lower().startswith("left"):
            left = hand
        elif str(label).lower().startswith("right"):
            right = hand
        else:
            if left is None:
                left = hand
            else:
                right = hand
    return left, right


def landmark_out_path(video_id: str) -> Path:
    return PATHS.landmarks / f"{video_id}.npz"


def save_landmarks(video_id: str, landmarks: np.ndarray, extra: dict | None = None) -> Path:
    PATHS.landmarks.mkdir(parents=True, exist_ok=True)
    out = landmark_out_path(video_id)
    payload = {"landmarks": landmarks.astype(np.float32), "video_id": np.array(video_id)}
    if extra:
        for k, v in extra.items():
            payload[k] = v
    np.savez_compressed(out, **payload)
    return out


def load_landmarks(video_id: str | Path) -> np.ndarray:
    path = Path(video_id)
    if path.suffix != ".npz":
        path = landmark_out_path(str(video_id))
    with np.load(path, allow_pickle=False) as data:
        return data["landmarks"].astype(np.float32)


_WORKER: LandmarkExtractor | None = None


def _worker_init(use_separate: bool | None, use_gpu: bool) -> None:
    global _WORKER
    _WORKER = LandmarkExtractor(use_separate=use_separate, use_gpu=use_gpu)


def _worker_job(row: tuple[str, str, str, str]) -> str:
    video_id, path, signer_id, sentence_id = row
    assert _WORKER is not None
    seq = _WORKER.extract_video(Path(path))
    out = save_landmarks(
        video_id,
        seq,
        extra={
            "signer_id": np.array(signer_id),
            "sentence_id": np.array(sentence_id),
        },
    )
    return str(out)


def extract_index(
    index,
    *,
    max_videos: int | None = None,
    skip_existing: bool = True,
    use_separate: bool | None = None,
    use_gpu: bool | None = None,
    workers: int | None = None,
) -> list[Path]:
    subset = index if max_videos is None else index.head(max_videos)
    written: list[Path] = []
    used_gpu = CFG.extract.use_gpu if use_gpu is None else use_gpu
    n_workers = workers if workers is not None else max(1, min(10, (os.cpu_count() or 2) - 2))

    jobs: list[tuple[str, str, str, str]] = []
    for _, row in subset.iterrows():
        video_id = str(row["video_id"])
        out = landmark_out_path(video_id)
        if skip_existing and out.exists():
            written.append(out)
            continue
        jobs.append(
            (video_id, str(row["path"]), str(row["signer_id"]), str(row["sentence_id"]))
        )
    print(f"[extract] already done={len(written)}  remaining={len(jobs)}  workers={n_workers}")
    if not jobs:
        used_separate = USE_SEPARATE_MODELS
    elif n_workers <= 1:
        used_separate, used_gpu = _extract_sequential(jobs, written, use_separate, used_gpu)
    else:
        used_separate = bool(use_separate) or USE_SEPARATE_MODELS
        used_gpu = False
        print("[extract] parallel CPU workers (MediaPipe Windows wheels cannot use NVIDIA GPU)")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(use_separate, False),
        ) as pool:
            futs = {pool.submit(_worker_job, job): job[0] for job in jobs}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="extract landmarks"):
                written.append(Path(fut.result()))

    meta = {
        "feature_dim": FEATURE_DIM,
        "n_videos": len(written),
        "use_separate_models": used_separate,
        "use_gpu": used_gpu,
        "max_frame_width": CFG.extract.max_frame_width,
        "workers": n_workers,
    }
    (PATHS.landmarks / "extraction_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return written


def _extract_sequential(jobs, written, use_separate, used_gpu):
    try:
        extractor_cm = LandmarkExtractor(use_separate=use_separate, use_gpu=used_gpu)
        extractor = extractor_cm.__enter__()
    except Exception as exc:
        if used_gpu:
            print(f"[extract] GPU delegate failed ({exc}); retrying on CPU.")
            used_gpu = False
            extractor_cm = LandmarkExtractor(use_separate=use_separate, use_gpu=False)
            extractor = extractor_cm.__enter__()
        else:
            raise
    try:
        used_separate = extractor.use_separate
        used_gpu = extractor.use_gpu
        for video_id, path, signer_id, sentence_id in tqdm(jobs, desc="extract landmarks"):
            seq = extractor.extract_video(Path(path))
            written.append(
                save_landmarks(
                    video_id,
                    seq,
                    extra={
                        "signer_id": np.array(signer_id),
                        "sentence_id": np.array(sentence_id),
                    },
                )
            )
    finally:
        extractor_cm.__exit__(None, None, None)
    return used_separate, used_gpu
