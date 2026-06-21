from __future__ import annotations

import importlib.util
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import paths  # noqa: F401  — configure sys.path before local imports
from paths import BBOX_DIR

_yolo_spec = importlib.util.spec_from_file_location(
    "yolo_inference",
    BBOX_DIR / "yolo_inference.py",
)
if _yolo_spec is None or _yolo_spec.loader is None:
    raise ImportError(f"Cannot load YOLO inference from {BBOX_DIR}")
_yolo_mod = importlib.util.module_from_spec(_yolo_spec)
_yolo_spec.loader.exec_module(_yolo_mod)
detect_largest_person_bbox = _yolo_mod.detect_largest_person_bbox
load_yolo_model = _yolo_mod.load_yolo_model

from config_loader import AppConfig, load_config
from mmpose_vitpose_estimator import MMPoseEmbeddingSource, MMPoseVitPoseEstimator
from model import EmbeddingStandardScaler, ID_TO_LABEL, LSTMActivityClassifier, predict_label

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def crop_person_image(
    image: Image.Image,
    bbox_xyxy: tuple[float, float, float, float],
) -> Image.Image:
    x1, y1, x2, y2 = bbox_xyxy
    width, height = image.size
    left = max(0, min(width, int(round(x1))))
    top = max(0, min(height, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def sorted_image_paths(folder: Path) -> list[Path]:
    paths_list = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths_list, key=lambda path: path.name)


def bbox_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_diagonal(bbox: list[float]) -> float:
    w = max(0.0, bbox[2] - bbox[0])
    h = max(0.0, bbox[3] - bbox[1])
    return float((w * w + h * h) ** 0.5)


def find_last_fall_frame_index(frames: list[FrameResult]) -> int | None:
    """Return list index of the last frame predicted as fall."""

    last_index: int | None = None
    for list_index, frame in enumerate(frames):
        if frame.prediction and frame.prediction.get("label") == "fall":
            last_index = list_index
    return last_index


def build_fall_confirmation(frames: list[FrameResult], config: AppConfig) -> dict[str, Any]:
    last_list_index = find_last_fall_frame_index(frames)
    if last_list_index is None:
        return {
            "last_fall_frame_index": None,
            "last_fall_frame_name": None,
            "stability": None,
            "trigger_agent": False,
            "reason": "no_fall_prediction_in_sequence",
        }

    last_fall_frame = frames[last_list_index]
    stability = analyze_bbox_stability(
        frames,
        last_list_index,
        num_frames=config.stability_frames,
        min_mean_iou=config.stability_min_mean_iou,
        max_center_shift=config.stability_max_center_shift,
    )
    return {
        "last_fall_frame_index": last_fall_frame.index,
        "last_fall_frame_name": last_fall_frame.name,
        "stability": stability,
        "trigger_agent": bool(stability.get("confirmed_fall")),
    }


def analyze_bbox_stability(
    frames: list[FrameResult],
    start_index: int,
    *,
    num_frames: int,
    min_mean_iou: float,
    max_center_shift: float,
) -> dict[str, Any]:
    end_exclusive = min(start_index + num_frames, len(frames))
    segment = frames[start_index:end_exclusive]
    valid = [f for f in segment if f.bbox_xyxy is not None and f.embedding_ok]

    result: dict[str, Any] = {
        "start_index": start_index,
        "end_index": end_exclusive - 1 if end_exclusive > start_index else start_index,
        "frames_requested": num_frames,
        "frames_available": len(segment),
        "frames_with_bbox": len(valid),
        "mean_iou": None,
        "mean_center_shift_ratio": None,
        "is_stable": False,
        "confirmed_fall": False,
        "reason": None,
    }

    if not valid:
        result["reason"] = "no_valid_bbox_frames"
        return result

    if len(segment) < num_frames:
        result["reason"] = "not_enough_frames_after_last_fall"
        return result

    if len(valid) < 2:
        result["reason"] = "need_at_least_two_bbox_frames"
        return result

    ious: list[float] = []
    shifts: list[float] = []
    for prev, curr in zip(valid, valid[1:]):
        assert prev.bbox_xyxy is not None and curr.bbox_xyxy is not None
        ious.append(bbox_iou(prev.bbox_xyxy, curr.bbox_xyxy))
        c_prev = bbox_center(prev.bbox_xyxy)
        c_curr = bbox_center(curr.bbox_xyxy)
        dist = ((c_curr[0] - c_prev[0]) ** 2 + (c_curr[1] - c_prev[1]) ** 2) ** 0.5
        ref = max(bbox_diagonal(prev.bbox_xyxy), 1.0)
        shifts.append(float(dist / ref))

    mean_iou = float(np.mean(ious))
    mean_shift = float(np.mean(shifts))
    is_stable = mean_iou >= min_mean_iou and mean_shift <= max_center_shift

    result["mean_iou"] = mean_iou
    result["mean_center_shift_ratio"] = mean_shift
    result["is_stable"] = is_stable
    result["confirmed_fall"] = is_stable
    if not is_stable:
        result["reason"] = "bbox_changed_too_much"
    return result


@dataclass
class FrameResult:
    index: int
    name: str
    bbox_xyxy: list[float] | None
    bbox_fallback: bool
    embedding_ok: bool
    buffer_size: int
    prediction: dict[str, Any] | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "index": self.index,
            "name": self.name,
            "bbox_xyxy": self.bbox_xyxy,
            "bbox_fallback": self.bbox_fallback,
            "embedding_ok": self.embedding_ok,
            "buffer_size": self.buffer_size,
            "prediction": self.prediction,
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload


class FallDetectionPipeline:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.device = resolve_device(self.config.device)
        self._detector_model = None
        self._vitpose: MMPoseVitPoseEstimator | None = None
        self._lstm: LSTMActivityClassifier | None = None
        self._scaler: EmbeddingStandardScaler | None = None
        self._checkpoint_meta: dict[str, Any] = {}
        self.reload()

    def reload(self, config: AppConfig | None = None) -> None:
        if config is not None:
            self.config = config
        self.device = resolve_device(self.config.device)

        if not self.config.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.config.checkpoint}")

        ckpt = torch.load(self.config.checkpoint, map_location=self.device, weights_only=False)
        args = ckpt.get("args") or {}

        self._lstm = LSTMActivityClassifier(
            input_dim=self.config.embedding_dim,
            hidden_dim=int(args.get("hidden_dim", 256)),
            num_layers=int(args.get("num_layers", 2)),
            bidirectional=bool(args.get("bidirectional", True)),
            dropout=float(args.get("dropout", 0.2)),
            pooling=args.get("pooling", "last"),
        ).to(self.device)
        self._lstm.load_state_dict(ckpt["model_state"])
        self._lstm.eval()

        scaler_data = ckpt.get("scaler")
        self._scaler = (
            EmbeddingStandardScaler.from_dict(scaler_data) if scaler_data is not None else None
        )

        self._checkpoint_meta = {
            "epoch": ckpt.get("epoch"),
            "val_loss": float(ckpt.get("val_loss", float("nan"))),
            "val_acc": float(ckpt.get("val_acc", float("nan"))),
        }

        if not self.config.mmpose_config.is_file():
            raise FileNotFoundError(f"MMPose config not found: {self.config.mmpose_config}")
        if not self.config.mmpose_checkpoint.is_file():
            raise FileNotFoundError(
                f"MMPose checkpoint not found: {self.config.mmpose_checkpoint}"
            )

        self._detector_model = load_yolo_model(self.config.rtdetr_model)
        self._vitpose = MMPoseVitPoseEstimator(
            config_path=self.config.mmpose_config,
            checkpoint_path=self.config.mmpose_checkpoint,
            device=self.device,
        )

    @property
    def status(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "config": self.config.to_public_dict(),
            "checkpoint_meta": self._checkpoint_meta,
            "models_loaded": all(
                item is not None for item in (self._detector_model, self._vitpose, self._lstm)
            ),
        }

    def _extract_embedding(
        self,
        image: Image.Image,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> np.ndarray:
        assert self._vitpose is not None
        cropped = crop_person_image(image, bbox_xyxy)
        source: MMPoseEmbeddingSource = self.config.embedding_source  # type: ignore[assignment]
        return self._vitpose.extract_embedding(
            image=cropped,
            source=source,
        )

    @torch.no_grad()
    def _predict_window(self, window: deque[np.ndarray]) -> dict[str, Any]:
        assert self._lstm is not None

        stacked = np.stack(list(window), axis=0).astype(np.float32)
        x = torch.from_numpy(stacked).unsqueeze(0).to(self.device)
        lengths = torch.tensor([x.shape[1]], dtype=torch.long, device=self.device)

        if self._scaler is not None:
            x = self._scaler.transform(x)

        logits = self._lstm(x, lengths)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().tolist()
        label = predict_label(logits)[0]
        label_id = 1 if label == "fall" else 0

        return {
            "label": label,
            "confidence": float(probs[label_id]),
            "probabilities": {
                ID_TO_LABEL[0]: float(probs[0]),
                ID_TO_LABEL[1]: float(probs[1]),
            },
        }

    def process_one_frame(
        self,
        rgb_image: Image.Image,
        *,
        index: int,
        name: str,
        window: deque[np.ndarray],
    ) -> FrameResult:
        if self._detector_model is None or self._vitpose is None or self._lstm is None:
            raise RuntimeError("Pipeline models are not loaded.")

        bbox_xyxy: list[float] | None = None
        bbox_fallback = False
        embedding_ok = False
        prediction: dict[str, Any] | None = None

        try:
            width, height = rgb_image.size
            bbox = detect_largest_person_bbox(
                image=rgb_image,
                model=self._detector_model,
                conf=self.config.rtdetr_conf,
                iou=self.config.rtdetr_iou,
                device=str(self.device),
            )
            if bbox is None:
                bbox_fallback = True
                bbox = (0.0, 0.0, float(width), float(height))

            bbox_xyxy = [float(v) for v in bbox]
            embedding = self._extract_embedding(rgb_image, bbox)
            if embedding.shape != (self.config.embedding_dim,):
                raise ValueError(
                    f"Expected embedding ({self.config.embedding_dim},), got {embedding.shape}"
                )

            window.append(embedding)
            embedding_ok = True

            if len(window) == self.config.window_size:
                prediction = self._predict_window(window)
        except Exception as exc:
            return FrameResult(
                index=index,
                name=name,
                bbox_xyxy=bbox_xyxy,
                bbox_fallback=bbox_fallback,
                embedding_ok=False,
                buffer_size=len(window),
                prediction=None,
                error=str(exc),
            )

        return FrameResult(
            index=index,
            name=name,
            bbox_xyxy=bbox_xyxy,
            bbox_fallback=bbox_fallback,
            embedding_ok=embedding_ok,
            buffer_size=len(window),
            prediction=prediction,
        )

    def process_folder(self, folder: Path) -> dict[str, Any]:
        image_paths = sorted_image_paths(folder)
        return self.process_images(image_paths)

    def process_images(self, image_paths: list[Path]) -> dict[str, Any]:
        if self._detector_model is None or self._vitpose is None or self._lstm is None:
            raise RuntimeError("Pipeline models are not loaded.")

        window: deque[np.ndarray] = deque(maxlen=self.config.window_size)
        frames: list[FrameResult] = []
        predictions: list[dict[str, Any]] = []

        for index, image_path in enumerate(image_paths):
            try:
                with Image.open(image_path) as image:
                    rgb_image = image.convert("RGB")
                    frame = self.process_one_frame(
                        rgb_image,
                        index=index,
                        name=image_path.name,
                        window=window,
                    )
            except Exception as exc:
                frames.append(
                    FrameResult(
                        index=index,
                        name=image_path.name,
                        bbox_xyxy=None,
                        bbox_fallback=False,
                        embedding_ok=False,
                        buffer_size=len(window),
                        prediction=None,
                        error=str(exc),
                    )
                )
                continue

            frames.append(frame)
            if frame.prediction is not None:
                predictions.append(
                    {
                        "frame_index": index,
                        "frame_name": image_path.name,
                        **frame.prediction,
                        "bbox_xyxy": frame.bbox_xyxy,
                        "bbox_fallback": frame.bbox_fallback,
                    }
                )

        fall_confirmation = build_fall_confirmation(frames, self.config)

        return {
            "frames_total": len(image_paths),
            "frames_processed": sum(1 for frame in frames if frame.embedding_ok),
            "window_size": self.config.window_size,
            "predictions": predictions,
            "frames": [frame.to_dict() for frame in frames],
            "fall_confirmation": fall_confirmation,
        }
