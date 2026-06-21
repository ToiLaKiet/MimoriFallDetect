from __future__ import annotations

import importlib.util
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

import paths  # noqa: F401
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


def is_bbox_pair_stable(
    prev: list[float],
    curr: list[float],
    *,
    min_iou: float,
    max_center_shift: float,
) -> bool:
    iou = bbox_iou(prev, curr)
    c_prev = bbox_center(prev)
    c_curr = bbox_center(curr)
    dist = ((c_curr[0] - c_prev[0]) ** 2 + (c_curr[1] - c_prev[1]) ** 2) ** 0.5
    ref = max(bbox_diagonal(prev), 1.0)
    shift_ratio = float(dist / ref)
    return iou >= min_iou and shift_ratio <= max_center_shift


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
    # reload the pipeline models
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
        return self._vitpose.extract_embedding(image=cropped, source=source)

    @torch.no_grad()
    def _predict_window(self, window: deque[np.ndarray]) -> dict[str, Any]:
        assert self._lstm is not None

        stacked = np.stack(list(window), axis=0).astype(np.float32)
        x = torch.from_numpy(stacked).unsqueeze(0).to(self.device)
        lengths = torch.tensor([x.shape[1]], dtype=torch.long, device=self.device)

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
