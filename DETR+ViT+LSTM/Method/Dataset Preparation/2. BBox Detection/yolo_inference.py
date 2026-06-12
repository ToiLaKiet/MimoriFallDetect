from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_YOLO_MODEL = "yolo26x.pt"
PERSON_CLASS_ID = 0


def load_yolo_model(model_path: str | Path = DEFAULT_YOLO_MODEL) -> Any:
    """Load a YOLO model from a local .pt file or an Ultralytics model name."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'ultralytics'. Install it before using YOLO inference."
        ) from exc

    return YOLO(str(model_path))


def detect_person_bboxes(
    image: str | Path | np.ndarray,
    model: Any | None = None,
    model_path: str | Path = DEFAULT_YOLO_MODEL,
    conf: float = 0.5,
    iou: float = 0.7,
    imgsz: int | tuple[int, int] | None = None,
    device: str | int | None = None,
    max_persons: int | None = None,
) -> np.ndarray:
    """Detect person bounding boxes with YOLO and return boxes as xyxy float32.

    Returns:
        np.ndarray with shape (N, 4), where each row is [x1, y1, x2, y2].
    """

    yolo_model = model if model is not None else load_yolo_model(model_path)
    predict_kwargs: dict[str, Any] = {
        "source": image,
        "classes": [PERSON_CLASS_ID],
        "conf": conf,
        "iou": iou,
        "verbose": False,
    }
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz
    if device is not None:
        predict_kwargs["device"] = device

    results = yolo_model.predict(**predict_kwargs)
    if not results:
        return np.empty((0, 4), dtype=np.float32)

    boxes_obj = results[0].boxes
    
    if boxes_obj is None or len(boxes_obj) == 0:
        return np.empty((0, 4), dtype=np.float32)

    boxes = boxes_obj.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = boxes_obj.conf.detach().cpu().numpy().astype(np.float32)
    order = np.argsort(scores)[::-1]
    if max_persons is not None and max_persons > 0:
        order = order[:max_persons]
    return boxes[order]


def detect_largest_person_bbox(
    image: str | Path | np.ndarray,
    model: Any | None = None,
    model_path: str | Path = DEFAULT_YOLO_MODEL,
    conf: float = 0.5,
    iou: float = 0.7,
    imgsz: int | tuple[int, int] | None = None,
    device: str | int | None = None,
) -> tuple[float, float, float, float] | None:
    """Detect people with YOLO and return the largest person bbox as xyxy."""

    boxes = detect_person_bboxes(
        image=image,
        model=model,
        model_path=model_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
    )
    if len(boxes) == 0:
        return None

    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    largest_index = int(np.argmax(widths * heights))
    return tuple(float(value) for value in boxes[largest_index])
