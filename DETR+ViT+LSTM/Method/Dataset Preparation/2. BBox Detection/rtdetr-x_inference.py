from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import RTDETR


DEFAULT_RTDETR_MODEL = "rtdetr-x.pt"
PERSON_CLASS_ID = 0


def load_rtdetr_model(model_path: str | Path = DEFAULT_RTDETR_MODEL) -> Any:
    """Load an Ultralytics RT-DETR model."""

    rtdetr = RTDETR(str(model_path))
    return rtdetr


def get_rtdetr_head(rtdetr: Any) -> Any:
    """Return the Ultralytics RT-DETR detection head."""

    return rtdetr.model.model[-1]


def detect_person_bboxes(
    image: str | Path | np.ndarray,
    model: Any | None = None,
    model_path: str | Path = DEFAULT_RTDETR_MODEL,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int | tuple[int, int] | None = None,
    device: str | int | None = None,
    max_persons: int | None = None,
) -> np.ndarray:
    """Detect person bounding boxes with RT-DETR-X and return xyxy float32 boxes."""

    detections = detect_person_detections(
        image=image,
        model=model,
        model_path=model_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        max_persons=max_persons,
    )
    return detections[:, :4]


def detect_person_detections(
    image: str | Path | np.ndarray,
    model: Any | None = None,
    model_path: str | Path = DEFAULT_RTDETR_MODEL,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int | tuple[int, int] | None = None,
    device: str | int | None = None,
    max_persons: int | None = None,
) -> np.ndarray:
    """Detect people with RT-DETR-X and return xyxy, confidence, class rows."""

    rtdetr_model = (
        model
        if model is not None
        else load_rtdetr_model(model_path)
    )
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

    results = rtdetr_model.predict(**predict_kwargs)
    if not results:
        return np.empty((0, 6), dtype=np.float32)

    boxes_obj = results[0].boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return np.empty((0, 6), dtype=np.float32)

    boxes = boxes_obj.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = boxes_obj.conf.detach().cpu().numpy().astype(np.float32)
    classes = boxes_obj.cls.detach().cpu().numpy().astype(np.float32)
    order = np.argsort(scores)[::-1]
    if max_persons is not None and max_persons > 0:
        order = order[:max_persons]
    return np.column_stack(
        (boxes[order], scores[order], classes[order])
    ).astype(np.float32)


def detect_largest_person_bbox(
    image: str | Path | np.ndarray,
    model: Any | None = None,
    model_path: str | Path = DEFAULT_RTDETR_MODEL,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int | tuple[int, int] | None = None,
    device: str | int | None = None,
) -> tuple[float, float, float, float] | None:
    """Detect people with RT-DETR-X and return the largest person bbox as xyxy."""

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
