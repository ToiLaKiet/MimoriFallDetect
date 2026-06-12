"""Reusable realtime pose helpers for the MVP."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
for cache_path in (
    CACHE_DIR,
    CACHE_DIR / "matplotlib",
    CACHE_DIR / "ultralytics",
    CACHE_DIR / "xdg",
    CACHE_DIR / "xdg" / "fontconfig",
):
    cache_path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_DIR / "ultralytics"))
default_hf_home = Path.home() / ".cache" / "huggingface"
if default_hf_home.exists():
    os.environ.setdefault("HF_HOME", str(default_hf_home))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))

import cv2
import numpy as np
import torch
from torch import nn
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
    VitPoseImageProcessor,
)


DEFAULT_DETECTOR_MODEL = "PekingU/rtdetr_r50vd_coco_o365"

COCO_LINKS = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)


class SkeletonSequenceClassifier(nn.Module):
    """CNN encoder + 2-layer LSTM classifier matching vitpose_lstm_best.pt."""

    def __init__(self, num_classes: int = 11) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.encoder.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Linear(128, num_classes),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = frames.shape
        x = frames.reshape(batch * steps, channels, height, width)
        x = self.encoder.cnn(x)
        x = self.encoder.projection(x)
        x = x.reshape(batch, steps, -1)
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1])


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(f"Checkpoint must be a state_dict dict, got {type(loaded)!r}")
    return loaded


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RTDetrPersonDetector:
    def __init__(
        self,
        detector_model_name: str,
        device: torch.device,
        allow_download: bool,
        threshold: float = 0.5,
        person_label: str = "person",
        max_persons: int = 1,
    ) -> None:
        model_source = resolve_pretrained_model_source(detector_model_name, allow_download)
        self.det_processor = AutoProcessor.from_pretrained(
            model_source,
            local_files_only=not allow_download,
            use_fast=False,
        )
        self.det_model = (
            RTDetrForObjectDetection.from_pretrained(
                model_source,
                local_files_only=not allow_download,
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.threshold = threshold
        self.person_label = person_label
        self.max_persons = max_persons

    def detect(self, frame_bgr: np.ndarray) -> list[list[float]]:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.det_processor(images=frame_rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.det_model(**inputs)

        result = self.det_processor.post_process_object_detection(
            outputs,
            threshold=self.threshold,
            target_sizes=[(height, width)],
        )[0]
        boxes = select_person_boxes(
            result,
            self.det_model.config.id2label,
            self.person_label,
            self.max_persons,
        )
        if not boxes:
            return [full_frame_box(width, height)]
        return [xyxy_to_coco_box(box, width, height) for box in boxes]


def resolve_pretrained_model_source(model_name_or_path: str, allow_download: bool) -> str:
    path = Path(model_name_or_path).expanduser()
    if path.exists():
        return str(path)
    if allow_download:
        return model_name_or_path

    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo_cache = cache_root / f"models--{model_name_or_path.replace('/', '--')}"
    refs_main = repo_cache / "refs" / "main"
    if refs_main.exists():
        snapshot = repo_cache / "snapshots" / refs_main.read_text().strip()
        if snapshot.exists():
            return str(snapshot)

    snapshots_root = repo_cache / "snapshots"
    if snapshots_root.exists():
        for snapshot in snapshots_root.iterdir():
            if (snapshot / "config.json").exists():
                return str(snapshot)

    return model_name_or_path


def select_person_boxes(
    detection_result: dict,
    id2label: dict[int, str],
    person_label: str,
    max_persons: int,
) -> list[torch.Tensor]:
    selected: list[tuple[float, torch.Tensor]] = []
    for score, label, box in zip(
        detection_result["scores"],
        detection_result["labels"],
        detection_result["boxes"],
    ):
        label_name = id2label.get(int(label), str(int(label))).lower()
        if label_name != person_label.lower():
            continue
        selected.append((float(score), box.detach().cpu()))

    selected.sort(key=lambda item: item[0], reverse=True)
    if max_persons > 0:
        selected = selected[:max_persons]
    return [box for _, box in selected]


def xyxy_to_coco_box(box: torch.Tensor, image_width: int, image_height: int) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box.tolist()]
    x1 = min(max(x1, 0.0), float(image_width - 1))
    y1 = min(max(y1, 0.0), float(image_height - 1))
    x2 = min(max(x2, x1 + 1.0), float(image_width))
    y2 = min(max(y2, y1 + 1.0), float(image_height))
    return [x1, y1, x2 - x1, y2 - y1]


def full_frame_box(image_width: int, image_height: int) -> list[float]:
    return [0.0, 0.0, float(image_width), float(image_height)]


class VitPoseRunner:
    def __init__(
        self,
        model_name_or_path: str,
        device: torch.device,
        allow_download: bool,
    ) -> None:
        local_only = not allow_download
        self.processor = VitPoseImageProcessor.from_pretrained(
            model_name_or_path,
            local_files_only=local_only,
        )
        self.model = VitPoseForPoseEstimation.from_pretrained(
            model_name_or_path,
            local_files_only=local_only,
        )
        self.model.to(device)
        self.model.eval()
        self.device = device

    def estimate(self, frame_bgr: np.ndarray, boxes: list[list[float]]) -> list[dict]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        batch_boxes = [boxes]
        inputs = self.processor(frame_rgb, boxes=batch_boxes, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return self.processor.post_process_pose_estimation(outputs, boxes=batch_boxes)[0]


def draw_skeleton_on_black(
    frame_shape: tuple[int, int, int],
    poses: Iterable[dict],
    keypoint_threshold: float,
) -> np.ndarray:
    canvas = np.zeros(frame_shape, dtype=np.uint8)
    draw_skeleton(canvas, poses, keypoint_threshold)
    return canvas


def draw_skeleton_overlay(
    frame_bgr: np.ndarray,
    poses: Iterable[dict],
    keypoint_threshold: float,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    draw_skeleton(canvas, poses, keypoint_threshold)
    return canvas


def draw_skeleton(
    canvas: np.ndarray,
    poses: Iterable[dict],
    keypoint_threshold: float,
) -> None:
    height, width = canvas.shape[:2]

    for pose in poses:
        keypoints = pose["keypoints"].detach().cpu().numpy()
        scores = pose["scores"].detach().cpu().numpy()

        for start, end in COCO_LINKS:
            if scores[start] < keypoint_threshold or scores[end] < keypoint_threshold:
                continue
            p1 = tuple(np.round(keypoints[start]).astype(int))
            p2 = tuple(np.round(keypoints[end]).astype(int))
            if not points_in_frame((p1, p2), width, height):
                continue
            cv2.line(canvas, p1, p2, (0, 255, 0), 3, lineType=cv2.LINE_AA)

        for point, score in zip(keypoints, scores):
            if score < keypoint_threshold:
                continue
            x, y = np.round(point).astype(int)
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(canvas, (x, y), 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)


def points_in_frame(points: Iterable[tuple[int, int]], width: int, height: int) -> bool:
    return all(0 <= x < width and 0 <= y < height for x, y in points)


def prepare_classifier_tensor(
    frames_bgr: Iterable[np.ndarray],
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    frames = []
    for frame in frames_bgr:
        resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    array = np.stack(frames).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).unsqueeze(0)
    return tensor.to(device)


def load_labels(labels_path: Path | None, expected_count: int) -> list[str]:
    if labels_path is None or not labels_path.exists():
        return [f"class_{index}" for index in range(expected_count)]
    labels = [line.strip() for line in labels_path.read_text().splitlines() if line.strip()]
    if len(labels) != expected_count:
        raise ValueError(
            f"{labels_path} has {len(labels)} labels, expected {expected_count}"
        )
    return labels


def fall_label_for_class(class_index: int) -> str:
    return "Fall" if 0 <= class_index <= 4 else "No Fall"


def put_status(
    image: np.ndarray,
    fps: float,
    buffer_size: int,
    action_text: str,
) -> None:
    text = f"FPS {fps:4.1f} | buffer {buffer_size}/10"
    if action_text:
        text += f" | {action_text}"
    cv2.putText(
        image,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
