#!/usr/bin/env python3
"""Realtime OpenCV demo for the same RT-DETR + ViTPose pipeline as vitpose.ipynb."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Iterable


DETECTOR_MODEL = "PekingU/rtdetr_r50vd_coco_o365"
POSE_MODEL = "usyd-community/vitpose-base-simple"

COCO_EDGES = [
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
    (3, 5),
    (4, 6),
]

LEFT_KEYPOINTS = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT_KEYPOINTS = {2, 4, 6, 8, 10, 12, 14, 16}


def load_runtime_dependencies() -> None:
    cache_root = Path(tempfile.gettempdir()) / "vitpose-opencv-cache"
    for child in ("matplotlib", "xdg"):
        (cache_root / child).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

    global AutoProcessor
    global Image
    global RTDetrForObjectDetection
    global VitPoseForPoseEstimation
    global cv2
    global np
    global torch

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo realtime OpenCV using RT-DETR person detection and ViTPose."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index such as 0, or a path to a video file. Default: 0.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device. Default: auto.",
    )
    parser.add_argument(
        "--det-thr",
        type=float,
        default=0.3,
        help="Person detection threshold. Default: 0.3.",
    )
    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=0.3,
        help="Keypoint score threshold for drawing. Default: 0.3.",
    )
    parser.add_argument(
        "--det-every",
        type=int,
        default=3,
        help="Run person detector every N frames and reuse boxes between runs. Default: 3.",
    )
    parser.add_argument(
        "--max-persons",
        type=int,
        default=4,
        help="Maximum number of detected people to send to ViTPose. Default: 4.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=640,
        help="Resize long side before inference; 0 keeps original frame size. Default: 640.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Requested camera capture width. 0 keeps camera default.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Requested camera capture height. 0 keeps camera default.",
    )
    parser.add_argument(
        "--save",
        default="",
        help="Optional path to save the annotated video.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not show a GUI window. Useful when only saving output video.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        requested = "cpu"

    return torch.device(requested)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def source_arg(value: str) -> int | str:
    if value.isdigit():
        return int(value)
    return value


def resize_for_inference(frame_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    if max_side <= 0:
        return frame_bgr, 1.0

    height, width = frame_bgr.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame_bgr, 1.0

    scale = max_side / float(longest)
    resized = cv2.resize(
        frame_bgr,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def bgr_to_pil_rgb(frame_bgr: np.ndarray) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def xyxy_to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    boxes_xywh = boxes_xyxy.astype(np.float32, copy=True)
    boxes_xywh[:, 2] = boxes_xywh[:, 2] - boxes_xywh[:, 0]
    boxes_xywh[:, 3] = boxes_xywh[:, 3] - boxes_xywh[:, 1]
    return boxes_xywh


def load_models(device: torch.device):
    print(f"Loading detector: {DETECTOR_MODEL}")
    det_processor = AutoProcessor.from_pretrained(DETECTOR_MODEL)
    det_model = RTDetrForObjectDetection.from_pretrained(DETECTOR_MODEL).to(device).eval()

    print(f"Loading pose model: {POSE_MODEL}")
    pose_processor = AutoProcessor.from_pretrained(POSE_MODEL)
    pose_model = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL).to(device).eval()

    return det_processor, det_model, pose_processor, pose_model


def detect_people(
    image: Image.Image,
    det_processor,
    det_model,
    device: torch.device,
    threshold: float,
    max_persons: int,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = det_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = det_model(**inputs)
    results = det_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([(image.height, image.width)], device=device),
        threshold=threshold,
    )[0]

    labels = results["labels"].detach().cpu().numpy()
    person_indices = np.flatnonzero(labels == 0)
    if len(person_indices) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    boxes = results["boxes"][person_indices].detach().cpu().numpy()
    scores = results["scores"][person_indices].detach().cpu().numpy()
    order = np.argsort(scores)[::-1][:max_persons]

    return xyxy_to_xywh(boxes[order]), scores[order].astype(np.float32)


def estimate_pose(
    image: Image.Image,
    boxes_xywh: np.ndarray,
    pose_processor,
    pose_model,
    device: torch.device,
) -> list[dict[str, np.ndarray]]:
    if len(boxes_xywh) == 0:
        return []

    pose_inputs = pose_processor(image, boxes=[boxes_xywh], return_tensors="pt").to(device)
    with torch.no_grad():
        pose_outputs = pose_model(**pose_inputs)
    pose_results = pose_processor.post_process_pose_estimation(
        pose_outputs,
        boxes=[boxes_xywh],
    )[0]

    persons = []
    for person_id, pose in enumerate(pose_results):
        persons.append(
            {
                "person_id": person_id,
                "bbox_xywh": boxes_xywh[person_id].astype(np.float32),
                "keypoints": pose["keypoints"].detach().cpu().numpy().astype(np.float32),
                "scores": pose["scores"].detach().cpu().numpy().astype(np.float32),
            }
        )
    return persons


def scale_people_to_original(
    persons: Iterable[dict[str, np.ndarray]],
    scale: float,
) -> list[dict[str, np.ndarray]]:
    if scale == 1.0:
        return list(persons)

    restored = []
    for person in persons:
        restored_person = dict(person)
        restored_person["bbox_xywh"] = person["bbox_xywh"] / scale
        restored_person["keypoints"] = person["keypoints"] / scale
        restored.append(restored_person)
    return restored


def keypoint_color(idx: int) -> tuple[int, int, int]:
    if idx in LEFT_KEYPOINTS:
        return (25, 160, 255)
    if idx in RIGHT_KEYPOINTS:
        return (255, 80, 80)
    return (60, 220, 255)


def draw_pose(
    frame_bgr: np.ndarray,
    persons: list[dict[str, np.ndarray]],
    box_scores: np.ndarray,
    kpt_thr: float,
    fps: float,
    device: torch.device,
) -> np.ndarray:
    canvas = frame_bgr.copy()

    for person_index, person in enumerate(persons):
        x, y, w, h = person["bbox_xywh"]
        pt1 = (int(round(x)), int(round(y)))
        pt2 = (int(round(x + w)), int(round(y + h)))
        cv2.rectangle(canvas, pt1, pt2, (0, 180, 255), 2)

        if person_index < len(box_scores):
            label = f"person {box_scores[person_index]:.2f}"
            cv2.putText(
                canvas,
                label,
                (pt1[0], max(20, pt1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )

        keypoints = person["keypoints"]
        scores = person["scores"]

        for a, b in COCO_EDGES:
            if a >= len(keypoints) or b >= len(keypoints):
                continue
            if scores[a] < kpt_thr or scores[b] < kpt_thr:
                continue
            p1 = tuple(np.rint(keypoints[a]).astype(int))
            p2 = tuple(np.rint(keypoints[b]).astype(int))
            cv2.line(canvas, p1, p2, (40, 220, 140), 3, cv2.LINE_AA)

        for idx, point in enumerate(keypoints):
            if idx >= len(scores) or scores[idx] < kpt_thr:
                continue
            center = tuple(np.rint(point).astype(int))
            cv2.circle(canvas, center, 5, keypoint_color(idx), -1, cv2.LINE_AA)
            cv2.circle(canvas, center, 5, (255, 255, 255), 1, cv2.LINE_AA)

    status = f"ViTPose | {device.type} | {fps:4.1f} FPS | people: {len(persons)} | q: quit"
    cv2.rectangle(canvas, (8, 8), (620, 40), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        status,
        (16, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def build_writer(path: str, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter | None:
    if not path:
        return None

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, max(fps, 1.0), frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {out_path}")
    print(f"Saving annotated video to: {out_path}")
    return writer


def main() -> None:
    args = parse_args()
    args.det_every = max(1, args.det_every)
    args.max_persons = max(1, args.max_persons)
    
    load_runtime_dependencies()

    device = choose_device(args.device)
    print(f"Using device: {device}")
    det_processor, det_model, pose_processor, pose_model = load_models(device)

    cap = cv2.VideoCapture(source_arg(args.source))
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    writer = None

    cached_boxes = np.empty((0, 4), dtype=np.float32)
    cached_box_scores = np.empty((0,), dtype=np.float32)
    frame_index = 0
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            sync_device(device)
            frame_start = perf_counter()

            infer_frame, scale = resize_for_inference(frame, args.max_side)
            image = bgr_to_pil_rgb(infer_frame)

            if frame_index % args.det_every == 0 or len(cached_boxes) == 0:
                cached_boxes, cached_box_scores = detect_people(
                    image,
                    det_processor,
                    det_model,
                    device,
                    args.det_thr,
                    args.max_persons,
                )

            persons = estimate_pose(image, cached_boxes, pose_processor, pose_model, device)
            persons = scale_people_to_original(persons, scale)
            box_scores = cached_box_scores.copy()

            sync_device(device)
            elapsed = perf_counter() - frame_start
            current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            fps = current_fps if fps == 0.0 else (0.85 * fps + 0.15 * current_fps)

            annotated = draw_pose(frame, persons, box_scores, args.kpt_thr, fps, device)

            if args.save and writer is None:
                height, width = annotated.shape[:2]
                writer = build_writer(args.save, source_fps or 20.0, (width, height))

            if writer is not None:
                writer.write(annotated)

            if not args.no_window:
                cv2.imshow("OpenCV ViTPose realtime demo", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
