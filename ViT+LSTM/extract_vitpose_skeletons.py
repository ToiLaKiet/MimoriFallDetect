#!/usr/bin/env python3
"""Extract ViTPose skeleton images from raw frames."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_vitpose_lstm import (  # noqa: E402
    FRAME_COLUMNS,
    choose_device,
    iter_images,
    load_manifest_rows,
    required_row_value,
    resolve_manifest_image_path,
    skeleton_cache_path,
)


DEFAULT_DETECTOR_MODEL = "PekingU/rtdetr_r50vd_coco_o365"
DEFAULT_POSE_MODEL = "usyd-community/vitpose-base-simple"

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


def pil_bilinear_resample():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR


def configure_runtime_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "vitpose-extract-cache"
    for child in ("matplotlib", "xdg"):
        (cache_root / child).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))


def xyxy_to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    boxes_xywh = boxes_xyxy.astype(np.float32, copy=True)
    boxes_xywh[:, 2] = boxes_xywh[:, 2] - boxes_xywh[:, 0]
    boxes_xywh[:, 3] = boxes_xywh[:, 3] - boxes_xywh[:, 1]
    return boxes_xywh


def resize_for_inference(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image

    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image

    scale = max_side / float(longest)
    resized_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(resized_size, pil_bilinear_resample())


def keypoint_color(index: int) -> tuple[int, int, int]:
    if index in LEFT_KEYPOINTS:
        return (25, 160, 255)
    if index in RIGHT_KEYPOINTS:
        return (255, 80, 80)
    return (60, 220, 255)


def draw_skeleton(
    image_size: tuple[int, int],
    persons: list[dict[str, np.ndarray]],
    keypoint_threshold: float,
) -> Image.Image:
    canvas = Image.new("RGB", image_size, (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    width, height = image_size
    line_width = max(2, int(round(max(width, height) / 220)))
    radius = max(3, int(round(max(width, height) / 180)))

    for person in persons:
        keypoints = np.asarray(person["keypoints"], dtype=np.float32)
        scores = np.asarray(person["scores"], dtype=np.float32)
        if keypoints.size == 0:
            continue

        for a, b in COCO_EDGES:
            if a >= len(keypoints) or b >= len(keypoints):
                continue
            if scores[a] < keypoint_threshold or scores[b] < keypoint_threshold:
                continue

            x1, y1 = keypoints[a]
            x2, y2 = keypoints[b]
            draw.line((x1, y1, x2, y2), fill=(40, 220, 140), width=line_width)

        for index, (x, y) in enumerate(keypoints):
            if index >= len(scores) or scores[index] < keypoint_threshold:
                continue
            color = keypoint_color(index)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=(255, 255, 255),
            )

    return canvas


class VitPoseSkeletonExtractor:
    def __init__(
        self,
        detector_model_name: str,
        pose_model_name: str,
        device: torch.device,
        detection_threshold: float,
        keypoint_threshold: float,
        max_persons: int,
        max_side: int,
        save_blank_on_miss: bool,
    ) -> None:
        configure_runtime_cache()

        from transformers import (  # noqa: PLC0415
            AutoProcessor,
            RTDetrForObjectDetection,
            VitPoseForPoseEstimation,
        )

        self.device = device
        self.detection_threshold = detection_threshold
        self.keypoint_threshold = keypoint_threshold
        self.max_side = max_side
        self.save_blank_on_miss = save_blank_on_miss

        print(f"Loading detector: {detector_model_name}")
        self.det_processor = AutoProcessor.from_pretrained(detector_model_name)
        self.det_model = (
            RTDetrForObjectDetection.from_pretrained(detector_model_name)
            .to(device)
            .eval()
        )

        print(f"Loading pose model: {pose_model_name}")
        self.pose_processor = AutoProcessor.from_pretrained(pose_model_name)
        self.pose_model = (
            VitPoseForPoseEstimation.from_pretrained(pose_model_name).to(device).eval()
        )

    @torch.no_grad()
    def detect_people(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        inputs = self.det_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.det_model(**inputs)
        results = self.det_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(image.height, image.width)], device=self.device),
            threshold=self.detection_threshold,
        )[0]

        labels = results["labels"].detach().cpu().numpy()
        person_indices = np.flatnonzero(labels == 0)
        if len(person_indices) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        boxes = results["boxes"][person_indices].detach().cpu().numpy()
        scores = results["scores"][person_indices].detach().cpu().numpy()
        widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        largest_index = int(np.argmax(widths * heights))

        return xyxy_to_xywh(boxes[[largest_index]]), scores[[largest_index]].astype(np.float32)

    @torch.no_grad()
    def estimate_pose(
        self,
        image: Image.Image,
        boxes_xywh: np.ndarray,
    ) -> list[dict[str, np.ndarray]]:
        if len(boxes_xywh) == 0:
            return []

        pose_inputs = self.pose_processor(
            image,
            boxes=[boxes_xywh],
            return_tensors="pt",
        ).to(self.device)
        pose_outputs = self.pose_model(**pose_inputs)
        pose_results = self.pose_processor.post_process_pose_estimation(
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

    def extract_one(self, image_path: Path, output_path: Path) -> bool:
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        image = resize_for_inference(image, self.max_side)
        boxes_xywh, _ = self.detect_people(image)
        persons = self.estimate_pose(image, boxes_xywh)

        if not persons and not self.save_blank_on_miss:
            return False

        skeleton = draw_skeleton(image.size, persons, self.keypoint_threshold)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skeleton.save(output_path)
        return bool(persons)


def image_paths_from_manifest(
    manifest_path: Path,
    image_dir: Path | None,
    frame_col: str,
    limit: int,
) -> list[Path]:
    manifest_path = Path(manifest_path).resolve()
    raw_rows = load_manifest_rows(manifest_path)
    if limit > 0:
        raw_rows = raw_rows[:limit]

    image_paths = []
    for row_index, row in enumerate(raw_rows):
        frame_value = required_row_value(
            row=row,
            preferred=frame_col,
            candidates=FRAME_COLUMNS,
            row_index=row_index,
            kind="frame",
        )
        image_paths.append(
            resolve_manifest_image_path(
                frame_value=frame_value,
                image_dir=image_dir,
                manifest_dir=manifest_path.parent,
            )
        )
    return image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract skeleton images from raw frames with RT-DETR + ViTPose."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Raw frame root. With manifest, relative frame paths resolve from here first.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional CSV/JSON/JSONL manifest containing frame paths.",
    )
    parser.add_argument(
        "--frame-col",
        default="",
        help="Manifest frame/path column. Empty value auto-detects common names.",
    )
    parser.add_argument(
        "--skeleton-dir",
        "--pose-cache-dir",
        dest="skeleton_dir",
        type=Path,
        default=SCRIPT_DIR / "vitpose_cache",
        help="Output directory for skeleton PNG files. Default: ViT+LSTM/vitpose_cache.",
    )
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--save-blank-on-miss", action="store_true")
    parser.add_argument("--detector-model", default=DEFAULT_DETECTOR_MODEL)
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Inference device. Default: auto.",
    )
    parser.add_argument("--det-thr", type=float, default=0.3)
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    parser.add_argument(
        "--max-persons",
        type=int,
        default=1,
        help="Deprecated; extractor always keeps only the largest detected person box.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=640,
        help="Resize long side before ViTPose extraction; 0 keeps original size.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.manifest_path is not None:
        args.manifest_path = args.manifest_path.resolve()
    if args.image_dir is None:
        if args.manifest_path is None:
            raise RuntimeError("--image-dir is required when --manifest-path is not provided.")
        args.image_dir = args.manifest_path.parent
    args.image_dir = args.image_dir.resolve()
    args.skeleton_dir = args.skeleton_dir.resolve()
    args.max_persons = max(1, args.max_persons)
    args.log_every = max(1, args.log_every)

    if args.manifest_path is not None:
        image_paths = image_paths_from_manifest(
            manifest_path=args.manifest_path,
            image_dir=args.image_dir,
            frame_col=args.frame_col,
            limit=args.limit,
        )
    else:
        image_paths = iter_images(args.image_dir, args.limit)

    if not image_paths:
        raise RuntimeError("No raw image frames found.")

    device = choose_device(args.device)
    print(f"Using device: {device}")
    extractor = VitPoseSkeletonExtractor(
        detector_model_name=args.detector_model,
        pose_model_name=args.pose_model,
        device=device,
        detection_threshold=args.det_thr,
        keypoint_threshold=args.kpt_thr,
        max_persons=args.max_persons,
        max_side=args.max_side,
        save_blank_on_miss=args.save_blank_on_miss,
    )

    total = len(image_paths)
    extracted = 0
    skipped_existing = 0
    missing_images = 0
    missed_people = 0

    for index, image_path in enumerate(image_paths, start=1):
        skeleton_path = skeleton_cache_path(image_path, args.image_dir, args.skeleton_dir)
        if not image_path.is_file():
            missing_images += 1
        elif skeleton_path.exists() and not args.overwrite_cache:
            skipped_existing += 1
        else:
            found_person = extractor.extract_one(image_path, skeleton_path)
            if found_person:
                extracted += 1
            else:
                missed_people += 1

        if index == total or index % args.log_every == 0:
            print(
                "ViTPose skeletons "
                f"{index}/{total} | new={extracted} existing={skipped_existing} "
                f"missing_images={missing_images} missed_people={missed_people}"
            )

    print(f"Skeleton directory: {args.skeleton_dir}")


if __name__ == "__main__":
    main()
