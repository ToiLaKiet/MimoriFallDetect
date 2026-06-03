#!/usr/bin/env python3
"""Train the CNN+LSTM classifier from raw images via ViTPose skeletons."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import SkeletonImageLSTMClassifier  # noqa: E402
from utils import train_model  # noqa: E402


DEFAULT_DETECTOR_MODEL = "PekingU/rtdetr_r50vd_coco_o365"
DEFAULT_POSE_MODEL = "usyd-community/vitpose-base-simple"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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


@dataclass(frozen=True)
class LabelRow:
    label: int
    group_key: str
    sort_key: str


@dataclass(frozen=True)
class FrameItem:
    image_path: Path
    skeleton_path: Path
    label: int
    group_key: str
    sort_key: str


@dataclass(frozen=True)
class SequenceItem:
    skeleton_paths: tuple[Path, ...]
    label: int
    group_key: str


def pil_bilinear_resample():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        requested = "cpu"
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print("MPS is not available; falling back to CPU.")
        requested = "cpu"

    return torch.device(requested)


def configure_runtime_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "vitpose-train-cache"
    for child in ("matplotlib", "xdg"):
        (cache_root / child).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))


def iter_images(image_dir: Path, limit: int = 0) -> list[Path]:
    image_dir = Path(image_dir)
    paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit > 0:
        paths = paths[:limit]
    return paths


def timestamp_variants(value: str | Path) -> set[str]:
    text = str(value).strip()
    stem = Path(text).stem
    candidates = {text, stem}

    if stem.startswith("pose_"):
        candidates.add(stem[len("pose_") :])
    if text.startswith("pose_"):
        candidates.add(text[len("pose_") :])

    expanded = set(candidates)
    for candidate in list(candidates):
        expanded.add(candidate.replace(":", "_"))
        expanded.add(candidate.replace("_", ":", 2))
        if candidate.startswith("pose_"):
            stripped = candidate[len("pose_") :]
            expanded.add(stripped)
            expanded.add(stripped.replace(":", "_"))
            expanded.add(stripped.replace("_", ":", 2))

    return {item for item in expanded if item}


def get_row_value(row: dict[str, str], name: str, default: str = "") -> str:
    if name in row:
        return row[name]
    lowered = name.lower()
    for key, value in row.items():
        if key.lower() == lowered:
            return value
    return default


def parse_label(value: str, offset: int) -> int:
    return int(float(value)) + offset


def group_key_from_row(row: dict[str, str], label: int) -> str:
    source_file = get_row_value(row, "Source_File")
    if source_file:
        return source_file

    parts = [
        get_row_value(row, "Subject"),
        get_row_value(row, "Activity"),
        get_row_value(row, "Trial"),
    ]
    if all(parts):
        return "Subject{}Activity{}Trial{}".format(*parts)

    return f"label-{label}"


def read_labels(labels_csv: Path, label_col: str, label_offset: int) -> dict[str, LabelRow]:
    labels_csv = Path(labels_csv)
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    label_map: dict[str, LabelRow] = {}
    duplicate_keys = 0

    with labels_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Labels CSV has no header: {labels_csv}")

        for row_index, row in enumerate(reader):
            raw_label = get_row_value(row, label_col)
            if raw_label == "":
                raise KeyError(f"Missing label column '{label_col}' at row {row_index}.")

            label = parse_label(raw_label, label_offset)
            timestamp = get_row_value(row, "Timestamp") or get_row_value(row, "image")
            sort_key = timestamp or f"{row_index:012d}"
            label_row = LabelRow(
                label=label,
                group_key=group_key_from_row(row, label),
                sort_key=sort_key,
            )

            key_sources = [timestamp]
            for optional_col in ("Image", "image", "Filename", "filename", "File", "Path"):
                value = get_row_value(row, optional_col)
                if value:
                    key_sources.append(value)

            for key_source in key_sources:
                for key in timestamp_variants(key_source):
                    if key in label_map:
                        duplicate_keys += 1
                    label_map[key] = label_row

    if duplicate_keys:
        print(f"Warning: {duplicate_keys} duplicate label lookup keys were overwritten.")
    return label_map


def find_label_for_image(image_path: Path, label_map: dict[str, LabelRow]) -> LabelRow | None:
    key_sources = [
        image_path.name,
        image_path.stem,
        str(image_path),
    ]
    for key_source in key_sources:
        for key in timestamp_variants(key_source):
            if key in label_map:
                return label_map[key]
    return None


def skeleton_cache_path(image_path: Path, image_root: Path, cache_root: Path) -> Path:
    rel_path = image_path.relative_to(image_root)
    return cache_root / rel_path.with_suffix(".png")


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
        self.max_persons = max_persons
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
        order = np.argsort(scores)[::-1][: self.max_persons]

        return xyxy_to_xywh(boxes[order]), scores[order].astype(np.float32)

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


def precompute_skeletons(
    frame_items: list[FrameItem],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
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

    total = len(frame_items)
    extracted = 0
    skipped_existing = 0
    missed = 0

    for index, item in enumerate(frame_items, start=1):
        if item.skeleton_path.exists() and not args.overwrite_cache:
            skipped_existing += 1
        else:
            found_person = extractor.extract_one(item.image_path, item.skeleton_path)
            if found_person:
                extracted += 1
            else:
                missed += 1

        if index == total or index % args.log_every == 0:
            print(
                "ViTPose cache "
                f"{index}/{total} | new={extracted} "
                f"existing={skipped_existing} missed={missed}"
            )


class SkeletonSequenceDataset(Dataset):
    def __init__(self, sequences: list[SequenceItem], image_size: int) -> None:
        self.sequences = sequences
        self.image_size = image_size
        self.resample = pil_bilinear_resample()

    def __len__(self) -> int:
        return len(self.sequences)

    def load_image_tensor(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        image = image.resize((self.image_size, self.image_size), self.resample)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return np.transpose(array, (2, 0, 1))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.sequences[index]
        frames = [self.load_image_tensor(path) for path in item.skeleton_paths]
        x = torch.from_numpy(np.stack(frames, axis=0))
        y = torch.tensor(item.label, dtype=torch.long)
        return x, y


def make_frame_items(
    image_paths: list[Path],
    image_root: Path,
    pose_cache_dir: Path,
    label_map: dict[str, LabelRow],
    use_input_images_as_skeletons: bool,
) -> tuple[list[FrameItem], list[Path]]:
    items: list[FrameItem] = []
    missing_labels: list[Path] = []

    for image_path in image_paths:
        label_row = find_label_for_image(image_path, label_map)
        if label_row is None:
            missing_labels.append(image_path)
            continue

        skeleton_path = (
            image_path
            if use_input_images_as_skeletons
            else skeleton_cache_path(image_path, image_root, pose_cache_dir)
        )
        items.append(
            FrameItem(
                image_path=image_path,
                skeleton_path=skeleton_path,
                label=label_row.label,
                group_key=label_row.group_key,
                sort_key=label_row.sort_key,
            )
        )

    return items, missing_labels


def filter_existing_skeletons(frame_items: list[FrameItem]) -> list[FrameItem]:
    existing = [item for item in frame_items if item.skeleton_path.is_file()]
    missing_count = len(frame_items) - len(existing)
    if missing_count:
        print(f"Skipped {missing_count} frames without skeleton cache.")
    return existing


def sequence_label(labels: Iterable[int], mode: str) -> int:
    labels = list(labels)
    if mode == "last":
        return labels[-1]
    if mode == "majority":
        return Counter(labels).most_common(1)[0][0]
    raise ValueError(f"Unsupported sequence label mode: {mode}")


def build_sequences(
    frame_items: list[FrameItem],
    sequence_length: int,
    stride: int,
    label_mode: str,
) -> list[SequenceItem]:
    grouped: dict[str, list[FrameItem]] = defaultdict(list)
    for item in frame_items:
        grouped[item.group_key].append(item)

    sequences: list[SequenceItem] = []
    for group_key, group_items in grouped.items():
        group_items = sorted(group_items, key=lambda item: (item.sort_key, item.image_path.name))
        if len(group_items) < sequence_length:
            continue

        for start in range(0, len(group_items) - sequence_length + 1, stride):
            window = group_items[start : start + sequence_length]
            label = sequence_label((item.label for item in window), label_mode)
            sequences.append(
                SequenceItem(
                    skeleton_paths=tuple(item.skeleton_path for item in window),
                    label=label,
                    group_key=group_key,
                )
            )

    return sequences


def split_sequences(
    sequences: list[SequenceItem],
    val_split: float,
    seed: int,
    group_split: bool,
) -> tuple[list[SequenceItem], list[SequenceItem]]:
    if val_split <= 0:
        return sequences, []

    rng = random.Random(seed)
    if group_split:
        by_group: dict[str, list[SequenceItem]] = defaultdict(list)
        for item in sequences:
            by_group[item.group_key].append(item)

        groups = list(by_group)
        rng.shuffle(groups)
        val_group_count = max(1, int(round(len(groups) * val_split)))
        val_groups = set(groups[:val_group_count])

        train_sequences = [
            item for item in sequences if item.group_key not in val_groups
        ]
        val_sequences = [item for item in sequences if item.group_key in val_groups]
        return train_sequences, val_sequences

    shuffled = list(sequences)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_split)))
    val_sequences = shuffled[:val_count]
    train_sequences = shuffled[val_count:]
    return train_sequences, val_sequences


def write_history(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def args_for_json(args: argparse.Namespace) -> dict[str, object]:
    data = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            data[key] = str(value)
        else:
            data[key] = value
    return data


def save_metadata(
    args: argparse.Namespace,
    sequences: list[SequenceItem],
    train_sequences: list[SequenceItem],
    val_sequences: list[SequenceItem],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "args": args_for_json(args),
        "total_sequences": len(sequences),
        "train_sequences": len(train_sequences),
        "val_sequences": len(val_sequences),
        "class_counts": dict(Counter(item.label for item in sequences)),
    }
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train SkeletonImageLSTMClassifier from raw image frames. "
            "Raw images are first converted to skeleton images with RT-DETR + ViTPose."
        )
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing raw image frames. File names should match labels CSV timestamps.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=SCRIPT_DIR / "labels.csv",
        help="CSV with Timestamp and Label columns. Default: ViT+LSTM/labels.csv.",
    )
    parser.add_argument(
        "--pose-cache-dir",
        type=Path,
        default=None,
        help="Directory for cached ViTPose skeleton images. Default: ViT+LSTM/vitpose_cache.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip ViTPose extraction and train from existing skeletons in pose-cache-dir.",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Regenerate skeleton cache even when output files already exist.",
    )
    parser.add_argument(
        "--save-blank-on-miss",
        action="store_true",
        help="Save a blank skeleton frame when no person is detected.",
    )
    parser.add_argument("--detector-model", default=DEFAULT_DETECTOR_MODEL)
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Training/inference device. Default: auto.",
    )
    parser.add_argument("--det-thr", type=float, default=0.3)
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    parser.add_argument("--max-persons", type=int, default=1)
    parser.add_argument(
        "--max-side",
        type=int,
        default=640,
        help="Resize long side before ViTPose extraction; 0 keeps original size.",
    )
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--sequence-label",
        choices=("majority", "last"),
        default="majority",
        help="How to choose a label for each sequence window.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument("--label-col", default="Label")
    parser.add_argument(
        "--label-offset",
        type=int,
        default=0,
        help="Add this offset to labels. Use -1 when label-col is one-based Activity.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument(
        "--group-split",
        action="store_true",
        help="Split train/val by trial/source group instead of random sequence split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "vitpose_lstm_best.pt",
        help="Best validation checkpoint path.",
    )
    parser.add_argument(
        "--final-checkpoint-path",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "vitpose_lstm_final.pt",
        help="Final checkpoint path.",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "vitpose_lstm_history.csv",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "vitpose_lstm_metadata.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    args.image_dir = args.image_dir.resolve()
    args.labels_csv = args.labels_csv.resolve()
    if args.pose_cache_dir is None:
        args.pose_cache_dir = (
            args.image_dir if args.no_extract else SCRIPT_DIR / "vitpose_cache"
        )
    args.pose_cache_dir = args.pose_cache_dir.resolve()
    args.sequence_length = max(1, args.sequence_length)
    args.stride = max(1, args.stride)
    args.max_persons = max(1, args.max_persons)
    args.log_every = max(1, args.log_every)

    image_paths = iter_images(args.image_dir, args.limit)
    if not image_paths:
        raise RuntimeError(f"No images found in {args.image_dir}")

    label_map = read_labels(args.labels_csv, args.label_col, args.label_offset)
    use_input_images_as_skeletons = args.no_extract and (
        args.pose_cache_dir == args.image_dir
    )
    frame_items, missing_labels = make_frame_items(
        image_paths=image_paths,
        image_root=args.image_dir,
        pose_cache_dir=args.pose_cache_dir,
        label_map=label_map,
        use_input_images_as_skeletons=use_input_images_as_skeletons,
    )

    print(
        f"Found {len(image_paths)} images, matched {len(frame_items)} labels, "
        f"missing labels for {len(missing_labels)} images."
    )
    if missing_labels:
        preview = ", ".join(path.name for path in missing_labels[:5])
        print(f"First missing-label images: {preview}")

    if not frame_items:
        raise RuntimeError("No image frames matched labels; check image file names and labels CSV.")

    device = choose_device(args.device)
    print(f"Using device: {device}")

    if not args.no_extract:
        precompute_skeletons(frame_items, args, device)
    else:
        print(f"Skipping ViTPose extraction; using skeletons from {args.pose_cache_dir}")

    frame_items = filter_existing_skeletons(frame_items)
    if not frame_items:
        raise RuntimeError("No skeleton frames available for training.")

    sequences = build_sequences(
        frame_items=frame_items,
        sequence_length=args.sequence_length,
        stride=args.stride,
        label_mode=args.sequence_label,
    )
    if not sequences:
        raise RuntimeError(
            "No trainable sequences were built; reduce sequence length or check grouping."
        )

    train_sequences, val_sequences = split_sequences(
        sequences=sequences,
        val_split=args.val_split,
        seed=args.seed,
        group_split=args.group_split,
    )
    if not train_sequences:
        raise RuntimeError("Train split is empty; reduce val-split.")

    print(
        f"Built {len(sequences)} sequences: "
        f"train={len(train_sequences)} val={len(val_sequences)}"
    )
    print(f"Class counts: {dict(Counter(item.label for item in sequences))}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        SkeletonSequenceDataset(train_sequences, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_sequences:
        val_loader = DataLoader(
            SkeletonSequenceDataset(val_sequences, args.image_size),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    model = SkeletonImageLSTMClassifier(
        sequence_length=args.sequence_length,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        grad_clip=args.grad_clip,
        checkpoint_path=args.checkpoint_path if val_loader is not None else None,
    )

    args.final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.final_checkpoint_path)
    write_history(history, args.history_csv)
    save_metadata(
        args=args,
        sequences=sequences,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        output_path=args.metadata_json,
    )

    if val_loader is not None:
        print(f"Best validation checkpoint: {args.checkpoint_path}")
    print(f"Final checkpoint: {args.final_checkpoint_path}")
    print(f"Training history: {args.history_csv}")
    print(f"Training metadata: {args.metadata_json}")


if __name__ == "__main__":
    main()
