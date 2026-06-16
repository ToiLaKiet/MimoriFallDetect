from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from vitpose_estimator import VitPoseEstimator


SCRIPT_DIR = Path(__file__).resolve().parent


DEFAULT_POSE_MODEL = "usyd-community/vitpose-huge-simple"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}(?:\.\d+)?$"
)
FOLDER_PATTERNS = {
    "Subject": re.compile(r"^Subject\D*(\d+)\D*$", re.IGNORECASE),
    "Activity": re.compile(r"^Activity\D*(\d+)\D*$", re.IGNORECASE),
    "Trial": re.compile(r"^Trial\D*(\d+)\D*$", re.IGNORECASE),
    "Camera": re.compile(r"^Camera\D*(\d+)\D*$", re.IGNORECASE),
}

COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class SequenceMetadata:
    subject: str
    activity: str
    trial: str
    camera: str


@dataclass(frozen=True)
class FolderStats:
    folder_key: str
    total_images: int
    processed_images: int
    missed_images: int
    skipped_existing: int


def natural_sort_key(value: str | Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


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


def pil_bilinear_resample():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR


def resize_for_inference(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image

    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image

    scale = max_side / float(longest)
    resized = (
        int(round(width * scale)),
        int(round(height * scale)),
    )
    return image.resize(resized, pil_bilinear_resample())


def is_timestamp_image(path: Path) -> bool:
    return bool(IMAGE_TIMESTAMP_PATTERN.match(path.stem))


def list_timestamp_images(folder: Path) -> list[Path]:
    images = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and is_timestamp_image(path)
    ]
    return sorted(images, key=lambda path: natural_sort_key(path.stem))


def iter_camera_folders(image_root: Path) -> list[Path]:
    image_root = Path(image_root)
    camera_folders: list[Path] = []

    for subject_dir in sorted(
        (path for path in image_root.iterdir() if path.is_dir()),
        key=lambda path: natural_sort_key(path.name),
    ):
        if not FOLDER_PATTERNS["Subject"].match(subject_dir.name):
            continue
        for activity_dir in sorted(
            (path for path in subject_dir.iterdir() if path.is_dir()),
            key=lambda path: natural_sort_key(path.name),
        ):
            if not FOLDER_PATTERNS["Activity"].match(activity_dir.name):
                continue
            for trial_dir in sorted(
                (path for path in activity_dir.iterdir() if path.is_dir()),
                key=lambda path: natural_sort_key(path.name),
            ):
                if not FOLDER_PATTERNS["Trial"].match(trial_dir.name):
                    continue
                for camera_dir in sorted(
                    (path for path in trial_dir.iterdir() if path.is_dir()),
                    key=lambda path: natural_sort_key(path.name),
                ):
                    if FOLDER_PATTERNS["Camera"].match(camera_dir.name):
                        camera_folders.append(camera_dir)

    return camera_folders


def parse_sequence_metadata(camera_folder: Path, image_root: Path) -> SequenceMetadata:
    relative_parts = camera_folder.relative_to(image_root).parts
    if len(relative_parts) != 4:
        raise ValueError(
            f"Expected Subject/Activity/Trial/Camera path, got: {camera_folder}"
        )
    subject, activity, trial, camera = relative_parts
    return SequenceMetadata(
        subject=subject,
        activity=activity,
        trial=trial,
        camera=camera,
    )


def folder_key_from_path(camera_folder: Path, image_root: Path) -> str:
    return camera_folder.relative_to(image_root).as_posix()


def csv_output_path_for_folder(
    camera_folder: Path,
    image_root: Path,
    output_root: Path,
) -> Path:
    relative_folder = camera_folder.relative_to(image_root)
    return output_root / relative_folder / "keypoints.csv"


def has_valid_keypoints(
    scores: np.ndarray,
    keypoint_threshold: float,
    min_visible_keypoints: int,
) -> bool:
    if scores.size == 0:
        return False
    visible_count = int(np.sum(scores >= keypoint_threshold))
    return visible_count >= min_visible_keypoints


def scale_pose_to_original(
    bbox_xywh: list[float] | None,
    keypoints: np.ndarray,
    original_size: tuple[int, int],
    inference_size: tuple[int, int],
) -> tuple[list[float] | None, np.ndarray]:
    original_width, original_height = original_size
    inference_width, inference_height = inference_size
    if (original_width, original_height) == (inference_width, inference_height):
        return bbox_xywh, keypoints

    scale_x = original_width / float(inference_width)
    scale_y = original_height / float(inference_height)

    scaled_bbox = None
    if bbox_xywh is not None:
        scaled_bbox = [
            bbox_xywh[0] * scale_x,
            bbox_xywh[1] * scale_y,
            bbox_xywh[2] * scale_x,
            bbox_xywh[3] * scale_y,
        ]

    if keypoints.size == 0:
        return scaled_bbox, keypoints

    scaled_keypoints = keypoints.copy()
    scaled_keypoints[:, 0] *= scale_x
    scaled_keypoints[:, 1] *= scale_y
    return scaled_bbox, scaled_keypoints


def normalize_keypoints(
    keypoints: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    if keypoints.size == 0 or image_width <= 0 or image_height <= 0:
        return np.empty((0, 2), dtype=np.float32)

    normalized = keypoints.copy()
    normalized[:, 0] /= float(image_width) # normalize the x coordinate to the range [0, 1]
    normalized[:, 1] /= float(image_height) # normalize the y coordinate to the range [0, 1]
    return normalized


def keypoint_csv_fieldnames() -> list[str]:
    fields = [
        "Subject",
        "Activity",
        "Trial",
        "Camera",
        "timestamp",
        "image_name",
        "image_width",
        "image_height",
        "has_keypoints",
        "visible_keypoint_count",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]
    for name in COCO_KEYPOINT_NAMES:
        fields.extend(
            [
                f"{name}_x",
                f"{name}_y",
                f"{name}_x_norm",
                f"{name}_y_norm",
                f"{name}_score",
            ]
        )
    return fields


def keypoints_to_csv_row(
    metadata: SequenceMetadata,
    image_path: Path,
    image_size: tuple[int, int],
    bbox_xywh: list[float] | None,
    keypoints: np.ndarray,
    scores: np.ndarray,
    keypoint_threshold: float,
    min_visible_keypoints: int,
) -> dict[str, object]:
    image_width, image_height = image_size
    normalized_keypoints = normalize_keypoints(
        keypoints=keypoints,
        image_width=image_width,
        image_height=image_height,
    )
    visible_count = int(np.sum(scores >= keypoint_threshold)) if scores.size else 0
    has_keypoints = has_valid_keypoints(
        scores=scores,
        keypoint_threshold=keypoint_threshold,
        min_visible_keypoints=min_visible_keypoints,
    )

    row: dict[str, object] = {
        "Subject": metadata.subject,
        "Activity": metadata.activity,
        "Trial": metadata.trial,
        "Camera": metadata.camera,
        "timestamp": image_path.stem,
        "image_name": image_path.name,
        "image_width": image_width,
        "image_height": image_height,
        "has_keypoints": int(has_keypoints),
        "visible_keypoint_count": visible_count,
        "bbox_x": "",
        "bbox_y": "",
        "bbox_w": "",
        "bbox_h": "",
    }
    if bbox_xywh is not None:
        row["bbox_x"] = float(bbox_xywh[0])
        row["bbox_y"] = float(bbox_xywh[1])
        row["bbox_w"] = float(bbox_xywh[2])
        row["bbox_h"] = float(bbox_xywh[3])

    for index, name in enumerate(COCO_KEYPOINT_NAMES): # Literal 
        if index < len(keypoints):
            x, y = keypoints[index]
            norm_x, norm_y = normalized_keypoints[index]
            score = float(scores[index]) if index < len(scores) else ""
        else:
            x, y, norm_x, norm_y, score = "", "", "", "", ""
        row[f"{name}_x"] = float(x) if x != "" else ""
        row[f"{name}_y"] = float(y) if y != "" else ""
        row[f"{name}_x_norm"] = float(norm_x) if norm_x != "" else ""
        row[f"{name}_y_norm"] = float(norm_y) if norm_y != "" else ""
        row[f"{name}_score"] = score

    return row


def write_folder_keypoints_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = keypoint_csv_fieldnames()
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_camera_folder(
    camera_folder: Path,
    image_root: Path,
    output_root: Path,
    estimator: VitPoseEstimator,
    keypoint_threshold: float,
    min_visible_keypoints: int,
    max_side: int,
    overwrite: bool,
) -> FolderStats:
    folder_key = folder_key_from_path(camera_folder, image_root)
    metadata = parse_sequence_metadata(camera_folder, image_root)
    image_paths = list_timestamp_images(camera_folder)
    csv_output_path = csv_output_path_for_folder(
        camera_folder=camera_folder,
        image_root=image_root,
        output_root=output_root,
    )

    if not overwrite and csv_output_path.is_file():
        return FolderStats(
            folder_key=folder_key,
            total_images=len(image_paths),
            processed_images=0,
            missed_images=0,
            skipped_existing=len(image_paths),
        )

    processed_images = 0
    missed_images = 0
    rows: list[dict[str, object]] = []

    for image_path in tqdm(image_paths, desc=folder_key, leave=False):
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        inference_image = resize_for_inference(image, max_side)
        bbox_xywh, keypoints, scores = estimator.estimate(inference_image)
        bbox_xywh, keypoints = scale_pose_to_original(
            bbox_xywh=bbox_xywh,
            keypoints=keypoints,
            original_size=image.size,
            inference_size=inference_image.size,
        )
        has_keypoints = has_valid_keypoints(
            scores=scores,
            keypoint_threshold=keypoint_threshold,
            min_visible_keypoints=min_visible_keypoints,
        )

        rows.append(
            keypoints_to_csv_row(
                metadata=metadata,
                image_path=image_path,
                image_size=image.size,
                bbox_xywh=bbox_xywh,
                keypoints=keypoints,
                scores=scores,
                keypoint_threshold=keypoint_threshold,
                min_visible_keypoints=min_visible_keypoints,
            )
        )

        if has_keypoints:
            processed_images += 1
        else:
            missed_images += 1

    write_folder_keypoints_csv(csv_output_path, rows)

    return FolderStats(
        folder_key=folder_key,
        total_images=len(image_paths),
        processed_images=processed_images,
        missed_images=missed_images,
        skipped_existing=0,
    )


def write_folder_summary(
    output_root: Path,
    folder_stats: list[FolderStats],
) -> Path:
    summary_path = output_root / "pose_estimate_folder_summary.csv"
    output_root.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "folder",
                "total_images",
                "processed_images",
                "missed_images",
                "skipped_existing",
            ],
        )
        writer.writeheader()
        for stats in folder_stats:
            writer.writerow(
                {
                    "folder": stats.folder_key,
                    "total_images": stats.total_images,
                    "processed_images": stats.processed_images,
                    "missed_images": stats.missed_images,
                    "skipped_existing": stats.skipped_existing,
                }
            )

    totals = Counter()
    for stats in folder_stats:
        totals["total_images"] += stats.total_images
        totals["processed_images"] += stats.processed_images
        totals["missed_images"] += stats.missed_images
        totals["skipped_existing"] += stats.skipped_existing

    metadata = {
        "folder_count": len(folder_stats),
        "totals": dict(totals),
        "folders": [
            {
                "folder": stats.folder_key,
                "total_images": stats.total_images,
                "processed_images": stats.processed_images,
                "missed_images": stats.missed_images,
                "skipped_existing": stats.skipped_existing,
            }
            for stats in folder_stats
        ],
    }
    with (output_root / "pose_estimate_summary.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    return summary_path


def build_pose_dataset(
    image_root: Path,
    output_root: Path,
    pose_model_name: str,
    device_name: str,
    keypoint_threshold: float,
    min_visible_keypoints: int,
    max_side: int,
    dataset_index: int,
    allow_download: bool,
    overwrite: bool,
) -> dict[str, int]:
    image_root = Path(image_root)
    output_root = Path(output_root)

    if not image_root.is_dir():
        raise ValueError(f"Image root does not exist: {image_root}")

    camera_folders = iter_camera_folders(image_root)
    if not camera_folders:
        raise ValueError(
            f"No Subject/Activity/Trial/Camera folders found under: {image_root}"
        )

    device = choose_device(device_name)
    estimator = VitPoseEstimator(
        pose_model_name=pose_model_name,
        device=device,
        dataset_index=dataset_index,
        allow_download=allow_download,
    )

    folder_stats: list[FolderStats] = []
    for camera_folder in camera_folders:
        folder_stats.append(
            process_camera_folder(
                camera_folder=camera_folder,
                image_root=image_root,
                output_root=output_root,
                estimator=estimator,
                keypoint_threshold=keypoint_threshold,
                min_visible_keypoints=min_visible_keypoints,
                max_side=max_side,
                overwrite=overwrite,
            )
        )

    summary_path = write_folder_summary(output_root, folder_stats)

    totals = Counter()
    for stats in folder_stats:
        totals["folder_count"] += 1
        totals["total_images"] += stats.total_images
        totals["processed_images"] += stats.processed_images
        totals["missed_images"] += stats.missed_images
        totals["skipped_existing"] += stats.skipped_existing

    print(f"Image root: {image_root}")
    print(f"Output root: {output_root}")
    print(f"Pose model: {pose_model_name}")
    print(f"Folder summary: {summary_path}")
    print(f"Stats: {dict(sorted(totals.items()))}")

    return dict(sorted(totals.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ViTPose-H keypoints from Subject/Activity/Trial/Camera folders. "
            "Each sequence folder writes one keypoints.csv containing all frames."
        )
    )
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
    )
    parser.add_argument("--keypoint-threshold", type=float, default=0.25)
    parser.add_argument("--min-visible-keypoints", type=int, default=1)
    parser.add_argument(
        "--max-side",
        type=int,
        default=1280,
        help="Resize long side before inference. Use 0 to keep original size.",
    )
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=0,
        help="Expert index for ViTPose++ models. Ignored for plain ViTPose models.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow transformers to download ViTPose weights if they are not cached.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_pose_dataset(
        image_root=args.image_root,
        output_root=args.output_root,
        pose_model_name=args.pose_model,
        device_name=args.device,
        keypoint_threshold=args.keypoint_threshold,
        min_visible_keypoints=args.min_visible_keypoints,
        max_side=args.max_side,
        dataset_index=args.dataset_index,
        allow_download=args.allow_download,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
