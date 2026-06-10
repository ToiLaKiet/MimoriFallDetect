#!/usr/bin/env python3
"""Extract largest-person DETR crops and write a bbox manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import (
    SCRIPT_DIR,
    add_old_pipeline_to_path,
    choose_device,
    configure_runtime_cache,
    resolve_hf_model_source,
)

add_old_pipeline_to_path()
from sequence_data import (  # noqa: E402
    FRAME_COLUMNS,
    LABEL_COLUMNS,
    filename_stem_from_path_text,
    iter_images,
    load_manifest_rows,
    normalize_image_timestamp_strict,
    parse_label,
    required_row_value,
    resolve_manifest_image_path,
    trial_key_from_path,
)


DEFAULT_DETECTOR_MODEL = "PekingU/rtdetr_r50vd_coco_o365"


@dataclass(frozen=True)
class BBoxRecord:
    frame_path: Path
    crop_path: Path
    label: int
    group_key: str
    timestamp: str
    bbox_xyxy: tuple[float, float, float, float]
    score: float


def cache_path_for_crop(image_path: Path, image_root: Path, crop_root: Path) -> Path:
    """Mirror a raw frame path under crop_root, falling back to a digest for external paths."""

    resolved_image = image_path.resolve()
    resolved_root = image_root.resolve()
    try:
        rel_path = resolved_image.relative_to(resolved_root)
    except ValueError:
        digest = hashlib.sha1(str(resolved_image).encode("utf-8")).hexdigest()[:12]
        rel_path = Path("_external") / f"{image_path.stem}_{digest}{image_path.suffix}"
    return crop_root / rel_path.with_suffix(".jpg")


def resize_for_detection(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    """Resize long side for detector speed and return the scale used."""

    if max_side <= 0:
        return image, 1.0
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.BILINEAR), scale


def clamp_xyxy(
    box: np.ndarray,
    width: int,
    height: int,
    padding_ratio: float,
) -> tuple[float, float, float, float]:
    """Clamp and optionally pad an xyxy box to image bounds."""

    x1, y1, x2, y2 = [float(value) for value in box.tolist()]
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio

    x1 = max(0.0, x1 - pad_x)
    y1 = max(0.0, y1 - pad_y)
    x2 = min(float(width), x2 + pad_x)
    y2 = min(float(height), y2 + pad_y)
    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)
    return x1, y1, x2, y2


class LargestPersonBBoxExtractor:
    """RT-DETR person detector that keeps only the largest person box."""

    def __init__(
        self,
        detector_model_name: str,
        device: torch.device,
        threshold: float,
        allow_download: bool,
        max_side: int,
        token: str | None = None,
    ) -> None:
        configure_runtime_cache()

        from transformers import AutoProcessor, RTDetrForObjectDetection  # noqa: PLC0415

        model_source = resolve_hf_model_source(detector_model_name, allow_download)
        print(f"Loading detector: {model_source}")
        self.processor = AutoProcessor.from_pretrained(
            model_source,
            local_files_only=not allow_download,
            use_fast=False,
            token=token
        )
        self.model = (
            RTDetrForObjectDetection.from_pretrained(
                model_source,
                local_files_only=not allow_download,
                token=token
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.threshold = threshold
        self.max_side = max_side

    @torch.no_grad()
    def detect_largest_person(self, image: Image.Image) -> tuple[np.ndarray, float] | None:
        """Return the largest detected person box in original image xyxy coordinates."""

        detector_image, scale = resize_for_detection(image, self.max_side)
        inputs = self.processor(images=detector_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=[(detector_image.height, detector_image.width)],
            threshold=self.threshold,
        )[0]

        labels = results["labels"].detach().cpu().numpy()
        person_indices = np.flatnonzero(labels == 0)
        if len(person_indices) == 0:
            return None

        boxes = results["boxes"][person_indices].detach().cpu().numpy().astype(np.float32)
        scores = results["scores"][person_indices].detach().cpu().numpy().astype(np.float32)
        if scale != 1.0:
            boxes = boxes / scale

        widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        largest_index = int(np.argmax(widths * heights))
        return boxes[largest_index], float(scores[largest_index])

    def crop_largest_person(
        self,
        image_path: Path,
        crop_path: Path,
        padding_ratio: float,
        overwrite: bool,
    ) -> tuple[tuple[float, float, float, float], float] | None:
        """Detect, crop, and save the largest person from one frame."""

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        detection = self.detect_largest_person(image)
        if detection is None:
            return None

        box, score = detection
        bbox_xyxy = clamp_xyxy(box, image.width, image.height, padding_ratio)
        if overwrite or not crop_path.exists():
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            left, top, right, bottom = [int(round(value)) for value in bbox_xyxy]
            image.crop((left, top, right, bottom)).save(crop_path, quality=95)
        return bbox_xyxy, score


def manifest_image_rows(
    manifest_path: Path,
    image_dir: Path,
    frame_col: str,
    label_col: str,
    label_offset: int,
    limit: int,
) -> list[tuple[Path, int, str, str]]:
    """Resolve manifest rows into frame path, label, group key, and timestamp."""

    raw_rows = load_manifest_rows(manifest_path)
    if limit > 0:
        raw_rows = raw_rows[:limit]

    rows = []
    for row_index, row in enumerate(raw_rows):
        frame_value = required_row_value(
            row=row,
            preferred=frame_col,
            candidates=FRAME_COLUMNS,
            row_index=row_index,
            kind="frame",
        )
        label_value = required_row_value(
            row=row,
            preferred=label_col,
            candidates=LABEL_COLUMNS,
            row_index=row_index,
            kind="label",
        )
        timestamp = normalize_image_timestamp_strict(
            filename_stem_from_path_text(frame_value)
        )
        if timestamp is None:
            continue

        image_path = resolve_manifest_image_path(
            frame_value=frame_value,
            image_dir=image_dir,
            manifest_dir=manifest_path.parent,
        )
        group_key = trial_key_from_path(image_path.parent, image_dir)
        rows.append((image_path, parse_label(label_value, label_offset), group_key, timestamp))
    return rows


def image_rows_without_manifest(
    image_dir: Path,
    limit: int,
) -> list[tuple[Path, int, str, str]]:
    """Allow smoke testing without labels by assigning label 0."""
    rows = []
    for image_path in iter_images(image_dir, limit):
        timestamp = normalize_image_timestamp_strict(image_path.stem) or image_path.stem
        rows.append((image_path, 0, trial_key_from_path(image_path.parent, image_dir), timestamp))
    return rows


def write_bbox_manifest(records: list[BBoxRecord], output_path: Path) -> None:
    """Write crop metadata for sequence construction."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame",
        "crop_path",
        "Label",
        "group_key",
        "timestamp",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "det_score",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            x1, y1, x2, y2 = record.bbox_xyxy
            writer.writerow(
                {
                    "frame": str(record.frame_path),
                    "crop_path": str(record.crop_path),
                    "Label": record.label,
                    "group_key": record.group_key,
                    "timestamp": record.timestamp,
                    "bbox_x1": f"{x1:.3f}",
                    "bbox_y1": f"{y1:.3f}",
                    "bbox_x2": f"{x2:.3f}",
                    "bbox_y2": f"{y2:.3f}",
                    "det_score": f"{record.score:.6f}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract largest-person RT-DETR crops and bbox manifest."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--frame-col", default="")
    parser.add_argument("--label-col", default="Label")
    parser.add_argument("--label-offset", type=int, default=0)
    parser.add_argument("--crop-dir", type=Path, default=SCRIPT_DIR / "detr_crops")
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=SCRIPT_DIR / "bbox_manifest.csv",
    )
    parser.add_argument("--detector-model", default=DEFAULT_DETECTOR_MODEL)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--det-thr", type=float, default=0.3)
    parser.add_argument("--padding-ratio", type=float, default=0.08)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--overwrite-crops", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token for private models or to increase rate limits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.image_dir = args.image_dir.resolve()
    args.crop_dir = args.crop_dir.resolve()
    args.output_manifest = args.output_manifest.resolve()
    args.log_every = max(1, args.log_every)

    if args.manifest_path is not None:
        args.manifest_path = args.manifest_path.resolve()
        rows = manifest_image_rows(
            manifest_path=args.manifest_path,
            image_dir=args.image_dir,
            frame_col=args.frame_col,
            label_col=args.label_col,
            label_offset=args.label_offset,
            limit=args.limit,
        )
    else:
        rows = image_rows_without_manifest(args.image_dir, args.limit)
    if not rows:
        raise RuntimeError("No input image rows found.")

    device = choose_device(args.device)
    print(f"Using device: {device}")
    extractor = LargestPersonBBoxExtractor(
        detector_model_name=args.detector_model,
        device=device,
        threshold=args.det_thr,
        allow_download=args.allow_download,
        max_side=args.max_side,
        token=args.token
    )

    records: list[BBoxRecord] = []
    missing_images = 0
    missed_people = 0

    for index, (image_path, label, group_key, timestamp) in enumerate(rows, start=1):
        crop_path = cache_path_for_crop(image_path, args.image_dir, args.crop_dir)
        if not image_path.is_file():
            missing_images += 1
        else:
            result = extractor.crop_largest_person(
                image_path=image_path,
                crop_path=crop_path,
                padding_ratio=args.padding_ratio,
                overwrite=args.overwrite_crops,
            )
            if result is None:
                missed_people += 1
            else:
                bbox_xyxy, score = result
                records.append(
                    BBoxRecord(
                        frame_path=image_path,
                        crop_path=crop_path,
                        label=label,
                        group_key=group_key,
                        timestamp=timestamp,
                        bbox_xyxy=bbox_xyxy,
                        score=score,
                    )
                )

        if index == len(rows) or index % args.log_every == 0:
            print(
                "DETR crops "
                f"{index}/{len(rows)} | kept={len(records)} "
                f"missing_images={missing_images} missed_people={missed_people}"
            )

    write_bbox_manifest(records, args.output_manifest)
    print(f"Crop directory: {args.crop_dir}")
    print(f"BBox manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
