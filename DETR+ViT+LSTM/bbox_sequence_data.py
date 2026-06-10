#!/usr/bin/env python3
"""Create/load sequence data from DETR crop manifests."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import SCRIPT_DIR, add_old_pipeline_to_path

add_old_pipeline_to_path()
from sequence_data import (  # noqa: E402
    FRAME_COLUMNS,
    LABEL_COLUMNS,
    filename_stem_from_path_text,
    load_manifest_rows,
    natural_sort_key,
    normalize_image_timestamp_strict,
    parse_image_size,
    parse_label,
    required_row_value,
    resolve_manifest_image_path,
    trial_key_from_path,
)


CROP_COLUMNS = ("crop_path", "crop", "bbox_crop", "person_crop", "image_path", "frame")
ImageSize = tuple[int, int]


@dataclass(frozen=True)
class BBoxFrameItem:
    crop_path: Path
    label: int
    group_key: str
    sort_key: str


@dataclass(frozen=True)
class BBoxSequenceItem:
    crop_paths: tuple[Path, ...]
    label: int
    group_key: str


@dataclass(frozen=True)
class BBoxSequenceDataBundle:
    sequences: list[BBoxSequenceItem]
    train_sequences: list[BBoxSequenceItem]
    val_sequences: list[BBoxSequenceItem]
    test_sequences: list[BBoxSequenceItem]
    inference_dataset: BBoxCropSequenceDataset
    train_dataset: BBoxCropSequenceDataset | None
    val_dataset: BBoxCropSequenceDataset | None
    test_dataset: BBoxCropSequenceDataset | None
    inference_loader: DataLoader
    train_loader: DataLoader | None
    train_eval_loader: DataLoader | None
    val_loader: DataLoader | None
    test_loader: DataLoader | None
    total_inputs: int
    matched_frames: int
    trial_count: int
    missing_crops: tuple[Path, ...]
    invalid_rows: tuple[str, ...]


def resolve_crop_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidate = (manifest_dir / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def sequence_label(labels: list[int], mode: str) -> int:
    if mode == "last":
        return labels[-1]
    if mode == "majority":
        return Counter(labels).most_common(1)[0][0]
    raise ValueError(f"Unsupported sequence label mode: {mode}")


def build_sequences(
    frames: list[BBoxFrameItem],
    sequence_length: int,
    stride: int,
    label_mode: str,
) -> list[BBoxSequenceItem]:
    if len(frames) < sequence_length:
        return []

    sequences: list[BBoxSequenceItem] = []
    for start in range(0, len(frames) - sequence_length + 1, stride):
        window = frames[start : start + sequence_length]
        sequences.append(
            BBoxSequenceItem(
                crop_paths=tuple(item.crop_path for item in window),
                label=sequence_label([item.label for item in window], label_mode),
                group_key=window[0].group_key,
            )
        )
    return sequences


def build_trial_sequences(
    frame_items: list[BBoxFrameItem],
    sequence_length: int,
    stride: int,
    label_mode: str,
) -> list[BBoxSequenceItem]:
    """Build sliding-window sequences inside one Trial/Camera group only."""

    return build_sequences(frame_items, sequence_length, stride, label_mode)


def build_bbox_manifest_sequence_groups(
    bbox_manifest: Path,
    image_dir: Path,
    crop_col: str,
    frame_col: str,
    label_col: str,
    label_offset: int,
    sequence_length: int,
    stride: int,
    label_mode: str,
    limit: int,
) -> tuple[list[tuple[str, list[BBoxSequenceItem]]], list[Path], list[str], int, int]:
    """Build sequence groups from DETR crop rows using the original Trial grouping logic."""

    bbox_manifest = Path(bbox_manifest).resolve()
    raw_rows = load_manifest_rows(bbox_manifest)
    if limit > 0:
        raw_rows = raw_rows[:limit]

    grouped_frames: dict[str, list[BBoxFrameItem]] = {}
    missing_crops: list[Path] = []
    invalid_timestamps: list[str] = []
    matched_frames = 0

    for row_index, row in enumerate(raw_rows):
        frame_value = required_row_value(
            row=row,
            preferred=frame_col,
            candidates=FRAME_COLUMNS,
            row_index=row_index,
            kind="frame",
        )
        timestamp = normalize_image_timestamp_strict(
            filename_stem_from_path_text(frame_value)
        )
        if timestamp is None:
            invalid_timestamps.append(frame_value)
            continue

        label_value = required_row_value(
            row=row,
            preferred=label_col,
            candidates=LABEL_COLUMNS,
            row_index=row_index,
            kind="label",
        )
        crop_value = required_row_value(
            row=row,
            preferred=crop_col,
            candidates=CROP_COLUMNS,
            row_index=row_index,
            kind="crop",
        )
        image_path = resolve_manifest_image_path(
            frame_value=frame_value,
            image_dir=image_dir,
            manifest_dir=bbox_manifest.parent,
        )
        crop_path = resolve_crop_path(crop_value, bbox_manifest.parent)
        if not crop_path.is_file():
            missing_crops.append(crop_path)
            continue

        group_key = trial_key_from_path(image_path.parent, image_dir)
        grouped_frames.setdefault(group_key, []).append(
            BBoxFrameItem(
                crop_path=crop_path,
                label=parse_label(label_value, label_offset),
                group_key=group_key,
                sort_key=timestamp,
            )
        )
        matched_frames += 1

    sequence_groups: list[tuple[str, list[BBoxSequenceItem]]] = []
    for group_key in sorted(grouped_frames, key=natural_sort_key):
        frame_items = sorted(
            grouped_frames[group_key],
            key=lambda item: (item.sort_key, natural_sort_key(item.crop_path.name)),
        )
        sequences = build_trial_sequences(
            frame_items=frame_items,
            sequence_length=sequence_length,
            stride=stride,
            label_mode=label_mode,
        )
        if sequences:
            sequence_groups.append((group_key, sequences))

    return (
        sequence_groups,
        missing_crops,
        invalid_timestamps,
        len(raw_rows),
        matched_frames,
    )


def split_count(total: int, fraction: float) -> int:
    if fraction <= 0 or total <= 0:
        return 0
    return max(1, int(round(total * fraction)))


def split_sequences(
    sequences: list[BBoxSequenceItem],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[BBoxSequenceItem], list[BBoxSequenceItem], list[BBoxSequenceItem]]:
    items = list(sequences)
    random.Random(seed).shuffle(items)
    total = len(items)
    test_count = split_count(total, test_split)
    val_count = split_count(total, val_split)

    while total > 0 and test_count + val_count >= total:
        if test_count >= val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            break

    test_sequences = items[:test_count]
    val_sequences = items[test_count : test_count + val_count]
    train_sequences = items[test_count + val_count :]
    return train_sequences, val_sequences, test_sequences


class BBoxCropSequenceDataset(Dataset):
    """Loads DETR person crops as B,T,C,H,W tensors for frozen ViT."""

    def __init__(
        self,
        sequences: list[BBoxSequenceItem],
        image_size: ImageSize | int,
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.sequences = sequences
        self.image_size = parse_image_size(image_size)
        self.resample = Image.Resampling.BILINEAR
        self.image_mean = np.asarray(image_mean, dtype=np.float32).reshape(3, 1, 1)
        self.image_std = np.asarray(image_std, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.sequences)

    def load_crop_tensor(self, crop_path: Path) -> np.ndarray:
        with Image.open(crop_path) as image_file:
            image = image_file.convert("RGB")
        image = image.resize(self.image_size, self.resample)
        array = np.asarray(image, dtype=np.float32) / 255.0
        chw = np.transpose(array, (2, 0, 1))
        return (chw - self.image_mean) / self.image_std

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.sequences[index]
        frames = [self.load_crop_tensor(path) for path in item.crop_paths]
        x = torch.from_numpy(np.stack(frames, axis=0))
        y = torch.tensor(item.label, dtype=torch.long)
        return x, y


def make_loader(
    sequences: list[BBoxSequenceItem],
    image_size: ImageSize | int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> tuple[BBoxCropSequenceDataset, DataLoader]:
    dataset = BBoxCropSequenceDataset(sequences, image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataset, loader


def attach_loaders(
    sequences: list[BBoxSequenceItem],
    train_sequences: list[BBoxSequenceItem],
    val_sequences: list[BBoxSequenceItem],
    test_sequences: list[BBoxSequenceItem],
    image_size: ImageSize | int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    total_inputs: int,
    matched_frames: int,
    trial_count: int,
    missing_crops: list[Path],
    invalid_rows: list[str],
) -> BBoxSequenceDataBundle:
    inference_dataset, inference_loader = make_loader(
        sequences, image_size, batch_size, False, num_workers, pin_memory
    )

    train_dataset = train_loader = train_eval_loader = None
    if train_sequences:
        train_dataset, train_loader = make_loader(
            train_sequences, image_size, batch_size, True, num_workers, pin_memory
        )
        train_eval_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    val_dataset = val_loader = None
    if val_sequences:
        val_dataset, val_loader = make_loader(
            val_sequences, image_size, batch_size, False, num_workers, pin_memory
        )

    test_dataset = test_loader = None
    if test_sequences:
        test_dataset, test_loader = make_loader(
            test_sequences, image_size, batch_size, False, num_workers, pin_memory
        )

    return BBoxSequenceDataBundle(
        sequences=sequences,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
        inference_dataset=inference_dataset,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        inference_loader=inference_loader,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        total_inputs=total_inputs,
        matched_frames=matched_frames,
        trial_count=trial_count,
        missing_crops=tuple(missing_crops),
        invalid_rows=tuple(invalid_rows),
    )


def prepare_bbox_sequence_data(
    bbox_manifest: Path,
    image_dir: Path,
    crop_col: str,
    frame_col: str,
    label_col: str,
    label_offset: int,
    sequence_length: int,
    stride: int,
    sequence_label_mode: str,
    val_split: float,
    test_split: float,
    seed: int,
    limit: int,
    image_size: ImageSize | int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> BBoxSequenceDataBundle:
    image_dir = Path(image_dir).resolve()
    sequence_groups, missing_crops, invalid_rows, total_inputs, matched_frames = (
        build_bbox_manifest_sequence_groups(
            bbox_manifest=bbox_manifest,
            image_dir=image_dir,
            crop_col=crop_col,
            frame_col=frame_col,
            label_col=label_col,
            label_offset=label_offset,
            sequence_length=sequence_length,
            stride=stride,
            label_mode=sequence_label_mode,
            limit=limit,
        )
    )

    sequences = [item for _, group in sequence_groups for item in group]

    train_sequences, val_sequences, test_sequences = split_sequences(
        sequences,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )

    return attach_loaders(
        sequences=sequences,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        total_inputs=total_inputs,
        matched_frames=matched_frames,
        trial_count=len(sequence_groups),
        missing_crops=missing_crops,
        invalid_rows=invalid_rows,
    )


def sequence_item_to_dict(item: BBoxSequenceItem) -> dict[str, object]:
    return {
        "crop_paths": [str(path) for path in item.crop_paths],
        "label": item.label,
        "group_key": item.group_key,
    }


def sequence_item_from_dict(item: dict[str, object]) -> BBoxSequenceItem:
    crop_paths = item.get("crop_paths")
    if not isinstance(crop_paths, list):
        raise ValueError("Sequence item must contain a crop_paths list.")
    return BBoxSequenceItem(
        crop_paths=tuple(Path(path) for path in crop_paths),
        label=int(item["label"]),
        group_key=str(item["group_key"]),
    )


def save_bbox_sequence_data(bundle: BBoxSequenceDataBundle, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "source_kind": "bbox_crop_manifest",
        "total_inputs": bundle.total_inputs,
        "matched_frames": bundle.matched_frames,
        "trial_count": bundle.trial_count,
        "missing_crops": [str(path) for path in bundle.missing_crops],
        "invalid_rows": list(bundle.invalid_rows),
        "sequences": [sequence_item_to_dict(item) for item in bundle.sequences],
        "train_sequences": [sequence_item_to_dict(item) for item in bundle.train_sequences],
        "val_sequences": [sequence_item_to_dict(item) for item in bundle.val_sequences],
        "test_sequences": [sequence_item_to_dict(item) for item in bundle.test_sequences],
    }
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_bbox_sequence_data(
    sequence_data_path: Path,
    image_size: ImageSize | int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> BBoxSequenceDataBundle:
    data = json.loads(Path(sequence_data_path).read_text(encoding="utf-8"))
    sequences = [sequence_item_from_dict(item) for item in data.get("sequences", [])]
    train_sequences = [
        sequence_item_from_dict(item) for item in data.get("train_sequences", [])
    ]
    val_sequences = [sequence_item_from_dict(item) for item in data.get("val_sequences", [])]
    test_sequences = [
        sequence_item_from_dict(item) for item in data.get("test_sequences", [])
    ]
    return attach_loaders(
        sequences=sequences,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        total_inputs=int(data.get("total_inputs", len(sequences))),
        matched_frames=int(data.get("matched_frames", len(sequences))),
        trial_count=int(data.get("trial_count", 0)),
        missing_crops=[Path(path) for path in data.get("missing_crops", [])],
        invalid_rows=[str(row) for row in data.get("invalid_rows", [])],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create frozen-ViT sequence data from a DETR bbox manifest."
    )
    parser.add_argument("--bbox-manifest", type=Path, required=True)
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Raw frame root. Trial grouping is derived from frame parent relative to this root.",
    )
    parser.add_argument("--crop-col", default="crop_path")
    parser.add_argument("--frame-col", default="")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--label-offset", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sequence-label", choices=("majority", "last"), default="last")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--image-size",
        type=parse_image_size,
        default=parse_image_size("192x256"),
        help="ViTPose backbone input size as WIDTHxHEIGHT. Default: 192x256.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output",
        "--sequence-data",
        dest="output",
        type=Path,
        default=SCRIPT_DIR / "bbox_sequence_data.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.bbox_manifest = args.bbox_manifest.resolve()
    args.image_dir = args.image_dir.resolve()
    args.output = args.output.resolve()
    args.sequence_length = max(1, args.sequence_length)
    args.stride = max(1, args.stride)

    bundle = prepare_bbox_sequence_data(
        bbox_manifest=args.bbox_manifest,
        image_dir=args.image_dir,
        crop_col=args.crop_col,
        frame_col=args.frame_col,
        label_col=args.label_col,
        label_offset=args.label_offset,
        sequence_length=args.sequence_length,
        stride=args.stride,
        sequence_label_mode=args.sequence_label,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        limit=args.limit,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    save_bbox_sequence_data(bundle, args.output)
    print(
        f"Wrote {len(bundle.sequences)} bbox sequences "
        f"(train={len(bundle.train_sequences)} val={len(bundle.val_sequences)} "
        f"test={len(bundle.test_sequences)}) to {args.output}"
    )
    print(
        f"Groups={bundle.trial_count} frames={bundle.total_inputs} "
        f"matched_frames={bundle.matched_frames} "
        f"missing_crops={len(bundle.missing_crops)} invalid_rows={len(bundle.invalid_rows)}"
    )


if __name__ == "__main__":
    main()
