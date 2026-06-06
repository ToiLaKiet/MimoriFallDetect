"""Create and load sequence datasets for skeleton-image ViT+LSTM workflows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_COLUMNS = ("frame", "image", "image_path", "path", "filename", "file")
LABEL_COLUMNS = ("label", "Label", "target", "class", "class_id", "activity")
SORT_COLUMNS = ("frame_index", "index", "idx", "timestamp", "time", "sort_key")
TIMESTAMP_KEY_PREFIX = "timestamp:"
IMAGE_TIMESTAMP_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?"
)


@dataclass(frozen=True)
class LabelRow:
    """Manifest label metadata for one frame."""

    label: int
    sort_key: str


@dataclass(frozen=True)
class FrameItem:
    """One usable frame inside a Trial folder with its skeleton and label."""

    image_path: Path
    skeleton_path: Path
    label: int
    trial_key: str
    sort_key: str


@dataclass(frozen=True)
class SequenceItem:
    """One sliding-window sequence of skeleton images."""

    skeleton_paths: tuple[Path, ...]
    label: int
    group_key: str


@dataclass(frozen=True)
class SequenceDataBundle:
    """In-memory dataset bundle used by train_vitpose_lstm.py."""

    sequences: list[SequenceItem]
    train_sequences: list[SequenceItem]
    val_sequences: list[SequenceItem]
    test_sequences: list[SequenceItem]
    inference_dataset: SkeletonSequenceDataset
    train_dataset: SkeletonSequenceDataset | None
    val_dataset: SkeletonSequenceDataset | None
    test_dataset: SkeletonSequenceDataset | None
    inference_loader: DataLoader
    train_loader: DataLoader | None
    val_loader: DataLoader | None
    test_loader: DataLoader | None
    source_kind: str
    total_inputs: int
    matched_frames: int
    trial_count: int
    missing_labels: tuple[Path, ...]
    missing_skeletons: tuple[Path, ...]


def pil_bilinear_resample():
    """Return the Pillow bilinear resize enum for old and new Pillow versions."""

    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR


def natural_sort_key(value: str | Path) -> tuple[object, ...]:
    """Sort strings with embedded numbers in human order, e.g. Trial2 before Trial10."""

    text = str(value)
    parts = re.split(r"(\d+)", text)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def normalize_image_timestamp(filename_stem: str | Path) -> str:
    """Normalize timestamp-like image names from files/URLs to microsecond precision."""

    text = str(filename_stem).strip()
    if "T" in text:
        date_part, time_part = text.split("T", 1)
        time_part = time_part.replace("_", ":", 2)
        text = date_part + "T" + time_part

    match = IMAGE_TIMESTAMP_PATTERN.search(text)
    if match is None:
        raise ValueError(f"Cannot parse timestamp from: {filename_stem}")

    timestamp = match.group()
    if "." not in timestamp:
        return timestamp

    main_part, decimal_part = timestamp.split(".", 1)
    if len(decimal_part) <= 6:
        return timestamp

    first_six = decimal_part[:6]
    seventh_digit = int(decimal_part[6])
    dt = datetime.strptime(
        main_part + "." + first_six,
        "%Y-%m-%dT%H:%M:%S.%f",
    )
    if seventh_digit >= 5:
        dt = dt + timedelta(microseconds=1)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")


def filename_stem_from_path_text(value: str | Path) -> str:
    """Extract a filename stem from a local path or URL-like manifest value."""

    text = unquote(str(value).strip())
    parsed = urlparse(text)
    path_text = parsed.path if parsed.scheme or parsed.netloc else text
    path_text = path_text.split("?", 1)[0].split("#", 1)[0]
    return Path(path_text).stem


def timestamp_lookup_keys(*values: str | Path) -> tuple[str, ...]:
    """Build normalized timestamp lookup keys from path strings, URLs, or filenames."""

    keys: list[str] = []
    for value in values:
        candidates = (str(value), filename_stem_from_path_text(value))
        for candidate in candidates:
            if not candidate:
                continue
            try:
                keys.append(TIMESTAMP_KEY_PREFIX + normalize_image_timestamp(candidate))
            except ValueError:
                pass
    return tuple(dict.fromkeys(keys))


def trial_image_sort_key(path: Path) -> tuple[object, ...]:
    """Sort Trial images by normalized timestamp when filenames are timestamp-based."""

    try:
        return (0, normalize_image_timestamp(path.stem))
    except ValueError:
        return (1, natural_sort_key(path.name))


def iter_images(image_dir: Path, limit: int = 0) -> list[Path]:
    """Return all image files below image_dir, sorted by natural path order."""

    image_dir = Path(image_dir)
    paths = sorted(
        (
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )
    if limit > 0:
        paths = paths[:limit]
    return paths


def get_row_value(row: dict[str, object], name: str, default: str = "") -> str:
    """Read a manifest value by column name with case-insensitive fallback."""

    if name in row:
        value = row[name]
        return default if value is None else str(value).strip()
    lowered = name.lower()
    for key, value in row.items():
        if key.lower() == lowered:
            return default if value is None else str(value).strip()
    return default


def first_row_value(
    row: dict[str, object],
    preferred: str,
    candidates: Iterable[str],
) -> str:
    """Return the first non-empty value from preferred and candidate columns."""

    if preferred:
        value = get_row_value(row, preferred)
        if value:
            return value

    for candidate in candidates:
        value = get_row_value(row, candidate)
        if value:
            return value
    return ""


def required_row_value(
    row: dict[str, object],
    preferred: str,
    candidates: Iterable[str],
    row_index: int,
    kind: str,
) -> str:
    """Return a required manifest value or raise a clear column-missing error."""

    value = first_row_value(row, preferred, candidates)
    if value:
        return value

    names = [preferred] if preferred else []
    names.extend(candidate for candidate in candidates if candidate not in names)
    raise KeyError(
        f"Missing {kind} column at manifest row {row_index}. "
        f"Tried: {', '.join(names)}"
    )


def parse_label(value: str, offset: int) -> int:
    """Parse numeric labels from manifest text and apply label_offset."""

    return int(float(value)) + offset


def normalize_sort_key(value: str, row_index: int) -> str:
    """Normalize numeric or timestamp values so lexical sort keeps chronological order."""

    if not value:
        return f"{row_index:012d}"
    try:
        return f"{int(float(value)):012d}"
    except ValueError:
        try:
            return normalize_image_timestamp(value)
        except ValueError:
            return value


def rows_from_json_manifest(data: object) -> list[dict[str, object]]:
    """Convert supported JSON manifest shapes into a list of row dictionaries."""

    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = None
        for key in ("frames", "items", "annotations", "data", "manifest"):
            value = data.get(key)
            if isinstance(value, list):
                rows = value
                break

        if rows is None:
            rows = []
            for frame_name, value in data.items():
                if isinstance(value, dict):
                    item = dict(value)
                    if not any(get_row_value(item, col) for col in FRAME_COLUMNS):
                        item["frame"] = frame_name
                else:
                    item = {"frame": frame_name, "label": value}
                rows.append(item)
    else:
        raise ValueError("JSON manifest must be a list or object.")

    converted = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Manifest row {index} must be an object/dict.")
        converted.append(dict(row))
    return converted


def load_manifest_rows(manifest_path: Path) -> list[dict[str, object]]:
    """Load CSV, JSON, JSONL, or NDJSON manifest rows."""

    manifest_path = Path(manifest_path)
    suffix = manifest_path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with manifest_path.open("r", encoding="utf-8", errors="replace") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL line {line_number} must be an object.")
                rows.append(row)
        return rows

    if suffix == ".json":
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return rows_from_json_manifest(data)

    with manifest_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest CSV has no header: {manifest_path}")
        return [dict(row) for row in reader]


def resolve_manifest_image_path(
    frame_value: str,
    image_dir: Path | None,
    manifest_dir: Path,
) -> Path:
    """Resolve a frame path from the manifest against image_dir, then manifest_dir."""

    frame_path = Path(frame_value).expanduser()
    if frame_path.is_absolute():
        return frame_path.resolve()

    roots = []
    if image_dir is not None:
        roots.append(image_dir)
    roots.append(manifest_dir)

    seen = set()
    for root in roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        candidate = (root / frame_path).resolve()
        if candidate.exists():
            return candidate

    return ((image_dir or manifest_dir) / frame_path).resolve()


def manifest_lookup_keys(
    frame_value: str,
    image_path: Path,
    image_dir: Path,
) -> tuple[str, ...]:
    """Create path and timestamp lookup keys for manifest-to-frame matching."""

    keys = [image_path.resolve().as_posix(), str(image_path.resolve())]
    try:
        keys.append(image_path.resolve().relative_to(image_dir.resolve()).as_posix())
    except ValueError:
        pass

    raw_path = Path(frame_value).expanduser()
    keys.append(raw_path.as_posix())
    keys.append(str(raw_path))
    keys.extend(timestamp_lookup_keys(frame_value))
    return tuple(dict.fromkeys(key for key in keys if key))


def read_manifest_label_map(
    manifest_path: Path,
    image_dir: Path,
    frame_col: str,
    label_col: str,
    label_offset: int,
    sort_col: str,
    limit: int,
) -> dict[str, LabelRow]:
    """Build a path-keyed label map from manifest rows for frame lookup."""

    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    raw_rows = load_manifest_rows(manifest_path)
    if limit > 0:
        raw_rows = raw_rows[:limit]

    label_map: dict[str, LabelRow] = {}
    ambiguous_keys: set[str] = set()
    duplicate_keys = 0
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
        image_path = resolve_manifest_image_path(
            frame_value=frame_value,
            image_dir=image_dir,
            manifest_dir=manifest_path.parent,
        )
        sort_value = first_row_value(row, sort_col, SORT_COLUMNS) or image_path.name
        label_row = LabelRow(
            label=parse_label(label_value, label_offset),
            sort_key=normalize_sort_key(sort_value, row_index),
        )

        for key in manifest_lookup_keys(frame_value, image_path, image_dir):
            if key in ambiguous_keys:
                duplicate_keys += 1
                continue

            existing = label_map.get(key)
            if existing is not None:
                duplicate_keys += 1
                if existing.label != label_row.label:
                    label_map.pop(key, None)
                    ambiguous_keys.add(key)
                continue

            label_map[key] = label_row

    if duplicate_keys:
        print(f"Warning: {duplicate_keys} duplicate manifest lookup keys were repeated.")
    if ambiguous_keys:
        print(
            "Warning: "
            f"{len(ambiguous_keys)} manifest lookup keys had conflicting labels and were ignored."
        )
    return label_map


def find_label_for_image(
    image_path: Path,
    image_dir: Path,
    label_map: dict[str, LabelRow],
) -> LabelRow | None:
    """Find the manifest label row for an image path discovered in a Trial folder."""

    resolved = image_path.resolve()
    keys = [resolved.as_posix(), str(resolved)]
    try:
        keys.append(resolved.relative_to(image_dir.resolve()).as_posix())
    except ValueError:
        pass
    keys.extend(timestamp_lookup_keys(resolved.as_posix()))

    for key in dict.fromkeys(keys):
        if key in label_map:
            return label_map[key]
    return None


def skeleton_cache_path(image_path: Path, image_root: Path, cache_root: Path) -> Path:
    """Map a raw image path to the mirrored skeleton PNG path under cache_root."""

    resolved_image = image_path.resolve()
    resolved_root = image_root.resolve()
    try:
        rel_path = resolved_image.relative_to(resolved_root)
    except ValueError:
        digest = hashlib.sha1(str(resolved_image).encode("utf-8")).hexdigest()[:12]
        rel_path = Path("_external") / f"{image_path.stem}_{digest}{image_path.suffix}"
    return cache_root / rel_path.with_suffix(".png")


def find_trial_dirs(image_dir: Path) -> list[Path]:
    """Find Subject*/Activity*/Camera*/Trial* directories that contain raw frames."""

    image_dir = Path(image_dir)
    trial_dirs = []
    for subject_dir in sorted(image_dir.glob("Subject*"), key=natural_sort_key):
        if not subject_dir.is_dir():
            continue
        for activity_dir in sorted(subject_dir.glob("Activity*"), key=natural_sort_key):
            if not activity_dir.is_dir():
                continue
            for camera_dir in sorted(activity_dir.glob("Camera*"), key=natural_sort_key):
                if not camera_dir.is_dir():
                    continue
                for trial_dir in sorted(camera_dir.glob("Trial*"), key=natural_sort_key):
                    if trial_dir.is_dir() and iter_trial_images(trial_dir):
                        trial_dirs.append(trial_dir)
    return trial_dirs


def trial_key_from_path(trial_dir: Path, image_dir: Path) -> str:
    """Return a stable split/group key for one Trial directory."""

    try:
        return trial_dir.resolve().relative_to(image_dir.resolve()).as_posix()
    except ValueError:
        return trial_dir.resolve().as_posix()


def iter_trial_images(trial_dir: Path) -> list[Path]:
    """Return immediate image children of one Trial directory sorted by timestamp-like names."""

    return sorted(
        (
            path
            for path in Path(trial_dir).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=trial_image_sort_key,
    )


def frame_items_for_trial(
    trial_dir: Path,
    image_dir: Path,
    skeleton_dir: Path,
    label_map: dict[str, LabelRow],
) -> tuple[list[FrameItem], list[Path], list[Path]]:
    """Create ordered FrameItem objects for one Trial, skipping missing labels/skeletons."""

    trial_key = trial_key_from_path(trial_dir, image_dir)
    frame_items: list[FrameItem] = []
    missing_labels: list[Path] = []
    missing_skeletons: list[Path] = []

    for image_path in iter_trial_images(trial_dir):
        label_row = find_label_for_image(image_path, image_dir, label_map)
        if label_row is None:
            missing_labels.append(image_path)
            continue

        skeleton_path = skeleton_cache_path(image_path, image_dir, skeleton_dir)
        if not skeleton_path.is_file():
            missing_skeletons.append(skeleton_path)
            continue

        frame_items.append(
            FrameItem(
                image_path=image_path,
                skeleton_path=skeleton_path,
                label=label_row.label,
                trial_key=trial_key,
                sort_key=label_row.sort_key or image_path.name,
            )
        )

    return (
        sorted(frame_items, key=lambda item: (item.sort_key, natural_sort_key(item.image_path.name))),
        missing_labels,
        missing_skeletons,
    )


def sequence_label(labels: Iterable[int], mode: str) -> int:
    """Choose one label for a sliding-window sequence from its frame labels."""

    labels = list(labels)
    if mode == "last":
        return labels[-1]
    if mode == "majority":
        return Counter(labels).most_common(1)[0][0]
    raise ValueError(f"Unsupported sequence label mode: {mode}")


def build_trial_sequences(
    frame_items: list[FrameItem],
    sequence_length: int,
    stride: int,
    label_mode: str,
) -> list[SequenceItem]:
    """Build sliding-window sequences inside one Trial without crossing Trial boundaries."""

    if len(frame_items) < sequence_length:
        return []

    sequences: list[SequenceItem] = []
    for start in range(0, len(frame_items) - sequence_length + 1, stride):
        window = frame_items[start : start + sequence_length]
        sequences.append(
            SequenceItem(
                skeleton_paths=tuple(item.skeleton_path for item in window),
                label=sequence_label((item.label for item in window), label_mode),
                group_key=window[0].trial_key,
            )
        )
    return sequences


def build_trial_sequence_groups(
    trial_dirs: list[Path],
    image_dir: Path,
    skeleton_dir: Path,
    label_map: dict[str, LabelRow],
    sequence_length: int,
    stride: int,
    label_mode: str,
) -> tuple[list[tuple[str, list[SequenceItem]]], list[Path], list[Path], int]:
    """Build sequence groups per Trial and collect missing label/skeleton diagnostics."""

    groups: list[tuple[str, list[SequenceItem]]] = []
    missing_labels: list[Path] = []
    missing_skeletons: list[Path] = []
    matched_frames = 0

    for trial_dir in trial_dirs:
        frame_items, trial_missing_labels, trial_missing_skeletons = frame_items_for_trial(
            trial_dir=trial_dir,
            image_dir=image_dir,
            skeleton_dir=skeleton_dir,
            label_map=label_map,
        )
        missing_labels.extend(trial_missing_labels)
        missing_skeletons.extend(trial_missing_skeletons)
        matched_frames += len(frame_items)

        trial_sequences = build_trial_sequences(
            frame_items=frame_items,
            sequence_length=sequence_length,
            stride=stride,
            label_mode=label_mode,
        )
        if trial_sequences:
            groups.append((trial_key_from_path(trial_dir, image_dir), trial_sequences))

    return groups, missing_labels, missing_skeletons, matched_frames


def split_count(total: int, fraction: float) -> int:
    """Convert a split fraction into a count while keeping zero fractions at zero."""

    if fraction <= 0 or total <= 0:
        return 0
    return max(1, int(round(total * fraction)))


def split_trial_sequence_groups(
    groups: list[tuple[str, list[SequenceItem]]],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[SequenceItem], list[SequenceItem], list[SequenceItem]]:
    """Split whole Trial groups into train/val/test so windows from one Trial stay together."""
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled) # Đảm bảo rằng việc xáo trộn các nhóm Trial là ngẫu nhiên nhưng có thể tái tạo được bằng cách sử dụng seed. Điều này giúp đảm bảo rằng việc phân chia dữ liệu thành train/val/test không bị lệch và có thể được tái tạo trong các lần chạy khác nhau.
    total = len(shuffled)
    test_count = split_count(total, test_split)
    val_count = split_count(total, val_split)

    while total > 0 and test_count + val_count >= total:
        if test_count >= val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            break

    test_groups = shuffled[:test_count]
    val_groups = shuffled[test_count : test_count + val_count]
    train_groups = shuffled[test_count + val_count :]

    train_sequences = [item for _, group in train_groups for item in group]
    val_sequences = [item for _, group in val_groups for item in group]
    test_sequences = [item for _, group in test_groups for item in group]
    return train_sequences, val_sequences, test_sequences


class SkeletonSequenceDataset(Dataset):
    """PyTorch Dataset that loads one skeleton-image sequence per item."""

    def __init__(self, sequences: list[SequenceItem], image_size: int) -> None:
        """Store sequence metadata and image resize settings."""

        self.sequences = sequences
        self.image_size = image_size
        self.resample = pil_bilinear_resample()

    def __len__(self) -> int:
        """Return the number of sequence windows in the dataset."""

        return len(self.sequences)

    def load_image_tensor(self, image_path: Path) -> np.ndarray:
        """Load one skeleton PNG/JPG as a CHW float tensor array in [0, 1]."""

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        image = image.resize((self.image_size, self.image_size), self.resample)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return np.transpose(array, (2, 0, 1))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (sequence_tensor, label_tensor) for one sequence window."""

        item = self.sequences[index]
        frames = [self.load_image_tensor(path) for path in item.skeleton_paths]
        x = torch.from_numpy(np.stack(frames, axis=0))
        y = torch.tensor(item.label, dtype=torch.long)
        return x, y


def make_sequence_loader(
    sequences: list[SequenceItem],
    image_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> tuple[SkeletonSequenceDataset, DataLoader]:
    """Create a SkeletonSequenceDataset and matching DataLoader."""

    dataset = SkeletonSequenceDataset(sequences, image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataset, loader


def attach_datasets_and_loaders(
    sequences: list[SequenceItem],
    train_sequences: list[SequenceItem],
    val_sequences: list[SequenceItem],
    test_sequences: list[SequenceItem],
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    source_kind: str,
    total_inputs: int,
    matched_frames: int,
    trial_count: int,
    missing_labels: Iterable[Path],
    missing_skeletons: Iterable[Path],
) -> SequenceDataBundle:
    """Attach PyTorch Dataset/DataLoader objects to sequence splits."""

    inference_dataset, inference_loader = make_sequence_loader(
        sequences=sequences,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_dataset = None
    train_loader = None
    if train_sequences:
        train_dataset, train_loader = make_sequence_loader(
            sequences=train_sequences,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    val_dataset = None
    val_loader = None
    if val_sequences:
        val_dataset, val_loader = make_sequence_loader(
            sequences=val_sequences,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    test_dataset = None
    test_loader = None
    if test_sequences:
        test_dataset, test_loader = make_sequence_loader(
            sequences=test_sequences,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return SequenceDataBundle(
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
        val_loader=val_loader,
        test_loader=test_loader,
        source_kind=source_kind,
        total_inputs=total_inputs,
        matched_frames=matched_frames,
        trial_count=trial_count,
        missing_labels=tuple(missing_labels),
        missing_skeletons=tuple(missing_skeletons),
    )


def prepare_sequence_data(
    image_dir: Path,
    manifest_path: Path,
    frame_col: str,
    label_col: str,
    label_offset: int,
    sort_col: str,
    skeleton_dir: Path,
    sequence_length: int,
    stride: int,
    sequence_label_mode: str,
    val_split: float,
    test_split: float,
    seed: int,
    limit: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> SequenceDataBundle:
    """Create Trial-bounded sequences, split them, and attach Dataset/DataLoader objects."""

    image_dir = Path(image_dir).resolve()
    skeleton_dir = Path(skeleton_dir).resolve()
    trial_dirs = find_trial_dirs(image_dir) # Hàm này return các thư mục Trial* có chứa ảnh, được sắp xếp theo thứ tự tự nhiên. Nó sẽ đi qua cấu trúc Subject*/Activity*/Camera*/Trial* và kiểm tra xem có ảnh nào trong mỗi thư mục Trial* không. Nếu có, nó sẽ thêm thư mục đó vào danh sách trial_dirs.
    
    if limit > 0:
        trial_dirs = trial_dirs[:limit] 
    if not trial_dirs:
        raise RuntimeError(f"No Trial directories with images found in {image_dir}")

    label_map = read_manifest_label_map(
        manifest_path=manifest_path,
        image_dir=image_dir,
        frame_col=frame_col,
        label_col=label_col,
        label_offset=label_offset,
        sort_col=sort_col,
        limit=0,
    )
    sequence_groups, missing_labels, missing_skeletons, matched_frames = (
        build_trial_sequence_groups(
            trial_dirs=trial_dirs,
            image_dir=image_dir,
            skeleton_dir=skeleton_dir,
            label_map=label_map,
            sequence_length=sequence_length,
            stride=stride,
            label_mode=sequence_label_mode,
        )
    )

    # Cần kiểm tra coi là nó chia như thế nào, có hợp lý không.
    train_sequences, val_sequences, test_sequences = split_trial_sequence_groups(
        groups=sequence_groups,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )
    
    sequences = [item for _, group in sequence_groups for item in group]
    total_inputs = sum(len(iter_trial_images(trial_dir)) for trial_dir in trial_dirs)

    return attach_datasets_and_loaders(
        sequences=sequences,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        source_kind="trial_folders",
        total_inputs=total_inputs,
        matched_frames=matched_frames,
        trial_count=len(trial_dirs),
        missing_labels=missing_labels,
        missing_skeletons=missing_skeletons,
    )


def sequence_item_to_dict(item: SequenceItem) -> dict[str, object]:
    """Serialize one SequenceItem into JSON-compatible data."""

    return {
        "skeleton_paths": [str(path) for path in item.skeleton_paths],
        "label": item.label,
        "group_key": item.group_key,
    }


def sequence_item_from_dict(item: dict[str, object]) -> SequenceItem:
    """Deserialize one SequenceItem from JSON-compatible data."""

    skeleton_paths = item.get("skeleton_paths")
    if not isinstance(skeleton_paths, list):
        raise ValueError("Sequence item must contain a skeleton_paths list.")
    return SequenceItem(
        skeleton_paths=tuple(Path(path) for path in skeleton_paths),
        label=int(item["label"]),
        group_key=str(item["group_key"]),
    )


def save_sequence_data(bundle: SequenceDataBundle, output_path: Path) -> None:
    """Write sequence metadata and train/val/test splits to JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "source_kind": bundle.source_kind,
        "total_inputs": bundle.total_inputs,
        "matched_frames": bundle.matched_frames,
        "trial_count": bundle.trial_count,
        "missing_labels": [str(path) for path in bundle.missing_labels],
        "missing_skeletons": [str(path) for path in bundle.missing_skeletons],
        "sequences": [sequence_item_to_dict(item) for item in bundle.sequences],
        "train_sequences": [sequence_item_to_dict(item) for item in bundle.train_sequences],
        "val_sequences": [sequence_item_to_dict(item) for item in bundle.val_sequences],
        "test_sequences": [sequence_item_to_dict(item) for item in bundle.test_sequences],
    }
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_sequence_data(
    sequence_data_path: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> SequenceDataBundle:
    """Load a serialized sequence dataset and recreate Dataset/DataLoader objects."""

    data = json.loads(Path(sequence_data_path).read_text(encoding="utf-8"))
    sequences = [sequence_item_from_dict(item) for item in data.get("sequences", [])]
    train_sequences = [
        sequence_item_from_dict(item) for item in data.get("train_sequences", [])
    ]
    val_sequences = [sequence_item_from_dict(item) for item in data.get("val_sequences", [])]
    test_sequences = [
        sequence_item_from_dict(item) for item in data.get("test_sequences", [])
    ]

    return attach_datasets_and_loaders(
        sequences=sequences,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        source_kind=str(data.get("source_kind", "sequence_data")),
        total_inputs=int(data.get("total_inputs", len(sequences))),
        matched_frames=int(data.get("matched_frames", len(sequences))),
        trial_count=int(data.get("trial_count", 0)),
        missing_labels=tuple(Path(path) for path in data.get("missing_labels", [])),
        missing_skeletons=tuple(Path(path) for path in data.get("missing_skeletons", [])),
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for creating the serialized sequence dataset."""

    parser = argparse.ArgumentParser(
        description="Create a serialized Trial-bounded skeleton sequence dataset."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--frame-col", default="")
    parser.add_argument("--sort-col", default="")
    parser.add_argument(
        "--skeleton-dir",
        "--pose-cache-dir",
        dest="skeleton_dir",
        type=Path,
        default=SCRIPT_DIR / "vitpose_cache",
    )
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--sequence-label",
        choices=("majority", "last"),
        default="last",
    )
    parser.add_argument("--label-col", default="Label")
    parser.add_argument("--label-offset", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of Trial directories for quick debugging.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output",
        "--sequence-data",
        dest="output",
        type=Path,
        default=SCRIPT_DIR / "sequence_data.json",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point that builds and writes sequence_data.json."""

    args = parse_args()
    args.image_dir = args.image_dir.resolve()
    args.manifest_path = args.manifest_path.resolve()
    args.skeleton_dir = args.skeleton_dir.resolve()
    args.output = args.output.resolve()
    args.sequence_length = max(1, args.sequence_length)
    args.stride = max(1, args.stride)

    bundle = prepare_sequence_data(
        image_dir=args.image_dir,
        manifest_path=args.manifest_path,
        frame_col=args.frame_col,
        label_col=args.label_col,
        label_offset=args.label_offset,
        sort_col=args.sort_col,
        skeleton_dir=args.skeleton_dir,
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
    save_sequence_data(bundle, args.output)
    print(
        f"Wrote {len(bundle.sequences)} sequences "
        f"(train={len(bundle.train_sequences)} val={len(bundle.val_sequences)} "
        f"test={len(bundle.test_sequences)}) to {args.output}"
    )
    print(
        f"Trials={bundle.trial_count} frames={bundle.total_inputs} "
        f"matched_frames={bundle.matched_frames} "
        f"missing_labels={len(bundle.missing_labels)} "
        f"missing_skeletons={len(bundle.missing_skeletons)}"
    )


if __name__ == "__main__":
    main()
