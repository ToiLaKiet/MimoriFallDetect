#!/usr/bin/env python3
"""Train the CNN+LSTM classifier from precomputed skeleton images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import SkeletonImageLSTMClassifier  # noqa: E402
from utils import train_model  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_COLUMNS = ("frame", "image", "image_path", "path", "filename", "file")
LABEL_COLUMNS = ("label", "Label", "target", "class", "class_id", "activity")
GROUP_COLUMNS = ("group", "video", "clip", "sequence", "trial", "source_file")
SORT_COLUMNS = ("frame_index", "index", "idx", "timestamp", "time", "sort_key")


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
class ManifestRow:
    image_path: Path
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


def label_lookup_keys(value: str | Path) -> tuple[str, ...]:
    text = str(value).strip()
    if not text:
        return ()

    path = Path(text)
    candidates = [text]
    if path.name and path.name != text:
        candidates.append(path.name)
    if path.suffix.lower() in IMAGE_EXTENSIONS and path.stem:
        candidates.append(path.stem)

    return tuple(dict.fromkeys(candidates))


def get_row_value(row: dict[str, object], name: str, default: str = "") -> str:
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
    return int(float(value)) + offset


def normalize_sort_key(value: str, row_index: int) -> str:
    if not value:
        return f"{row_index:012d}"
    try:
        return f"{int(float(value)):012d}"
    except ValueError:
        return value


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


def expand_group_columns(row: dict[str, object], group_cols: Iterable[str]) -> tuple[str, ...]:
    if isinstance(group_cols, str):
        group_cols = (group_cols,)

    expanded = []
    for group_col in group_cols:
        group_col = str(group_col).strip()
        if not group_col:
            continue
        if get_row_value(row, group_col):
            expanded.append(group_col)
            continue

        separators = "," if "," in group_col else None
        candidates = group_col.split(separators)
        expanded.extend(candidate.strip() for candidate in candidates if candidate.strip())

    return tuple(dict.fromkeys(expanded))


def manifest_group_key(
    row: dict[str, object],
    group_cols: Iterable[str],
    label: int,
    image_path: Path,
    row_index: int,
) -> str:
    explicit_group_cols = expand_group_columns(row, group_cols)
    if explicit_group_cols:
        parts = []
        missing_cols = []
        for group_col in explicit_group_cols:
            value = get_row_value(row, group_col)
            if value:
                parts.append((group_col, value))
            else:
                missing_cols.append(group_col)

        if missing_cols:
            raise KeyError(
                f"Missing group column(s) at manifest row {row_index}: "
                f"{', '.join(missing_cols)}"
            )
        if len(parts) == 1:
            return parts[0][1]
        return "|".join(f"{name}={value}" for name, value in parts)

    group_key = first_row_value(row, "", GROUP_COLUMNS)
    if group_key:
        return group_key
    if image_path.parent.name:
        return image_path.parent.as_posix()
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
            for optional_col in FRAME_COLUMNS:
                value = get_row_value(row, optional_col)
                if value:
                    key_sources.append(value)

            for key_source in key_sources:
                for key in label_lookup_keys(key_source):
                    if key in label_map:
                        duplicate_keys += 1
                    label_map[key] = label_row

    if duplicate_keys:
        print(f"Warning: {duplicate_keys} duplicate label lookup keys were overwritten.")
    return label_map


def rows_from_json_manifest(data: object) -> list[dict[str, object]]:
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
    #suffix là phần mở rộng của file, ví dụ .json, .csv, .jsonl
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


def read_manifest(
    manifest_path: Path,
    image_dir: Path | None,
    frame_col: str,
    label_col: str,
    label_offset: int,
    group_cols: Iterable[str],
    sort_col: str,
    limit: int,
) -> list[ManifestRow]:
    # resolve là để tìm file ảnh dựa trên giá trị frame trong manifest, có thể là đường dẫn tuyệt đối hoặc tương đối với image_dir hoặc manifest_dir
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

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
        label = parse_label(label_value, label_offset)
        image_path = resolve_manifest_image_path(
            frame_value=frame_value,
            image_dir=image_dir,
            manifest_dir=manifest_path.parent,
        )
        group_key = manifest_group_key(
            row=row,
            group_cols=group_cols,
            label=label,
            image_path=image_path,
            row_index=row_index,
        )

        sort_key = normalize_sort_key(
            first_row_value(row, sort_col, SORT_COLUMNS),
            row_index,
        )

        rows.append(
            ManifestRow(
                image_path=image_path,
                label=label,
                group_key=group_key,
                sort_key=sort_key,
            )
        )

    return rows


def find_label_for_image(image_path: Path, label_map: dict[str, LabelRow]) -> LabelRow | None:
    key_sources = [
        image_path.name,
        image_path.stem,
        str(image_path),
    ]
    for key_source in key_sources:
        for key in label_lookup_keys(key_source):
            if key in label_map:
                return label_map[key]
    return None


def skeleton_cache_path(image_path: Path, image_root: Path, cache_root: Path) -> Path:
    resolved_image = image_path.resolve()
    resolved_root = image_root.resolve()
    try:
        # relative_to trả về đường dẫn tương đối từ resolved_root đến resolved_image nếu resolved_image nằm trong resolved_root, ngược lại sẽ ném ValueError
        rel_path = resolved_image.relative_to(resolved_root)
    except ValueError:
        digest = hashlib.sha1(str(resolved_image).encode("utf-8")).hexdigest()[:12]
        rel_path = Path("_external") / f"{image_path.stem}_{digest}{image_path.suffix}"
    return cache_root / rel_path.with_suffix(".png")


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
    skeleton_dir: Path,
    label_map: dict[str, LabelRow],
) -> tuple[list[FrameItem], list[Path], list[Path]]:
    items: list[FrameItem] = []
    missing_labels: list[Path] = []
    missing_skeletons: list[Path] = []

    for image_path in image_paths:
        label_row = find_label_for_image(image_path, label_map)
        if label_row is None:
            missing_labels.append(image_path)
            continue
        
        skeleton_path = skeleton_cache_path(image_path, image_root, skeleton_dir)
        if not skeleton_path.is_file():
            missing_skeletons.append(skeleton_path)
            continue

        items.append(
            FrameItem(
                image_path=image_path,
                skeleton_path=skeleton_path,
                label=label_row.label,
                group_key=label_row.group_key,
                sort_key=label_row.sort_key,
            )
        )

    return items, missing_labels, missing_skeletons


def make_frame_items_from_manifest(
    manifest_rows: list[ManifestRow],
    image_root: Path,
    skeleton_dir: Path,
) -> tuple[list[FrameItem], list[Path]]:
    items: list[FrameItem] = []
    missing_skeletons: list[Path] = []

    for row in manifest_rows:
        skeleton_path = skeleton_cache_path(row.image_path, image_root, skeleton_dir)
        if not skeleton_path.is_file():
            missing_skeletons.append(skeleton_path)
            continue

        items.append(
            FrameItem(
                image_path=row.image_path,
                skeleton_path=skeleton_path,
                label=row.label,
                group_key=row.group_key,
                sort_key=row.sort_key,
            )
        )

    return items, missing_skeletons


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
        description="Train SkeletonImageLSTMClassifier from precomputed skeleton images."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help=(
            "Original frame root used to map frames to skeleton paths. "
            "Required without --manifest-path."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help=(
            "CSV/JSON/JSONL manifest where each row maps one frame to one label. "
            "Common frame columns are auto-detected: frame, image_path, path, filename."
        ),
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=SCRIPT_DIR / "labels.csv",
        help=(
            "Timestamp-wise label CSV used when --manifest-path is not provided. "
            "Default: ViT+LSTM/labels.csv."
        ),
    )
    parser.add_argument(
        "--frame-col",
        default="",
        help="Manifest frame/path column. Empty value auto-detects common names.",
    )
    parser.add_argument(
        "--group-col",
        nargs="*",
        default=(),
        help="Optional manifest group column(s), e.g. Subject Activity Trial Camera.",
    )
    parser.add_argument(
        "--sort-col",
        default="",
        help="Optional manifest sort column, e.g. frame_index/timestamp.",
    )
    parser.add_argument(
        "--skeleton-dir",
        "--pose-cache-dir",
        dest="skeleton_dir",
        type=Path,
        default=SCRIPT_DIR / "vitpose_cache",
        help="Directory containing precomputed skeleton PNG files. Default: ViT+LSTM/vitpose_cache.",
    )
    parser.add_argument("--no-extract", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Training device. Default: auto.",
    )
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--sequence-label",
        # majority là lấy nhãn xuất hiện nhiều nhất trong cửa sổ, last là lấy nhãn cuối cùng trong cửa sổ
        choices=("majority", "last"),
        default="last",
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
    #group_split nhận giá trị True để chia tập train/val theo nhóm trial/source thay vì chia ngẫu nhiên theo chuỗi
    parser.add_argument(
        "--group-split",
        action="store_true",
        help="Split train/val by trial/source group instead of random sequence split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
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

    if args.manifest_path is not None:
        args.manifest_path = args.manifest_path.resolve()
    if args.image_dir is None:
        if args.manifest_path is None:
            raise RuntimeError("--image-dir is required when --manifest-path is not provided.")
        args.image_dir = args.manifest_path.parent
    args.image_dir = args.image_dir.resolve()
    args.labels_csv = args.labels_csv.resolve()
    args.skeleton_dir = args.skeleton_dir.resolve()
    args.sequence_length = max(1, args.sequence_length)
    args.stride = max(1, args.stride)

    if args.manifest_path is not None:
        manifest_rows = read_manifest(
            manifest_path=args.manifest_path,
            image_dir=args.image_dir,
            frame_col=args.frame_col,
            label_col=args.label_col,
            label_offset=args.label_offset,
            group_cols=args.group_col,
            sort_col=args.sort_col,
            limit=args.limit,
        )
        frame_items, missing_images = make_frame_items_from_manifest(
            manifest_rows=manifest_rows,
            image_root=args.image_dir,
            skeleton_dir=args.skeleton_dir,
        )
        print(
            f"Read {len(manifest_rows)} manifest rows, matched {len(frame_items)} skeletons, "
            f"missing skeleton files for {len(missing_images)} rows."
        )
        if missing_images:
            preview = ", ".join(str(path) for path in missing_images[:5])
            print(f"First missing skeleton files: {preview}")
    else:
        image_paths = iter_images(args.image_dir, args.limit)
        if not image_paths:
            raise RuntimeError(f"No images found in {args.image_dir}")

        label_map = read_labels(args.labels_csv, args.label_col, args.label_offset)
        frame_items, missing_labels, missing_skeletons = make_frame_items(
            image_paths=image_paths,
            image_root=args.image_dir,
            skeleton_dir=args.skeleton_dir,
            label_map=label_map,
        )

        print(
            f"Found {len(image_paths)} frames, matched {len(frame_items)} skeletons, "
            f"missing labels for {len(missing_labels)} frames, "
            f"missing skeleton files for {len(missing_skeletons)} frames."
        )
        if missing_labels:
            preview = ", ".join(path.name for path in missing_labels[:5])
            print(f"First missing-label images: {preview}")
        if missing_skeletons:
            preview = ", ".join(str(path) for path in missing_skeletons[:5])
            print(f"First missing skeleton files: {preview}")

    if not frame_items:
        raise RuntimeError(
            "No skeleton frames are trainable; run extract_vitpose_skeletons.py first."
        )

    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Using skeletons from {args.skeleton_dir}")

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
