from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


SUBJECT_PATTERN = re.compile(r"^Subject\D*(\d+)\D*$", re.IGNORECASE)
CLASS_NAMES = {0: "normal", 1: "fall"}


@dataclass(frozen=True)
class ValidSequence:
    record: dict[str, object]
    resolved_paths: tuple[Path, ...]


def parse_subject_number(subject: str) -> int:
    match = SUBJECT_PATTERN.match(subject.strip())
    if not match:
        raise ValueError(f"Invalid subject format: {subject!r}")
    return int(match.group(1))


def load_records(json_path: Path) -> list[dict[str, object]]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {json_path}")
    return data


def validate_sequence(
    record: dict[str, object],
    image_folder: Path,
) -> ValidSequence | None:
    img_paths = record.get("img_paths")
    if not isinstance(img_paths, list) or not img_paths:
        return None

    resolved_paths: list[Path] = []
    for rel_path in img_paths:
        if not isinstance(rel_path, str) or not rel_path.strip():
            return None
        full_path = image_folder / rel_path
        if not full_path.is_file():
            return None
        resolved_paths.append(full_path)

    return ValidSequence(record=record, resolved_paths=tuple(resolved_paths))


def split_val_test(
    fall_sequences: list[ValidSequence],
    normal_sequences: list[ValidSequence],
    val_ratio: float,
    seed: int,
) -> tuple[list[ValidSequence], list[ValidSequence]]:
    sample_size = min(len(fall_sequences), len(normal_sequences))
    if sample_size == 0:
        return [], []

    rng = random.Random(seed)
    fall_sample = rng.sample(fall_sequences, sample_size)
    normal_sample = rng.sample(normal_sequences, sample_size)
    rng.shuffle(fall_sample)
    rng.shuffle(normal_sample)

    val_count_per_class = round(sample_size * val_ratio)
    val_sequences = (
        fall_sample[:val_count_per_class] + normal_sample[:val_count_per_class]
    )
    test_sequences = (
        fall_sample[val_count_per_class:] + normal_sample[val_count_per_class:]
    )
    rng.shuffle(val_sequences)
    rng.shuffle(test_sequences)
    return val_sequences, test_sequences


def write_sequence(
    sequence: ValidSequence,
    destination_dir: Path,
    seq_name: str,
) -> None:
    seq_dir = destination_dir / seq_name
    seq_dir.mkdir(parents=True, exist_ok=True)

    for frame_index, source_path in enumerate(sequence.resolved_paths):
        destination_path = seq_dir / f"frame_{frame_index:03d}.jpg"
        shutil.copy2(source_path, destination_path)


def export_split(
    sequences: list[ValidSequence],
    output_dir: Path,
    split_name: str,
) -> Counter:
    stats: Counter = Counter()
    class_counters = {class_name: 0 for class_name in CLASS_NAMES.values()}

    for sequence in sequences:
        class_name = CLASS_NAMES[int(sequence.record["fall_alert"])]
        class_counters[class_name] += 1
        seq_name = f"seq_{class_counters[class_name]:04d}"
        write_sequence(
            sequence=sequence,
            destination_dir=output_dir / split_name / class_name,
            seq_name=seq_name,
        )
        stats[f"{split_name}_{class_name}"] += 1
        stats[f"{split_name}_total"] += 1

    return stats


def build_dataset(
    json_path: Path,
    image_folder: Path,
    output_dir: Path,
    val_subject: int = 11,
    val_ratio: float = 0.5,
    seed: int = 42,
) -> dict[str, int]:
    json_path = Path(json_path)
    image_folder = Path(image_folder)
    output_dir = Path(output_dir)

    if not json_path.is_file():
        raise ValueError(f"JSON file does not exist: {json_path}")
    if not image_folder.is_dir():
        raise ValueError(f"Image folder does not exist: {image_folder}")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    records = load_records(json_path)
    stats: Counter = Counter()
    stats["records_total"] = len(records)

    train_sequences: list[ValidSequence] = []
    val_subject_fall: list[ValidSequence] = []
    val_subject_normal: list[ValidSequence] = []

    for record in records:
        subject_text = str(record.get("Subject", "")).strip()
        if not subject_text:
            stats["records_missing_subject"] += 1
            continue

        try:
            subject_number = parse_subject_number(subject_text)
        except ValueError:
            stats["records_invalid_subject"] += 1
            continue

        validated = validate_sequence(record, image_folder)
        if validated is None:
            stats["records_missing_images"] += 1
            continue

        stats["records_valid"] += 1
        if subject_number == val_subject:
            if int(record["fall_alert"]) == 1:
                val_subject_fall.append(validated)
            else:
                val_subject_normal.append(validated)
        else:
            train_sequences.append(validated)

    stats["val_subject_fall_total"] = len(val_subject_fall)
    stats["val_subject_normal_total"] = len(val_subject_normal)

    val_sequences, test_sequences = split_val_test(
        fall_sequences=val_subject_fall,
        normal_sequences=val_subject_normal,
        val_ratio=val_ratio,
        seed=seed,
    )
    stats["val_total"] = len(val_sequences)
    stats["test_total"] = len(test_sequences)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats.update(export_split(train_sequences, output_dir, "train"))
    stats.update(export_split(val_sequences, output_dir, "val"))
    stats.update(export_split(test_sequences, output_dir, "test"))

    return dict(sorted(stats.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build train/val/test image-sequence folders from sequences JSON. "
            "Sequences with missing frames are skipped. Subject 11 is split "
            "between val and test with equal fall/normal counts in each split; "
            "other subjects go to train."
        )
    )
    parser.add_argument("--json-path", type=Path, required=True)
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-subject", type=int, default=11)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.5,
        help=(
            "Fraction of each class from Subject 11 assigned to val; "
            "rest go to test. Val and test each keep equal fall/normal counts."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_dataset(
        json_path=args.json_path,
        image_folder=args.image_folder,
        output_dir=args.output_dir,
        val_subject=args.val_subject,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"JSON: {args.json_path}")
    print(f"Image folder: {args.image_folder}")
    print(f"Output: {args.output_dir}")
    print(f"Val/Test subject: Subject{args.val_subject}")
    print(f"Val ratio: {args.val_ratio}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
