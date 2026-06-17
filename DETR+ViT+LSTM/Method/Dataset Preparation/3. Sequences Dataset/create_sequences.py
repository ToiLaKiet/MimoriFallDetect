from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SUBJECT_PATTERN = re.compile(r"^Subject\D*(\d+)\D*$", re.IGNORECASE) # eg: Subject 11. No space between Subject and the number. 
CLASS_NAMES = {0: "normal", 1: "fall"}
SplitMode = Literal["train", "val-test"]


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


def parse_subject_set(values: list[str] | None) -> set[int] | None:
    if not values:
        return None
    return {int(value) for value in values}


def clear_split_dirs(output_dir: Path, split_names: tuple[str, ...]) -> None:
    for split_name in split_names:
        split_dir = output_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)


def existing_class_counters(split_dir: Path) -> dict[str, int]:
    counters = {class_name: 0 for class_name in CLASS_NAMES.values()}
    for class_name in CLASS_NAMES.values():
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        for path in class_dir.iterdir():
            if not path.is_dir() or not path.name.startswith("seq_"):
                continue
            try:
                counters[class_name] = max(counters[class_name], int(path.name[4:])) # path.name[4:] is the number of the sequence.
            except ValueError:
                continue
    return counters


def split_class(
    sequences: list[ValidSequence],
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[ValidSequence], list[ValidSequence]]:
    if not sequences:
        return [], []

    shuffled = list(sequences)
    rng.shuffle(shuffled)
    val_count = round(len(shuffled) * val_ratio)
    return shuffled[:val_count], shuffled[val_count:]


def split_val_test(
    fall_sequences: list[ValidSequence],
    normal_sequences: list[ValidSequence],
    val_ratio: float,
    seed: int,
) -> tuple[list[ValidSequence], list[ValidSequence]]:
    rng = random.Random(seed)
    fall_val, fall_test = split_class(fall_sequences, val_ratio, rng)
    normal_val, normal_test = split_class(normal_sequences, val_ratio, rng)

    val_sequences = fall_val + normal_val
    test_sequences = fall_test + normal_test
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

    record = sequence.record
    subject = str(record.get("Subject", "")).strip()
    activity = str(record.get("Activity", "")).strip()
    trial = str(record.get("Trial", "")).strip()
    camera = str(record.get("Camera", "")).strip()
    fall_alert = int(record.get("fall_alert", 0))
    img_paths = record.get("img_paths")

    frames_meta: list[dict[str, object]] = []
    for frame_index, source_path in enumerate(sequence.resolved_paths):
        destination_path = seq_dir / f"frame_{frame_index:03d}.jpg"
        shutil.copy2(source_path, destination_path) # to copy is to copy the content of the source path to the destination path.

        rel_path = ""
        if isinstance(img_paths, list) and frame_index < len(img_paths):
            rel_path = str(img_paths[frame_index])
        timestamp = Path(rel_path or source_path.name).stem

        frames_meta.append(
            {
                "frame_index": frame_index,
                "frame_file": destination_path.name,
                "timestamp": timestamp,
                "source_rel_path": rel_path,
                "source_abs_path": str(source_path),
            }
        )

    # Per-sequence metadata to keep frame provenance & identifiers.
    metadata = {
        "Subject": subject,
        "Activity": activity,
        "Trial": trial,
        "Camera": camera,
        "fall_alert": fall_alert,
        "sequence_name": seq_name,
        "frame_count": len(sequence.resolved_paths),
        "frames": frames_meta,
    }
    with (seq_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")


def export_split(
    sequences: list[ValidSequence],
    output_dir: Path,
    split_name: str,
    class_counters: dict[str, int] | None = None,
) -> Counter:
    stats: Counter = Counter()
    if class_counters is None:
        class_counters = {class_name: 0 for class_name in CLASS_NAMES.values()}

    for class_name in CLASS_NAMES.values():
        (output_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)

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
    split_mode: SplitMode = "train",
    train_subjects: set[int] | None = None,
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
    if split_mode == "val-test" and not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if split_mode != "train" and train_subjects is not None:
        raise ValueError("--train-subjects is only supported with --split-mode train.")

    append_output = split_mode == "train" and train_subjects is not None

    records = load_records(json_path)
    stats: Counter = Counter()
    stats["records_total"] = len(records)

    train_sequences: list[ValidSequence] = []
    val_subject_fall: list[ValidSequence] = []
    val_subject_normal: list[ValidSequence] = []

    for record in records:
        subject_text = str(record.get("Subject", "")).strip() # eg: Subject 11
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
        elif (
            split_mode == "train"
            and train_subjects is not None 
            and subject_number not in train_subjects # why not in train_subjects? because we want to export only the subjects that are not in train_subjects.
        ):
            stats["records_filtered_subject"] += 1
        else: # if train_subjects is None, then all subjects are exported to train/.
            train_sequences.append(validated)

    stats["split_mode"] = split_mode
    stats["val_subject_fall_total"] = len(val_subject_fall)
    stats["val_subject_normal_total"] = len(val_subject_normal)
    stats["train_total"] = len(train_sequences)

    output_dir.mkdir(parents=True, exist_ok=True)
    if split_mode == "train" and not append_output:
        clear_split_dirs(output_dir, ("train",))
    elif split_mode == "val-test":
        clear_split_dirs(output_dir, ("val", "test"))

    if split_mode == "train":
        class_counters = None
        if append_output:
            class_counters = existing_class_counters(output_dir / "train")
        stats.update(
            export_split(
                train_sequences,
                output_dir,
                "train",
                class_counters=class_counters,
            )
        )
        return dict(sorted(stats.items()))

    val_sequences, test_sequences = split_val_test(
        fall_sequences=val_subject_fall,
        normal_sequences=val_subject_normal,
        val_ratio=val_ratio,
        seed=seed,
    )
    stats["val_total"] = len(val_sequences)
    stats["test_total"] = len(test_sequences)
    stats.update(export_split(val_sequences, output_dir, "val"))
    stats.update(export_split(test_sequences, output_dir, "test"))

    return dict(sorted(stats.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build image-sequence folders from sequences JSON. "
            "Every split is written under {split}/{fall|normal}/seq_XXXX/. "
            "Sequences with missing frames are skipped. Use --split-mode train "
            "to export all subjects except the held-out subject into "
            "train/fall and train/normal, or --split-mode val-test to export "
            "only that subject into val/ and test/ (all sequences kept)."
        )
    )
    parser.add_argument("--json-path", type=Path, required=True)
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split-mode",
        choices=("train", "val-test"),
        default="train",
        help=(
            "train: export subjects other than --val-subject into "
            "train/fall/ and train/normal/ (by fall_alert). "
            "Use --train-subjects to export one or more subjects at a time "
            "and append to an existing train/ folder. "
            "val-test: export only --val-subject into val/{fall,normal}/ "
            "and test/{fall,normal}/."
        ),
    )
    parser.add_argument(
        "--train-subjects",
        nargs="+",
        type=int,
        default=None,
        metavar="SUBJECT",
        help=(
            "Only for --split-mode train. Export specific subject numbers, "
            "e.g. --train-subjects 7 or --train-subjects 7 8 9 12. "
            "When set, existing output is kept and new sequences are appended "
            "with continued seq_XXXX numbering."
        ),
    )
    parser.add_argument("--val-subject", type=int, default=11)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.5,
        help=(
            "Fraction of each class from the held-out subject assigned to val; "
            "the rest go to test. All fall and normal sequences are kept."
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
        split_mode=args.split_mode,
        train_subjects=parse_subject_set(args.train_subjects),
        val_subject=args.val_subject,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"JSON: {args.json_path}")
    print(f"Image folder: {args.image_folder}")
    print(f"Output: {args.output_dir}")
    print(f"Split mode: {args.split_mode}")
    if args.split_mode == "train" and args.train_subjects:
        print(f"Train subjects: {args.train_subjects}")
    print(f"Held-out subject: Subject{args.val_subject}")
    if args.split_mode == "val-test":
        print(f"Val ratio: {args.val_ratio}")
        print(
            "Val layout: "
            f"fall={stats.get('val_fall', 0)}, "
            f"normal={stats.get('val_normal', 0)} | "
            "Test layout: "
            f"fall={stats.get('test_fall', 0)}, "
            f"normal={stats.get('test_normal', 0)}"
        )
    if args.split_mode == "train":
        print(
            "Train layout: "
            f"fall={stats.get('train_fall', 0)}, "
            f"normal={stats.get('train_normal', 0)}"
        )
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
