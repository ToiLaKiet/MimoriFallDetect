from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_FALL_TAGS = {1, 2, 3, 4, 5}
FOLDER_PATTERNS = {
    "Subject": re.compile(r"^Subject\D*(\d+)\D*$", re.IGNORECASE),
    "Activity": re.compile(r"^Activity\D*(\d+)\D*$", re.IGNORECASE),
    "Trial": re.compile(r"^Trial\D*(\d+)\D*$", re.IGNORECASE),
    "Camera": re.compile(r"^Camera\D*(\d+)\D*$", re.IGNORECASE),
}


@dataclass(frozen=True)
class ManifestFrame:
    image_path: str
    label: int
    subject: str
    activity: str
    trial: str
    camera: str
    sort_key: tuple[object, ...]
    row_index: int


def natural_sort_key(value: str | Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def split_path_text(value: str | Path) -> tuple[str, ...]:
    text = str(value).strip().replace("\\", "/")
    return tuple(part for part in text.split("/") if part and part != ".")


def sequence_parts_from_path(value: str | Path) -> tuple[str, str, str, str, str]:
    parts = split_path_text(value)
    for index in range(0, len(parts) - 4):
        subject, activity, trial, camera, filename = parts[index:index + 5]
        if (
            FOLDER_PATTERNS["Subject"].match(subject)
            and FOLDER_PATTERNS["Activity"].match(activity)
            and FOLDER_PATTERNS["Trial"].match(trial)
            and FOLDER_PATTERNS["Camera"].match(camera)
            and filename
        ):
            return subject, activity, trial, camera, filename

    raise ValueError(
        "image_path must contain Subject/Activity/Trial/Camera/file "
        f"structure: {value}"
    )


def get_row_value(row: dict[str, str], column_name: str) -> str:
    if column_name in row:
        return str(row[column_name]).strip()

    lowered = column_name.lower()
    for key, value in row.items():
        if key.lower() == lowered:
            return str(value).strip()
    return ""


def parse_int_text(value: str, row_index: int, column_name: str) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Missing {column_name} at manifest row {row_index}.")
    try:
        return int(float(text))
    except ValueError:
        match = re.search(r"(\d+)", text)
        if match:
            return int(match.group(1))
        raise ValueError(
            f"Invalid {column_name} value {value!r} at manifest row {row_index}."
        )


def normalize_group_value(
    value: str,
    prefix: str,
    row_index: int,
    column_name: str,
) -> str:
    return f"{prefix}{parse_int_text(value, row_index, column_name)}"


def group_values_from_row(
    row: dict[str, str],
    row_index: int,
    image_path: str,
    subject_col: str,
    activity_col: str,
    trial_col: str,
    camera_col: str,
) -> tuple[str, str, str, str]:
    subject = get_row_value(row, subject_col)
    activity = get_row_value(row, activity_col)
    trial = get_row_value(row, trial_col)
    camera = get_row_value(row, camera_col)

    if not all((subject, activity, trial, camera)):
        subject, activity, trial, camera, _ = sequence_parts_from_path(image_path)

    return (
        normalize_group_value(subject, "Subject", row_index, subject_col),
        normalize_group_value(activity, "Activity", row_index, activity_col),
        normalize_group_value(trial, "Trial", row_index, trial_col),
        normalize_group_value(camera, "Camera", row_index, camera_col),
    )


def frame_sort_key(
    row: dict[str, str],
    image_path: str,
    row_index: int,
    timestamp_col: str,
) -> tuple[object, ...]:
    timestamp = get_row_value(row, timestamp_col) if timestamp_col else ""
    primary = timestamp or Path(split_path_text(image_path)[-1]).stem
    return natural_sort_key(primary) + (row_index,)


def parse_int_set(value: str) -> set[int]:
    tags = set()
    for part in str(value).split(","):
        text = part.strip()
        if not text:
            continue
        tags.add(int(text))
    if not tags:
        raise argparse.ArgumentTypeError("Tag set cannot be empty.")
    return tags


def resolve_output_format(output_path: Path, output_format: str) -> str:
    if output_format != "auto":
        return output_format
    if output_path.suffix.lower() in {".jsonl", ".ndjson"}:
        return "jsonl"
    return "json"


def load_manifest_frames(
    manifest_csv_path: Path,
    image_path_col: str,
    label_col: str,
    subject_col: str,
    activity_col: str,
    trial_col: str,
    camera_col: str,
    timestamp_col: str,
    limit: int = 0,
) -> tuple[list[ManifestFrame], Counter]:
    frames: list[ManifestFrame] = []
    stats = Counter()

    with Path(manifest_csv_path).open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"Manifest CSV has no header: {manifest_csv_path}")

        for row_index, row in enumerate(reader, start=2):
            if limit > 0 and len(frames) >= limit:
                break

            image_path = get_row_value(row, image_path_col)
            if not image_path:
                raise ValueError(
                    f"Missing {image_path_col} at manifest row {row_index}."
                )

            label = parse_int_text(
                get_row_value(row, label_col),
                row_index,
                label_col,
            )
            subject, activity, trial, camera = group_values_from_row(
                row=row,
                row_index=row_index,
                image_path=image_path,
                subject_col=subject_col,
                activity_col=activity_col,
                trial_col=trial_col,
                camera_col=camera_col,
            )
            frames.append(
                ManifestFrame(
                    image_path=image_path,
                    label=label,
                    subject=subject,
                    activity=activity,
                    trial=trial,
                    camera=camera,
                    sort_key=frame_sort_key(
                        row=row,
                        image_path=image_path,
                        row_index=row_index,
                        timestamp_col=timestamp_col,
                    ),
                    row_index=row_index,
                )
            )
            stats["manifest_rows"] += 1

    return frames, stats


def group_frames_by_sequence(
    frames: Iterable[ManifestFrame],
) -> dict[tuple[str, str, str, str], list[ManifestFrame]]:
    groups: dict[tuple[str, str, str, str], list[ManifestFrame]] = defaultdict(list)
    for frame in frames:
        group_key = (frame.subject, frame.activity, frame.trial, frame.camera)
        groups[group_key].append(frame)

    for group_key, group_frames in groups.items():
        groups[group_key] = sorted(
            group_frames,
            key=lambda frame: frame.sort_key,
        )

    return dict(groups)


def sequence_record(
    group_key: tuple[str, str, str, str],
    window_frames: list[ManifestFrame],
    fall_tags: set[int],
    fall_min_frames: int,
) -> dict[str, object]:
    subject, activity, trial, camera = group_key
    labels = [frame.label for frame in window_frames]
    fall_count = sum(label in fall_tags for label in labels)
    return {
        "Subject": subject,
        "Activity": activity,
        "Trial": trial,
        "Camera": camera,
        "img_paths": [frame.image_path for frame in window_frames],
        "label_array": labels,
        "fall_alert": int(fall_count >= fall_min_frames),
    }


def write_records(
    records: Iterable[dict[str, object]],
    output_path: Path,
    output_format: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "jsonl":
        count = 0
        with output_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        return count

    record_list = list(records)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(record_list, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return len(record_list)


def build_sequences_from_manifest(
    manifest_csv_path: Path,
    output_path: Path,
    sequence_length: int = 10,
    stride: int = 1,
    fall_tags: set[int] | None = None,
    fall_min_frames: int = 2,
    image_path_col: str = "image_path",
    label_col: str = "Label",
    subject_col: str = "Subject",
    activity_col: str = "Activity",
    trial_col: str = "Trial",
    camera_col: str = "Camera",
    timestamp_col: str = "Timestamp",
    output_format: str = "auto",
    limit: int = 0,
) -> dict[str, int]:
    manifest_csv_path = Path(manifest_csv_path)
    output_path = Path(output_path)
    fall_tags = set(fall_tags or DEFAULT_FALL_TAGS)

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if fall_min_frames <= 0:
        raise ValueError("fall_min_frames must be positive.")
    if not manifest_csv_path.is_file():
        raise ValueError(f"Manifest CSV does not exist: {manifest_csv_path}")

    frames, manifest_stats = load_manifest_frames(
        manifest_csv_path=manifest_csv_path,
        image_path_col=image_path_col,
        label_col=label_col,
        subject_col=subject_col,
        activity_col=activity_col,
        trial_col=trial_col,
        camera_col=camera_col,
        timestamp_col=timestamp_col,
        limit=limit,
    )
    groups = group_frames_by_sequence(frames)
    output_format = resolve_output_format(output_path, output_format)

    stats = Counter(manifest_stats)
    stats["sequence_groups"] = len(groups)

    records: list[dict[str, object]] = []
    for group_key in sorted(groups, key=lambda key: natural_sort_key("/".join(key))):
        group_frames = groups[group_key]
        if len(group_frames) < sequence_length:
            stats["sequence_groups_too_short"] += 1
            continue

        for start in range(0, len(group_frames) - sequence_length + 1, stride):
            window_frames = group_frames[start:start + sequence_length]
            records.append(
                sequence_record(
                    group_key=group_key,
                    window_frames=window_frames,
                    fall_tags=fall_tags,
                    fall_min_frames=fall_min_frames,
                )
            )
            stats["windows_written"] += 1

    written_count = write_records(
        records=records,
        output_path=output_path,
        output_format=output_format,
    )
    stats["records_written"] = written_count
    stats["fall_alert_positive"] = sum(
        int(record["fall_alert"]) for record in records
    )
    stats["fall_alert_negative"] = written_count - stats["fall_alert_positive"]

    print(f"Manifest CSV: {manifest_csv_path}")
    print(f"Output: {output_path}")
    print(f"Output format: {output_format}")
    print(f"Sequence length: {sequence_length}")
    print(f"Stride: {stride}")
    print(f"Fall tags: {sorted(fall_tags)}")
    print(f"Fall min frames: {fall_min_frames}")
    print(f"Stats: {dict(sorted(stats.items()))}")

    return dict(stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed-length image sequences from manifest.csv. "
            "Frames are grouped by Subject/Activity/Trial/Camera; image files "
            "are not checked on disk."
        )
    )
    parser.add_argument("--manifest-csv-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fall-tags", type=parse_int_set, default=DEFAULT_FALL_TAGS)
    parser.add_argument("--fall-min-frames", type=int, default=2)
    parser.add_argument("--image-path-col", default="image_path")
    parser.add_argument("--label-col", default="Label")
    parser.add_argument("--subject-col", default="Subject")
    parser.add_argument("--activity-col", default="Activity")
    parser.add_argument("--trial-col", default="Trial")
    parser.add_argument("--camera-col", default="Camera")
    parser.add_argument("--timestamp-col", default="Timestamp")
    parser.add_argument(
        "--output-format",
        choices=("auto", "json", "jsonl"),
        default="auto",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_sequences_from_manifest(
        manifest_csv_path=args.manifest_csv_path,
        output_path=args.output_path,
        sequence_length=args.sequence_length,
        stride=args.stride,
        fall_tags=args.fall_tags,
        fall_min_frames=args.fall_min_frames,
        image_path_col=args.image_path_col,
        label_col=args.label_col,
        subject_col=args.subject_col,
        activity_col=args.activity_col,
        trial_col=args.trial_col,
        camera_col=args.camera_col,
        timestamp_col=args.timestamp_col,
        output_format=args.output_format,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
