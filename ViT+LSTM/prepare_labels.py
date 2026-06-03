import argparse
import csv
import re
from collections import Counter
from pathlib import Path


FILENAME_PATTERN = re.compile(
    r"^Subject(?P<subject>\d+)Activity(?P<activity>\d+)Trial(?P<trial>\d+)\.csv$"
)


def parse_filename(path):
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Unexpected HAR-UP filename: {path.name}")

    return {
        "subject": int(match.group("subject")),
        "activity": int(match.group("activity")),
        "trial": int(match.group("trial")),
    }


def parse_int(value):
    return int(float(value))


def csv_sort_key(csv_file):
    meta = parse_filename(csv_file)
    return meta["subject"], meta["activity"], meta["trial"]


def iter_timestamp_labels(csv_file, label_source="tag"):
    file_meta = parse_filename(csv_file)

    with csv_file.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # Header row: sensor/group names
        next(reader, None)  # Header row: axis/unit names

        for row_index, row in enumerate(reader):
            if not row or not row[0].strip():
                continue

            # Data rows have 47 columns:
            # TimeStamps, sensor values..., Subject, Activity, Trial, Tag
            # The first header row misses the trailing Tag column.
            if len(row) < 47:
                raise ValueError(
                    f"Malformed row in {csv_file.name} at data row {row_index}: "
                    f"expected at least 47 columns, got {len(row)}."
                )

            subject = parse_int(row[-4])
            activity = parse_int(row[-3])
            trial = parse_int(row[-2])
            tag = parse_int(row[-1])

            if (
                subject != file_meta["subject"]
                or activity != file_meta["activity"]
                or trial != file_meta["trial"]
            ):
                raise ValueError(
                    f"Metadata mismatch in {csv_file.name} at data row {row_index}: "
                    f"row has Subject{subject} Activity{activity} Trial{trial}, "
                    f"filename has Subject{file_meta['subject']} "
                    f"Activity{file_meta['activity']} Trial{file_meta['trial']}."
                )

            if label_source == "tag":
                class_id_1_based = tag
            elif label_source == "activity":
                class_id_1_based = activity
            else:
                raise ValueError("label_source must be either 'tag' or 'activity'.")

            yield {
                "timestamp": row[0],
                "subject": subject,
                "activity": activity,
                "trial": trial,
                "tag": tag,
                "class_id_1_based": class_id_1_based,
                "label": class_id_1_based - 1,
                "source_file": csv_file.name,
                "row_index": row_index,
            }


def prepare_labels(dataset_csv_path, output_csv_path, label_source="activity", num_classes=11):
    dataset_csv_path = Path(dataset_csv_path)
    output_csv_path = Path(output_csv_path)

    if not dataset_csv_path.is_dir():
        raise ValueError(f"Provided dataset_csv_path {dataset_csv_path} is not a directory.")

    rows_written = 0
    skipped_empty = []
    skipped_out_of_range = Counter()

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=[
                "Timestamp",
                "Subject",
                "Activity",
                "Trial",
                "Tag",
                "Label",
                "Source_File",
                "Row_Index",
            ],
        )
        writer.writeheader()

        for csv_file in sorted(dataset_csv_path.glob("*.csv"), key=csv_sort_key):
            file_rows = 0
            for row in iter_timestamp_labels(csv_file, label_source=label_source):
                if row["label"] < 0 or row["label"] >= num_classes:
                    skipped_out_of_range[row["class_id_1_based"]] += 1
                    continue

                writer.writerow(
                    {
                        "Timestamp": row["timestamp"],
                        "Subject": row["subject"],
                        "Activity": row["activity"],
                        "Trial": row["trial"],
                        "Tag": row["tag"],
                        "Label": row["label"],
                        "Source_File": row["source_file"],
                        "Row_Index": row["row_index"],
                    }
                )
                file_rows += 1
                rows_written += 1

            if file_rows == 0:
                skipped_empty.append(csv_file.name)

    print(f"Wrote {rows_written} timestamp label rows to {output_csv_path}")
    if skipped_empty:
        print(f"Skipped {len(skipped_empty)} empty CSV files: {', '.join(skipped_empty)}")
    if skipped_out_of_range:
        skipped_summary = ", ".join(
            f"{label_source}={key}: {count}"
            for key, count in sorted(skipped_out_of_range.items())
        )
        print(f"Skipped out-of-range labels for {num_classes} classes: {skipped_summary}")

    return rows_written


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare timestamp-wise HAR-UP labels from DataSetCSV files."
    )
    parser.add_argument(
        "--dataset-csv-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "crawler" / "DataSetCSV",
        help="Directory containing Subject*Activity*Trial*.csv files.",
    )
    parser.add_argument(
        "--output-csv-path",
        type=Path,
        default=Path(__file__).with_name("labels.csv"),
        help="Output timestamp-wise labels CSV path.",
    )
    parser.add_argument(
        "--label-source",
        choices=("tag", "activity"),
        default="activity",
        help=(
            "Use Activity or Tag as the timestamp-wise label source. "
            "Default: activity."
        ),
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=11,
        help="Number of classifier output classes. Default: 11.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_labels(
        dataset_csv_path=args.dataset_csv_path,
        output_csv_path=args.output_csv_path,
        label_source=args.label_source,
        num_classes=args.num_classes,
    )
