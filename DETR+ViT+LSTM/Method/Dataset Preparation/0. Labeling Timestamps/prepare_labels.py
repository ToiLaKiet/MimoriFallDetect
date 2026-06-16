import argparse
import csv
from collections import Counter
from pathlib import Path


DEFAULT_INPUT_CSV = Path(__file__).with_name("CompleteDataSet.csv")
MIN_DATA_COLUMNS = 47
OUTPUT_FIELDNAMES = [
    "Row_Index",
    "Timestamp",
    "Subject",
    "Activity",
    "Trial",
    "Tag",
    "Label"
]


def parse_int(value, column_name, source_line):
    text = str(value).strip()
    if not text:
        raise ValueError(f"Missing {column_name} at source line {source_line}.")
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(
            f"Invalid {column_name} value {value!r} at source line {source_line}."
        ) from exc


def resolve_input_csv(input_csv_path):
    input_csv_path = Path(input_csv_path)
    if input_csv_path.is_file():
        return input_csv_path

    if input_csv_path.is_dir():
        complete_dataset = input_csv_path / "CompleteDataSet.csv"
        if complete_dataset.is_file():
            return complete_dataset
        raise ValueError(
            f"Provided input path {input_csv_path} is a directory, but "
            "CompleteDataSet.csv was not found inside it."
        )

    raise ValueError(f"Input CSV path does not exist: {input_csv_path}")



def iter_timestamp_labels(input_csv_path):
    """Yield timestamp labels from one integrated CompleteDataSet.csv file."""

    input_csv_path = resolve_input_csv(input_csv_path)
    with input_csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # Header row: sensor/group names.
        next(reader, None)  # Header row: axis/unit names.

        for row_index, row in enumerate(reader):
            source_line = row_index +  3  # Account for the two header lines and 0-based index.
            if not row or not row[0].strip():
                continue

            if len(row) < MIN_DATA_COLUMNS:
                raise ValueError(
                    f"Malformed row in {input_csv_path.name} at source line "
                    f"{source_line}: expected at least {MIN_DATA_COLUMNS} columns, "
                    f"got {len(row)}."
                )

            timestamp = row[0].strip()
            subject = parse_int(row[-4], "Subject", source_line)
            activity = parse_int(row[-3], "Activity", source_line)
            trial = parse_int(row[-2], "Trial", source_line)
            tag = parse_int(row[-1], "Tag", source_line)

            # yield là từ khóa tạo generator, cho phép hàm trả về một giá trị và tạm dừng trạng thái của nó, sau đó tiếp tục từ điểm đó khi được gọi lại. Điều này rất hữu ích để xử lý dữ liệu lớn hoặc luồng dữ liệu mà không cần tải tất cả vào bộ nhớ cùng một lúc.
            yield {
                "Timestamp": timestamp,
                "Subject": subject,
                "Activity": activity,
                "Trial": trial,
                "Tag": tag,
                "Label": tag,
                "Row_Index": row_index,
            }


def prepare_labels(input_csv_path, output_csv_path, num_classes=11):
    input_csv_path = resolve_input_csv(input_csv_path)
    output_csv_path = Path(output_csv_path)

    rows_written = 0
    skipped_out_of_range = Counter()
    label_counts = Counter()
    key_counts = Counter()

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()

        for row in iter_timestamp_labels(input_csv_path):
            # Kiểm tra nếu nhãn (Label) nằm ngoài phạm vi hợp lệ (1 đến num_classes). Nếu có, đếm số lần bị bỏ qua cho mỗi Tag và tiếp tục vòng lặp mà không ghi dòng đó vào CSV đầu ra.
            if row["Label"] < 1 or row["Label"] > num_classes:
                skipped_out_of_range[row["Tag"]] += 1
                continue

            key = (
                row["Timestamp"],
                row["Subject"],
                row["Activity"],
                row["Trial"],
            )
            key_counts[key] += 1 # Đếm số lần xuất hiện của mỗi key (Timestamp, Subject, Activity, Trial) để phát hiện trùng lặp.
            label_counts[row["Label"]] += 1 # Đếm số lượng mỗi nhãn (Label) để có thống kê về phân phối nhãn trong dữ liệu đầu ra.
            writer.writerow(
                {
                    "Row_Index": row["Row_Index"],
                    "Timestamp": row["Timestamp"],
                    "Subject": row["Subject"],
                    "Activity": row["Activity"],
                    "Trial": row["Trial"],
                    "Tag": row["Tag"],
                    "Label": row["Label"]
                }
            )
            rows_written += 1

    duplicate_rows = sum(count - 1 for count in key_counts.values() if count > 1) 

    print(f"Input CSV: {input_csv_path}")
    print(f"Wrote {rows_written} timestamp label rows to {output_csv_path}")
    print(f"Label counts: {dict(sorted(label_counts.items()))}")
    if duplicate_rows:
        print(
            "Duplicate label keys "
            f"(Timestamp, Subject, Activity, Trial): {duplicate_rows}"
        )
    if skipped_out_of_range:
        skipped_summary = ", ".join(
            f"Tag={key}: {count}"
            for key, count in sorted(skipped_out_of_range.items())
        )
        print(f"Skipped out-of-range labels for {num_classes} classes: {skipped_summary}")

    return rows_written


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare timestamp-wise labels from integrated CompleteDataSet.csv."
    )
    parser.add_argument(
        "--input-csv-path",
        "--dataset-csv-path",
        dest="input_csv_path",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=(
            "Integrated sensor CSV file. A directory is also accepted if it contains "
            "CompleteDataSet.csv."
        ),
    )
    parser.add_argument(
        "--output-csv-path",
        type=Path,
        default=Path(__file__).with_name("Labels.csv"),
        help="Output timestamp-wise labels CSV path.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=12,
        help="Number of valid Tag labels. Default: 11.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_labels(
        input_csv_path=args.input_csv_path,
        output_csv_path=args.output_csv_path,
        num_classes=args.num_classes,
    )
