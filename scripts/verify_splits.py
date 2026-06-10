"""Validate split CSVs against the expected coronary image layout."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.label_maps import LABEL_MAPS
from src.utils.paths import build_image_path


REQUIRED_COLUMNS = ("filename", "label", "study_id", "artery", "split", "task", "fold")
VALID_ARTERIES = {"LCA", "RCA"}
VALID_SPLITS = {"train", "val", "test"}
MAX_EXAMPLES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate coronary split CSV files.")
    parser.add_argument("--data_root", required=True, type=Path, help="Root directory containing image data.")
    parser.add_argument("--split_root", required=True, type=Path, help="Root directory containing split CSVs.")
    return parser.parse_args()


def validate_row(row: dict[str, str], csv_path: Path, row_number: int, data_root: Path) -> list[str]:
    errors: list[str] = []
    task = row.get("task", "")
    artery = row.get("artery", "")
    split = row.get("split", "")
    label = row.get("label", "")

    if task not in LABEL_MAPS:
        errors.append(f"{csv_path}:{row_number} invalid task '{task}'")
    if artery not in VALID_ARTERIES:
        errors.append(f"{csv_path}:{row_number} invalid artery '{artery}'")
    if split not in VALID_SPLITS:
        errors.append(f"{csv_path}:{row_number} invalid split '{split}'")
    if task in LABEL_MAPS and label not in LABEL_MAPS[task]:
        valid_labels = ", ".join(sorted(LABEL_MAPS[task]))
        errors.append(f"{csv_path}:{row_number} invalid label '{label}' for task '{task}' ({valid_labels})")

    if not errors:
        image_path = build_image_path(
            data_root,
            task=task,
            artery=artery,
            label=label,
            filename=row.get("filename", ""),
        )
        if not image_path.is_file():
            errors.append(f"{csv_path}:{row_number} missing image file '{image_path}'")

    return errors


def validate_csv(csv_path: Path, data_root: Path) -> tuple[int, Counter[str], list[str]]:
    counts: Counter[str] = Counter()
    examples: list[str] = []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing_columns:
            counts["csv_with_missing_columns"] += 1
            examples.append(f"{csv_path} missing required columns: {', '.join(missing_columns)}")
            return 0, counts, examples

        row_count = 0
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            row_errors = validate_row(row, csv_path, row_number, data_root)
            if row_errors:
                if any("missing image file" in error for error in row_errors):
                    counts["missing_files"] += 1
                if any("missing image file" not in error for error in row_errors):
                    counts["invalid_rows"] += 1
                examples.extend(row_errors[: max(0, MAX_EXAMPLES - len(examples))])

    return row_count, counts, examples


def main() -> int:
    args = parse_args()
    split_root = args.split_root
    data_root = args.data_root

    csv_paths = sorted(split_root.rglob("*.csv"))
    total_rows = 0
    total_counts: Counter[str] = Counter()
    examples: list[str] = []

    for csv_path in csv_paths:
        row_count, counts, csv_examples = validate_csv(csv_path, data_root)
        total_rows += row_count
        total_counts.update(counts)
        examples.extend(csv_examples[: max(0, MAX_EXAMPLES - len(examples))])

    checked_csvs = len(csv_paths)
    invalid_rows = total_counts["invalid_rows"]
    missing_files = total_counts["missing_files"]
    csvs_with_missing_columns = total_counts["csv_with_missing_columns"]
    failed = checked_csvs == 0 or invalid_rows > 0 or missing_files > 0 or csvs_with_missing_columns > 0

    print("Split verification summary")
    print(f"  split_root: {split_root}")
    print(f"  data_root: {data_root}")
    print(f"  checked CSV files: {checked_csvs}")
    print(f"  total rows: {total_rows}")
    print(f"  CSVs with missing columns: {csvs_with_missing_columns}")
    print(f"  invalid rows: {invalid_rows}")
    print(f"  missing files: {missing_files}")

    if checked_csvs == 0:
        print("  error: no CSV files found under split_root")

    if examples:
        print("\nExamples:")
        for example in examples[:MAX_EXAMPLES]:
            print(f"  - {example}")

    if failed:
        print("\nValidation failed.")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
