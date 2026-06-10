"""PyTorch Dataset for coronary task split CSVs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from src.data.label_maps import get_class_id, get_label_map
from src.utils.paths import build_image_path


REQUIRED_COLUMNS = ("filename", "label", "study_id", "artery", "split", "task", "fold")


class CoronaryDataset(Dataset):
    """Dataset backed by a single coronary split CSV."""

    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        transform: Any | None = None,
        target_transform: Any | None = None,
        image_mode: str = "L",
        return_metadata: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.transform = transform
        self.target_transform = target_transform
        self.image_mode = image_mode
        self.return_metadata = return_metadata
        self.rows = self._load_rows()

    def _load_rows(self) -> list[dict[str, Any]]:
        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            missing_columns = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
            if missing_columns:
                missing = ", ".join(missing_columns)
                raise ValueError(f"{self.csv_path} is missing required columns: {missing}")

            rows: list[dict[str, Any]] = []
            for row_number, row in enumerate(reader, start=2):
                task = row["task"]
                label = row["label"]
                get_label_map(task)
                class_id = get_class_id(task, label)
                image_path = build_image_path(
                    self.data_root,
                    task=task,
                    artery=row["artery"],
                    label=label,
                    filename=row["filename"],
                )
                rows.append(
                    {
                        "filename": row["filename"],
                        "label": label,
                        "class_id": class_id,
                        "study_id": row["study_id"],
                        "artery": row["artery"],
                        "split": row["split"],
                        "task": task,
                        "fold": row["fold"],
                        "image_path": image_path,
                        "row_number": row_number,
                    }
                )
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, int] | dict[str, Any]:
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            image = image.convert(self.image_mode)

        label = row["class_id"]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.return_metadata:
            metadata = {
                key: row[key]
                for key in ("filename", "study_id", "artery", "split", "task", "fold", "image_path")
            }
            return {"image": image, "label": label, "metadata": metadata}

        return image, label
