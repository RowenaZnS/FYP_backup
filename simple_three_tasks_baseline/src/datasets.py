import csv
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


@dataclass
class CsvSchema:
    """
    CSV 每行至少包含:
      - path: 相对 train_root 的图片路径，或绝对路径
      - id: identity 类别（int）
      - age: 年龄（int 或 float，默认当作 0..100 的连续值）
    可选:
      - mix_id: 若你想让 mixture 用另一套 identity label（默认不用）
    """

    path_col: str = "path"
    id_col: str = "id"
    age_col: str = "age"


def build_transform(image_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


class FaceIdAgeCsvDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        train_root: Optional[str] = None,
        image_size: int = 224,
        train: bool = True,
        schema: CsvSchema = CsvSchema(),
    ):
        self.csv_path = csv_path
        self.train_root = train_root
        self.schema = schema
        self.transform = build_transform(image_size=image_size, train=train)

        self.rows = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(r)

        if len(self.rows) == 0:
            raise ValueError(f"CSV is empty: {csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        if self.train_root is None:
            return p
        return os.path.join(self.train_root, p)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = self.rows[idx]
        img_path = self._resolve_path(r[self.schema.path_col])
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        x = self.transform(im)

        y_id = torch.tensor(int(r[self.schema.id_col]), dtype=torch.long)
        y_age = torch.tensor(float(r[self.schema.age_col]), dtype=torch.float32)
        return x, y_id, y_age

