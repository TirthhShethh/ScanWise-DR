"""
dataset.py – IDRiD dataset loader for ScanWise.

Primary task: Diabetic Retinopathy (DR) severity grading (5 classes 0-4).
Secondary task: Macular Edema (ME) risk grading (3 classes 0-2).

The dataset uses the official IDRiD train/test split.
All paths are resolved relative to DATA_ROOT so the project is portable.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Dataset root resolution
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT                               # dataset lives here

DISEASE_GRADING_ROOT = DATA_ROOT / "B.%20Disease%20Grading" / "B. Disease Grading"
IMG_TRAIN = DISEASE_GRADING_ROOT / "1. Original Images" / "a. Training Set"
IMG_TEST  = DISEASE_GRADING_ROOT / "1. Original Images" / "b. Testing Set"
GT_DIR    = DISEASE_GRADING_ROOT / "2. Groundtruths"
TRAIN_CSV = GT_DIR / "a. IDRiD_Disease Grading_Training Labels.csv"
TEST_CSV  = GT_DIR / "b. IDRiD_Disease Grading_Testing Labels.csv"

DR_GRADES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

ME_GRADES = {
    0: "No ME Risk",
    1: "Moderate ME Risk",
    2: "High ME Risk",
}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _load_csv(csv_path: Path) -> pd.DataFrame:
    """Load IDRiD label CSV, strip trailing empty columns."""
    df = pd.read_csv(csv_path, skipinitialspace=True)
    # Drop fully-empty columns (artefact from training CSV)
    df = df.dropna(axis=1, how="all")
    df.columns = [c.strip() for c in df.columns]
    df["Image name"] = df["Image name"].str.strip()
    return df


def load_labels(split: str = "train") -> pd.DataFrame:
    """Return a DataFrame with columns: image_id, retinopathy_grade, me_grade, image_path."""
    csv_path = TRAIN_CSV if split == "train" else TEST_CSV
    img_dir  = IMG_TRAIN  if split == "train" else IMG_TEST

    df = _load_csv(csv_path)
    df = df.rename(columns={
        "Image name":          "image_id",
        "Retinopathy grade":   "retinopathy_grade",
        "Risk of macular edema": "me_grade",
    })
    df["retinopathy_grade"] = pd.to_numeric(df["retinopathy_grade"], errors="coerce").astype("Int64")
    df["me_grade"]          = pd.to_numeric(df["me_grade"],          errors="coerce").astype("Int64")

    def _resolve(img_id: str) -> Optional[str]:
        p = img_dir / f"{img_id}.jpg"
        return str(p) if p.exists() else None

    df["image_path"] = df["image_id"].apply(_resolve)
    missing = df["image_path"].isna().sum()
    if missing:
        import warnings
        warnings.warn(f"[dataset] {missing} images missing from {img_dir}")
    df = df.dropna(subset=["image_path"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class IDRiDDataset(Dataset):
    """
    PyTorch Dataset for IDRiD Disease Grading images.

    Args:
        split:      "train" | "test"
        task:       "dr" for retinopathy grade | "me" for macular edema risk
        transform:  torchvision transforms applied to the PIL image
    """

    def __init__(self, split: str = "train", task: str = "dr", transform=None):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        if task not in ("dr", "me"):
            raise ValueError(f"task must be 'dr' or 'me', got {task!r}")

        self.split     = split
        self.task      = task
        self.transform = transform

        self.df = load_labels(split)
        self._label_col = "retinopathy_grade" if task == "dr" else "me_grade"

        # Drop rows where the target label is NaN
        self.df = self.df.dropna(subset=[self._label_col]).reset_index(drop=True)

        self.class_names: Dict[int, str] = DR_GRADES if task == "dr" else ME_GRADES
        self.num_classes: int = len(self.class_names)
        self.labels: List[int] = self.df[self._label_col].astype(int).tolist()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row   = self.df.iloc[idx]
        label = int(row[self._label_col])

        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    # ------------------------------------------------------------------
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for weighted loss."""
        counts = np.bincount(self.labels, minlength=self.num_classes).astype(float)
        counts = np.where(counts == 0, 1.0, counts)           # avoid /0
        weights = 1.0 / counts
        weights = weights / weights.sum() * self.num_classes  # scale
        return torch.tensor(weights, dtype=torch.float32)

    def weighted_sampler(self) -> WeightedRandomSampler:
        """Return a WeightedRandomSampler for class-balanced mini-batches."""
        cw = self.class_weights().numpy()
        sample_weights = [cw[lbl] for lbl in self.labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            f"IDRiD {self.split.upper()} split | task={self.task.upper()} | n={len(self)}",
        ]
        for grade, name in self.class_names.items():
            n = self.labels.count(grade)
            lines.append(f"  Grade {grade} ({name}): {n}")
        return "\n".join(lines)
