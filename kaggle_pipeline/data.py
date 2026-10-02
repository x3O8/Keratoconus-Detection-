"""Dataset and leakage-safe split helpers for four-map ORBSCAN samples."""

from __future__ import annotations

import math
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

MAP_NAMES = ("anterior", "axial", "pachymetry", "posterior")
MAP_COLUMNS = tuple(f"{name}_path" for name in MAP_NAMES)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find_data_root(requested: str | Path | None) -> Path:
    if requested:
        root = Path(requested)
        if not (root / "metadata" / "eye_manifest.csv").is_file():
            raise FileNotFoundError(f"No metadata/eye_manifest.csv under {root}")
        return root
    candidates = list(Path("/kaggle/input").glob("**/metadata/eye_manifest.csv"))
    if not candidates:
        local = Path("keratoconus_better/labeled_dataset/metadata/eye_manifest.csv")
        candidates = [local] if local.is_file() else []
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Could not uniquely discover the dataset. Pass --data-root. "
            f"Found: {[str(path) for path in candidates]}"
        )
    return candidates[0].parent.parent


class SynchronizedTransform:
    """Apply the same mild geometry to all four maps in an eye sample."""

    def __init__(self, image_size: int, training: bool) -> None:
        self.image_size = image_size
        self.training = training

    def __call__(self, images: list[Image.Image]) -> torch.Tensor:
        angle = random.uniform(-6.0, 6.0) if self.training else 0.0
        translate = (
            (
                random.randint(-round(self.image_size * 0.025), round(self.image_size * 0.025)),
                random.randint(-round(self.image_size * 0.025), round(self.image_size * 0.025)),
            )
            if self.training
            else (0, 0)
        )
        scale = random.uniform(0.97, 1.03) if self.training else 1.0
        tensors = []
        for image in images:
            image = TF.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            if self.training:
                image = TF.affine(
                    image,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )
            tensor = TF.to_tensor(image)
            tensors.append(TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD))
        return torch.stack(tensors)


class OrbscanDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        root: Path,
        image_size: int = 224,
        training: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.root = root
        self.transform = SynchronizedTransform(image_size, training)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        images = []
        for column in MAP_COLUMNS:
            with Image.open(self.root / row[column]) as image:
                images.append(image.convert("RGB"))
        return {
            "images": self.transform(images),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
            "eye": row["eye"],
        }


def load_manifest(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        root / "metadata" / "eye_manifest.csv",
        dtype={"patient_id": str, "sample_id": str},
    )
    missing = {"patient_id", "sample_id", "eye", "label", *MAP_COLUMNS} - set(frame)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return frame


def make_splits(frame: pd.DataFrame, fold: int, seed: int) -> tuple:
    """Use provided patient-consistent split/fold metadata when available."""
    if {"split", "fold"}.issubset(frame.columns):
        development = frame[frame["split"].str.lower() == "train"].copy()
        test = frame[frame["split"].str.lower() == "test"].copy()
        numeric_fold = pd.to_numeric(development["fold"], errors="coerce")
        if numeric_fold.notna().any():
            val = development[numeric_fold == fold].copy()
            train = development[numeric_fold != fold].copy()
            if not train.empty and not val.empty and not test.empty:
                _assert_patient_disjoint(train, val, test)
                return train, val, test

    from sklearn.model_selection import StratifiedGroupKFold

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(outer.split(frame, frame["label"], groups=frame["patient_id"]))
    train_val_idx, test_idx = splits[(fold - 1) % 5]
    train_val, test = frame.iloc[train_val_idx], frame.iloc[test_idx]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    train_idx, val_idx = next(
        inner.split(
            train_val,
            train_val["label"],
            groups=train_val["patient_id"],
        )
    )
    train, val = train_val.iloc[train_idx], train_val.iloc[val_idx]
    _assert_patient_disjoint(train, val, test)
    return train.copy(), val.copy(), test.copy()


def _assert_patient_disjoint(*frames: pd.DataFrame) -> None:
    patient_sets = [set(frame["patient_id"]) for frame in frames]
    for left in range(len(patient_sets)):
        for right in range(left + 1, len(patient_sets)):
            overlap = patient_sets[left] & patient_sets[right]
            if overlap:
                raise RuntimeError(
                    f"Patient leakage between splits: {sorted(overlap)[:10]}"
                )

