"""One-cell Kaggle fine-tuning pipeline for the independent seven-map dataset.

Upload this script and the dataset containing Normal/Keratoconus/Suspect case
folders to Kaggle alongside the supplied checkpoint package. Enable a GPU and
run this file once; no arguments or additional commands are required.

The pipeline trains only on ``Train_Validation sets``. Five-fold out-of-fold
predictions determine probability calibration and the decision threshold, then
the supplied ``Independent Test Set`` is evaluated exactly once. It trains one
practical deployment model on all 300 development cases and also evaluates a
five-model ensemble. Suspect cases are never used as labelled training data.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import random
import shutil
import subprocess
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path


# ---------------------------------------------------------------------------
# Runs as-is on Kaggle. These are the only settings normally worth changing.
# ---------------------------------------------------------------------------
SEED = 42
OUTPUT_ROOT = Path("/kaggle/working/keratoconus_external_finetune")
# The first path is the exact Kaggle location supplied by the user. Kaggle may
# alternatively mount the same dataset at the shorter slug-only path, so both
# are checked before generic discovery.
EXTERNAL_DATASET_ROOTS = (
    Path("/kaggle/input/datasets/eatradium/testset-kera"),
    Path("/kaggle/input/testset-kera"),
)
# The package is named mobilenet-v2-kera on Kaggle, but its best_model.pth
# contains the prior EfficientNetV2-S four-map attention-fusion model.
MODEL_ROOTS = (
    Path("/kaggle/input/models/eatradium/mobilenet-v2-kera/pytorch/default/1"),
    Path("/kaggle/input/mobilenet-v2-kera"),
)
SOURCE_CHECKPOINT_PATH = Path(
    "/kaggle/input/models/eatradium/mobilenet-v2-kera/pytorch/default/1/best_model.pth"
)
IMAGE_SIZE = 224
N_FOLDS = 5
MAX_EPOCHS = 18
PATIENCE = 4
FREEZE_EPOCHS = 2
BATCH_SIZE = 4
GRAD_ACCUMULATION = 2
WORKERS = 2
BACKBONE_LR = 2e-5
HEAD_LR = 2e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
ATTENTION_ENTROPY_WEIGHT = 0.01
MODALITY_DROPOUT = 0.20
NORMAL_CLASS_WEIGHT = 1.20
FOCAL_GAMMA = 1.5
AUXILIARY_MAP_LOSS_WEIGHT = 0.15
HARD_NEGATIVE_FRACTION = 0.20
HARD_NEGATIVE_WEIGHT = 2.50
HARD_NEGATIVE_REFRESH_EPOCHS = (2, 6)
USE_HORIZONTAL_FLIP_TTA = True
# Suppresses text/legends near the scan edge while preserving the central
# corneal map. Training randomizes the rim slightly so the model cannot rely
# on a fixed border pattern.
TRAIN_ROI_RADIUS_RANGE = (0.470, 0.500)
EVAL_ROI_RADIUS = 0.485
# The classification threshold is selected only from OOF development
# predictions. It is the most specific cutoff that retains this sensitivity.
SCREENING_MIN_SENSITIVITY = 0.95
# Triage bands are also OOF-derived: below low is low-risk, above high is
# high-risk, and the interval is clinician-review territory.
LOW_RISK_MIN_SENSITIVITY = 0.99
HIGH_RISK_MIN_SPECIFICITY = 0.90
MAP_NAMES = ("CT_A", "EC_A", "EC_P", "Elv_A", "Elv_P", "Sag_A", "Sag_P")
CLASS_FOLDERS = {"Normal": 0, "Keratoconus": 1}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def ensure_dependencies():
    packages = {
        "timm": "timm>=1.0.15,<2",
        "pytorch_grad_cam": "grad-cam>=1.5.5",
    }
    missing = [pip_name for module, pip_name in packages.items() if importlib.util.find_spec(module) is None]
    if missing:
        print("Installing:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_dependencies()

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def discover_dataset_roots():
    explicit_roots = []
    for root in EXTERNAL_DATASET_ROOTS:
        training = root / "Train_Validation sets" / "Train_Validation sets"
        independent = root / "Independent Test Set" / "Independent Test Set"
        if (
            (training / "Normal").is_dir()
            and (training / "Keratoconus").is_dir()
            and (training / "Suspect").is_dir()
            and (independent / "Normal").is_dir()
            and (independent / "Keratoconus").is_dir()
            and (independent / "Suspect").is_dir()
        ):
            explicit_roots.append((training, independent))
    if explicit_roots:
        training, independent = explicit_roots[0]
        print("Using configured testset-kera dataset root:", training.parent.parent)
        return training, independent

    candidates = []
    for normal in Path("/kaggle/input").glob("**/Normal"):
        parent = normal.parent
        if (parent / "Keratoconus").is_dir() and (parent / "Suspect").is_dir():
            candidates.append(parent)
    candidates = sorted(set(candidates), key=lambda path: (len(path.parts), str(path)))
    training = [
        path for path in candidates
        if "train validation" in str(path).lower().replace("_", " ").replace("-", " ")
    ]
    independent = [
        path for path in candidates
        if "independent test" in str(path).lower().replace("_", " ").replace("-", " ")
    ]
    if not training or not independent:
        raise FileNotFoundError(
            "Expected both Train_Validation sets and Independent Test Set roots, each containing "
            "Normal, Keratoconus, and Suspect folders, under /kaggle/input. Upload the complete archive. "
            f"Detected candidate roots: {candidates}"
        )
    if len(training) > 1 or len(independent) > 1:
        warnings.warn(f"Multiple roots detected; using training={training[0]}, independent={independent[0]}")
    return training[0], independent[0]


def _extract_state_dict(raw):
    """Support common PyTorch checkpoint layouts without trusting file names."""
    if isinstance(raw, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            value = raw.get(key)
            if isinstance(value, dict) and value:
                return value
        if raw and all(isinstance(value, torch.Tensor) for value in raw.values()):
            return raw
    if hasattr(raw, "state_dict"):
        return raw.state_dict()
    raise RuntimeError("The model file does not contain a recognised PyTorch state_dict")


def discover_source_checkpoint():
    if SOURCE_CHECKPOINT_PATH.is_file():
        raw = torch.load(SOURCE_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        _extract_state_dict(raw)  # Fail clearly here if this is not a PyTorch checkpoint.
        print("Using configured EfficientNetV2-S checkpoint:", SOURCE_CHECKPOINT_PATH)
        return SOURCE_CHECKPOINT_PATH, raw

    candidates = []
    discovered_files = []
    for root in MODEL_ROOTS:
        if root.is_dir():
            files = [path for path in root.glob("**/*") if path.is_file()]
            discovered_files.extend(files)
            candidates.extend(
                path for path in files
                if path.suffix.lower() in {".pth", ".pt", ".bin", ".ckpt", ".pkl", ".tar"}
            )
            # Some Kaggle model packages use an extensionless `model` file.
            candidates.extend(path for path in files if path.name.lower() in {"model", "checkpoint", "weights"})
    # Fallback allows a Kaggle mount which omits the owner/model hierarchy.
    if not candidates:
        for pattern in ("**/*mobilenet*.pth", "**/*mobilenet*.pt", "**/*model*.pth"):
            candidates.extend(Path("/kaggle/input").glob(pattern))
    seen = set()
    inspected = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        try:
            raw = torch.load(path, map_location="cpu", weights_only=False)
            state = _extract_state_dict(raw)
            if any(key.startswith("backbone.blocks.") for key in state):
                return path, raw
            inspected.append(f"{path}: state_dict found but no torchvision features.* keys; examples={sorted(cleaned_keys)[:5]}")
        except Exception as error:
            inspected.append(f"{path}: unreadable by torch.load ({error})")
    existing_roots = [str(root) for root in MODEL_ROOTS if root.is_dir()]
    file_names = [str(path) for path in discovered_files[:25]]
    raise FileNotFoundError(
        "Could not find an EfficientNetV2-S attention-fusion state_dict under the configured model path: "
        f"{MODEL_ROOTS}\nExisting roots: {existing_roots}\n"
        f"Files found (first 25): {file_names}\nInspection details: {inspected[:10]}"
    )


def image_for_suffix(case_dir, suffix):
    matches = [
        path
        for path in case_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and path.stem.upper().endswith("_" + suffix.upper())
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one _{suffix} image in {case_dir}, found {matches}")
    return matches[0]


def build_manifest(root):
    rows, skipped = [], []
    for class_name, label in CLASS_FOLDERS.items():
        for case_dir in sorted(path for path in (root / class_name).iterdir() if path.is_dir()):
            try:
                paths = {name: image_for_suffix(case_dir, name) for name in MAP_NAMES}
            except FileNotFoundError as error:
                skipped.append(str(error))
                continue
            rows.append(
                {
                    "sample_id": f"{class_name}_{case_dir.name}",
                    "case_id": case_dir.name,
                    "source_class": class_name,
                    "label": label,
                    **{f"{name}_path": str(path) for name, path in paths.items()},
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty or frame.label.nunique() != 2:
        raise RuntimeError("The binary Normal/Keratoconus manifest is empty or has only one class")
    if frame.sample_id.duplicated().any():
        raise RuntimeError("Duplicate sample IDs found")
    return frame, skipped


def build_suspect_manifest(root):
    rows = []
    suspect_root = root / "Suspect"
    for case_dir in sorted(path for path in suspect_root.iterdir() if path.is_dir()):
        try:
            paths = {name: image_for_suffix(case_dir, name) for name in MAP_NAMES}
        except FileNotFoundError:
            continue
        rows.append(
            {
                "sample_id": f"Suspect_{case_dir.name}",
                "case_id": case_dir.name,
                "source_class": "Suspect",
                "label": -1,
                **{f"{name}_path": str(path) for name, path in paths.items()},
            }
        )
    return pd.DataFrame(rows)


def exact_hashes_by_sample(frame):
    """Return exact file-content hashes for leakage auditing."""
    output = {}
    for row in frame.itertuples(index=False):
        output[row.sample_id] = {
            hashlib.sha256(Path(getattr(row, f"{name}_path")).read_bytes()).hexdigest()
            for name in MAP_NAMES
        }
    return output


def perceptual_hash(path):
    """Return a 256-bit difference hash for strict near-duplicate detection."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((17, 16)), dtype=np.int16)
    bits = np.packbits((pixels[:, :-1] > pixels[:, 1:]).reshape(-1))
    return int.from_bytes(bits.tobytes(), "big")


def cross_partition_near_duplicates(development, independent, max_distance=8):
    pairs = []
    for name in MAP_NAMES:
        dev_hashes = [
            (row.sample_id, int(row.label), perceptual_hash(getattr(row, f"{name}_path")))
            for row in development.itertuples(index=False)
        ]
        for row in independent.itertuples(index=False):
            test_hash = perceptual_hash(getattr(row, f"{name}_path"))
            for sample_id, dev_label, dev_hash in dev_hashes:
                distance = (test_hash ^ dev_hash).bit_count()
                if distance <= max_distance:
                    pairs.append(
                        {
                            "development_label": dev_label,
                            "independent_label": int(row.label),
                            "map": name,
                            "development_sample": sample_id,
                            "independent_sample": row.sample_id,
                            "hamming_distance_256": distance,
                        }
                    )
    return pairs


class SevenMapTransform:
    """Synchronized mild geometry plus a circular ROI mask for chart corners."""

    def __init__(self, training):
        self.training = training
        yy, xx = torch.meshgrid(torch.arange(IMAGE_SIZE), torch.arange(IMAGE_SIZE), indexing="ij")
        center = (IMAGE_SIZE - 1) / 2
        self.radial_distance = torch.sqrt((xx - center) ** 2 + (yy - center) ** 2) / IMAGE_SIZE

    def __call__(self, images):
        angle = random.uniform(-5, 5) if self.training else 0.0
        translate = (
            (random.randint(-4, 4), random.randint(-4, 4)) if self.training else (0, 0)
        )
        scale = random.uniform(0.98, 1.02) if self.training else 1.0
        radius = random.uniform(*TRAIN_ROI_RADIUS_RANGE) if self.training else EVAL_ROI_RADIUS
        roi = (self.radial_distance <= radius).float()
        tensors = []
        for image in images:
            image = TF.resize(image, [IMAGE_SIZE, IMAGE_SIZE], InterpolationMode.BICUBIC, antialias=True)
            if self.training:
                image = TF.affine(
                    image,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=tuple(round(value * 255) for value in MEAN),
                )
            tensor = TF.to_tensor(image)
            # ImageNet mean outside the map becomes exactly zero after normalization.
            mean_tensor = torch.tensor(MEAN).view(3, 1, 1)
            tensor = tensor * roi + mean_tensor * (1.0 - roi)
            tensors.append(TF.normalize(tensor, MEAN, STD))
        output = torch.stack(tensors)
        if self.training and random.random() < MODALITY_DROPOUT:
            output[random.randrange(len(MAP_NAMES))].zero_()
        return output


class SevenMapDataset(Dataset):
    def __init__(self, frame, training=False):
        self.frame = frame.reset_index(drop=True)
        self.transform = SevenMapTransform(training)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        images = []
        for name in MAP_NAMES:
            with Image.open(row[f"{name}_path"]) as image:
                images.append(image.convert("RGB"))
        return {
            "images": self.transform(images),
            "label": torch.tensor(int(row.label), dtype=torch.long),
            "sample_id": row.sample_id,
        }


class SevenMapModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s.in21k_ft_in1k",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        self.feature_dim = int(self.backbone.num_features)
        self.map_embeddings = nn.Parameter(torch.zeros(1, len(MAP_NAMES), self.feature_dim))
        hidden = max(64, self.feature_dim // 4)
        self.attention = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim), nn.Dropout(0.30), nn.Linear(self.feature_dim, 2)
        )
        # Shared auxiliary prediction head makes every modality learn useful
        # disease features instead of allowing attention to ignore most maps.
        self.map_classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim), nn.Dropout(0.15), nn.Linear(self.feature_dim, 2)
        )

    def encode(self, images):
        batch, maps, channels, height, width = images.shape
        features = self.backbone(images.reshape(batch * maps, channels, height, width))
        return features.reshape(batch, maps, self.feature_dim)

    def fuse(self, features):
        features = features + self.map_embeddings
        attention = torch.softmax(self.attention(features).squeeze(-1), dim=1)
        logits = self.classifier((features * attention.unsqueeze(-1)).sum(1))
        return logits, attention

    def forward(self, images, return_attention=False, return_auxiliary=False):
        features = self.encode(images)
        logits, attention = self.fuse(features)
        if return_auxiliary:
            return logits, attention, self.map_classifier(features)
        return (logits, attention) if return_attention else logits

    def freeze_backbone(self, frozen):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen


def initialize_from_efficientnet(model, raw_checkpoint):
    source = _extract_state_dict(raw_checkpoint)
    target = model.state_dict()
    copied = {}
    for key, value in source.items():
        if key != "map_embeddings" and key in target and target[key].shape == value.shape:
            copied[key] = value
    result = model.load_state_dict(copied, strict=False)
    copied_backbone = sum(key.startswith("backbone.") for key in copied)
    backbone_total = sum(key.startswith("backbone.") for key in target)
    coverage = copied_backbone / max(1, backbone_total)
    if coverage < 0.80:
        raise RuntimeError(
            f"EfficientNetV2-S checkpoint compatibility is only {coverage:.1%}. "
            "Expected timm EfficientNetV2-S keys such as backbone.blocks.0.0.conv.weight. "
            f"Checkpoint key examples: {sorted(source.keys())[:30]}"
        )
    source_embeddings = source.get("map_embeddings")
    if source_embeddings is None or source_embeddings.shape[1] != 4:
        raise RuntimeError("Expected four source map embeddings in the EfficientNet checkpoint")
    source_map_index = {
        "CT_A": 2, "EC_A": 0, "EC_P": 3, "Elv_A": 0,
        "Elv_P": 3, "Sag_A": 1, "Sag_P": 1,
    }
    with torch.no_grad():
        for target_index, name in enumerate(MAP_NAMES):
            model.map_embeddings[:, target_index].copy_(source_embeddings[:, source_map_index[name]])
        # Initialize the per-map auxiliary head from the source fused classifier.
        if "classifier.0.weight" in source:
            model.map_classifier[0].weight.copy_(source["classifier.0.weight"])
            model.map_classifier[0].bias.copy_(source["classifier.0.bias"])
        if "classifier.2.weight" in source:
            model.map_classifier[2].weight.copy_(source["classifier.2.weight"])
            model.map_classifier[2].bias.copy_(source["classifier.2.bias"])
    print(
        f"Transferred {copied_backbone}/{backbone_total} EfficientNetV2-S backbone tensors "
        f"({coverage:.1%}) and expanded four map embeddings to seven native maps"
    )


def make_loader(frame, training, sample_weights=None):
    sampler = None
    if training and sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(frame),
            replacement=True,
        )
    return DataLoader(
        SevenMapDataset(frame, training),
        batch_size=BATCH_SIZE,
        shuffle=training and sampler is None,
        sampler=sampler,
        drop_last=training and len(frame) >= BATCH_SIZE,
        num_workers=WORKERS,
        pin_memory=True,
        persistent_workers=WORKERS > 0,
    )


def make_optimizer(model, backbone_trainable):
    head_parameters = [
        *model.attention.parameters(),
        *model.classifier.parameters(),
        *model.map_classifier.parameters(),
        model.map_embeddings,
    ]
    groups = [{"params": head_parameters, "lr": HEAD_LR}]
    if backbone_trainable:
        groups.insert(0, {"params": model.backbone.parameters(), "lr": BACKBONE_LR})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    class_weights = torch.tensor([NORMAL_CLASS_WEIGHT, 1.0], dtype=torch.float32, device=device)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for step, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with context:
            logits, attention, map_logits = model(images, return_auxiliary=True)
            ce = F.cross_entropy(
                logits,
                labels,
                weight=class_weights,
                label_smoothing=LABEL_SMOOTHING,
                reduction="none",
            )
            true_probability = torch.softmax(logits, 1).gather(1, labels[:, None]).squeeze(1)
            classification_loss = (((1.0 - true_probability) ** FOCAL_GAMMA) * ce).mean()
            repeated_labels = labels[:, None].expand(-1, len(MAP_NAMES)).reshape(-1)
            auxiliary_loss = F.cross_entropy(
                map_logits.reshape(-1, 2),
                repeated_labels,
                weight=class_weights,
                label_smoothing=LABEL_SMOOTHING,
            )
            entropy = -(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()).sum(1).mean()
            loss = (
                classification_loss
                + AUXILIARY_MAP_LOSS_WEIGHT * auxiliary_loss
                - ATTENTION_ENTROPY_WEIGHT * entropy
            ) / GRAD_ACCUMULATION
        scaler.scale(loss).backward()
        if (step + 1) % GRAD_ACCUMULATION == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach()) * GRAD_ACCUMULATION * len(labels)
    return total_loss / len(loader.dataset)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        logits, attention = model(images, return_attention=True)
        if USE_HORIZONTAL_FLIP_TTA:
            flipped_logits, flipped_attention = model(torch.flip(images, dims=[-1]), return_attention=True)
            logits = (logits + flipped_logits) / 2.0
            attention = (attention + flipped_attention) / 2.0
        margins = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        raw_probabilities = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        for index, sample_id in enumerate(batch["sample_id"]):
            row = {
                "sample_id": sample_id,
                "logit_margin": float(margins[index]),
                "raw_probability": float(raw_probabilities[index]),
            }
            for map_index, name in enumerate(MAP_NAMES):
                row[f"attention_{name}"] = float(attention[index, map_index].cpu())
            rows.append(row)
    return pd.DataFrame(rows)


def hard_negative_sample_weights(frame, predictions):
    """Upweight only difficult normal examples, using training-fold data."""
    scored = frame[["sample_id", "label"]].merge(
        predictions[["sample_id", "raw_probability"]], on="sample_id", validate="one_to_one"
    )
    normal_scores = scored.loc[scored.label == 0, "raw_probability"]
    cutoff = float(normal_scores.quantile(1.0 - HARD_NEGATIVE_FRACTION))
    hard = (scored.label == 0) & (scored.raw_probability >= cutoff)
    weights = np.ones(len(scored), dtype=np.float64)
    weights[hard.to_numpy()] = HARD_NEGATIVE_WEIGHT
    return weights, int(hard.sum()), cutoff


def binary_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def youden_threshold(labels, probabilities):
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][np.argmax((tpr - fpr)[finite])])


def threshold_for_min_sensitivity(labels, probabilities, minimum_sensitivity):
    """Highest cutoff retaining the requested sensitivity; maximizes specificity."""
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    finite = np.isfinite(thresholds)
    eligible = np.where(finite & (tpr >= minimum_sensitivity))[0]
    if not len(eligible):
        raise RuntimeError(f"Could not reach requested sensitivity {minimum_sensitivity:.1%} on OOF data")
    # Larger threshold is stricter, hence has no worse specificity.
    return float(np.max(thresholds[eligible]))


def threshold_for_min_specificity(labels, probabilities, minimum_specificity):
    """Lowest cutoff attaining a specificity target; preserves sensitivity."""
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    finite = np.isfinite(thresholds)
    eligible = np.where(finite & ((1.0 - fpr) >= minimum_specificity))[0]
    if not len(eligible):
        raise RuntimeError(f"Could not reach requested specificity {minimum_specificity:.1%} on OOF data")
    return float(np.min(thresholds[eligible]))


def add_triage(frame, policy):
    """Add low-risk/review/high-risk labels without changing binary metrics."""
    low, high = policy["low_risk_threshold"], policy["high_risk_threshold"]
    if low > high:
        # Degenerate but safe fallback for unusually overlapping OOF scores.
        low = high = policy["screening_decision_threshold"]
    frame["triage"] = np.select(
        [frame.calibrated_probability < low, frame.calibrated_probability >= high],
        ["low_risk", "high_risk"],
        default="clinician_review",
    )
    return frame


def bootstrap_intervals(labels, probabilities, threshold, iterations=2000):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(SEED + 999)
    names = ("auroc", "accuracy", "sensitivity", "specificity")
    samples = {name: [] for name in names}
    for _ in range(iterations):
        indices = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) < 2:
            continue
        result = binary_metrics(labels[indices], probabilities[indices], threshold)
        for name in names:
            samples[name].append(result[name])
    return {
        name: {"low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5))}
        for name, values in samples.items()
    }


def fit_fold(fold, train_frame, val_frame, checkpoint, device):
    fold_dir = OUTPUT_ROOT / "folds" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model = SevenMapModel().to(device)
    initialize_from_efficientnet(model, checkpoint)
    model.freeze_backbone(True)
    optimizer = make_optimizer(model, False)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_loader = make_loader(train_frame, True)
    val_loader = make_loader(val_frame, False)
    best_objective, stale, history, best_epoch = -1.0, 0, [], 0
    checkpoint_path = fold_dir / "best_model.pth"

    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == FREEZE_EPOCHS + 1:
            model.freeze_backbone(False)
            optimizer = make_optimizer(model, True)
        started = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        validation = predict(model, val_loader, device)
        val_auc = roc_auc_score(val_frame.label, validation.raw_probability)
        val_threshold = threshold_for_min_sensitivity(
            val_frame.label, validation.raw_probability, SCREENING_MIN_SENSITIVITY
        )
        val_metrics = binary_metrics(val_frame.label, validation.raw_probability, val_threshold)
        balanced_accuracy = (val_metrics["sensitivity"] + val_metrics["specificity"]) / 2.0
        # Ranking remains primary, while the operating-point term prevents an
        # early perfect-ish AUROC checkpoint from winning despite worse class decisions.
        validation_objective = 0.70 * val_auc + 0.30 * balanced_accuracy
        hard_count, hard_cutoff = 0, None
        if epoch in HARD_NEGATIVE_REFRESH_EPOCHS and epoch < MAX_EPOCHS:
            training_predictions = predict(model, make_loader(train_frame, False), device)
            sample_weights, hard_count, hard_cutoff = hard_negative_sample_weights(
                train_frame, training_predictions
            )
            train_loader = make_loader(train_frame, True, sample_weights)
            print(
                f"Fold {fold}: hard-negative replay now upweights {hard_count} normal cases "
                f"with training probability >= {hard_cutoff:.4f}"
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auroc": val_auc,
                "val_balanced_accuracy": balanced_accuracy,
                "val_sensitivity": val_metrics["sensitivity"],
                "val_specificity": val_metrics["specificity"],
                "validation_objective": validation_objective,
                "hard_negatives_upweighted": hard_count,
                "hard_negative_cutoff": hard_cutoff,
                "seconds": time.time() - started,
            }
        )
        print(
            f"Fold {fold} epoch {epoch:02d}: loss={train_loss:.4f} val_auc={val_auc:.4f} "
            f"balanced_acc={balanced_accuracy:.4f} objective={validation_objective:.4f}"
        )
        if validation_objective > best_objective + 1e-5:
            best_objective, stale, best_epoch = validation_objective, 0, epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "encoder": "efficientnetv2_s",
                        "map_order": list(MAP_NAMES),
                        "image_size": IMAGE_SIZE,
                        "evaluation_roi_radius_fraction": EVAL_ROI_RADIUS,
                    },
                    "epoch": epoch,
                    "validation_auroc": float(val_auc),
                    "validation_balanced_accuracy": float(balanced_accuracy),
                    "validation_objective": float(validation_objective),
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    oof = predict(model, val_loader, device)
    oof = val_frame[["sample_id", "label"]].merge(oof, on="sample_id", validate="one_to_one")
    oof["fold"] = fold
    del model, optimizer, scaler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    return checkpoint_path, oof, best_epoch


def load_finetuned(path, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    model = SevenMapModel().to(device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    return model.eval()


def ensemble_predict(checkpoint_paths, frame, device):
    base = frame[["sample_id"]].copy()
    margins, probabilities, attentions = [], [], []
    loader = make_loader(frame, False)
    for path in checkpoint_paths:
        model = load_finetuned(path, device)
        result = predict(model, loader, device).set_index("sample_id").loc[base.sample_id]
        margins.append(result.logit_margin.to_numpy())
        probabilities.append(result.raw_probability.to_numpy())
        attentions.append(result[[f"attention_{name}" for name in MAP_NAMES]].to_numpy())
        del model
        gc.collect()
        torch.cuda.empty_cache()
    output = base.copy()
    output["logit_margin"] = np.mean(margins, axis=0)
    output["raw_probability"] = np.mean(probabilities, axis=0)
    mean_attention = np.mean(attentions, axis=0)
    for index, name in enumerate(MAP_NAMES):
        output[f"attention_{name}"] = mean_attention[:, index]
    return output


def fit_calibrator(oof):
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=SEED)
    calibrator.fit(oof[["logit_margin"]], oof.label)
    calibrated = calibrator.predict_proba(oof[["logit_margin"]])[:, 1]
    screening_threshold = threshold_for_min_sensitivity(
        oof.label, calibrated, SCREENING_MIN_SENSITIVITY
    )
    low_risk_threshold = min(
        screening_threshold,
        threshold_for_min_sensitivity(oof.label, calibrated, LOW_RISK_MIN_SENSITIVITY),
    )
    high_risk_threshold = max(
        screening_threshold,
        threshold_for_min_specificity(oof.label, calibrated, HIGH_RISK_MIN_SPECIFICITY),
    )
    policy = {
        "selection_data": "five-fold out-of-fold development predictions only",
        "screening_decision_threshold": screening_threshold,
        "screening_min_sensitivity": SCREENING_MIN_SENSITIVITY,
        "low_risk_threshold": low_risk_threshold,
        "low_risk_min_sensitivity": LOW_RISK_MIN_SENSITIVITY,
        "high_risk_threshold": high_risk_threshold,
        "high_risk_min_specificity": HIGH_RISK_MIN_SPECIFICITY,
        "youden_threshold_reference_only": youden_threshold(oof.label, calibrated),
    }
    return calibrator, calibrated, policy


def apply_calibrator(frame, calibrator):
    return calibrator.predict_proba(frame[["logit_margin"]])[:, 1]


def train_deployment_model(dev_frame, checkpoint, device, epochs):
    print(f"Training single deployment model on all {len(dev_frame)} development cases for {epochs} epochs")
    model = SevenMapModel().to(device)
    initialize_from_efficientnet(model, checkpoint)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loader = make_loader(dev_frame, True)
    model.freeze_backbone(True)
    optimizer = make_optimizer(model, False)
    history = []
    for epoch in range(1, epochs + 1):
        if epoch == FREEZE_EPOCHS + 1:
            model.freeze_backbone(False)
            optimizer = make_optimizer(model, True)
        loss = train_epoch(model, loader, optimizer, scaler, device)
        hard_count, hard_cutoff = 0, None
        if epoch in HARD_NEGATIVE_REFRESH_EPOCHS and epoch < epochs:
            training_predictions = predict(model, make_loader(dev_frame, False), device)
            sample_weights, hard_count, hard_cutoff = hard_negative_sample_weights(
                dev_frame, training_predictions
            )
            loader = make_loader(dev_frame, True, sample_weights)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "hard_negatives_upweighted": hard_count,
                "hard_negative_cutoff": hard_cutoff,
            }
        )
        print(f"Deployment epoch {epoch:02d}/{epochs}: loss={loss:.4f}")
    path = OUTPUT_ROOT / "deployment_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "encoder": "efficientnetv2_s",
                "map_order": list(MAP_NAMES),
                "image_size": IMAGE_SIZE,
                "evaluation_roi_radius_fraction": EVAL_ROI_RADIUS,
                "source_checkpoint": "EfficientNetV2-S four-map attention-fusion checkpoint",
            },
            "training_cases": len(dev_frame),
            "epochs": epochs,
        },
        path,
    )
    pd.DataFrame(history).to_csv(OUTPUT_ROOT / "deployment_history.csv", index=False)
    del model, optimizer, scaler, loader
    gc.collect()
    torch.cuda.empty_cache()
    return path


def save_gradcams(model_path, frame, output_dir, device):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    class CamWrapper(nn.Module):
        def __init__(self, model, reference, map_index):
            super().__init__()
            self.model, self.map_index = model, map_index
            with torch.no_grad():
                self.register_buffer("fixed", model.encode(reference))

        def forward(self, selected):
            features = self.fixed.repeat(selected.shape[0], 1, 1).clone()
            features[:, self.map_index] = self.model.backbone(selected)
            return self.model.fuse(features)[0]

    model = load_finetuned(model_path, device)
    transform = SevenMapTransform(False)
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in frame.itertuples(index=False):
        originals = []
        for name in MAP_NAMES:
            with Image.open(getattr(row, f"{name}_path")) as image:
                originals.append(image.convert("RGB"))
        reference = transform(originals).unsqueeze(0).to(device)
        case_dir = output_dir / row.sample_id
        case_dir.mkdir(parents=True, exist_ok=True)
        for index, (name, original) in enumerate(zip(MAP_NAMES, originals)):
            wrapper = CamWrapper(model, reference, index).to(device).eval()
            with GradCAM(model=wrapper, target_layers=[model.backbone.blocks[-1][-1]]) as cam:
                heatmap = cam(reference[:, index], [ClassifierOutputTarget(1)])[0]
            base = np.asarray(original.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32) / 255.0
            Image.fromarray(show_cam_on_image(base, heatmap, use_rgb=True)).save(case_dir / f"{name}.png")
            del wrapper
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator before running this script")
    seed_everything()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    training_root, independent_root = discover_dataset_roots()
    source_path, source_checkpoint = discover_source_checkpoint()
    dev, training_skipped = build_manifest(training_root)
    test, test_skipped = build_manifest(independent_root)
    training_suspects = build_suspect_manifest(training_root)
    independent_suspects = build_suspect_manifest(independent_root)
    print("Training/validation dataset:", training_root)
    print("Independent test dataset:", independent_root)
    print("EfficientNetV2-S source checkpoint:", source_path)
    print("Development binary cases:", dev.label.value_counts().sort_index().to_dict())
    print("Independent binary cases:", test.label.value_counts().sort_index().to_dict())
    print("Suspects (development/independent):", len(training_suspects), len(independent_suspects))
    print("GPU:", torch.cuda.get_device_name(0))

    # Folder names repeat across the two official partitions, so file paths—not
    # case1/case2 identifiers—establish that no actual image is reused.
    development_paths = {row[f"{name}_path"] for _, row in dev.iterrows() for name in MAP_NAMES}
    independent_paths = {row[f"{name}_path"] for _, row in test.iterrows() for name in MAP_NAMES}
    if development_paths & independent_paths:
        raise RuntimeError("Image path leakage detected between development and independent test")
    development_hashes = exact_hashes_by_sample(dev)
    independent_hashes = exact_hashes_by_sample(test)
    binary_hash_overlap = set().union(*development_hashes.values()) & set().union(*independent_hashes.values())
    if binary_hash_overlap:
        raise RuntimeError(
            f"Exact image-content leakage detected across binary partitions: {len(binary_hash_overlap)} hashes"
        )
    binary_near_duplicates = cross_partition_near_duplicates(dev, test)
    if binary_near_duplicates:
        raise RuntimeError(
            "Strict perceptual near-duplicate audit failed across binary partitions: "
            f"{binary_near_duplicates[:10]}"
        )

    # The supplied archive contains a few repeated Suspect charts across its
    # two partitions. Suspects are not used for fitting, but exclude those
    # repeated cases from the independent exploratory report for honesty.
    repeated_independent_suspects = []
    if not training_suspects.empty and not independent_suspects.empty:
        training_suspect_hashes = exact_hashes_by_sample(training_suspects)
        independent_suspect_hashes = exact_hashes_by_sample(independent_suspects)
        training_hash_pool = set().union(*training_suspect_hashes.values())
        repeated_independent_suspects = [
            sample_id
            for sample_id, hashes in independent_suspect_hashes.items()
            if hashes & training_hash_pool
        ]
        if repeated_independent_suspects:
            warnings.warn(
                "Excluding independent Suspect cases with exact chart reuse: "
                f"{repeated_independent_suspects}"
            )
            independent_suspects = independent_suspects[
                ~independent_suspects.sample_id.isin(repeated_independent_suspects)
            ].reset_index(drop=True)
    dev.assign(split="development").to_csv(OUTPUT_ROOT / "development_manifest.csv", index=False)
    test.assign(split="independent_test").to_csv(OUTPUT_ROOT / "independent_test_manifest.csv", index=False)

    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    checkpoint_paths, oof_parts, best_epochs = [], [], []
    for fold, (train_indices, val_indices) in enumerate(folds.split(dev, dev.label), 1):
        print(f"\n{'=' * 70}\nFOLD {fold}/{N_FOLDS}\n{'=' * 70}")
        path, oof, best_epoch = fit_fold(
            fold,
            dev.iloc[train_indices].reset_index(drop=True),
            dev.iloc[val_indices].reset_index(drop=True),
            source_checkpoint,
            torch.device("cuda"),
        )
        checkpoint_paths.append(path)
        oof_parts.append(oof)
        best_epochs.append(best_epoch)

    oof = pd.concat(oof_parts, ignore_index=True)
    if set(oof.sample_id) != set(dev.sample_id) or oof.sample_id.duplicated().any():
        raise RuntimeError("OOF integrity check failed")
    calibrator, oof_calibrated, threshold_policy = fit_calibrator(oof)
    threshold = threshold_policy["screening_decision_threshold"]
    oof["calibrated_probability"] = oof_calibrated
    oof["prediction"] = (oof_calibrated >= threshold).astype(int)
    oof = add_triage(oof, threshold_policy)
    oof.to_csv(OUTPUT_ROOT / "oof_predictions.csv", index=False)
    oof_metrics = binary_metrics(oof.label, oof_calibrated, threshold)

    device = torch.device("cuda")
    ensemble_test = test[["sample_id", "label", "source_class"]].merge(
        ensemble_predict(checkpoint_paths, test, device), on="sample_id", validate="one_to_one"
    )
    ensemble_test["calibrated_probability"] = apply_calibrator(ensemble_test, calibrator)
    ensemble_test["prediction"] = (ensemble_test.calibrated_probability >= threshold).astype(int)
    ensemble_test = add_triage(ensemble_test, threshold_policy)
    ensemble_metrics = binary_metrics(test.label, ensemble_test.calibrated_probability, threshold)
    ensemble_metrics["bootstrap_95_ci"] = bootstrap_intervals(
        test.label, ensemble_test.calibrated_probability, threshold
    )
    ensemble_test.to_csv(OUTPUT_ROOT / "independent_test_predictions_ensemble.csv", index=False)

    # Upper-quartile fold stopping time is more stable than the previous median,
    # which under-trained the single deployment model (3 epochs in the last run).
    final_epochs = max(FREEZE_EPOCHS + 1, int(round(np.percentile(best_epochs, 75))))
    deployment_path = train_deployment_model(dev, source_checkpoint, device, final_epochs)
    deployment_model = load_finetuned(deployment_path, device)
    deployment_test = test[["sample_id", "label", "source_class"]].merge(
        predict(deployment_model, make_loader(test, False), device), on="sample_id", validate="one_to_one"
    )
    deployment_test["calibrated_probability"] = apply_calibrator(deployment_test, calibrator)
    deployment_test["prediction"] = (deployment_test.calibrated_probability >= threshold).astype(int)
    deployment_test = add_triage(deployment_test, threshold_policy)
    deployment_metrics = binary_metrics(test.label, deployment_test.calibrated_probability, threshold)
    deployment_metrics["bootstrap_95_ci"] = bootstrap_intervals(
        test.label, deployment_test.calibrated_probability, threshold
    )
    deployment_test.to_csv(OUTPUT_ROOT / "independent_test_predictions_deployment.csv", index=False)
    del deployment_model
    torch.cuda.empty_cache()

    for partition, suspects in (
        ("development", training_suspects),
        ("independent", independent_suspects),
    ):
        if not suspects.empty:
            suspect_predictions = ensemble_predict(checkpoint_paths, suspects, device)
            suspect_predictions["calibrated_probability"] = apply_calibrator(suspect_predictions, calibrator)
            suspect_predictions["prediction_at_binary_threshold"] = (
                suspect_predictions.calibrated_probability >= threshold
            ).astype(int)
            suspect_predictions = add_triage(suspect_predictions, threshold_policy)
            suspect_predictions.to_csv(
                OUTPUT_ROOT / f"{partition}_suspect_predictions_exploratory.csv", index=False
            )

    # Save calibration parameters with the single deployment model.
    deployment_saved = torch.load(deployment_path, map_location="cpu", weights_only=False)
    deployment_saved["calibration"] = {
        "method": "OOF logistic/Platt calibration",
        "coefficient": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
        **{key: float(value) if isinstance(value, (float, np.floating)) else value for key, value in threshold_policy.items()},
    }
    deployment_saved["class_names"] = ["normal", "keratoconus"]
    torch.save(deployment_saved, deployment_path)

    # Explain representative true cases and up to two deployment errors.
    selected_ids = []
    for label in (0, 1):
        subset = deployment_test[deployment_test.label == label]
        median = subset.calibrated_probability.median()
        selected_ids.append(subset.iloc[(subset.calibrated_probability - median).abs().argsort()[:1]].sample_id.iloc[0])
    selected_ids.extend(deployment_test.loc[deployment_test.label != deployment_test.prediction, "sample_id"].head(2))
    explanation_frame = test[test.sample_id.isin(dict.fromkeys(selected_ids))]
    save_gradcams(deployment_path, explanation_frame, OUTPUT_ROOT / "gradcam", device)

    summary = {
        "training_root": str(training_root),
        "independent_test_root": str(independent_root),
        "source_checkpoint": str(source_path),
        "framework": "PyTorch timm EfficientNetV2-S + custom seven-map attention fusion",
        "split": {
            "development_cases": len(dev),
            "independent_test_cases": len(test),
            "development_class_counts": dev.label.value_counts().sort_index().to_dict(),
            "test_class_counts": test.label.value_counts().sort_index().to_dict(),
            "folds": N_FOLDS,
        },
        "map_order": list(MAP_NAMES),
        "preprocessing": {
            "evaluation": f"central circular ROI radius={EVAL_ROI_RADIUS:.3f} image widths",
            "training": f"randomized ROI radius={TRAIN_ROI_RADIUS_RANGE} + mild synchronized affine + modality dropout",
            "purpose": "suppress scan borders, corner labels, legends, and fixed-layout shortcuts",
        },
        "training_strategy": {
            "loss": "normal-weighted focal cross-entropy + per-map auxiliary supervision",
            "normal_class_weight": NORMAL_CLASS_WEIGHT,
            "focal_gamma": FOCAL_GAMMA,
            "auxiliary_map_loss_weight": AUXILIARY_MAP_LOSS_WEIGHT,
            "hard_negative_fraction": HARD_NEGATIVE_FRACTION,
            "hard_negative_weight": HARD_NEGATIVE_WEIGHT,
            "hard_negative_refresh_epochs": list(HARD_NEGATIVE_REFRESH_EPOCHS),
            "horizontal_flip_tta": USE_HORIZONTAL_FLIP_TTA,
            "checkpoint_selection": "70% validation AUROC + 30% validation balanced accuracy",
        },
        "oof_metrics": oof_metrics,
        "ensemble_independent_test": ensemble_metrics,
        "deployment_independent_test": deployment_metrics,
        "deployment_model": str(deployment_path),
        "ensemble_checkpoints": [str(path) for path in checkpoint_paths],
        "calibration": {
            "coefficient": float(calibrator.coef_[0, 0]),
            "intercept": float(calibrator.intercept_[0]),
            **{key: float(value) if isinstance(value, (float, np.floating)) else value for key, value in threshold_policy.items()},
        },
        "best_fold_epochs": best_epochs,
        "deployment_epochs": final_epochs,
        "suspect_cases_used_for_training": 0,
        "development_suspect_cases_scored": len(training_suspects),
        "independent_suspect_cases_scored": len(independent_suspects),
        "incomplete_cases_skipped": {
            "training": training_skipped,
            "independent_test": test_skipped,
        },
        "leakage_audit": {
            "binary_exact_hashes_crossing_partitions": 0,
            "binary_strict_perceptual_duplicates_crossing_partitions": 0,
            "perceptual_hash_definition": "256-bit dHash, failure at Hamming distance <= 8 within class/map",
            "independent_suspect_cases_excluded_for_exact_chart_reuse": repeated_independent_suspects,
        },
        "limitations": [
            "Case folders do not expose patient IDs, so patient-level grouping cannot be verified.",
            "The supplied independent test partition is separate from training but its patient/site provenance is not documented here.",
            "Suspect outputs are exploratory and are not a validated third class.",
        ],
    }
    (OUTPUT_ROOT / "run_summary.json").write_text(json.dumps(summary, indent=2))
    archive = shutil.make_archive(
        "/kaggle/working/keratoconus_external_finetune_results", "zip", OUTPUT_ROOT
    )
    print("\nOOF metrics:\n", json.dumps(oof_metrics, indent=2))
    print("\nUntouched test ensemble:\n", json.dumps(ensemble_metrics, indent=2))
    print("\nUntouched test deployment model:\n", json.dumps(deployment_metrics, indent=2))
    print("\nDownload:", archive)


if __name__ == "__main__":
    main()
