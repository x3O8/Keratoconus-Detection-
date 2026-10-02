"""One-click Kaggle pipeline for the ORBSCAN keratoconus dataset.

The runner accepts either the prepared ``labeled_dataset`` or the original
ORBSCAN upload containing ``ORBSCAN_Dataset`` and
``clinical_data_and_labels.csv``. No CLI arguments are required. It builds an
eye-level manifest in memory for the original layout, then trains all five
encoders, evaluates the patient-held-out test split, creates Grad-CAM examples,
and zips outputs.

Kaggle Internet must be enabled for pretrained timm and RETFound weights.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path


# ---------------------------------------------------------------------------
# User-adjustable defaults. The pipeline runs as-is without changing these.
# ---------------------------------------------------------------------------
SEED = 42
FOLD = 1
OUTPUT_ROOT = Path("/kaggle/working/keratoconus_complete_run")
MODEL_RUNS = {
    "efficientnetv2_s": {"epochs": 18, "batch": 8, "accum": 1},
    "convnext_b": {"epochs": 12, "batch": 2, "accum": 4},
    "swin_b": {"epochs": 12, "batch": 2, "accum": 4},
    "dinov2": {"epochs": 12, "batch": 2, "accum": 4},
    "retfound": {"epochs": 8, "batch": 1, "accum": 8},
}
PATIENCE = 4
FREEZE_EPOCHS = 2
HEAD_LR = 2e-4
BACKBONE_LR = 2e-5
WEIGHT_DECAY = 1e-4
WORKERS = 2
MAP_NAMES = ("anterior", "axial", "pachymetry", "posterior")
MAP_COLUMNS = tuple(f"{name}_path" for name in MAP_NAMES)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
RETF_GDRIVE_ID = "1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE"
# Kaggle normally mounts the dataset slug below as /kaggle/input/orbscan.
# The second path supports the path supplied by the user in case it is mounted
# with the owner hierarchy preserved.
KAGGLE_ORBSCAN_ROOTS = (
    Path("/kaggle/input/orbscan"),
    Path("/kaggle/input/datasets/eatradium/orbscan"),
)


def ensure_dependencies():
    packages = {
        "timm": "timm>=1.0.15,<2",
        "pytorch_grad_cam": "grad-cam>=1.5.5",
        "gdown": "gdown>=5.2",
    }
    missing = [pip_name for module, pip_name in packages.items() if importlib.util.find_spec(module) is None]
    if missing:
        print("Installing:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_dependencies()

import gdown
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


ENCODERS = {
    "efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "convnext_b": "convnext_base.fb_in22k_ft_in1k",
    "swin_b": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    "dinov2": "vit_base_patch14_dinov2.lvd142m",
    "retfound": "vit_large_patch16_224",
}


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _validate_manifest(root, frame):
    missing_columns = {"patient_id", "sample_id", "eye", "label", *MAP_COLUMNS} - set(frame)
    if missing_columns:
        raise RuntimeError(f"Manifest missing columns: {sorted(missing_columns)}")
    missing_files = [
        str(root / row[column])
        for _, row in frame.iterrows()
        for column in MAP_COLUMNS
        if not (root / row[column]).is_file()
    ]
    if missing_files:
        raise RuntimeError(f"Manifest references missing files: {missing_files[:10]}")
    print(f"Dataset: {root} | eyes={len(frame)} | patients={frame.patient_id.nunique()}")
    return root, frame


def _resolve_patient_code(csv_code, patient_ids):
    """Resolve an exact patient ID plus spreadsheet-style '+' corruption."""
    value = str(csv_code).strip()
    if value in patient_ids:
        return value
    candidates = {
        value.replace("+", ""),
        value.replace("E+", "E"),
        value.replace("e+", "e"),
    }
    matches = sorted(candidates & patient_ids)
    return matches[0] if len(matches) == 1 else None


def _manifest_from_raw_orbscan(source_root):
    """Create one deterministic image sample per available patient eye."""
    images_root = source_root / "ORBSCAN_Dataset"
    labels_path = source_root / "clinical_data_and_labels.csv"
    if not images_root.is_dir() or not labels_path.is_file():
        raise FileNotFoundError(f"Not an ORBSCAN source dataset: {source_root}")

    raw = pd.read_csv(labels_path, dtype={"patient_code": str})
    required = {"patient_code", "eye", "label", "split_80_20", "fold"}
    missing_columns = required - set(raw.columns)
    if missing_columns:
        raise RuntimeError(f"ORBSCAN CSV missing columns: {sorted(missing_columns)}")
    raw["patient_code"] = raw["patient_code"].str.strip()
    raw["eye"] = raw["eye"].str.strip().str.upper()
    raw["label"] = pd.to_numeric(raw["label"], errors="raise").astype(int)
    if not set(raw["label"]).issubset({0, 1}):
        raise RuntimeError("ORBSCAN labels must contain only 0 (normal) and 1 (keratoconus)")

    patient_ids = {path.name for path in images_root.iterdir() if path.is_dir()}
    raw["resolved_patient_code"] = raw["patient_code"].map(
        lambda patient: _resolve_patient_code(patient, patient_ids)
    )
    unresolved = raw.loc[raw["resolved_patient_code"].isna(), "patient_code"].unique().tolist()
    if unresolved:
        raise RuntimeError(f"CSV patient IDs not found in ORBSCAN_Dataset: {unresolved[:20]}")
    conflicting = raw.groupby("resolved_patient_code")["label"].nunique()
    conflicting = conflicting[conflicting > 1]
    if not conflicting.empty:
        raise RuntimeError(f"Patients with conflicting labels: {conflicting.index.tolist()[:20]}")

    # The source CSV contains repeated clinical measurements for some eyes.
    # Images are unique, so train once per patient/eye and keep the first record.
    eyes = raw.sort_values(["resolved_patient_code", "eye"]).drop_duplicates(
        ["resolved_patient_code", "eye"], keep="first"
    )
    rows, skipped = [], []
    for record in eyes.itertuples(index=False):
        patient = record.resolved_patient_code
        eye = record.eye
        eye_folder = images_root / patient / eye
        paths = {}
        for map_name in MAP_NAMES:
            image_path = eye_folder / f"{patient}_{eye}_{map_name.title()}.png"
            if not image_path.is_file():
                paths = None
                break
            paths[f"{map_name}_path"] = image_path.relative_to(source_root).as_posix()
        if paths is None:
            skipped.append(f"{patient}/{eye}")
            continue
        fold = None if pd.isna(record.fold) else int(record.fold)
        rows.append(
            {
                "patient_id": patient,
                "sample_id": f"{patient}_{eye}",
                "eye": eye,
                "label": int(record.label),
                "split": str(record.split_80_20).strip().lower(),
                "fold": fold,
                **paths,
            }
        )
    frame = pd.DataFrame(rows)
    if skipped:
        print(f"Skipped {len(skipped)} incomplete eye folders: {skipped[:10]}")
    aliases = raw.loc[
        raw["patient_code"] != raw["resolved_patient_code"],
        ["patient_code", "resolved_patient_code"],
    ].drop_duplicates()
    print(
        "Built in-memory manifest from original ORBSCAN files | "
        f"eyes={len(frame)} | duplicate CSV eye rows removed={len(raw) - len(eyes)} | "
        f"ID aliases={aliases.to_dict('records')}"
    )
    return _validate_manifest(source_root, frame)


def discover_dataset():
    # First honor the requested Kaggle ORBSCAN mount, whether it contains the
    # original dataset or an already-prepared labelled dataset.
    candidates = [path for path in KAGGLE_ORBSCAN_ROOTS if path.is_dir()]
    local_source = Path("keratoconus_better")
    if local_source.is_dir():
        candidates.append(local_source)
    for candidate in candidates:
        manifest = candidate / "metadata" / "eye_manifest.csv"
        if manifest.is_file():
            frame = pd.read_csv(manifest, dtype={"patient_id": str, "sample_id": str})
            return _validate_manifest(candidate, frame)
        if (candidate / "labeled_dataset" / "metadata" / "eye_manifest.csv").is_file():
            root = candidate / "labeled_dataset"
            frame = pd.read_csv(root / "metadata" / "eye_manifest.csv", dtype={"patient_id": str, "sample_id": str})
            return _validate_manifest(root, frame)
        if (candidate / "ORBSCAN_Dataset").is_dir():
            return _manifest_from_raw_orbscan(candidate)

    manifests = list(Path("/kaggle/input").glob("**/metadata/eye_manifest.csv"))
    if len(manifests) == 1:
        root = manifests[0].parent.parent
        frame = pd.read_csv(manifests[0], dtype={"patient_id": str, "sample_id": str})
        return _validate_manifest(root, frame)
    raw_csvs = list(Path("/kaggle/input").glob("**/clinical_data_and_labels.csv"))
    if len(raw_csvs) == 1:
        return _manifest_from_raw_orbscan(raw_csvs[0].parent)
    raise RuntimeError(
        "Could not find the ORBSCAN dataset. Expected either metadata/eye_manifest.csv "
        "or ORBSCAN_Dataset/ plus clinical_data_and_labels.csv under /kaggle/input. "
        f"Found manifests={manifests}, raw CSVs={raw_csvs}"
    )


def split_manifest(frame):
    development = frame[frame["split"].str.lower() == "train"].copy()
    test = frame[frame["split"].str.lower() == "test"].copy()
    folds = pd.to_numeric(development["fold"], errors="coerce")
    train = development[folds != FOLD].copy()
    val = development[folds == FOLD].copy()
    patient_sets = [set(part.patient_id) for part in (train, val, test)]
    for left in range(3):
        for right in range(left + 1, 3):
            if patient_sets[left] & patient_sets[right]:
                raise RuntimeError("Patient leakage detected between splits")
    if min(map(len, (train, val, test))) == 0:
        raise RuntimeError("Train, validation, or test split is empty")
    print(
        f"Split eyes: train={len(train)}, val={len(val)}, test={len(test)} | "
        f"patients={train.patient_id.nunique()}/{val.patient_id.nunique()}/{test.patient_id.nunique()}"
    )
    return train, val, test


class FourMapTransform:
    def __init__(self, training):
        self.training = training

    def __call__(self, images):
        angle = random.uniform(-6, 6) if self.training else 0
        translate = (
            (random.randint(-6, 6), random.randint(-6, 6)) if self.training else (0, 0)
        )
        scale = random.uniform(0.97, 1.03) if self.training else 1.0
        output = []
        for image in images:
            image = TF.resize(image, [224, 224], InterpolationMode.BICUBIC, antialias=True)
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
            output.append(TF.normalize(TF.to_tensor(image), MEAN, STD))
        return torch.stack(output)


class OrbscanDataset(Dataset):
    def __init__(self, frame, root, training=False):
        self.frame = frame.reset_index(drop=True)
        self.root = root
        self.transform = FourMapTransform(training)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        images = []
        for column in MAP_COLUMNS:
            with Image.open(self.root / row[column]) as image:
                images.append(image.convert("RGB"))
        return {
            "images": self.transform(images),
            "label": torch.tensor(int(row.label), dtype=torch.long),
            "sample_id": row.sample_id,
            "patient_id": row.patient_id,
            "eye": row.eye,
        }


def retfound_checkpoint():
    # Prefer an uploaded official checkpoint, otherwise download the public CFP weights.
    candidates = list(Path("/kaggle/input").glob("**/RETFound_cfp_weights.pth"))
    if candidates:
        print("Using uploaded RETFound checkpoint:", candidates[0])
        return candidates[0]
    # Keep the source checkpoint outside OUTPUT_ROOT so it is not duplicated in
    # the final results archive alongside the fine-tuned RETFound checkpoint.
    destination = Path("/kaggle/working/retfound_pretrained/RETFound_cfp_weights.pth")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        print("Downloading official RETFound CFP weights (about 1.2 GB)...")
        result = gdown.download(id=RETF_GDRIVE_ID, output=str(destination), quiet=False)
        if not result or not destination.is_file():
            raise RuntimeError(
                "RETFound download failed. Enable Kaggle Internet or attach "
                "RETFound_cfp_weights.pth as an additional input dataset."
            )
    return destination


def load_retfound_weights(backbone, path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw.get("model", raw.get("model_state_dict", raw))
    target = backbone.state_dict()
    compatible = {}
    for key, value in state.items():
        for prefix in ("module.", "model.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key in target and target[key].shape == value.shape:
            compatible[key] = value
    coverage = len(compatible) / max(1, len(target))
    if coverage < 0.70:
        raise RuntimeError(f"RETFound checkpoint compatibility only {coverage:.1%}")
    backbone.load_state_dict(compatible, strict=False)
    print(f"Loaded RETFound weights: {len(compatible)} tensors ({coverage:.1%} coverage)")


class MultiMapModel(nn.Module):
    def __init__(self, encoder, retfound_path=None):
        super().__init__()
        import timm

        kwargs = {"pretrained": encoder != "retfound", "num_classes": 0, "global_pool": "avg"}
        if encoder == "dinov2":
            kwargs["img_size"] = 224
        self.backbone = timm.create_model(ENCODERS[encoder], **kwargs)
        if encoder == "retfound":
            load_retfound_weights(self.backbone, retfound_path)
        self.encoder_name = encoder
        self.feature_dim = int(self.backbone.num_features)
        self.map_embeddings = nn.Parameter(torch.zeros(1, 4, self.feature_dim))
        nn.init.trunc_normal_(self.map_embeddings, std=0.02)
        hidden = max(64, self.feature_dim // 4)
        self.attention = nn.Sequential(
            nn.LayerNorm(self.feature_dim), nn.Linear(self.feature_dim, hidden),
            nn.GELU(), nn.Linear(hidden, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim), nn.Dropout(0.30), nn.Linear(self.feature_dim, 2)
        )

    def encode(self, images):
        batch, maps, channels, height, width = images.shape
        features = self.backbone(images.reshape(batch * maps, channels, height, width))
        return features.reshape(batch, maps, self.feature_dim)

    def fuse(self, features):
        features = features + self.map_embeddings
        attention = torch.softmax(self.attention(features).squeeze(-1), dim=1)
        fused = (features * attention.unsqueeze(-1)).sum(1)
        return self.classifier(fused), attention

    def forward(self, images, return_attention=False):
        logits, attention = self.fuse(self.encode(images))
        return (logits, attention) if return_attention else logits

    def freeze_backbone(self, frozen):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen


def loaders(root, parts, batch_size):
    common = {
        "batch_size": batch_size,
        "num_workers": WORKERS,
        "pin_memory": True,
        "persistent_workers": WORKERS > 0,
    }
    return (
        DataLoader(OrbscanDataset(parts[0], root, True), shuffle=True, drop_last=True, **common),
        DataLoader(OrbscanDataset(parts[1], root, False), shuffle=False, **common),
        DataLoader(OrbscanDataset(parts[2], root, False), shuffle=False, **common),
    )


def optimizer_for(model, backbone_trainable):
    if backbone_trainable:
        return torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
                {
                    "params": [*model.attention.parameters(), *model.classifier.parameters(), model.map_embeddings],
                    "lr": HEAD_LR,
                },
            ],
            weight_decay=WEIGHT_DECAY,
        )
    return torch.optim.AdamW(
        [*model.attention.parameters(), *model.classifier.parameters(), model.map_embeddings],
        lr=HEAD_LR,
        weight_decay=WEIGHT_DECAY,
    )


def metrics(labels, probabilities, threshold=0.5):
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


def train_epoch(model, loader, optimizer, criterion, device, scaler, accumulation):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, labels_all, probabilities = 0.0, [], []
    for step, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with context:
            logits = model(images)
            loss = criterion(logits, labels) / accumulation
        scaler.scale(loss).backward()
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total += loss.item() * accumulation * len(labels)
        labels_all.extend(labels.detach().cpu().tolist())
        probabilities.extend(torch.softmax(logits.detach(), 1)[:, 1].cpu().tolist())
    return total / len(loader.dataset), metrics(labels_all, probabilities)


@torch.inference_mode()
def evaluate(model, loader, criterion, device, threshold=0.5, attention=False):
    model.eval()
    total, labels_all, probabilities, rows = 0.0, [], [], []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        output = model(images, return_attention=attention)
        logits, weights = output if attention else (output, None)
        probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        total += criterion(logits, labels).item() * len(labels)
        labels_np = labels.cpu().numpy()
        labels_all.extend(labels_np.tolist())
        probabilities.extend(probs.tolist())
        for index, sample_id in enumerate(batch["sample_id"]):
            row = {
                "sample_id": sample_id, "patient_id": batch["patient_id"][index],
                "eye": batch["eye"][index], "label": int(labels_np[index]),
                "probability": float(probs[index]),
            }
            if weights is not None:
                for map_index, name in enumerate(MAP_NAMES):
                    row[f"attention_{name}"] = float(weights[index, map_index].cpu())
            rows.append(row)
    return total / len(loader.dataset), metrics(labels_all, probabilities, threshold), rows


def youden_threshold(rows):
    labels = [row["label"] for row in rows]
    probabilities = [row["probability"] for row in rows]
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][np.argmax((tpr - fpr)[finite])])


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


def cam_layer(model):
    if model.encoder_name == "swin_b":
        return model.backbone.layers[-1].blocks[-1].norm1
    if model.encoder_name == "convnext_b":
        return model.backbone.stages[-1].blocks[-1]
    if model.encoder_name == "efficientnetv2_s":
        return model.backbone.blocks[-1][-1]
    return model.backbone.blocks[-1].norm1


def cam_reshape(model):
    prefix = int(getattr(model.backbone, "num_prefix_tokens", 1))

    def reshape(tensor):
        if tensor.ndim == 4:
            return tensor.permute(0, 3, 1, 2) if tensor.shape[-1] == model.feature_dim else tensor
        tensor = tensor[:, prefix:, :]
        side = int(math.sqrt(tensor.shape[1]))
        return tensor.reshape(tensor.shape[0], side, side, tensor.shape[2]).permute(0, 3, 1, 2)

    return reshape


def make_gradcam(model, root, test_frame, output, device):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    # Activation hooks require an ordinary forward graph rather than a
    # checkpoint-recomputed one.
    if hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(False)
    model.eval()
    row = test_frame[test_frame.label == 1].iloc[0]
    originals = []
    for column in MAP_COLUMNS:
        with Image.open(root / row[column]) as image:
            originals.append(image.convert("RGB"))
    reference = FourMapTransform(False)(originals).unsqueeze(0).to(device)
    output.mkdir(parents=True, exist_ok=True)
    reshape = cam_reshape(model) if model.encoder_name in {"swin_b", "dinov2", "retfound"} else None
    for index, (name, original) in enumerate(zip(MAP_NAMES, originals)):
        wrapper = CamWrapper(model, reference, index).to(device).eval()
        with GradCAM(model=wrapper, target_layers=[cam_layer(model)], reshape_transform=reshape) as cam:
            heatmap = cam(reference[:, index], [ClassifierOutputTarget(1)])[0]
        base = np.asarray(original.resize((224, 224)), dtype=np.float32) / 255.0
        Image.fromarray(show_cam_on_image(base, heatmap, use_rgb=True)).save(output / f"{name}.png")
        del wrapper


def train_model(name, root, parts, device, retfound_path):
    settings = MODEL_RUNS[name]
    output = OUTPUT_ROOT / "models" / name / f"fold_{FOLD}"
    output.mkdir(parents=True, exist_ok=True)
    train_frame, val_frame, test_frame = parts
    train_loader, val_loader, test_loader = loaders(root, parts, settings["batch"])
    model = MultiMapModel(name, retfound_path if name == "retfound" else None).to(device)
    # Reduce transformer activation memory on Kaggle GPUs.
    if name in {"swin_b", "dinov2", "retfound"} and hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(True)
    counts = train_frame.label.value_counts().reindex([0, 1], fill_value=1).to_numpy()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(len(train_frame) / (2 * counts), dtype=torch.float32, device=device)
    )
    model.freeze_backbone(FREEZE_EPOCHS > 0)
    optimizer = optimizer_for(model, FREEZE_EPOCHS == 0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_auc, stale, history = -1.0, 0, []
    checkpoint_path = output / "best_model.pth"

    for epoch in range(1, settings["epochs"] + 1):
        if epoch == FREEZE_EPOCHS + 1:
            model.freeze_backbone(False)
            optimizer = optimizer_for(model, True)
        started = time.time()
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, settings["accum"]
        )
        val_loss, val_metrics, val_rows = evaluate(model, val_loader, criterion, device)
        record = {
            "epoch": epoch, "seconds": time.time() - started,
            "train_loss": train_loss, "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        print(
            f"[{name}] {epoch:02d}/{settings['epochs']} val_auc={val_metrics['auroc']:.4f} "
            f"sens={val_metrics['sensitivity']:.3f} spec={val_metrics['specificity']:.3f} "
            f"({record['seconds']:.0f}s)"
        )
        if val_metrics["auroc"] > best_auc:
            best_auc, stale = val_metrics["auroc"], 0
            threshold = youden_threshold(val_rows)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {"encoder": name, "image_size": 224, "map_order": list(MAP_NAMES)},
                    "epoch": epoch, "decision_threshold": threshold,
                    "class_names": ["normal", "keratoconus"],
                    "validation_metrics": metrics(
                        [row["label"] for row in val_rows],
                        [row["probability"] for row in val_rows], threshold,
                    ),
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"[{name}] early stopping")
                break

    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    _, test_metrics, test_rows = evaluate(
        model, test_loader, criterion, device, saved["decision_threshold"], attention=True
    )
    pd.DataFrame(test_rows).to_csv(output / "test_predictions.csv", index=False)
    summary = {
        "encoder": name, "best_epoch": saved["epoch"],
        "validation": saved["validation_metrics"], "test": test_metrics,
        "checkpoint": str(checkpoint_path),
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    make_gradcam(model, root, test_frame, output / "gradcam", device)
    del model, optimizer, scaler, train_loader, val_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator before running this script")
    seed_everything()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    root, frame = discover_dataset()
    parts = split_manifest(frame)
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    summaries = []
    failures = []
    for index, name in enumerate(MODEL_RUNS, 1):
        print(f"\n{'=' * 78}\nMODEL {index}/5: {name}\n{'=' * 78}")
        try:
            # Resolve the 1.2 GB domain checkpoint only when its run begins, so
            # the first four experiments are retained if a network issue occurs.
            retfound_path = retfound_checkpoint() if name == "retfound" else None
            summaries.append(train_model(name, root, parts, device, retfound_path))
        except Exception as error:
            failures.append({"encoder": name, "error": repr(error)})
            warnings.warn(f"{name} failed: {error}")
            gc.collect()
            torch.cuda.empty_cache()
    if not summaries:
        raise RuntimeError(f"All models failed: {failures}")
    comparison = pd.DataFrame(
        [
            {
                "encoder": item["encoder"],
                "validation_auroc": item["validation"]["auroc"],
                **{f"test_{k}": v for k, v in item["test"].items()},
            }
            for item in summaries
        ]
    ).sort_values("validation_auroc", ascending=False)
    comparison.to_csv(OUTPUT_ROOT / "model_comparison.csv", index=False)
    (OUTPUT_ROOT / "run_summary.json").write_text(
        json.dumps({"models": summaries, "failures": failures}, indent=2)
    )
    archive = shutil.make_archive(
        "/kaggle/working/keratoconus_complete_results", "zip", OUTPUT_ROOT
    )
    print("\nFINAL MODEL COMPARISON\n", comparison.to_string(index=False))
    print("\nDownload this archive from Kaggle Output:", archive)
    if failures:
        print("Failures recorded in run_summary.json:", failures)


if __name__ == "__main__":
    main()
