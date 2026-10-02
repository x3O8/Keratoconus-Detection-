"""Kaggle entrypoint for training and comparing all five vision encoders."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
from torch.utils.data import DataLoader

from data import OrbscanDataset, find_data_root, load_manifest, make_splits
from model import ENCODER_SPECS, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--encoder",
        choices=(*ENCODER_SPECS, "all"),
        default="efficientnetv2_s",
    )
    parser.add_argument("--output-dir", default="/kaggle/working/keratoconus_models")
    parser.add_argument("--retfound-checkpoint", default=None)
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def make_loaders(root, train_frame, val_frame, test_frame, image_size, args):
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    train = DataLoader(
        OrbscanDataset(train_frame, root, image_size, training=True),
        shuffle=True,
        drop_last=True,
        **common,
    )
    val = DataLoader(
        OrbscanDataset(val_frame, root, image_size, training=False),
        shuffle=False,
        **common,
    )
    test = DataLoader(
        OrbscanDataset(test_frame, root, image_size, training=False),
        shuffle=False,
        **common,
    )
    return train, val, test


def class_weight(frame: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = frame["label"].value_counts().reindex([0, 1], fill_value=1).to_numpy()
    weights = len(frame) / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_optimizer(model, args, backbone_trainable):
    if backbone_trainable:
        return torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": args.backbone_learning_rate},
                {
                    "params": [
                        *model.attention.parameters(),
                        *model.classifier.parameters(),
                        model.map_embeddings,
                    ],
                    "lr": args.learning_rate,
                },
            ],
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(
        [
            *model.attention.parameters(),
            *model.classifier.parameters(),
            model.map_embeddings,
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def train_epoch(model, loader, optimizer, criterion, device, scaler, args):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, labels_all, probabilities = 0.0, [], []
    amp_enabled = device.type == "cuda"
    for step, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with context:
            logits = model(images)
            loss = criterion(logits, labels) / args.grad_accumulation
        scaler.scale(loss).backward()
        if (step + 1) % args.grad_accumulation == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * args.grad_accumulation * len(labels)
        labels_all.extend(labels.detach().cpu().tolist())
        probabilities.extend(torch.softmax(logits.detach(), 1)[:, 1].cpu().tolist())
    return total_loss / len(loader.dataset), compute_metrics(labels_all, probabilities)


@torch.inference_mode()
def evaluate(model, loader, criterion, device, threshold=0.5, include_attention=False):
    model.eval()
    total_loss, labels_all, probabilities, rows = 0.0, [], [], []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        output = model(images, return_attention=include_attention)
        logits, attention = output if include_attention else (output, None)
        total_loss += criterion(logits, labels).item() * len(labels)
        probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        labels_np = labels.cpu().numpy()
        labels_all.extend(labels_np.tolist())
        probabilities.extend(probs.tolist())
        for index, sample_id in enumerate(batch["sample_id"]):
            row = {
                "sample_id": sample_id,
                "patient_id": batch["patient_id"][index],
                "eye": batch["eye"][index],
                "label": int(labels_np[index]),
                "probability": float(probs[index]),
            }
            if attention is not None:
                for map_index, name in enumerate(
                    ("anterior", "axial", "pachymetry", "posterior")
                ):
                    row[f"attention_{name}"] = float(attention[index, map_index].cpu())
            rows.append(row)
    return (
        total_loss / len(loader.dataset),
        compute_metrics(labels_all, probabilities, threshold),
        rows,
    )


def compute_metrics(labels, probabilities, threshold=0.5):
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def optimal_threshold(rows) -> float:
    labels = [row["label"] for row in rows]
    probabilities = [row["probability"] for row in rows]
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][np.argmax((tpr - fpr)[finite])])


def run_encoder(encoder, root, manifest, args, device):
    output = Path(args.output_dir) / encoder / f"fold_{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    train_frame, val_frame, test_frame = make_splits(manifest, args.fold, args.seed)
    spec = ENCODER_SPECS[encoder]
    loaders = make_loaders(
        root, train_frame, val_frame, test_frame, spec["image_size"], args
    )
    model = build_model(
        encoder,
        pretrained=args.pretrained,
        retfound_checkpoint=args.retfound_checkpoint,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight(train_frame, device))
    frozen = args.freeze_epochs > 0
    model.set_backbone_trainable(not frozen)
    optimizer = make_optimizer(model, args, not frozen)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, best_auc, stale = [], -1.0, 0
    best_path = output / "best_model.pth"
    config = {
        **vars(args),
        "encoder": encoder,
        "data_root": str(root),
        "image_size": spec["image_size"],
        "map_order": ["anterior", "axial", "pachymetry", "posterior"],
    }

    for epoch in range(1, args.epochs + 1):
        if frozen and epoch == args.freeze_epochs + 1:
            model.set_backbone_trainable(True)
            optimizer = make_optimizer(model, args, True)
            frozen = False
        started = time.time()
        train_loss, train_metrics = train_epoch(
            model, loaders[0], optimizer, criterion, device, scaler, args
        )
        val_loss, val_metrics, val_rows = evaluate(
            model, loaders[1], criterion, device
        )
        record = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 1),
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        print(
            f"[{encoder}] epoch {epoch:02d} "
            f"loss={val_loss:.4f} auc={val_metrics['auroc']:.4f} "
            f"sens={val_metrics['sensitivity']:.3f} spec={val_metrics['specificity']:.3f}"
        )
        if val_metrics["auroc"] > best_auc:
            best_auc, stale = val_metrics["auroc"], 0
            threshold = optimal_threshold(val_rows)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": compute_metrics(
                        [row["label"] for row in val_rows],
                        [row["probability"] for row in val_rows],
                        threshold,
                    ),
                    "decision_threshold": threshold,
                    "class_names": ["normal", "keratoconus"],
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = checkpoint["decision_threshold"]
    _, test_metrics, test_rows = evaluate(
        model, loaders[2], criterion, device, threshold, include_attention=True
    )
    pd.DataFrame(test_rows).to_csv(output / "test_predictions.csv", index=False)
    summary = {
        "encoder": encoder,
        "fold": args.fold,
        "best_epoch": checkpoint["epoch"],
        "validation": checkpoint["val_metrics"],
        "test": test_metrics,
        "samples": {
            "train": len(train_frame),
            "validation": len(val_frame),
            "test": len(test_frame),
        },
        "patients": {
            "train": train_frame["patient_id"].nunique(),
            "validation": val_frame["patient_id"].nunique(),
            "test": test_frame["patient_id"].nunique(),
        },
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    del model, loaders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main():
    args = parse_args()
    set_seed(args.seed)
    root = find_data_root(args.data_root)
    manifest = load_manifest(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {root} | samples={len(manifest)} | device={device}")
    encoders = list(ENCODER_SPECS) if args.encoder == "all" else [args.encoder]
    summaries = []
    for encoder in encoders:
        summaries.append(run_encoder(encoder, root, manifest, args, device))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"comparison_fold_{args.fold}.json").write_text(
        json.dumps(summaries, indent=2)
    )


if __name__ == "__main__":
    main()
