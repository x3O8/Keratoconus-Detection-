"""Load the production image and clinical keratoconus models."""

from __future__ import annotations

import os
import json
from functools import lru_cache
from pathlib import Path

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0


ROOT = Path(__file__).resolve().parents[1]

MODALITIES = ["CT_A", "EC_A", "EC_P", "Elv_A", "Elv_P", "Sag_A", "Sag_P"]
MODALITY_LABELS = {
    "CT_A": "Corneal thickness — anterior",
    "EC_A": "Epithelial curvature — anterior",
    "EC_P": "Epithelial curvature — posterior",
    "Elv_A": "Elevation map — anterior",
    "Elv_P": "Elevation map — posterior",
    "Sag_A": "Sagittal curvature — anterior",
    "Sag_P": "Sagittal curvature — posterior",
}

# The serialized XGBoost estimator was trained from a NumPy matrix and therefore
# does not retain feature_names_in_. This is the exact order used by its training
# pipeline and the supplied clinical-data analysis.
CLINICAL_FEATURES = [
    "age_years",
    "gender",
    "astig_value_d",
    "astig_axis_deg",
    "kmax_value_d",
    "kmax_axis_deg",
    "pachy_central_um",
    "pachy_thinnest_um",
    "pachy_thinnest_x",
    "pachy_thinnest_y",
    "asphericity_anterior",
    "asphericity_posterior",
]

FEATURE_METADATA = {
    "age_years": {"label": "Age", "unit": "years", "group": "Demographics", "min": 1, "max": 100},
    "gender": {"label": "Gender", "unit": "0 = female, 1 = male", "group": "Demographics", "min": 0, "max": 1},
    "astig_value_d": {"label": "Astigmatism", "unit": "D", "group": "Keratometry", "min": -25, "max": 25},
    "astig_axis_deg": {"label": "Astigmatism axis", "unit": "°", "group": "Keratometry", "min": 0, "max": 180},
    "kmax_value_d": {"label": "Kmax", "unit": "D", "group": "Keratometry", "min": 20, "max": 100},
    "kmax_axis_deg": {"label": "Kmax axis", "unit": "°", "group": "Keratometry", "min": 0, "max": 180},
    "pachy_central_um": {"label": "Central pachymetry", "unit": "μm", "group": "Pachymetry", "min": 200, "max": 900},
    "pachy_thinnest_um": {"label": "Thinnest pachymetry", "unit": "μm", "group": "Pachymetry", "min": 200, "max": 900},
    "pachy_thinnest_x": {"label": "Thinnest-point X", "unit": "mm", "group": "Pachymetry", "min": -10, "max": 10},
    "pachy_thinnest_y": {"label": "Thinnest-point Y", "unit": "mm", "group": "Pachymetry", "min": -10, "max": 10},
    "asphericity_anterior": {"label": "Anterior asphericity", "unit": "Q", "group": "Asphericity", "min": -10, "max": 10},
    "asphericity_posterior": {"label": "Posterior asphericity", "unit": "Q", "group": "Asphericity", "min": -10, "max": 10},
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        mask = torch.rand((x.shape[0], x.shape[1], 1), dtype=x.dtype, device=x.device)
        return x * torch.floor(mask + keep_prob) / keep_prob


class MultiHeadModalityAttention(nn.Module):
    def __init__(self, feature_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = feature_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.v_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)
        self.modality_gate = nn.Parameter(torch.ones(1, len(MODALITIES), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, modalities, dimensions = x.shape
        normalized = self.norm(x)

        def split_heads(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch, modalities, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        queries = split_heads(self.q_proj(normalized))
        keys = split_heads(self.k_proj(normalized))
        values = split_heads(self.v_proj(normalized))
        attention = F.softmax(torch.matmul(queries, keys.transpose(-2, -1)) * self.scale, dim=-1)
        attention = self.attn_drop(attention)
        output = torch.matmul(attention, values)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch, modalities, dimensions)
        output = x + self.out_proj(output)
        return (output * torch.sigmoid(self.modality_gate)).mean(dim=1)


class KeraEfficientNet(nn.Module):
    """Architecture used to train ``model1.pth`` in ``model1.ipynb``."""

    def __init__(self, num_classes: int = 2, proj_dim: int = 320):
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        self.feature_dim = backbone.classifier[1].in_features
        self.backbone = backbone.features
        self.backbone_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.drop_path = DropPath(0.1)
        self.attention_fusion = MultiHeadModalityAttention(proj_dim, n_heads=4, dropout=0.1)
        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(proj_dim // 2, num_classes),
        )

    def extract_features_batch(self, images: torch.Tensor) -> torch.Tensor:
        batch, modalities, channels, height, width = images.shape
        flattened = images.view(batch * modalities, channels, height, width)
        features = self.backbone(flattened)
        features = self.backbone_pool(features).flatten(1)
        return self.proj(features).view(batch, modalities, -1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.drop_path(self.extract_features_batch(images))
        return self.head(self.attention_fusion(features))


def _configured_path(environment_name: str, fallbacks: list[Path]) -> Path | None:
    configured = os.environ.get(environment_name)
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(fallbacks)
    return next((path.resolve() for path in candidates if path.exists()), None)


@lru_cache(maxsize=1)
def load_image_model():
    device = get_device()
    path = _configured_path(
        "KERATOSCAN_IMAGE_MODEL",
        [ROOT / "model1.pth", ROOT / "models" / "model1.pth"],
    )
    if path is None:
        return None, device, False, None

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    model = KeraEfficientNet().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    metadata = {
        "file": path.name,
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "validation_accuracy": checkpoint.get("val_acc") if isinstance(checkpoint, dict) else None,
        "validation_f1": checkpoint.get("val_f1") if isinstance(checkpoint, dict) else None,
    }
    return model, device, True, metadata


@lru_cache(maxsize=1)
def load_tabular_model():
    path = _configured_path(
        "KERATOSCAN_CLINICAL_MODEL",
        [
            ROOT / "models" / "catboost_final_pipeline.pkl",
            ROOT / "journal_xgb_results" / "xgb_journal_model.pkl",
            ROOT / "models" / "xgb_final.pkl",
            ROOT / "xgb_final.pkl",
        ],
    )
    if path is None:
        return None, False, None

    model = joblib.load(path)
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Clinical model {path.name} does not implement predict_proba().")
    expected = getattr(model, "n_features_in_", len(CLINICAL_FEATURES))
    if expected != len(CLINICAL_FEATURES):
        raise ValueError(f"Clinical model expects {expected} features; configured schema has {len(CLINICAL_FEATURES)}.")
    sidecar_path = path.with_suffix(".metadata.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else {}
    return model, True, {
        "file": path.name,
        "type": sidecar.get("model_type", type(model).__name__),
        "features": expected,
        "validated_threshold": sidecar.get("validated_threshold", getattr(model, "threshold_", None)),
        "training_metadata": sidecar.get("training_metadata", getattr(model, "metadata_", None)),
    }
