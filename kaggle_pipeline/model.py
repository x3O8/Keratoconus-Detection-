"""Shared-encoder, attention-fusion models for four corneal maps."""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
from torch import nn

ENCODER_SPECS = {
    "swin_b": {
        "timm_name": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        "image_size": 224,
    },
    "convnext_b": {
        "timm_name": "convnext_base.fb_in22k_ft_in1k",
        "image_size": 224,
    },
    "efficientnetv2_s": {
        "timm_name": "tf_efficientnetv2_s.in21k_ft_in1k",
        "image_size": 224,
    },
    "dinov2": {
        "timm_name": "vit_base_patch14_dinov2.lvd142m",
        "image_size": 224,
    },
    "retfound": {
        "timm_name": "vit_large_patch16_224",
        "image_size": 224,
    },
}


class MultiMapClassifier(nn.Module):
    """Encode each map with shared weights, then learn map-level attention."""

    def __init__(
        self,
        encoder_name: str,
        pretrained: bool = True,
        retfound_checkpoint: str | Path | None = None,
        num_maps: int = 4,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        import timm

        if encoder_name not in ENCODER_SPECS:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        spec = ENCODER_SPECS[encoder_name]
        # RETFound is a ViT architecture plus its domain-specific checkpoint.
        timm_pretrained = pretrained and encoder_name != "retfound"
        create_kwargs = {
            "pretrained": timm_pretrained,
            "num_classes": 0,
            "global_pool": "avg",
        }
        # timm's DINOv2-B checkpoint defaults to its 518 px pretraining crop.
        # Positional embeddings are safely resampled when explicitly using 224.
        if encoder_name == "dinov2":
            create_kwargs["img_size"] = spec["image_size"]
        self.backbone = timm.create_model(spec["timm_name"], **create_kwargs)
        if encoder_name == "retfound":
            if pretrained and not retfound_checkpoint:
                raise ValueError(
                    "RETFound requires --retfound-checkpoint pointing to the "
                    "official CFP pretrained .pth checkpoint added to Kaggle."
                )
            if retfound_checkpoint:
                _load_retfound_weights(self.backbone, Path(retfound_checkpoint))
        self.encoder_name = encoder_name
        self.num_maps = num_maps
        self.feature_dim = int(self.backbone.num_features)
        self.map_embeddings = nn.Parameter(torch.zeros(1, num_maps, self.feature_dim))
        nn.init.trunc_normal_(self.map_embeddings, std=0.02)
        self.attention = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, max(64, self.feature_dim // 4)),
            nn.GELU(),
            nn.Linear(max(64, self.feature_dim // 4), 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 2),
        )

    def forward(self, images: torch.Tensor, return_attention: bool = False):
        batch, maps, channels, height, width = images.shape
        if maps != self.num_maps:
            raise ValueError(f"Expected {self.num_maps} maps, received {maps}")
        features = self.backbone(images.reshape(batch * maps, channels, height, width))
        features = features.reshape(batch, maps, self.feature_dim)
        logits, weights = self.fuse_embeddings(features)
        return (logits, weights) if return_attention else logits

    def fuse_embeddings(self, features: torch.Tensor):
        features = features + self.map_embeddings
        weights = torch.softmax(self.attention(features).squeeze(-1), dim=1)
        fused = torch.sum(features * weights.unsqueeze(-1), dim=1)
        return self.classifier(fused), weights

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable


def build_model(
    encoder_name: str,
    pretrained: bool = True,
    retfound_checkpoint: str | Path | None = None,
    dropout: float = 0.30,
) -> MultiMapClassifier:
    return MultiMapClassifier(
        encoder_name=encoder_name,
        pretrained=pretrained,
        retfound_checkpoint=retfound_checkpoint,
        dropout=dropout,
    )


def load_exported_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(
        config["encoder"],
        pretrained=False,
        dropout=config.get("dropout", 0.30),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def _load_retfound_weights(backbone: nn.Module, checkpoint_path: Path) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RETFound checkpoint not found: {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = raw.get("model", raw.get("model_state_dict", raw))
    cleaned = {}
    target = backbone.state_dict()
    for key, value in state.items():
        for prefix in ("module.", "model.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key in target and target[key].shape == value.shape:
            cleaned[key] = value
    result = backbone.load_state_dict(cleaned, strict=False)
    coverage = len(cleaned) / max(1, len(target))
    if coverage < 0.70:
        raise RuntimeError(
            f"Only {coverage:.1%} of RETFound weights matched. Ensure this is the "
            "official CFP ViT-L/16 checkpoint compatible with timm vit_large_patch16_224."
        )
    warnings.warn(
        f"Loaded {len(cleaned)} RETFound tensors; ignored head/decoder tensors. "
        f"Missing encoder keys: {len(result.missing_keys)}"
    )
