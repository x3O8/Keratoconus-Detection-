"""Generate per-map Grad-CAM overlays for a saved model and eye sample."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from data import MAP_NAMES, SynchronizedTransform
from model import load_exported_model


class SingleMapWrapper(nn.Module):
    """Expose one map as input while keeping the other map embeddings fixed."""

    def __init__(self, model, reference_images, map_index):
        super().__init__()
        self.model = model
        self.map_index = map_index
        with torch.no_grad():
            batch, maps, channels, height, width = reference_images.shape
            features = model.backbone(
                reference_images.reshape(batch * maps, channels, height, width)
            )
            self.register_buffer(
                "fixed_features", features.reshape(batch, maps, model.feature_dim)
            )

    def forward(self, selected_image):
        features = self.fixed_features.repeat(selected_image.shape[0], 1, 1).clone()
        features[:, self.map_index] = self.model.backbone(selected_image)
        logits, _ = self.model.fuse_embeddings(features)
        return logits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eye-dir", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/gradcam")
    parser.add_argument("--target-class", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def target_layer(model):
    encoder = model.encoder_name
    if encoder == "swin_b":
        return model.backbone.layers[-1].blocks[-1].norm1
    if encoder == "convnext_b":
        return model.backbone.stages[-1].blocks[-1]
    if encoder == "efficientnetv2_s":
        return model.backbone.blocks[-1][-1]
    return model.backbone.blocks[-1].norm1


def transformer_reshape(model):
    prefix_tokens = int(getattr(model.backbone, "num_prefix_tokens", 1))

    def reshape(tensor):
        if tensor.ndim == 4:
            # Swin commonly returns BHWC.
            if tensor.shape[-1] == model.feature_dim:
                return tensor.permute(0, 3, 1, 2)
            return tensor
        tensor = tensor[:, prefix_tokens:, :]
        side = int(math.sqrt(tensor.shape[1]))
        if side * side != tensor.shape[1]:
            raise ValueError(f"Cannot reshape {tensor.shape[1]} patch tokens to a square")
        return tensor.reshape(tensor.shape[0], side, side, tensor.shape[2]).permute(
            0, 3, 1, 2
        )

    return reshape


def load_images(eye_dir: Path):
    images, paths = [], []
    for name in MAP_NAMES:
        matches = list(eye_dir.glob(f"*_{name.title()}.png"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one {name} map in {eye_dir}")
        paths.append(matches[0])
        with Image.open(matches[0]) as image:
            images.append(image.convert("RGB"))
    return images, paths


def main():
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_exported_model(args.checkpoint, device)
    images, paths = load_images(Path(args.eye_dir))
    size = checkpoint["config"].get("image_size", 224)
    tensor = SynchronizedTransform(size, training=False)(images).unsqueeze(0).to(device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reshape = (
        transformer_reshape(model)
        if model.encoder_name in {"swin_b", "dinov2", "retfound"}
        else None
    )

    for map_index, (name, image, source_path) in enumerate(
        zip(MAP_NAMES, images, paths)
    ):
        wrapper = SingleMapWrapper(model, tensor, map_index).to(device).eval()
        with GradCAM(
            model=wrapper,
            target_layers=[target_layer(model)],
            reshape_transform=reshape,
        ) as cam:
            grayscale = cam(
                input_tensor=tensor[:, map_index],
                targets=[ClassifierOutputTarget(args.target_class)],
            )[0]
        base = np.asarray(image.resize((size, size)), dtype=np.float32) / 255.0
        overlay = show_cam_on_image(base, grayscale, use_rgb=True)
        Image.fromarray(overlay).save(output / f"{name}_gradcam.png")
        image.save(output / f"{name}_original.png")
        print(f"Saved {name} Grad-CAM for {source_path.name}")


if __name__ == "__main__":
    main()

