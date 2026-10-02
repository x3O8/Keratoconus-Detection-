"""Run one exported checkpoint on a patient eye folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from data import MAP_NAMES, SynchronizedTransform
from model import load_exported_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--eye-dir",
        required=True,
        help="Folder containing *Anterior.png, *Axial.png, *Pachymetry.png, *Posterior.png",
    )
    return parser.parse_args()


def load_eye(eye_dir: Path, image_size: int) -> torch.Tensor:
    images = []
    for map_name in MAP_NAMES:
        matches = list(eye_dir.glob(f"*_{map_name.title()}.png"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one *_{map_name.title()}.png in {eye_dir}; found {len(matches)}"
            )
        with Image.open(matches[0]) as image:
            images.append(image.convert("RGB"))
    return SynchronizedTransform(image_size, training=False)(images).unsqueeze(0)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_exported_model(args.checkpoint, device)
    tensor = load_eye(
        Path(args.eye_dir), checkpoint["config"].get("image_size", 224)
    ).to(device)
    with torch.inference_mode():
        logits, attention = model(tensor, return_attention=True)
        probability = torch.softmax(logits, dim=1)[0, 1].item()
    threshold = float(checkpoint.get("decision_threshold", 0.5))
    result = {
        "prediction": "keratoconus" if probability >= threshold else "normal",
        "keratoconus_probability": probability,
        "decision_threshold": threshold,
        "encoder": checkpoint["config"]["encoder"],
        "map_attention": {
            name: float(attention[0, index])
            for index, name in enumerate(MAP_NAMES)
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

