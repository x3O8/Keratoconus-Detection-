# Keratoconus image-model training on Kaggle

## One-click complete run

For the requested single-entry workflow, use
`unified_kaggle_pipeline.py`. Upload either the labelled dataset **or the
original ORBSCAN dataset** (`ORBSCAN_Dataset/` plus
`clinical_data_and_labels.csv`), enable a Kaggle GPU and Internet, paste or add
the script to the notebook, and run it once. It
automatically:

- installs `timm`, `grad-cam`, and `gdown` if absent;
- discovers and validates the labelled dataset or builds an in-memory manifest
  directly from the original ORBSCAN layout;
- trains EfficientNetV2-S, ConvNeXt-B, Swin-B, DINOv2, and RETFound;
- uses patient-disjoint train, validation, and test partitions;
- downloads the public official RETFound CFP weights if they were not attached;
- saves best checkpoints, histories, predictions, metrics, and Grad-CAM maps;
- writes a validation-ranked model comparison; and
- creates `/kaggle/working/keratoconus_complete_results.zip`.

No command-line arguments or additional training commands are required. The
longer instructions below remain useful for running just one architecture.

This pipeline treats one eye as one sample. Each sample contains four ORBSCAN
maps in this fixed order:

1. Anterior
2. Axial
3. Pachymetry
4. Posterior

A shared vision backbone encodes all four maps. A learned attention layer fuses
the four embeddings and returns both the binary prediction and map importance.
The five supported encoders are:

- `swin_b`
- `convnext_b`
- `efficientnetv2_s`
- `dinov2`
- `retfound`

## What to upload

Upload `keratoconus_better/labeled_dataset` as a private Kaggle Dataset. It
already contains:

- `normal/<patient>/<eye>/*.png`
- `keratoconus/<patient>/<eye>/*.png`
- `metadata/eye_manifest.csv`
- `metadata/patient_manifest.csv`
- `metadata/dataset_report.json`

Upload the `kaggle_pipeline` folder as a second small Kaggle Dataset, or paste it
into the notebook working directory.

## Kaggle notebook setup

Enable a GPU and Internet in notebook settings. Then run:

```python
!pip install -q timm grad-cam
```

Locate the two attached datasets:

```python
from pathlib import Path

manifest = next(Path("/kaggle/input").glob("**/metadata/eye_manifest.csv"))
DATA_ROOT = manifest.parent.parent
PIPELINE = next(Path("/kaggle/input").glob("**/kaggle_pipeline/train.py")).parent
print("Data:", DATA_ROOT)
print("Code:", PIPELINE)
```

Train one model:

```python
!cd {PIPELINE} && python train.py \
  --data-root "{DATA_ROOT}" \
  --encoder efficientnetv2_s \
  --output-dir /kaggle/working/keratoconus_models \
  --epochs 25 \
  --batch-size 4 \
  --grad-accumulation 2 \
  --fold 1
```

Train the other ImageNet/DINOv2 models by changing `--encoder` to `swin_b`,
`convnext_b`, or `dinov2`. `--encoder all` runs every model sequentially, but
separate Kaggle sessions are safer because the complete comparison is
compute-intensive.

## RETFound

RETFound is not a generic timm pretrained model. Download the official
`RETFound_cfp_weights.pth` checkpoint from the
[official RETFound repository](https://github.com/rmaphoh/RETFound), upload it
as a private Kaggle Dataset, and pass its location:

```python
RETF_CHECKPOINT = next(
    Path("/kaggle/input").glob("**/RETFound_cfp_weights.pth")
)

!cd {PIPELINE} && python train.py \
  --data-root "{DATA_ROOT}" \
  --encoder retfound \
  --retfound-checkpoint "{RETF_CHECKPOINT}" \
  --output-dir /kaggle/working/keratoconus_models \
  --epochs 20 \
  --batch-size 1 \
  --grad-accumulation 8 \
  --fold 1
```

RETFound uses a roughly 300M-parameter ViT-L/16 encoder. On a 16 GB Kaggle GPU,
start with batch size 1. Do not run RETFound without its official checkpoint
unless you intentionally want a randomly initialized ViT-L baseline.

## Outputs

Each run creates:

```text
keratoconus_models/<encoder>/fold_<n>/
├── best_model.pth
├── history.csv
├── metrics.json
└── test_predictions.csv
```

`best_model.pth` includes model weights, preprocessing/map order, the validation
decision threshold, class names, and validation metrics. Download this file for
dashboard integration.

The provided CSV's test split is held out until the final evaluation. Validation
uses the supplied patient-level fold. Because both eyes from a patient remain
in the same split, the pipeline avoids left/right-eye leakage.

## Generate Grad-CAM

After training, select any eye folder from the labelled dataset:

```python
CHECKPOINT = "/kaggle/working/keratoconus_models/efficientnetv2_s/fold_1/best_model.pth"
EYE_DIR = DATA_ROOT / "keratoconus" / "181ZN" / "OD"

!cd {PIPELINE} && python explain.py \
  --checkpoint "{CHECKPOINT}" \
  --eye-dir "{EYE_DIR}" \
  --output-dir /kaggle/working/gradcam
```

This writes original and Grad-CAM overlay PNGs for all four maps.

## Run inference

```python
!cd {PIPELINE} && python predict.py \
  --checkpoint "{CHECKPOINT}" \
  --eye-dir "{EYE_DIR}"
```

## Recommended experiment protocol

- Start with `efficientnetv2_s` to validate the full pipeline quickly.
- Train all five backbones on the same fold and compare test AUROC,
  sensitivity, specificity, and F1 from `metrics.json`.
- Once the pipeline is stable, repeat folds 1–5 and report mean plus standard
  deviation. Do not select a model based on the held-out test set.
- Save the highest-validation-AUROC checkpoint from every run.
- Review Grad-CAM maps with an ophthalmologist. Attention and Grad-CAM indicate
  model influence, not clinical causality.

This is research code, not a medical device or clinical diagnostic system.

## Fine-tuning on the independent seven-map dataset

Use `unified_external_finetune_kaggle.py` when the Kaggle inputs contain:

- the MobileNetV2 Kera PyTorch model package mounted at
  `/kaggle/input/models/eatradium/mobilenet-v2-kera/pytorch/default/1`; and
- a dataset root with `Normal`, `Keratoconus`, and `Suspect` folders, where
  every case contains `CT_A`, `EC_A`, `EC_P`, `Elv_A`, `Elv_P`, `Sag_A`, and
  `Sag_P` images.

Upload the complete archive, including both `Train_Validation sets` and
`Independent Test Set`. The script auto-discovers them and the checkpoint,
uses all seven maps, performs five-fold out-of-fold calibration strictly on
the training/validation partition, evaluates the supplied independent set once,
and exports a practical single `deployment_model.pth`. `Suspect` cases are
scored only after training and are never treated as ground-truth Normal or
Keratoconus examples.
