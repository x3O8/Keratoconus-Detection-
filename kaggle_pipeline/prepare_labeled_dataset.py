"""Build a class-labelled, Kaggle-ready image dataset.

The source CSV may contain repeated clinical measurements for an eye. Images
are unique per patient/eye, so the generated eye manifest contains one row per
eye while the original clinical CSV is copied unchanged for provenance.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

MAP_NAMES = ("Anterior", "Axial", "Pachymetry", "Posterior")
CLASS_NAMES = {0: "normal", 1: "keratoconus"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("keratoconus_better"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("keratoconus_better/labeled_dataset"),
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="Use copy for a portable Kaggle upload. Hardlink saves local disk space.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_patient_id(csv_id: str, available: set[str]) -> str | None:
    """Resolve exact IDs and the known spreadsheet-style '+' corruption."""
    value = str(csv_id).strip()
    if value in available:
        return value
    candidates = {
        value.replace("+", ""),
        value.replace("E+", "E"),
        value.replace("e+", "e"),
    }
    matches = sorted(candidates & available)
    return matches[0] if len(matches) == 1 else None


def transfer(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        destination.hardlink_to(source.resolve())


def build_dataset(source: Path, output: Path, mode: str, overwrite: bool) -> dict:
    images_root = source / "ORBSCAN_Dataset"
    labels_csv = source / "clinical_data_and_labels.csv"
    if not images_root.is_dir() or not labels_csv.is_file():
        raise FileNotFoundError(
            "Expected ORBSCAN_Dataset/ and clinical_data_and_labels.csv under "
            f"{source.resolve()}"
        )
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to rebuild it")
        shutil.rmtree(output)

    raw = pd.read_csv(labels_csv, dtype={"patient_code": str})
    required = {"patient_code", "eye", "label"}
    missing_columns = required - set(raw.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing columns: {sorted(missing_columns)}")
    raw["patient_code"] = raw["patient_code"].str.strip()
    raw["eye"] = raw["eye"].str.strip().str.upper()
    raw["label"] = pd.to_numeric(raw["label"], errors="raise").astype(int)
    if not set(raw["label"]).issubset(CLASS_NAMES):
        raise ValueError("Labels must contain only 0 and 1")

    available = {path.name for path in images_root.iterdir() if path.is_dir()}
    id_map = {
        patient: resolve_patient_id(patient, available)
        for patient in raw["patient_code"].unique()
    }
    unresolved = sorted(patient for patient, resolved in id_map.items() if resolved is None)
    if unresolved:
        raise ValueError(f"CSV patient IDs not found on disk: {unresolved[:20]}")
    raw["resolved_patient_code"] = raw["patient_code"].map(id_map)

    label_counts = raw.groupby("resolved_patient_code")["label"].nunique()
    discordant = label_counts[label_counts > 1]
    if not discordant.empty:
        raise ValueError(f"Patients with conflicting labels: {discordant.index.tolist()[:20]}")

    # One image sample per eye; repeated clinical measurements are not repeated.
    key_columns = ["resolved_patient_code", "eye"]
    eyes = raw.sort_values(key_columns).drop_duplicates(key_columns, keep="first").copy()
    patient_rows = []
    eye_rows = []
    copied = 0
    skipped_missing_eye = []

    for patient_code, patient_frame in eyes.groupby("resolved_patient_code", sort=True):
        label = int(patient_frame["label"].iloc[0])
        class_name = CLASS_NAMES[label]
        split = (
            str(patient_frame["split_80_20"].iloc[0])
            if "split_80_20" in patient_frame
            else "unspecified"
        )
        fold_value = patient_frame["fold"].iloc[0] if "fold" in patient_frame else None
        fold = None if pd.isna(fold_value) else int(fold_value)
        patient_rows.append(
            {
                "patient_id": patient_code,
                "label": label,
                "class_name": class_name,
                "split": split,
                "fold": fold,
            }
        )

        for row in patient_frame.itertuples():
            eye = row.eye
            eye_dir = images_root / patient_code / eye
            if not eye_dir.is_dir():
                skipped_missing_eye.append(f"{patient_code}/{eye}")
                continue
            paths = {}
            missing_maps = []
            for map_name in MAP_NAMES:
                source_image = eye_dir / f"{patient_code}_{eye}_{map_name}.png"
                if not source_image.is_file():
                    missing_maps.append(map_name)
                    continue
                relative = Path(class_name) / patient_code / eye / source_image.name
                transfer(source_image, output / relative, mode)
                paths[map_name] = relative.as_posix()
                copied += 1
            if missing_maps:
                raise ValueError(
                    f"Incomplete map set for {patient_code}/{eye}: {missing_maps}"
                )
            eye_rows.append(
                {
                    "patient_id": patient_code,
                    "eye": eye,
                    "sample_id": f"{patient_code}_{eye}",
                    "label": label,
                    "class_name": class_name,
                    "split": split,
                    "fold": fold,
                    **{f"{name.lower()}_path": paths[name] for name in MAP_NAMES},
                }
            )

    metadata = output / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    eye_manifest = pd.DataFrame(eye_rows)
    patient_manifest = pd.DataFrame(patient_rows).drop_duplicates("patient_id")
    eye_manifest.to_csv(metadata / "eye_manifest.csv", index=False)
    patient_manifest.to_csv(metadata / "patient_manifest.csv", index=False)
    shutil.copy2(labels_csv, metadata / "clinical_data_and_labels_original.csv")

    report = {
        "patients": int(patient_manifest["patient_id"].nunique()),
        "eye_samples": len(eye_manifest),
        "images": copied,
        "class_counts_patients": {
            CLASS_NAMES[int(k)]: int(v)
            for k, v in patient_manifest["label"].value_counts().sort_index().items()
        },
        "class_counts_eyes": {
            CLASS_NAMES[int(k)]: int(v)
            for k, v in eye_manifest["label"].value_counts().sort_index().items()
        },
        "split_counts_eyes": dict(Counter(eye_manifest["split"])),
        "skipped_missing_eye_folders": skipped_missing_eye,
        "resolved_id_aliases": {
            source_id: resolved
            for source_id, resolved in id_map.items()
            if source_id != resolved
        },
        "duplicate_csv_patient_eye_rows_removed_from_image_manifest": int(
            len(raw) - len(eyes)
        ),
        "transfer_mode": mode,
    }
    (metadata / "dataset_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    report = build_dataset(args.source, args.output, args.mode, args.overwrite)
    print(json.dumps(report, indent=2))
    print(f"\nKaggle-ready dataset created at: {args.output.resolve()}")


if __name__ == "__main__":
    main()
