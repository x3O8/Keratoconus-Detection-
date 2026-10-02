"""Notebook-ready visual report for the seven-map fine-tuning run.

Paste this file's contents into one Kaggle notebook cell after the training
cell. It reads the result CSVs already written by unified_external_finetune_kaggle.py
and writes PNG/CSV report files beside them.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve


RESULTS_DIR = Path("/kaggle/working/keratoconus_external_finetune")
ENSEMBLE_FILE = RESULTS_DIR / "independent_test_predictions_ensemble.csv"
DEPLOYMENT_FILE = RESULTS_DIR / "independent_test_predictions_deployment.csv"
PLOT_DIR = RESULTS_DIR / "visualizations"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

if not ENSEMBLE_FILE.is_file() or not DEPLOYMENT_FILE.is_file():
    raise FileNotFoundError(
        "Run the training cell first. Expected:\n"
        f"{ENSEMBLE_FILE}\n{DEPLOYMENT_FILE}"
    )

ensemble = pd.read_csv(ENSEMBLE_FILE)
deployment = pd.read_csv(DEPLOYMENT_FILE)
MODELS = {"Five-fold ensemble": ensemble, "Single deployment model": deployment}


def ece(labels, probabilities, bins=10):
    """Expected calibration error; lower is better."""
    edges = np.linspace(0, 1, bins + 1)
    total = len(labels)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= left) & (probabilities < right)
        if right == 1:
            mask = (probabilities >= left) & (probabilities <= right)
        if mask.any():
            value += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(value)


def metrics_row(name, frame):
    labels = frame.label.to_numpy()
    predictions = frame.prediction.to_numpy()
    probabilities = frame.calibrated_probability.to_numpy()
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "model": name,
        "AUROC": auc(*roc_curve(labels, probabilities)[:2]),
        "accuracy": (predictions == labels).mean(),
        "sensitivity": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "false_positives": fp,
        "false_negatives": fn,
        "ECE": ece(labels, probabilities),
        "threshold": frame.threshold.iloc[0] if "threshold" in frame else np.nan,
    }


metrics = pd.DataFrame([metrics_row(name, frame) for name, frame in MODELS.items()])
metrics.to_csv(PLOT_DIR / "model_metrics.csv", index=False)
print("Independent test metrics")
display(metrics.style.format({"AUROC": "{:.3f}", "accuracy": "{:.1%}", "sensitivity": "{:.1%}", "specificity": "{:.1%}", "ECE": "{:.3f}", "threshold": "{:.3f}"}))


sns.set_theme(style="whitegrid", context="notebook")
fig, axes = plt.subplots(2, 3, figsize=(19, 11), constrained_layout=True)

# 1. ROC
ax = axes[0, 0]
for name, frame in MODELS.items():
    fpr, tpr, _ = roc_curve(frame.label, frame.calibrated_probability)
    ax.plot(fpr, tpr, linewidth=2.5, label=f"{name} (AUROC {auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
ax.set(title="Independent-test ROC", xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
ax.legend(loc="lower right")

# 2-3. Confusion matrices
for ax, (name, frame) in zip(axes[0, 1:3], MODELS.items()):
    cm = confusion_matrix(frame.label, frame.prediction, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Keratoconus"]).plot(
        ax=ax, colorbar=False, cmap="Blues", values_format="d"
    )
    ax.set_title(name)

# 4. Calibrated score distributions
ax = axes[1, 0]
for name, frame in MODELS.items():
    for label, linestyle in ((0, "--"), (1, "-")):
        values = frame.loc[frame.label == label, "calibrated_probability"]
        ax.hist(values, bins=np.linspace(0, 1, 21), density=True, histtype="step", linewidth=2,
                linestyle=linestyle, label=f"{name}: {'KCN' if label else 'Normal'}")
threshold = ensemble.threshold.iloc[0] if "threshold" in ensemble else np.nan
if np.isfinite(threshold):
    ax.axvline(threshold, color="black", linewidth=1.5, label=f"Decision threshold ({threshold:.3f})")
ax.set(title="Independent-test score distribution", xlabel="Calibrated keratoconus score", ylabel="Density", xlim=(0, 1))
ax.legend(fontsize=8)

# 5. Reliability / calibration
ax = axes[1, 1]
for name, frame in MODELS.items():
    observed, predicted = calibration_curve(frame.label, frame.calibrated_probability, n_bins=8, strategy="quantile")
    ax.plot(predicted, observed, marker="o", linewidth=2, label=f"{name} (ECE {ece(frame.label.to_numpy(), frame.calibrated_probability.to_numpy()):.3f})")
ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
ax.set(title="Calibration reliability", xlabel="Mean predicted probability", ylabel="Observed KCN frequency", xlim=(0, 1), ylim=(0, 1))
ax.legend(fontsize=8)

# 6. Mean map attention by independent-test class, using the selected deployment model.
ax = axes[1, 2]
attention_columns = [column for column in deployment.columns if column.startswith("attention_")]
if attention_columns:
    attention = deployment.groupby("label")[attention_columns].mean().T
    attention.index = attention.index.str.replace("attention_", "", regex=False)
    attention.rename(columns={0: "Normal", 1: "Keratoconus"}).plot.bar(ax=ax, color=["#4C78A8", "#E45756"])
    ax.set(title="Mean map attention", xlabel="Map", ylabel="Attention weight")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="True class")
else:
    ax.text(0.5, 0.5, "No attention columns found", ha="center", va="center")
    ax.axis("off")

fig.suptitle("Seven-map keratoconus classifier: independent-test report", fontsize=16, y=1.02)
fig.savefig(PLOT_DIR / "independent_test_report.png", dpi=220, bbox_inches="tight")
plt.show()


# Error cases for clinical/Grad-CAM review.
errors = []
for name, frame in MODELS.items():
    subset = frame.loc[frame.label.ne(frame.prediction)].copy()
    subset.insert(0, "model", name)
    errors.append(subset)
errors = pd.concat(errors, ignore_index=True)
errors.to_csv(PLOT_DIR / "independent_test_errors.csv", index=False)
print("\nCases to review with Grad-CAM")
display(errors[["model", "sample_id", "source_class", "label", "calibrated_probability", "prediction", "triage"]].sort_values(["model", "label", "sample_id"]))


# Triage distribution is useful only if the run generated a genuine review band.
if "triage" in deployment:
    triage = pd.crosstab(deployment.triage, deployment.label).reindex(
        ["low_risk", "clinician_review", "high_risk"], fill_value=0
    )
    print("\nDeployment-model triage distribution")
    display(triage.rename(columns={0: "Normal", 1: "Keratoconus"}))

print(f"\nSaved visuals and review CSVs to: {PLOT_DIR}")
