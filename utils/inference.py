"""Inference and patient-specific explainability for both production models."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from utils.model_loader import CLINICAL_FEATURES, FEATURE_METADATA, MODALITIES


def _classification(kc_probability: float, threshold: float) -> dict:
    positive = kc_probability >= threshold
    return {
        "keratoconus_probability": float(kc_probability),
        "normal_probability": float(1 - kc_probability),
        "screen_positive": positive,
        "predicted_label": "Keratoconus" if positive else "Normal / non-KC",
        "threshold": float(threshold),
    }


def _image_batch(image_tensors: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    zero = torch.zeros(1, 3, 224, 224, device=device)
    ordered = []
    for modality in MODALITIES:
        tensor = image_tensors.get(modality)
        ordered.append(tensor.to(device) if tensor is not None else zero)
    return torch.stack([tensor.squeeze(0) for tensor in ordered], dim=0).unsqueeze(0)


def run_image_inference(
    model,
    image_tensors: Dict[str, torch.Tensor],
    device: torch.device,
    threshold: float = 0.5,
) -> Dict:
    """Run the image model and one-pass modality ablation analysis.

    The image training notebook defines class index 0 as Keratoconus and class
    index 1 as Normal/Suspect. Keeping that mapping explicit prevents accidental
    probability inversion.
    """
    if model is None:
        raise RuntimeError("The image model is not available.")
    model.eval()
    full = _image_batch(image_tensors, device)
    uploaded = [modality for modality in MODALITIES if modality in image_tensors]

    variants = [full]
    for modality in uploaded:
        ablated = full.clone()
        ablated[:, MODALITIES.index(modality)] = 0
        variants.append(ablated)
    batch = torch.cat(variants, dim=0)

    with torch.inference_mode():
        probabilities = F.softmax(model(batch), dim=-1).cpu().numpy()

    kc_probability = float(probabilities[0, 0])
    ablation_impacts = {
        modality: float(kc_probability - probabilities[index + 1, 0])
        for index, modality in enumerate(uploaded)
    }
    gates = torch.sigmoid(model.attention_fusion.modality_gate).detach().cpu().numpy().reshape(-1)
    gates = gates / gates.sum()

    return {
        **_classification(kc_probability, threshold),
        "uploaded_modalities": uploaded,
        "missing_modalities": [modality for modality in MODALITIES if modality not in image_tensors],
        "input_completeness": len(uploaded) / len(MODALITIES),
        "modality_ablation": ablation_impacts,
        "learned_modality_gates": dict(zip(MODALITIES, gates.astype(float).tolist())),
    }


def run_tabular_inference(model, frame: pd.DataFrame, threshold: float = 0.5) -> Dict:
    if model is None:
        raise RuntimeError("The clinical model is not available.")
    ordered = frame[CLINICAL_FEATURES].astype(float)
    # The current clinical artifact is a scikit-learn Pipeline. Preserve the
    # feature names and order instead of converting to a raw NumPy array.
    probabilities = np.asarray(model.predict_proba(ordered))[0]
    positive_index = list(getattr(model, "classes_", [0, 1])).index(1)
    kc_probability = float(probabilities[positive_index])
    explanation = explain_clinical_model(model, ordered)
    return {
        **_classification(kc_probability, threshold),
        "feature_values": {name: float(ordered.iloc[0][name]) for name in CLINICAL_FEATURES},
        "explanation": explanation,
    }


def explain_clinical_model(model, frame: pd.DataFrame) -> dict:
    """Return TreeSHAP contributions for the supported clinical model artifact."""
    try:
        values = frame[CLINICAL_FEATURES].to_numpy(dtype=np.float32)
        if hasattr(model, "shap_contributions"):
            contributions = model.shap_contributions(values)[0]
        else:
            steps = getattr(model, "named_steps", {})
            estimator = steps.get("model", model)
            imputer = steps.get("imputer")
            transformed = imputer.transform(frame) if imputer is not None else values
            if estimator.__class__.__name__ == "CatBoostClassifier":
                from catboost import Pool

                contributions = estimator.get_feature_importance(
                    Pool(transformed, feature_names=CLINICAL_FEATURES),
                    type="ShapValues",
                )[0]
            else:
                import xgboost as xgb

                contributions = estimator.get_booster().predict(
                    xgb.DMatrix(transformed, feature_names=CLINICAL_FEATURES), pred_contribs=True
                )[0]
        feature_contributions = contributions[:-1]
        base_value = float(contributions[-1])
    except Exception:
        feature_contributions = np.zeros(len(CLINICAL_FEATURES), dtype=float)
        base_value = 0.0

    items = []
    for name, contribution in zip(CLINICAL_FEATURES, feature_contributions):
        meta = FEATURE_METADATA[name]
        items.append({
            "feature": name,
            "label": meta["label"],
            "unit": meta["unit"],
            "value": float(frame.iloc[0][name]),
            "contribution": float(contribution),
            "direction": "toward_keratoconus" if contribution > 0 else "toward_normal",
        })
    items.sort(key=lambda item: abs(item["contribution"]), reverse=True)

    positive = [item for item in items if item["contribution"] > 0][:3]
    protective = [item for item in items if item["contribution"] < 0][:2]
    if positive:
        positive_names = ", ".join(item["label"].lower() for item in positive)
        summary = f"The clinical model was pushed most toward keratoconus by {positive_names}."
    else:
        summary = "No clinical feature strongly pushed this prediction toward keratoconus."
    if protective:
        normal_names = ", ".join(item["label"].lower() for item in protective)
        summary += f" Factors pushing toward a normal result included {normal_names}."

    estimator = getattr(model, "named_steps", {}).get("model", model)
    global_importance = getattr(estimator, "feature_importances_", np.zeros(len(CLINICAL_FEATURES)))
    global_importance = np.asarray(global_importance, dtype=float)
    if global_importance.sum() > 0:
        global_importance = global_importance / global_importance.sum()

    return {
        "method": "TreeSHAP",
        "scale": "raw model margin (log-odds)",
        "base_value": base_value,
        "prediction_margin": float(base_value + feature_contributions.sum()),
        "summary": summary,
        "contributions": items,
        "global_importance": dict(zip(CLINICAL_FEATURES, global_importance.astype(float).tolist())),
    }


def ensemble_fusion(
    image_result: Dict | None,
    tabular_result: Dict | None,
    image_weight: float = 0.65,
    threshold: float = 0.5,
) -> Dict:
    if image_result is None and tabular_result is None:
        raise ValueError("At least one model result is required.")

    if image_result is not None and tabular_result is not None:
        image_weight = min(1.0, max(0.0, image_weight))
        clinical_weight = 1 - image_weight
        probability = (
            image_weight * image_result["keratoconus_probability"]
            + clinical_weight * tabular_result["keratoconus_probability"]
        )
        agreement = image_result["screen_positive"] == tabular_result["screen_positive"]
        method = "weighted_fusion"
    elif image_result is not None:
        probability = image_result["keratoconus_probability"]
        image_weight, clinical_weight = 1.0, 0.0
        agreement = None
        method = "image_only"
    else:
        probability = tabular_result["keratoconus_probability"]
        image_weight, clinical_weight = 0.0, 1.0
        agreement = None
        method = "clinical_only"

    return {
        **_classification(float(probability), threshold),
        "method": method,
        "image_weight": image_weight,
        "clinical_weight": clinical_weight,
        "models_agree": agreement,
        "review_priority": _review_priority(float(probability), threshold, agreement),
    }


def _review_priority(probability: float, threshold: float, agreement: bool | None) -> str:
    if agreement is False or abs(probability - threshold) < 0.1:
        return "Priority review — borderline or discordant model evidence"
    if probability >= threshold:
        return "Ophthalmologist review recommended — screen positive"
    return "Routine clinician review — screen negative"


def compute_gradcam(model, image_tensors: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, np.ndarray]:
    """Compute a patient-specific Grad-CAM for every uploaded modality."""
    if model is None or not image_tensors:
        return {}

    activations = None
    gradients = None

    def forward_hook(_module, _inputs, output):
        nonlocal activations
        activations = output

    def backward_hook(_module, _grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    target_layer = model.backbone[-1]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    inputs = _image_batch(image_tensors, device).requires_grad_(True)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        logits[0, 0].backward()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cams = F.relu((weights * activations).sum(dim=1)).detach().cpu().numpy()
        result = {}
        for index, modality in enumerate(MODALITIES):
            if modality not in image_tensors:
                continue
            cam = cams[index]
            minimum, maximum = float(cam.min()), float(cam.max())
            result[modality] = (cam - minimum) / (maximum - minimum + 1e-8)
        return result
    finally:
        forward_handle.remove()
        backward_handle.remove()


def compute_integrated_gradients(
    model,
    image_tensors: Dict[str, torch.Tensor],
    device: torch.device,
    steps: int = 8,
) -> Dict[str, np.ndarray]:
    """Integrated Gradients for the KC logit using a zero-map baseline.

    The compact step count keeps interactive latency manageable. Attribution is
    aggregated across RGB channels and normalized independently per modality.
    """
    if model is None or not image_tensors:
        return {}
    model.eval()
    inputs = _image_batch(image_tensors, device)
    baseline = torch.zeros_like(inputs)
    gradient_sum = torch.zeros_like(inputs)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)[1:]
    for alpha in alphas:
        interpolated = (baseline + alpha * (inputs - baseline)).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        kc_logit = model(interpolated)[0, 0]
        gradient = torch.autograd.grad(kc_logit, interpolated, retain_graph=False)[0]
        gradient_sum += gradient
    attribution = ((inputs - baseline) * gradient_sum / len(alphas)).abs().sum(dim=2)[0]
    result = {}
    for index, modality in enumerate(MODALITIES):
        if modality not in image_tensors:
            continue
        heat = attribution[index].detach().cpu().numpy()
        low, high = float(heat.min()), float(heat.max())
        result[modality] = (heat - low) / (high - low + 1e-8)
    return result


def compute_attention_rollout(model, image_tensors: Dict[str, torch.Tensor], device: torch.device) -> dict:
    """Expose the single-layer modality self-attention as a residual rollout."""
    if model is None or not image_tensors:
        return {"modalities": MODALITIES, "matrix": [], "importance": {}}
    model.eval()
    with torch.inference_mode():
        features = model.extract_features_batch(_image_batch(image_tensors, device))
        attention = model.attention_fusion
        normalized = attention.norm(features)
        batch, modalities, _ = normalized.shape

        def heads(value):
            return value.view(batch, modalities, attention.n_heads, attention.head_dim).permute(0, 2, 1, 3)

        queries = heads(attention.q_proj(normalized))
        keys = heads(attention.k_proj(normalized))
        weights = torch.softmax(torch.matmul(queries, keys.transpose(-2, -1)) * attention.scale, dim=-1)
        weights = weights.mean(dim=1)[0]
        rollout = weights + torch.eye(modalities, device=device)
        rollout = rollout / rollout.sum(dim=-1, keepdim=True)
        gates = torch.sigmoid(attention.modality_gate)[0, :, 0]
        importance = rollout.mean(dim=0) * gates
        uploaded_mask = torch.tensor([1.0 if name in image_tensors else 0.0 for name in MODALITIES], device=device)
        importance = importance * uploaded_mask
        importance = importance / (importance.sum() + 1e-8)
    return {
        "modalities": MODALITIES,
        "matrix": rollout.cpu().numpy().astype(float).tolist(),
        "importance": dict(zip(MODALITIES, importance.cpu().numpy().astype(float).tolist())),
        "note": "Single-layer residual attention rollout; attention weight is not causal importance.",
    }


def overlay_gradcam(original: np.ndarray, cam: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    height, width = original.shape[:2]
    resized = cv2.resize(cam, (width, height))
    heatmap = cv2.applyColorMap(np.uint8(resized * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * original + alpha * heatmap)
