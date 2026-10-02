"""Exercise the actual saved models through the production HTTP contract."""
import importlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = dict(zip([
    "age_years", "gender", "astig_value_d", "astig_axis_deg", "kmax_value_d", "kmax_axis_deg", "pachy_central_um", "pachy_thinnest_um", "pachy_thinnest_x", "pachy_thinnest_y", "asphericity_anterior", "asphericity_posterior"
], [27, 1, -5.2, 84, 53.1, 96, 472, 438, .8, -.9, -1.35, -1.62]))


def test_api_entrypoint_exists():
    assert (ROOT / "server.py").is_file(), "The production inference service entry point is missing"


@pytest.fixture(scope="module")
def client():
    server = importlib.import_module("server")
    with TestClient(server.app) as api:
        yield api


def test_status_has_real_models_and_schema(client):
    status = client.get("/api/status").json()
    assert status["clinical_model"]["ready"]
    assert status["image_model"]["ready"]
    assert len(status["clinical_features"]) == 12
    assert len(status["modalities"]) == 7
    assert status["default_threshold"] == pytest.approx(.4293)


def test_clinical_http_prediction_matches_saved_model(client):
    from utils.model_loader import load_tabular_model, CLINICAL_FEATURES
    model, _, _ = load_tabular_model()
    expected = model.predict_proba(pd.DataFrame([EXAMPLE])[CLINICAL_FEATURES])[0, 1]
    response = client.post("/api/analyze", data={"clinical_json": json.dumps(EXAMPLE), "case_id": "TEST-RESEARCH", "laterality": "OD", "threshold": ".4293"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["combined"]["method"] == "clinical_only"
    assert result["combined"]["keratoconus_probability"] == pytest.approx(expected)
    assert result["case"]["case_id"] == "TEST-RESEARCH"
    assert any(abs(item["contribution"]) > 0 for item in result["clinical"]["explanation"]["contributions"])
    assert result["retrieved_evidence"]
    assert result["image"] is None


@pytest.mark.parametrize("record", [{**EXAMPLE, "age_years": 150}, {**EXAMPLE, "gender": .5}, {**EXAMPLE, "kmax_value_d": float("nan")}, {name: value for name, value in EXAMPLE.items() if name != "kmax_value_d"}])
def test_invalid_measurements_are_rejected(client, record):
    response = client.post("/api/analyze", data={"clinical_json": json.dumps(record)})
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)


def test_csv_and_manual_produce_the_same_probability(client):
    csv = pd.DataFrame([EXAMPLE]).to_csv(index=False).encode()
    response = client.post("/api/analyze", files={"csv": ("clinical.csv", csv, "text/csv")})
    manual = client.post("/api/analyze", data={"clinical_json": json.dumps(EXAMPLE)})
    assert response.status_code == 200, response.text
    assert response.json()["clinical"]["keratoconus_probability"] == pytest.approx(manual.json()["clinical"]["keratoconus_probability"])


def test_multirow_csv_is_not_silently_truncated(client):
    csv = pd.DataFrame([EXAMPLE, EXAMPLE]).to_csv(index=False).encode()
    assert client.post("/api/analyze", files={"csv": ("clinical.csv", csv, "text/csv")}).status_code == 422


def test_corrupt_image_is_rejected(client):
    assert client.post("/api/analyze", files={"CT_A": ("map.png", b"not an image", "image/png")}).status_code == 422


def test_empty_request_and_ambiguous_clinical_input_are_rejected(client):
    assert client.post("/api/analyze").status_code == 422
    csv = pd.DataFrame([EXAMPLE]).to_csv(index=False).encode()
    assert client.post("/api/analyze", data={"clinical_json": json.dumps(EXAMPLE)}, files={"csv": ("clinical.csv", csv, "text/csv")}).status_code == 422


def test_real_imaging_and_fusion_explanations(client):
    # Synthetic pixels exercise the actual checkpoint, never patient material.
    buffer = io.BytesIO()
    grid = np.indices((224, 224)).sum(axis=0).astype(np.uint8)
    Image.fromarray(np.stack([grid, grid[::-1], grid.T], axis=-1)).save(buffer, format="PNG")
    response = client.post("/api/analyze", data={"clinical_json": json.dumps(EXAMPLE), "image_weight": ".65"}, files={"CT_A": ("synthetic-map.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 200, response.text
    result = response.json()
    expected = .65 * result["image"]["keratoconus_probability"] + .35 * result["clinical"]["keratoconus_probability"]
    assert result["combined"]["keratoconus_probability"] == pytest.approx(expected)
    assert result["combined"]["method"] == "weighted_fusion"
    assert len(result["image"]["missing_modalities"]) == 6
    assert result["gradcam"]["CT_A"]["overlay"].startswith("data:image/png;base64,")
    assert result["integrated_gradients"]["CT_A"]["overlay"].startswith("data:image/png;base64,")
    assert sum(result["attention_rollout"]["importance"].values()) == pytest.approx(1)


def test_grounded_explanation_works_without_an_external_key(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = client.post("/api/analyze", data={"clinical_json": json.dumps(EXAMPLE)}).json()
    response = client.post("/api/explain/query", json={"question": "Which inputs influenced this result?", "combined": result["combined"], "clinical": result["clinical"], "image": None})
    assert response.status_code == 200, response.text
    assert response.json()["sources"]
    assert response.json()["answer"]
    assert response.json()["generation"] == "deterministic grounded fallback"


def test_site_and_unknown_api_routes(client):
    assert client.get("/").status_code == 200
    assert "<html" in client.get("/").text
    assert client.get("/api/not-a-route").status_code == 404
    assert client.get("/api/template.csv").text.splitlines()[0].split(",") == list(EXAMPLE)
