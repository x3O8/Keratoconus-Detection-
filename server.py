"""KeratoScan: real clinical/imaging inference and the built React website."""
from __future__ import annotations

import base64
import csv as csv_module
import json
import logging
import math
import os
import threading
import warnings
from contextlib import asynccontextmanager
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from utils.clinical_rag import answer_question, decision_explanation, retrieve, vector_index
from utils.inference import (compute_attention_rollout, compute_gradcam,
    compute_integrated_gradients, ensemble_fusion, overlay_gradcam,
    run_image_inference, run_tabular_inference)
from utils.model_loader import (CLINICAL_FEATURES, FEATURE_METADATA, MODALITIES,
    MODALITY_LABELS, load_image_model, load_tabular_model)
from utils.preprocessor import preprocess_image_pil

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "frontend" / "dist"
MAX_FILE_BYTES = 10 * 1024 * 1024
DISCLAIMER = "Research use only. This model assessment is not a diagnosis or treatment recommendation and requires independent ophthalmologist review."
LOGGER = logging.getLogger("keratoscan")
INFERENCE_LOCK = threading.Lock()  # Hooks/gradients on the shared image model must not overlap.
load_dotenv(ROOT / ".env", override=False)


@asynccontextmanager
async def lifespan(application: FastAPI):
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    application.state.models = await run_in_threadpool(_load_models)
    yield
    application.state.models = {}


def _load_models():
    models = {}
    for kind, loader in [("clinical", load_tabular_model), ("image", load_image_model)]:
        try:
            values = loader()
            if kind == "clinical":
                model, ready, metadata = values
            else:
                model, device, ready, metadata = values
                models["device"] = device
            models[kind] = {"model": model, "ready": ready, "metadata": metadata,
                            "error": None if ready else "Model checkpoint is missing."}
        except Exception:
            LOGGER.exception("Could not load %s model", kind)
            models[kind] = {"model": None, "ready": False, "metadata": None,
                            "error": "Model could not be loaded. Check the server log and configured checkpoint."}
    return models


app = FastAPI(title="Inference service", version="2.0.0", lifespan=lifespan)


@app.middleware("http")
async def response_policy(request: Request, call_next):
    # Bound multipart requests before Starlette spools uploads to temporary files.
    content_length = request.headers.get("content-length")
    if content_length and (not content_length.isdigit() or int(content_length) > 85 * 1024 * 1024):
        return JSONResponse({"detail": "Request exceeds the 85 MB upload limit."}, status_code=413)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(_request, error):
    messages = [f'{".".join(str(part) for part in item["loc"][1:])}: {item["msg"]}' for item in error.errors()]
    return JSONResponse({"detail": "; ".join(messages)}, status_code=422)


@app.get("/api/status")
def status(request: Request):
    models = request.app.state.models
    public = lambda kind: {key: value for key, value in models[kind].items() if key != "model"}
    return {"api_version": app.version, "device": str(models.get("device", "cpu")),
            "clinical_model": public("clinical"), "image_model": public("image"),
            "clinical_features": [{"name": name, **FEATURE_METADATA[name]} for name in CLINICAL_FEATURES],
            "modalities": [{"key": key, "label": MODALITY_LABELS[key]} for key in MODALITIES],
            "default_threshold": (models["clinical"].get("metadata") or {}).get("validated_threshold") or .5,
            "research_use_only": True, "rag": vector_index().status()}


@app.get("/api/template.csv")
def template():
    return Response(",".join(CLINICAL_FEATURES) + "\n", media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="clinical-template.csv"'})


async def read_upload(upload: UploadFile):
    data = await upload.read(MAX_FILE_BYTES + 1)
    if not data or len(data) > MAX_FILE_BYTES:
        raise HTTPException(422, "Choose a non-empty file under 10 MB.")
    return data


def clinical_frame(clinical_json: str | None, csv_bytes: bytes | None):
    if clinical_json and csv_bytes is not None:
        raise HTTPException(422, "Supply either manual clinical measurements or one CSV, not both.")
    if not clinical_json and csv_bytes is None:
        return None
    try:
        if csv_bytes is not None:
            text = csv_bytes.decode("utf-8-sig")
            headers = next(csv_module.reader(StringIO(text)))
            if len(headers) != len(set(headers)):
                raise ValueError("CSV column names must be unique.")
            frame = pd.read_csv(StringIO(text))
            if len(frame) != 1:
                raise ValueError("CSV must contain exactly one patient row.")
            record = frame.iloc[0].to_dict()
        else:
            record = json.loads(clinical_json)
            if not isinstance(record, dict):
                raise ValueError("Clinical measurements must be a JSON object.")
        missing = set(CLINICAL_FEATURES) - record.keys()
        if missing:
            raise ValueError(f'Missing clinical fields: {", ".join(sorted(missing))}.')
        values = {}
        for name in CLINICAL_FEATURES:
            raw = record[name]
            if isinstance(raw, bool) or raw is None:
                raise ValueError(f"{FEATURE_METADATA[name]['label']} requires a numeric value.")
            value = float(raw)
            meta = FEATURE_METADATA[name]
            if not math.isfinite(value) or not meta["min"] <= value <= meta["max"]:
                raise ValueError(f"{meta['label']} must be from {meta['min']} to {meta['max']}.")
            if name == "gender" and value not in (0, 1):
                raise ValueError("Gender must be encoded as 0 or 1.")
            values[name] = value
        return pd.DataFrame([values], columns=CLINICAL_FEATURES)
    except (ValueError, TypeError, UnicodeError, StopIteration, pd.errors.ParserError) as error:
        raise HTTPException(422, str(error)) from error


def decode_image(data: bytes):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in ("PNG", "JPEG"):
                    raise ValueError("Only PNG and JPEG corneal maps are accepted.")
                if source.width * source.height > 20_000_000:
                    raise ValueError("Image exceeds 20 megapixels. Resize the map before uploading.")
                source.load()
                return source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise HTTPException(422, f"Cannot read the corneal map: {error}") from error


def image_url(pixels):
    buffer = BytesIO()
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def attribution_payload(originals, attributions):
    return {name: {"original": image_url(original), "overlay": image_url(overlay_gradcam(original, attributions[name])) if name in attributions else None}
            for name, original in originals.items()}


def assess(models, frame, image_bytes, threshold, image_weight, case_id, laterality):
    if frame is None and not image_bytes:
        raise HTTPException(422, "Enter all 12 clinical fields, import one CSV row, or upload a corneal map.")
    if frame is not None and not models["clinical"]["ready"]:
        raise HTTPException(503, "The clinical model is unavailable. Check the service status.")
    if image_bytes and not models["image"]["ready"]:
        raise HTTPException(503, "The imaging model is unavailable. Check the service status.")
    originals, tensors = {}, {}
    for name, content in image_bytes.items():
        decoded = decode_image(content)
        tensors[name] = preprocess_image_pil(decoded)
        originals[name] = np.asarray(decoded.resize((448, 448)))
    quality = []
    with INFERENCE_LOCK:
        clinical = run_tabular_inference(models["clinical"]["model"], frame, threshold) if frame is not None else None
        image = run_image_inference(models["image"]["model"], tensors, models["device"], threshold) if tensors else None
        gradcam, integrated, attention = {}, {}, None
        if tensors:
            model, device = models["image"]["model"], models["device"]
            gradcam = attribution_payload(originals, compute_gradcam(model, tensors, device))
            integrated = attribution_payload(originals, compute_integrated_gradients(model, tensors, device))
            attention = compute_attention_rollout(model, tensors, device)
    combined = ensemble_fusion(image, clinical, image_weight, threshold)
    if clinical:
        quality.append({"level": "ok", "message": "All 12 clinical measurements validated; no missing values substituted."})
    if image:
        quality.append({"level": "ok", "message": f"{len(tensors)} of 7 imaging modalities supplied."})
        if image["missing_modalities"]:
            quality.append({"level": "warning", "message": "Missing imaging modalities were zero-filled: " + ", ".join(image["missing_modalities"]) + "."})
    if combined["models_agree"] is False:
        quality.append({"level": "review", "message": "Clinical and imaging models disagree at the selected threshold."})
    quality.append({"level": "review", "message": "Model probabilities and explanations require independent clinical review."})
    return {"case": {"case_id": case_id or None, "laterality": laterality}, "combined": combined,
            "clinical": clinical, "image": image, "gradcam": gradcam,
            "integrated_gradients": integrated, "attention_rollout": attention,
            "clinical_decision_explanation": decision_explanation(combined, clinical, image),
            "retrieved_evidence": retrieve("keratoconus tomography clinical model interpretation", clinical),
            "data_quality": quality, "disclaimer": DISCLAIMER}


@app.post("/api/analyze")
async def analyze(request: Request, clinical_json: str | None = Form(None),
                  case_id: str = Form("", max_length=128), laterality: str = Form("OD", pattern="^(OD|OS)$"),
                  threshold: float | None = Form(None, gt=0, lt=1), image_weight: float = Form(.65, ge=0, le=1),
                  csv: UploadFile | None = File(None), CT_A: UploadFile | None = File(None),
                  EC_A: UploadFile | None = File(None), EC_P: UploadFile | None = File(None),
                  Elv_A: UploadFile | None = File(None), Elv_P: UploadFile | None = File(None),
                  Sag_A: UploadFile | None = File(None), Sag_P: UploadFile | None = File(None)):
    csv_bytes = await read_upload(csv) if csv else None
    frame = clinical_frame(clinical_json, csv_bytes)
    images = {name: await read_upload(upload) for name, upload in zip(MODALITIES, [CT_A, EC_A, EC_P, Elv_A, Elv_P, Sag_A, Sag_P]) if upload}
    selected_threshold = threshold if threshold is not None else status(request)["default_threshold"]
    try:
        return await run_in_threadpool(assess, request.app.state.models, frame, images, selected_threshold, image_weight, case_id.strip(), laterality)
    except HTTPException:
        raise
    except Exception as error:
        LOGGER.exception("Assessment could not be completed")
        raise HTTPException(500, "The models could not complete the assessment. Your inputs have been preserved; check the service log and retry.") from error


class ExplanationQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=2000)
    combined: dict
    clinical: dict | None = None
    image: dict | None = None


@app.post("/api/explain/query")
def explain_query(payload: ExplanationQuery):
    combined = payload.combined
    required = {"keratoconus_probability", "threshold", "screen_positive", "predicted_label"}
    if not required.issubset(combined):
        raise HTTPException(422, "Include the completed assessment context with your question.")
    try:
        return answer_question(payload.question.strip(), combined, payload.clinical, payload.image)
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(422, "The assessment context is incomplete. Run an assessment and retry.") from error


if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str):
    if path == "api" or path.startswith("api/"):
        raise HTTPException(404, "Unknown API endpoint.")
    candidate = (DIST / path).resolve()
    if candidate.is_relative_to(DIST.resolve()) and candidate.is_file():
        return FileResponse(candidate)
    if path and "." in Path(path).name:
        raise HTTPException(404, "File not found.")
    if not (DIST / "index.html").is_file():
        raise HTTPException(503, "Build the frontend first: npm --prefix frontend run build.")
    return FileResponse(DIST / "index.html")
