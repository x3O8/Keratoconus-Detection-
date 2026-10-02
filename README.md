# Running the website

A local research website with real clinical and corneal-imaging inference. The React interface and FastAPI service run together at **http://127.0.0.1:8000**.

## Start the website

From the project directory:

```powershell
python run.py
```

Leave the terminal running. Both saved models load at startup. The website reconnects automatically after a temporary service interruption.

For a fresh environment, install dependencies and build the interface first:

```powershell
python -m pip install -r requirements.txt
npm --prefix frontend ci
npm --prefix frontend run build
python run.py
```

For frontend development, keep `python run.py` running and use `npm --prefix frontend run dev` in a second terminal. Open http://127.0.0.1:5173. Vite proxies `/api` to the same real service on port 8000.

## What works

- Manual entry of twelve clinical measurements, or a UTF-8 CSV containing exactly one patient row. The CSV template is available in the assessment form and at `/api/template.csv`.
- PNG/JPEG uploads for the seven corneal imaging modalities, with file-size and image-content validation. Missing modalities are explicitly reported and zero-filled by the existing model preprocessing.
- Clinical-only, imaging-only, and weighted combined assessments; configurable threshold and imaging weight; independent model outputs and data-quality feedback.
- Actual clinical TreeSHAP contributions, imaging modality ablation, Grad-CAM, integrated gradients, and attention rollout.
- Explanations grounded in the assessment and the local citation library. Retrieval and deterministic answers work without an external API key. Optional Gemini generation uses `GEMINI_API_KEY` in `.env`.
- Printable assessment results and downloadable saved model-evaluation evidence. Assessment inputs remain available when navigating between views or retrying a failed request.

## Models and interpretation

The clinical service loads `models/catboost_final_pipeline.pkl`; its stored development threshold is **0.4293**. The imaging service loads the original EfficientNet-B0 checkpoint at `models/model1.pth`, recovered from this repository's Git history. The separate XGBoost and EfficientNetV2-S experiments in the overview are evaluation reports, not alternate inference selections.

Optional overrides are `KERATOSCAN_CLINICAL_MODEL` and `KERATOSCAN_IMAGE_MODEL`. Model artifacts must come from a trusted source because their serialization formats can execute code while loading.

The current environment uses Python 3.14 and scikit-learn 1.9.0; the clinical pipeline was serialized with scikit-learn 1.6.1. This emits a compatibility warning. Integration checks confirm that the API agrees with the saved pipeline's direct predictions in this environment; they do not establish equivalence with the original training environment. Reproducing published evaluation claims requires the training environment and independent model validation.

**Research use only.** Model probabilities are not a diagnosis. The interface preserves checkpoint, dataset, missing-input, and interpretation limitations alongside results.

## API and verification

`server.py` provides `/api/status`, `/api/template.csv`, `/api/analyze`, and `/api/explain/query`, plus the compiled website. The service binds to localhost by default. Submitted measurements and maps are not saved to a patient database; responses use `Cache-Control: no-store`. Without an external generation key, assessment and explanation processing remains local.

```powershell
python -m pip install -r requirements-dev.txt
python -X utf8 -m pytest tests/test_server.py -q -p no:cacheprovider
npm --prefix frontend run build
```

The integration suite exercises the actual model artifacts, direct-prediction parity, combined fusion, image attribution output, grounded explanations, invalid inputs, CSV handling, and production page serving. Browser verification is documented in `frontend/test/verification.md`.
