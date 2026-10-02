# Page Dependency Trees

## `/` — Current browser dashboard

Entry: `web/index.html`

Dependencies:
- `web/styles.css`
- `web/app.js`
- `server.py`
  - `utils/model_loader.py`
  - `utils/inference.py`
  - `utils/preprocessor.py`

## Streamlit dashboard (legacy alternative entry)

Entry: `app.py`

Dependencies:
- `assets/style.css`
- `components/sidebar.py`
- `components/upload_tab.py`
  - `utils/preprocessor.py`
  - `utils/inference.py`
- `components/predictions_tab.py`
- `components/explainability_tab.py`
  - `utils/inference.py`
- `components/llm_tab.py`
  - `utils/gemini_client.py`
- `utils/model_loader.py`

## Requested React dashboard (new target)

No React source exists yet. It should reuse the current FastAPI inference contract and evolve it into distinct Patient Data, Results, and Explainability views.
