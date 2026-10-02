# Routes

The current FastAPI app serves one static page and switches views in the browser. There is no React router yet.

| URL / view | Source | Layout | Purpose |
|---|---|---|---|
| `/` → `workspace` | `web/index.html` | Browser application shell | Upload corneal images and a clinical CSV, select fusion balance, start analysis |
| `/` → `results` | `web/index.html` | Browser application shell | Ensemble finding, individual model probabilities, modality attention, feature importance |
| `/api/status` | `server.py` | API | Device, model availability, feature schema, supported image modalities |
| `/api/template.csv` | `server.py` | API | Download clinical-feature template |
| `/api/analyze` | `server.py` | API | Multipart image + clinical inference and weighted fusion |

The requested React target should use tab state (or nested routes) for: Patient Data, Results, and Explainability.
