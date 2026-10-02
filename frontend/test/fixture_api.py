"""Local UI verification only. Synthetic responses are never used by the app.

Run this file, then `npm run dev -- --config test/vite.config.ts`.
First assessment returns a recoverable error; retry returns a labelled fixture.
"""
import ast
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
constants = {}
for node in ast.parse((ROOT / "utils/model_loader.py").read_text(encoding="utf-8")).body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"FEATURE_METADATA", "MODALITY_LABELS", "CLINICAL_FEATURES"}:
                constants[target.id] = ast.literal_eval(node.value)

status = {
    "api_version": "UI FIXTURE", "device": "test", "research_use_only": True,
    "clinical_model": {"ready": True, "metadata": {"type": "Fixture clinical (CatBoost)"}},
    "image_model": {"ready": False}, "default_threshold": 0.4293,
    "clinical_features": [{"name": name, **constants["FEATURE_METADATA"][name]} for name in constants["CLINICAL_FEATURES"]],
    "modalities": [{"key": key, "label": label} for key, label in constants["MODALITY_LABELS"].items()],
}
classification = {"keratoconus_probability": .72, "normal_probability": .28, "screen_positive": True, "predicted_label": "UI FIXTURE", "threshold": .4293}
result = {
    "case": {"case_id": "UI-TEST-FIXTURE", "laterality": "OD"},
    "combined": {**classification, "method": "clinical_only", "image_weight": 0, "clinical_weight": 1, "models_agree": None, "review_priority": "UI fixture for rendering verification. No model was run."},
    "clinical": {**classification, "feature_values": {}, "explanation": {"method": "fixture", "scale": "log_odds", "base_value": 0, "prediction_margin": .9, "summary": "Synthetic attribution for UI verification.", "contributions": [{"feature": "kmax_value_d", "label": "Kmax", "unit": "D", "value": 53.1, "contribution": .6, "direction": "toward_keratoconus"}], "global_importance": {}}},
    "image": None, "gradcam": {}, "integrated_gradients": {}, "attention_rollout": None,
    "clinical_decision_explanation": {"headline": "UI fixture — not a model output", "model_evidence": ["Synthetic values used to verify the screen."], "uncertainty": ["No inference was performed."], "clinical_context": "Frontend testing only.", "suggested_review": "Review real model outputs separately."},
    "retrieved_evidence": [], "data_quality": [{"level": "ok", "message": "UI fixture payload rendered successfully."}], "disclaimer": "UI FIXTURE — NOT A PATIENT ASSESSMENT OR MODEL OUTPUT.",
}

class Handler(BaseHTTPRequestHandler):
    attempts = 0
    def log_message(self, *_):
        pass
    def respond(self, body, code=200):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def do_GET(self):
        self.respond(status) if self.path == "/api/status" else self.respond({"detail": "Not found"}, 404)
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.path == "/api/analyze":
            Handler.attempts += 1
            time.sleep(1)
            if Handler.attempts == 1:
                self.respond({"detail": "Fixture validation error. Correct the inputs and retry."}, 422)
            else:
                self.respond(result)
        elif self.path == "/api/explain/query":
            self.respond({"answer": "Synthetic response confirms the explanation interface handles a completed request.", "sources": [], "generation": "UI fixture", "disclaimer": "Not a model output."})
        else:
            self.respond({"detail": "Not found"}, 404)

if __name__ == "__main__":
    print("UI fixture API on 127.0.0.1:18000. No model inference.", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 18000), Handler).serve_forever()
