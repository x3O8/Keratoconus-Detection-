import { useState } from "react";
import {
  ArrowRight,
  ChevronDown,
  Download,
  FileSliders,
  ScanEye,
} from "lucide-react";
import { clinicalMetrics, evaluations } from "../data/catalog";
import type { StatusResponse } from "../types";
import { Topography } from "./Topography";

export function Overview({
  status,
  onStart,
  onExport,
}: {
  status: StatusResponse | null;
  onStart: () => void;
  onExport: () => void;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);
  return (
    <div className="tab-page overview-page">
      <section
        className="overview-hero landing-hero"
        aria-labelledby="landing-title"
      >
        <div className="hero-copy">
          <h1 id="landing-title">
            Small details.
            <br />
            Deeper <em>insight.</em>
          </h1>
          <p>Clinical measurements and corneal imaging in one workspace.</p>
          <div className="hero-actions">
            <button className="button primary-button" onClick={onStart}>
              Start assessment
              <ArrowRight size={17} />
            </button>
          </div>
        </div>
        <Topography />
      </section>
      <section
        className="library-grid landing-models"
        aria-label="Available models"
      >
        <article className="panel model-overview-card">
          <FileSliders size={24} className="model-overview-icon" />
          <div className="library-title">
            <h2>Clinical model</h2>
            <Readiness ready={status?.clinical_model.ready} />
          </div>
          <p>CatBoost · 12 clinical measurements</p>
          <span className="model-overview-detail">
            Feature contributions with TreeSHAP
          </span>
          <dl className="landing-metrics">
            <Metric label="AUROC" value={clinicalMetrics.auroc.toFixed(4)} />
            <Metric label="AUPRC" value={clinicalMetrics.auprc.toFixed(4)} />
          </dl>
          <p className="metric-scope">Locked internal test</p>
        </article>
        <article className="panel model-overview-card">
          <ScanEye size={24} className="model-overview-icon" />
          <div className="library-title">
            <h2>Imaging model</h2>
            <Readiness ready={status?.image_model.ready} />
          </div>
          <p>EfficientNet-B0 · 7 corneal maps</p>
          <span className="model-overview-detail">
            Grad-CAM and integrated gradients
          </span>
          <dl className="landing-metrics">
            <Metric
              label="Accuracy"
              value={checkpointMetric(
                status?.image_model.metadata?.validation_accuracy,
                true,
              )}
            />
            <Metric
              label="F1 score"
              value={checkpointMetric(
                status?.image_model.metadata?.validation_f1,
              )}
            />
          </dl>
          <p className="metric-scope">
            Saved checkpoint validation · not an external test
          </p>
        </article>
      </section>
      <section
        className="evidence-section panel"
        aria-labelledby="evidence-title"
      >
        <div className="section-title">
          <h2 id="evidence-title">Evaluation reports</h2>
          <button className="button ghost-button" onClick={onExport}>
            <Download size={15} />
            Export evidence
          </button>
        </div>
        <p className="evidence-intro">
          Separate datasets and checkpoints; scores are not directly comparable.
        </p>
        <div className="evidence-list">
          {evaluations.map((evaluation) => (
            <div
              key={evaluation.id}
              className={`evidence-item ${expanded === evaluation.id ? "expanded" : ""}`}
            >
              <button
                className="evidence-row"
                aria-expanded={expanded === evaluation.id}
                aria-controls={`evidence-${evaluation.id}`}
                onClick={() =>
                  setExpanded(expanded === evaluation.id ? null : evaluation.id)
                }
              >
                <span className="evaluation-name">
                  <strong>{evaluation.name}</strong>
                  <small>{evaluation.family}</small>
                </span>
                <span
                  className={`evaluation-scope ${evaluation.id === "imaging" ? "external" : ""}`}
                >
                  {evaluation.scope}
                </span>
                <span className="evaluation-auc">
                  <strong>{evaluation.auc.toFixed(4)}</strong>
                  <small>AUROC</small>
                </span>
                <ChevronDown size={17} className="disclosure-chevron" />
              </button>
              <div
                id={`evidence-${evaluation.id}`}
                hidden={expanded !== evaluation.id}
                className="evaluation-detail"
              >
                <p>{evaluation.note}</p>
                <div>
                  <span>{evaluation.detail}</span>
                  <span>
                    Recorded threshold:{" "}
                    {(evaluation.threshold * 100).toFixed(2)}%
                  </span>
                </div>
                <code>Source: {evaluation.source}</code>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function checkpointMetric(value: unknown, percentage = false) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return percentage ? `${(value * 100).toFixed(1)}%` : value.toFixed(4);
}

function Readiness({ ready }: { ready?: boolean }) {
  return (
    <span className="model-runtime">
      <span className={`status-dot ${ready ? "online" : ""}`} />
      {ready ? "Ready" : "Unavailable"}
    </span>
  );
}
