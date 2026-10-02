import {
  AlertTriangle,
  CheckCircle2,
  FileText,
  Image as ImageIcon,
  Printer,
  ShieldCheck,
} from "lucide-react";
import type { AnalysisResponse } from "../types";
import { EmptyState } from "./EmptyState";
import { ProbabilityRing } from "./ProbabilityRing";

interface ResultsProps {
  result: AnalysisResponse | null;
  onStart: () => void;
  clinicalModelName?: string;
}

export function Results({
  result,
  onStart,
  clinicalModelName = "Clinical model",
}: ResultsProps) {
  if (!result)
    return (
      <div className="tab-page">
        <EmptyState onStart={onStart} />
      </div>
    );
  const { combined } = result;
  const both = Boolean(result.image && result.clinical);
  const warnings = result.data_quality.filter(
    (message) => message.level === "warning",
  );
  return (
    <div className="tab-page results-page">
      <div className="page-intro results-intro">
        <div>
          <h1>Results</h1>
          <p>
            {result.case.case_id || "Current assessment"}
            {result.case.laterality ? ` · ${result.case.laterality}` : ""}
          </p>
        </div>
        <button className="button ghost-button" onClick={() => window.print()}>
          <Printer size={16} />
          Print
        </button>
      </div>
      <section
        className={`result-summary ${combined.screen_positive ? "high-risk" : "low-risk"}`}
      >
        <div className="result-copy">
          <div className="result-label-row">
            {combined.screen_positive ? (
              <AlertTriangle size={23} />
            ) : (
              <ShieldCheck size={23} />
            )}
            <h2>
              {combined.screen_positive ? "Screen positive" : "Screen negative"}
            </h2>
          </div>
          <p>{combined.review_priority}</p>
          <div className="threshold-line">
            <span>{formatMethod(combined.method)}</span>
            <span>Threshold {(combined.threshold * 100).toFixed(2)}%</span>
          </div>
        </div>
        <ProbabilityRing
          probability={combined.keratoconus_probability}
          positive={combined.screen_positive}
        />
      </section>
      {warnings.map((message, index) => (
        <p key={index} className="inline-notice warning">
          <AlertTriangle size={16} />
          {message.message}
        </p>
      ))}
      {both && !combined.models_agree ? (
        <p className="inline-notice warning">
          <AlertTriangle size={16} />
          Models disagree. Review the clinical and imaging outputs
          independently.
        </p>
      ) : null}
      <details className="panel compact-disclosure result-detail" open={both}>
        <summary>Model outputs</summary>
        <div className="model-result-grid">
          {result.image ? (
            <ModelCard
              name="Imaging model"
              icon={<ImageIcon size={18} />}
              result={result.image}
              weight={combined.image_weight}
              detail={`${result.image.uploaded_modalities.length}/7 maps supplied`}
            />
          ) : null}
          {result.clinical ? (
            <ModelCard
              name={clinicalModelName}
              icon={<FileText size={18} />}
              result={result.clinical}
              weight={combined.clinical_weight}
              detail="12 clinical measurements"
            />
          ) : null}
        </div>
        {!result.image ? (
          <p className="detail-note">No imaging was included.</p>
        ) : !result.clinical ? (
          <p className="detail-note">No clinical measurements were included.</p>
        ) : (
          <p className="detail-note">
            {combined.models_agree
              ? "Both models agree at the selected threshold."
              : "The models fall on opposite sides of the threshold."}
          </p>
        )}
      </details>
      <details className="panel compact-disclosure result-detail">
        <summary>Input checks</summary>
        <div className="quality-list">
          {result.data_quality.map((message, index) => (
            <div key={index} className={`quality-row ${message.level}`}>
              <CheckCircle2 size={16} />
              <span>{message.message}</span>
            </div>
          ))}
        </div>
      </details>
      <p className="interpretation-note">
        Model output requires independent clinical review and does not establish
        a diagnosis.
      </p>
    </div>
  );
}

function ModelCard({
  name,
  icon,
  result,
  weight,
  detail,
}: {
  name: string;
  icon: React.ReactNode;
  result: NonNullable<AnalysisResponse["image"] | AnalysisResponse["clinical"]>;
  weight: number;
  detail: string;
}) {
  return (
    <section className="model-result-card">
      <div className="model-card-head">
        {icon}
        <h3>{name}</h3>
      </div>
      <div className="model-number">
        <strong>{(result.keratoconus_probability * 100).toFixed(1)}%</strong>
        <span>KC probability</span>
      </div>
      <p className="detail-note">
        {(weight * 100).toFixed(0)}% weight · {detail}
      </p>
    </section>
  );
}

function formatMethod(method: string) {
  return method
    .replaceAll("_", " ")
    .replace(/^./, (letter) => letter.toUpperCase());
}
