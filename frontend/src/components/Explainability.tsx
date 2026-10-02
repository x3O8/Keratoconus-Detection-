import { useMemo, useState } from "react";
import { ArrowDownLeft, ArrowUpRight, LoaderCircle, Send } from "lucide-react";
import type {
  AnalysisResponse,
  ClinicalContribution,
  EvidenceSource,
  RagAnswer,
} from "../types";
import { EmptyState } from "./EmptyState";
import { askExplanation } from "../api";

export function Explainability({
  result,
  onStart,
}: {
  result: AnalysisResponse | null;
  onStart: () => void;
}) {
  const modalities = Object.keys(result?.gradcam || {});
  const [selectedModality, setSelectedModality] = useState<string>(
    modalities[0] || "",
  );
  const selected = modalities.includes(selectedModality)
    ? selectedModality
    : modalities[0] || "";
  const [attributionMethod, setAttributionMethod] = useState<
    "gradcam" | "integrated_gradients"
  >("gradcam");
  if (!result)
    return (
      <div className="tab-page">
        <EmptyState onStart={onStart} />
      </div>
    );
  const view =
    attributionMethod === "gradcam"
      ? result.gradcam[selected]
      : result.integrated_gradients[selected];

  return (
    <div className="tab-page explain-page">
      <div className="page-intro">
        <div>
          <h1>Explainability</h1>
          <p>What influenced this assessment.</p>
        </div>
      </div>
      {result.clinical ? (
        <ClinicalExplanation
          contributions={result.clinical.explanation.contributions}
          scale={result.clinical.explanation.scale}
          summary={result.clinical.explanation.summary}
        />
      ) : null}
      {result.image ? (
        <div className="image-explanation-grid">
          <section className="panel modality-impact-panel">
            <h2>Map contributions</h2>
            <p className="detail-note">
              Change in KC probability when a map is withheld.
            </p>
            <ModalityImpact impacts={result.image.modality_ablation} />
          </section>
          <section className="panel gradcam-panel">
            <h2>Image attribution</h2>
            {modalities.length ? (
              <>
                <div
                  className="method-switch"
                  role="group"
                  aria-label="Attribution method"
                >
                  <button
                    aria-pressed={attributionMethod === "gradcam"}
                    className={attributionMethod === "gradcam" ? "active" : ""}
                    onClick={() => setAttributionMethod("gradcam")}
                  >
                    Grad-CAM
                  </button>
                  <button
                    aria-pressed={attributionMethod === "integrated_gradients"}
                    className={
                      attributionMethod === "integrated_gradients"
                        ? "active"
                        : ""
                    }
                    onClick={() => setAttributionMethod("integrated_gradients")}
                  >
                    Integrated gradients
                  </button>
                </div>
                {modalities.length > 1 ? (
                  <div
                    className="modality-tabs"
                    role="group"
                    aria-label="Corneal map"
                  >
                    {modalities.map((modality) => (
                      <button
                        key={modality}
                        aria-pressed={selected === modality}
                        className={selected === modality ? "active" : ""}
                        onClick={() => setSelectedModality(modality)}
                      >
                        {modality}
                      </button>
                    ))}
                  </div>
                ) : null}
                {view ? (
                  <div className="gradcam-images">
                    <figure>
                      <img
                        src={view.original}
                        alt={`${selected} original corneal map`}
                      />
                      <figcaption>Original · {selected}</figcaption>
                    </figure>
                    <figure>
                      <img
                        src={view.overlay || view.original}
                        alt={`${selected} ${attributionMethod} overlay`}
                      />
                      <figcaption>
                        {attributionMethod === "gradcam"
                          ? "Grad-CAM"
                          : "Integrated gradients"}{" "}
                        · {selected}
                      </figcaption>
                    </figure>
                  </div>
                ) : null}
                <div className="heat-legend">
                  <span>Lower influence</span>
                  <i />
                  <span>Higher influence</span>
                </div>
              </>
            ) : (
              <p className="detail-note">No overlay was produced.</p>
            )}
          </section>
        </div>
      ) : null}
      {result.attention_rollout ? (
        <details className="panel compact-disclosure explanation-detail">
          <summary>Modality attention</summary>
          <AttentionRollout rollout={result.attention_rollout} />
        </details>
      ) : null}
      <EvidenceAssistant result={result} />
      <DecisionExplanation result={result} />
      <p className="interpretation-note">
        Attributions describe model behavior; they do not confirm disease or
        establish causation.
      </p>
    </div>
  );
}

function DecisionExplanation({ result }: { result: AnalysisResponse }) {
  const decision = result.clinical_decision_explanation;
  return (
    <details className="panel compact-disclosure explanation-detail">
      <summary>Clinical context and limitations</summary>
      <p className="detail-note">{decision.clinical_context}</p>
      <div className="decision-grid">
        <div>
          <h3>Evidence used</h3>
          <ul>
            {decision.model_evidence.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div>
          <h3>Uncertainty</h3>
          <ul>
            {decision.uncertainty.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div className="review-step">
          <h3>Clinical review</h3>
          <p>{decision.suggested_review}</p>
        </div>
      </div>
      <p className="detail-note">
        Correlated variables can share apparent importance. Heatmaps and
        attention weights do not segment disease. Retrieved sources provide
        context, not patient-specific advice.
      </p>
    </details>
  );
}

function AttentionRollout({
  rollout,
}: {
  rollout: NonNullable<AnalysisResponse["attention_rollout"]>;
}) {
  const entries = Object.entries(rollout.importance)
    .filter(([, value]) => value > 0)
    .sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map(([, value]) => value), 0.01);
  return (
    <>
      <p className="detail-note">{rollout.note}</p>
      <div className="rollout-bars">
        {entries.map(([name, value]) => (
          <div key={name}>
            <span>{name}</span>
            <i>
              <b style={{ width: `${(value / max) * 100}%` }} />
            </i>
            <strong>{(value * 100).toFixed(1)}%</strong>
          </div>
        ))}
      </div>
    </>
  );
}

function EvidenceAssistant({ result }: { result: AnalysisResponse }) {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<RagAnswer | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ask = async () => {
    if (loading || question.trim().length < 3) return;
    setLoading(true);
    setError(null);
    try {
      setResponse(await askExplanation(question.trim(), result));
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Evidence retrieval failed.",
      );
    } finally {
      setLoading(false);
    }
  };
  const sources = response?.sources || result.retrieved_evidence;
  return (
    <section className="panel evidence-panel">
      <h2>Ask a question</h2>
      <p className="detail-note">
        Answers use this result and published sources.
      </p>
      <div className="rag-input">
        <label className="sr-only" htmlFor="rag-question">
          Question about the explanation
        </label>
        <input
          id="rag-question"
          maxLength={2000}
          value={question}
          placeholder="What influenced this result?"
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.nativeEvent.isComposing)
              void ask();
          }}
        />
        <button
          onClick={() => void ask()}
          disabled={loading || question.trim().length < 3}
        >
          {loading ? (
            <LoaderCircle className="spin" size={17} />
          ) : (
            <Send size={17} />
          )}
          {loading ? "Asking…" : "Ask"}
        </button>
      </div>
      {error ? (
        <p className="rag-error" role="alert">
          {error}
        </p>
      ) : null}
      {response ? (
        <div className="rag-answer" aria-live="polite">
          <p>{response.answer}</p>
          {response.generation_warning &&
          !response.generation_warning.startsWith("LLM disabled:") ? (
            <small>{response.generation_warning}</small>
          ) : null}
        </div>
      ) : null}
      <details className="compact-disclosure source-disclosure">
        <summary>Sources and answer details ({sources.length})</summary>
        <EvidenceSources sources={sources} />
        <p className="detail-note">
          Answers use a curated local source library. If Gemini is enabled on
          the service, de-identified model context and retrieved passages are
          sent for citation-guided generation. Retrieved content is research
          context, not medical advice.
        </p>
      </details>
    </section>
  );
}

function EvidenceSources({ sources }: { sources: EvidenceSource[] }) {
  return (
    <div className="source-list">
      {sources.map((source) => (
        <a key={source.id} href={source.url} target="_blank" rel="noreferrer">
          <strong>{source.title}</strong>
          <small>{source.citation} ↗</small>
        </a>
      ))}
    </div>
  );
}

function ClinicalExplanation({
  contributions,
  scale,
  summary,
}: {
  contributions: ClinicalContribution[];
  scale: string;
  summary?: string;
}) {
  const sorted = useMemo(
    () =>
      [...contributions].sort(
        (a, b) => Math.abs(b.contribution) - Math.abs(a.contribution),
      ),
    [contributions],
  );
  const max = Math.max(
    ...sorted.map((item) => Math.abs(item.contribution)),
    0.01,
  );
  const row = (item: ClinicalContribution) => (
    <div className="shap-row" key={item.feature}>
      <div className="shap-label">
        <strong>{item.label}</strong>
        <small>
          {item.value.toLocaleString()}{" "}
          {item.unit.startsWith("0 =") ? "" : item.unit}
        </small>
      </div>
      <div className="shap-bar">
        <span className="zero-line" />
        <span
          className={item.contribution >= 0 ? "bar positive" : "bar negative"}
          style={{ width: `${(Math.abs(item.contribution) / max) * 48}%` }}
        />
        <b
          className={
            item.contribution >= 0 ? "positive-value" : "negative-value"
          }
        >
          {item.contribution > 0 ? "+" : ""}
          {item.contribution.toFixed(2)}
        </b>
      </div>
    </div>
  );
  return (
    <section className="panel shap-panel">
      <h2>Clinical contributors</h2>
      {summary ? <p className="contribution-summary">{summary}</p> : null}
      <div className="shap-axis">
        <span>
          <ArrowDownLeft size={14} />
          Toward normal
        </span>
        <span>Baseline</span>
        <span>
          Toward keratoconus
          <ArrowUpRight size={14} />
        </span>
      </div>
      <div className="shap-list">{sorted.slice(0, 6).map(row)}</div>
      {sorted.length > 6 ? (
        <details className="compact-disclosure more-features">
          <summary>{sorted.length - 6} more features</summary>
          <div className="shap-list">{sorted.slice(6).map(row)}</div>
        </details>
      ) : null}
      <p className="detail-note">TreeSHAP · {scale}</p>
    </section>
  );
}

function ModalityImpact({ impacts }: { impacts: Record<string, number> }) {
  const entries = Object.entries(impacts).sort(
    (a, b) => Math.abs(b[1]) - Math.abs(a[1]),
  );
  const max = Math.max(...entries.map(([, value]) => Math.abs(value)), 0.001);
  return (
    <div className="impact-list">
      {entries.map(([name, value]) => (
        <div className="impact-row" key={name}>
          <span>{name}</span>
          <div>
            <i
              className={value >= 0 ? "toward-kc" : "toward-normal"}
              style={{ width: `${(Math.abs(value) / max) * 100}%` }}
            />
          </div>
          <strong>
            {value > 0 ? "+" : ""}
            {(value * 100).toFixed(1)} pp
          </strong>
        </div>
      ))}
    </div>
  );
}
