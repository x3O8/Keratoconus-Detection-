import { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  ArrowRight,
  Check,
  Download,
  FileSpreadsheet,
  ImagePlus,
  LoaderCircle,
  Play,
  RotateCcw,
  UploadCloud,
  X,
} from "lucide-react";
import { runAnalysis } from "../api";
import type {
  AnalysisResponse,
  FeatureDefinition,
  ModalityDefinition,
  StatusResponse,
} from "../types";

interface PatientDataProps {
  status: StatusResponse | null;
  onComplete: (result: AnalysisResponse) => void;
}

const exampleValues: Record<string, string> = {
  age_years: "27",
  gender: "1",
  astig_value_d: "-5.2",
  astig_axis_deg: "84",
  kmax_value_d: "53.1",
  kmax_axis_deg: "96",
  pachy_central_um: "472",
  pachy_thinnest_um: "438",
  pachy_thinnest_x: "0.8",
  pachy_thinnest_y: "-0.9",
  asphericity_anterior: "-1.35",
  asphericity_posterior: "-1.62",
};

export function PatientData({ status, onComplete }: PatientDataProps) {
  const [caseId, setCaseId] = useState("");
  const [laterality, setLaterality] = useState("OD");
  const [inputMode, setInputMode] = useState<"manual" | "csv">("manual");
  const [values, setValues] = useState<Record<string, string>>({});
  const [csv, setCsv] = useState<File | null>(null);
  const [images, setImages] = useState<Record<string, File>>({});
  const [imageWeight, setImageWeight] = useState(65);
  const [threshold, setThreshold] = useState(50);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (status?.default_threshold != null)
      setThreshold(status.default_threshold * 100);
  }, [status?.default_threshold]);

  const features = status?.clinical_features || [];
  const modalities = status?.modalities || [];
  const clinicalReady =
    inputMode === "csv"
      ? Boolean(csv)
      : features.length > 0 &&
        features.every(
          (feature) =>
            values[feature.name]?.trim() &&
            !fieldError(feature, values[feature.name]),
        );
  const imageReady = Object.keys(images).length > 0;
  const runnable =
    (clinicalReady && status?.clinical_model.ready) ||
    (imageReady && status?.image_model.ready);

  useEffect(() => {
    if (!caseId && !csv && !imageReady && !Object.values(values).some(Boolean))
      return;
    const preventLoss = (event: BeforeUnloadEvent) => {
      event.preventDefault();
    };
    window.addEventListener("beforeunload", preventLoss);
    return () => window.removeEventListener("beforeunload", preventLoss);
  }, [caseId, csv, imageReady, values]);

  const groupedFeatures = useMemo(() => {
    return features.reduce<Record<string, FeatureDefinition[]>>(
      (groups, feature) => {
        (groups[feature.group] ||= []).push(feature);
        return groups;
      },
      {},
    );
  }, [features]);

  const submit = async () => {
    if (loading) return;
    if (!runnable) {
      setError(
        "Complete one model pathway: all 12 clinical fields (or a CSV), or at least one corneal image.",
      );
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const payload = new FormData();
      payload.set("case_id", caseId);
      payload.set("laterality", laterality);
      payload.set("image_weight", String(imageWeight / 100));
      payload.set("threshold", String(threshold / 100));
      if (inputMode === "csv" && csv && status?.clinical_model.ready)
        payload.set("csv", csv);
      if (
        inputMode === "manual" &&
        clinicalReady &&
        status?.clinical_model.ready
      ) {
        payload.set(
          "clinical_json",
          JSON.stringify(
            Object.fromEntries(
              features.map((feature) => [
                feature.name,
                Number(values[feature.name]),
              ]),
            ),
          ),
        );
      }
      if (status?.image_model.ready)
        Object.entries(images).forEach(([modality, file]) =>
          payload.set(modality, file),
        );
      onComplete(await runAnalysis(payload));
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The assessment could not be completed.",
      );
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setValues({});
    setCsv(null);
    setImages({});
    setError(null);
  };

  return (
    <div className="tab-page patient-page">
      <PageIntro />
      {!status?.clinical_model.ready && !status?.image_model.ready && (
        <div className="offline-notice">
          Connect the inference service to run an assessment.
        </div>
      )}

      <section className="case-strip panel">
        <div className="case-field wide">
          <label htmlFor="case-id">
            Case identifier <span>optional</span>
          </label>
          <input
            id="case-id"
            value={caseId}
            onChange={(event) => setCaseId(event.target.value)}
            placeholder="Optional reference"
          />
        </div>
        <div className="case-field">
          <label htmlFor="laterality">Eye</label>
          <select
            id="laterality"
            value={laterality}
            onChange={(event) => setLaterality(event.target.value)}
          >
            <option value="OD">Right eye · OD</option>
            <option value="OS">Left eye · OS</option>
          </select>
        </div>
        <div className="case-actions">
          <button
            className="button ghost-button"
            onClick={() => {
              setValues(exampleValues);
              setInputMode("manual");
              setError(null);
            }}
          >
            <Play size={15} />
            Use example
          </button>
          <button
            className="icon-button"
            onClick={reset}
            aria-label="Clear clinical and image inputs"
            title="Clear clinical and image inputs"
          >
            <RotateCcw size={17} />
          </button>
        </div>
      </section>

      <div className="input-columns">
        <section className="panel data-panel">
          <div className="panel-heading">
            <div className="heading-icon">
              <FileSpreadsheet size={19} />
            </div>
            <div>
              <h2>Clinical measurements</h2>
            </div>
            <ReadinessBadge
              ready={Boolean(clinicalReady)}
              label={
                clinicalReady
                  ? "Ready"
                  : `${features.filter((feature) => values[feature.name]?.trim() && !fieldError(feature, values[feature.name])).length}/12 valid`
              }
            />
          </div>

          <div
            className="segmented-control"
            role="group"
            aria-label="Clinical input method"
          >
            <button
              aria-pressed={inputMode === "manual"}
              className={inputMode === "manual" ? "active" : ""}
              onClick={() => setInputMode("manual")}
            >
              Manual entry
            </button>
            <button
              aria-pressed={inputMode === "csv"}
              className={inputMode === "csv" ? "active" : ""}
              onClick={() => setInputMode("csv")}
            >
              Import CSV
            </button>
          </div>

          {inputMode === "manual" ? (
            <div className="clinical-groups">
              {Object.entries(groupedFeatures).map(([group, groupFeatures]) => (
                <fieldset key={group}>
                  <legend>{group}</legend>
                  <div className="field-grid">
                    {groupFeatures.map((feature) => (
                      <ClinicalField
                        key={feature.name}
                        feature={feature}
                        value={values[feature.name] || ""}
                        onChange={(value) =>
                          setValues((current) => ({
                            ...current,
                            [feature.name]: value,
                          }))
                        }
                      />
                    ))}
                  </div>
                </fieldset>
              ))}
            </div>
          ) : (
            <CsvUploader
              file={csv}
              onChange={setCsv}
              onError={setError}
              features={features}
            />
          )}
        </section>

        <section className="panel data-panel imaging-panel">
          <div className="panel-heading">
            <div className="heading-icon">
              <ImagePlus size={19} />
            </div>
            <div>
              <h2>Corneal imaging</h2>
              <p>PNG or JPEG · up to 10 MB per map</p>
            </div>
            <ReadinessBadge
              ready={imageReady}
              label={`${Object.keys(images).length}/7`}
            />
          </div>
          <div className="modality-grid">
            {modalities.map((modality) => (
              <ImageUploader
                key={modality.key}
                modality={modality}
                file={images[modality.key]}
                onError={setError}
                onChange={(file) =>
                  setImages((current) => {
                    const next = { ...current };
                    if (file) next[modality.key] = file;
                    else delete next[modality.key];
                    return next;
                  })
                }
              />
            ))}
          </div>
          {Object.keys(images).length > 0 && Object.keys(images).length < 7 && (
            <div className="inline-notice warning">
              <AlertCircle size={16} />
              <span>
                Missing modalities are zero-filled by the image model. Interpret
                partial-input results cautiously.
              </span>
            </div>
          )}
        </section>
      </div>

      <details className="panel compact-disclosure assessment-settings">
        <summary>Analysis settings</summary>
        <div className="sliders">
          <label>
            <span>
              Imaging weight <strong>{imageWeight}%</strong>
            </span>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={imageWeight}
              disabled={!clinicalReady || !imageReady}
              onChange={(event) => setImageWeight(Number(event.target.value))}
            />
          </label>
          <label>
            <span>
              Decision threshold <strong>{threshold.toFixed(2)}%</strong>
            </span>
            <input
              type="range"
              min="0.01"
              max="99.99"
              step="0.01"
              value={threshold}
              onChange={(event) => setThreshold(Number(event.target.value))}
            />
          </label>
        </div>
      </details>
      <section className="analysis-dock">
        <div className="dock-status">
          <strong>
            {clinicalReady && imageReady
              ? "Dual-model fusion"
              : clinicalReady
                ? "Clinical model only"
                : imageReady
                  ? "Imaging model only"
                  : "Waiting for patient data"}
          </strong>
        </div>
        <button
          className="button primary-button run-button"
          disabled={!runnable || loading}
          aria-busy={loading}
          onClick={submit}
        >
          {loading ? (
            <>
              <LoaderCircle className="spin" size={18} />
              Running models…
            </>
          ) : (
            <>
              Run assessment
              <ArrowRight size={18} />
            </>
          )}
        </button>
      </section>
      {error && (
        <div className="error-banner" role="alert">
          <AlertCircle size={18} />
          <span>{error}</span>
          <button onClick={() => setError(null)} aria-label="Dismiss">
            <X size={16} />
          </button>
        </div>
      )}
    </div>
  );
}

function PageIntro() {
  return (
    <div className="page-intro">
      <div>
        <h1>Assessment</h1>
        <p>Add measurements, corneal maps, or both.</p>
      </div>
    </div>
  );
}

function ReadinessBadge({ ready, label }: { ready: boolean; label: string }) {
  return (
    <span className={`readiness-badge ${ready ? "ready" : ""}`}>
      {ready && <Check size={13} />}
      {label}
    </span>
  );
}

function ClinicalField({
  feature,
  value,
  onChange,
}: {
  feature: FeatureDefinition;
  value: string;
  onChange: (value: string) => void;
}) {
  const error = value.trim() ? fieldError(feature, value) : null;
  if (feature.name === "gender") {
    return (
      <label className="clinical-field">
        <span>
          {feature.label}
          <small>{feature.unit}</small>
        </span>
        <select
          value={value}
          onChange={(event) => onChange(event.target.value)}
        >
          <option value="">Select</option>
          <option value="0">Female · 0</option>
          <option value="1">Male · 1</option>
        </select>
      </label>
    );
  }
  return (
    <label className="clinical-field">
      <span>
        {feature.label}
        <small>{feature.unit}</small>
      </span>
      <input
        id={feature.name}
        type="number"
        min={feature.min}
        max={feature.max}
        step="any"
        value={value}
        aria-invalid={Boolean(error)}
        aria-describedby={error ? `${feature.name}-error` : undefined}
        onChange={(event) => onChange(event.target.value)}
        placeholder={`${feature.min}–${feature.max}`}
      />
      {error && (
        <span className="field-error" id={`${feature.name}-error`}>
          {error}
        </span>
      )}
    </label>
  );
}

function fieldError(feature: FeatureDefinition, value: string) {
  const number = Number(value);
  return !Number.isFinite(number) ||
    number < feature.min ||
    number > feature.max
    ? `Enter a value from ${feature.min} to ${feature.max}.`
    : null;
}

function CsvUploader({
  file,
  onChange,
  onError,
  features,
}: {
  file: File | null;
  onChange: (file: File | null) => void;
  onError: (error: string | null) => void;
  features: FeatureDefinition[];
}) {
  const selectFile = (selected: File | null) => {
    if (
      selected &&
      (!selected.name.toLowerCase().endsWith(".csv") ||
        selected.size > 10 * 1024 * 1024 ||
        selected.size === 0)
    ) {
      onError("Choose a non-empty CSV file under 10 MB.");
      return;
    }
    onError(null);
    onChange(selected);
  };
  const template = `data:text/csv;charset=utf-8,${encodeURIComponent(features.map((feature) => feature.name).join(",") + "\n" + features.map((feature) => exampleValues[feature.name]).join(",") + "\n")}`;
  return (
    <div className="csv-input-area">
      <label
        className={`csv-uploader ${file ? "has-file" : ""}`}
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          selectFile(event.dataTransfer.files[0] || null);
        }}
      >
        <input
          type="file"
          aria-label="Upload one-row clinical CSV"
          accept=".csv,text/csv"
          onChange={(event) => {
            selectFile(event.target.files?.[0] || null);
            event.target.value = "";
          }}
        />
        <span className="upload-circle">
          {file ? <Check size={22} /> : <UploadCloud size={23} />}
        </span>
        <strong>{file ? file.name : "Drop a one-row clinical CSV"}</strong>
        <small>
          {file
            ? `${(file.size / 1024).toFixed(1)} KB · ready to validate`
            : "or click to browse · CSV up to 10 MB"}
        </small>
        <span className="button ghost-button">
          {file ? "Replace file" : "Choose file"}
        </span>
      </label>
      {file && (
        <button className="button ghost-button" onClick={() => onChange(null)}>
          Remove CSV
        </button>
      )}
      <a
        className="template-link"
        href={template}
        download="clinical-example.csv"
      >
        <Download size={15} />
        Download CSV template
      </a>
    </div>
  );
}

function ImageUploader({
  modality,
  file,
  onChange,
  onError,
}: {
  modality: ModalityDefinition;
  file?: File;
  onChange: (file: File | null) => void;
  onError: (error: string | null) => void;
}) {
  const [preview, setPreview] = useState<string | null>(null);
  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  const selectFile = (selected: File | null) => {
    if (
      selected &&
      (!["image/png", "image/jpeg"].includes(selected.type) ||
        selected.size > 10 * 1024 * 1024 ||
        selected.size === 0)
    ) {
      onError("Choose a non-empty PNG or JPEG image under 10 MB.");
      return;
    }
    onError(null);
    onChange(selected);
  };

  return (
    <label
      className={`modality-uploader ${file ? "has-file" : ""}`}
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault();
        selectFile(event.dataTransfer.files[0] || null);
      }}
    >
      <input
        type="file"
        aria-label={`Upload ${modality.key}: ${modality.label}`}
        accept="image/png,image/jpeg"
        onChange={(event) => {
          selectFile(event.target.files?.[0] || null);
          event.target.value = "";
        }}
      />
      {preview && <img src={preview} alt={`${modality.label} preview`} />}
      <span className="modality-content">
        <span className="modality-icon">
          {file ? <Check size={16} /> : <ImagePlus size={16} />}
        </span>
        <strong>{modality.key}</strong>
        <small>{modality.label}</small>
      </span>
      {file && <small className="uploaded-filename">{file.name}</small>}
      {file && (
        <button
          type="button"
          className="remove-file"
          onClick={(event) => {
            event.preventDefault();
            onChange(null);
          }}
          aria-label={`Remove ${modality.key}`}
        >
          <X size={14} />
        </button>
      )}
    </label>
  );
}
