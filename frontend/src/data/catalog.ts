import catboost from "../../../models/catboost_final_pipeline.metadata.json";
import xgboost from "../../../journal_xgb_results/metrics.json";
import imaging from "../../../external_validation_results/external_validation_metrics.json";
import type { StatusResponse } from "../types";

// Recorded evidence is imported from the repository; it is never an inference result.
export const clinicalMetrics = {
  auroc: catboost.training_metadata.locked_internal_test_auroc,
  auprc: catboost.training_metadata.locked_internal_test_auprc,
};

export const evaluations = [
  {
    id: "catboost",
    name: "CatBoost",
    family: "Clinical · selected pipeline",
    scope: "Locked internal test",
    auc: catboost.training_metadata.locked_internal_test_auroc,
    detail: "12 clinical features",
    threshold: catboost.validated_threshold,
    note: "Selected by patient-grouped cross-validation. External validation has not been performed.",
    source: "models/catboost_final_pipeline.metadata.json",
    data: catboost,
  },
  {
    id: "xgboost",
    name: "XGBoost",
    family: "Clinical · journal experiment",
    scope: "Locked internal test",
    auc: xgboost.locked_test.roc_auc,
    detail: `${xgboost.locked_test.n} eyes · ${xgboost.design.locked_test_patients} patients`,
    threshold: xgboost.locked_test.threshold,
    note: "Patient-separated evaluation. Internal test performance does not establish external generalization.",
    source: "journal_xgb_results/metrics.json",
    data: xgboost,
  },
  {
    id: "imaging",
    name: "EfficientNetV2-S",
    family: "Imaging · separate experiment",
    scope: "Exploratory external test",
    auc: imaging.auroc,
    detail: `${imaging.cases_evaluated} cases · 4 mapped modalities`,
    threshold: imaging.threshold_from_orbscan_validation,
    note: `The transferred threshold produced ${(imaging.specificity * 100).toFixed(0)}% specificity. This experiment uses a separate checkpoint from the seven-map EfficientNet-B0 runtime.`,
    source: "external_validation_results/external_validation_metrics.json",
    data: imaging,
  },
];

export const offlineSchema: StatusResponse = {
  api_version: "—",
  device: "offline",
  image_model: { ready: false },
  clinical_model: { ready: false },
  default_threshold: catboost.validated_threshold,
  research_use_only: true,
  clinical_features: [
    {
      name: "age_years",
      label: "Age",
      unit: "years",
      group: "Demographics",
      min: 1,
      max: 100,
    },
    {
      name: "gender",
      label: "Gender",
      unit: "0 = female, 1 = male",
      group: "Demographics",
      min: 0,
      max: 1,
    },
    {
      name: "astig_value_d",
      label: "Astigmatism",
      unit: "D",
      group: "Keratometry",
      min: -25,
      max: 25,
    },
    {
      name: "astig_axis_deg",
      label: "Astigmatism axis",
      unit: "°",
      group: "Keratometry",
      min: 0,
      max: 180,
    },
    {
      name: "kmax_value_d",
      label: "Kmax",
      unit: "D",
      group: "Keratometry",
      min: 20,
      max: 100,
    },
    {
      name: "kmax_axis_deg",
      label: "Kmax axis",
      unit: "°",
      group: "Keratometry",
      min: 0,
      max: 180,
    },
    {
      name: "pachy_central_um",
      label: "Central pachymetry",
      unit: "μm",
      group: "Pachymetry",
      min: 200,
      max: 900,
    },
    {
      name: "pachy_thinnest_um",
      label: "Thinnest pachymetry",
      unit: "μm",
      group: "Pachymetry",
      min: 200,
      max: 900,
    },
    {
      name: "pachy_thinnest_x",
      label: "Thinnest-point X",
      unit: "mm",
      group: "Pachymetry",
      min: -10,
      max: 10,
    },
    {
      name: "pachy_thinnest_y",
      label: "Thinnest-point Y",
      unit: "mm",
      group: "Pachymetry",
      min: -10,
      max: 10,
    },
    {
      name: "asphericity_anterior",
      label: "Anterior asphericity",
      unit: "Q",
      group: "Asphericity",
      min: -10,
      max: 10,
    },
    {
      name: "asphericity_posterior",
      label: "Posterior asphericity",
      unit: "Q",
      group: "Asphericity",
      min: -10,
      max: 10,
    },
  ],
  modalities: [
    { key: "CT_A", label: "Corneal thickness · anterior" },
    { key: "EC_A", label: "Epithelial curvature · anterior" },
    { key: "EC_P", label: "Epithelial curvature · posterior" },
    { key: "Elv_A", label: "Elevation · anterior" },
    { key: "Elv_P", label: "Elevation · posterior" },
    { key: "Sag_A", label: "Sagittal curvature · anterior" },
    { key: "Sag_P", label: "Sagittal curvature · posterior" },
  ],
};

export function downloadEvidence() {
  const url = URL.createObjectURL(
    new Blob(
      [
        JSON.stringify(
          {
            purpose: "Recorded research evaluations; not patient predictions",
            evaluations: evaluations.map(({ name, source, data }) => ({
              name,
              source,
              data,
            })),
          },
          null,
          2,
        ),
      ],
      { type: "application/json" },
    ),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = "model-evidence.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
