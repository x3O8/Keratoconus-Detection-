export type AppTab = "overview" | "patient" | "results" | "explainability";

export interface FeatureDefinition {
  name: string;
  label: string;
  unit: string;
  group: string;
  min: number;
  max: number;
}

export interface ModalityDefinition {
  key: string;
  label: string;
}

export interface ModelStatus {
  ready: boolean;
  metadata?: Record<string, unknown> | null;
  error?: string | null;
}

export interface StatusResponse {
  api_version: string;
  device: string;
  image_model: ModelStatus;
  clinical_model: ModelStatus;
  clinical_features: FeatureDefinition[];
  modalities: ModalityDefinition[];
  default_threshold: number;
  research_use_only: boolean;
  rag?: {
    ready: boolean;
    backend: string;
    embedding_model: string;
    documents: number;
    error?: string | null;
  };
}

export interface ClassificationResult {
  keratoconus_probability: number;
  normal_probability: number;
  screen_positive: boolean;
  predicted_label: string;
  threshold: number;
}

export interface ClinicalContribution {
  feature: string;
  label: string;
  unit: string;
  value: number;
  contribution: number;
  direction: "toward_keratoconus" | "toward_normal";
}

export interface ClinicalResult extends ClassificationResult {
  feature_values: Record<string, number>;
  explanation: {
    method: string;
    scale: string;
    base_value: number;
    prediction_margin: number;
    summary: string;
    contributions: ClinicalContribution[];
    global_importance: Record<string, number>;
  };
}

export interface ImageResult extends ClassificationResult {
  uploaded_modalities: string[];
  missing_modalities: string[];
  input_completeness: number;
  modality_ablation: Record<string, number>;
  learned_modality_gates: Record<string, number>;
}

export interface AnalysisResponse {
  case: { case_id?: string | null; laterality?: string | null };
  combined: ClassificationResult & {
    method: "weighted_fusion" | "image_only" | "clinical_only";
    image_weight: number;
    clinical_weight: number;
    models_agree: boolean | null;
    review_priority: string;
  };
  image: ImageResult | null;
  clinical: ClinicalResult | null;
  gradcam: Record<string, { original: string; overlay: string | null }>;
  integrated_gradients: Record<
    string,
    { original: string; overlay: string | null }
  >;
  attention_rollout: {
    modalities: string[];
    matrix: number[][];
    importance: Record<string, number>;
    note: string;
  } | null;
  clinical_decision_explanation: {
    headline: string;
    model_evidence: string[];
    uncertainty: string[];
    clinical_context: string;
    suggested_review: string;
  };
  retrieved_evidence: EvidenceSource[];
  data_quality: Array<{ level: "ok" | "warning" | "review"; message: string }>;
  disclaimer: string;
}

export interface EvidenceSource {
  id: string;
  title: string;
  citation: string;
  url: string;
  topics: string[];
  text: string;
  relevance_score: number;
  semantic_score?: number;
  retrieval_method?: string;
}

export interface RagAnswer {
  answer: string;
  sources: EvidenceSource[];
  generation: string;
  retrieval?: {
    ready: boolean;
    backend: string;
    embedding_model: string;
    documents: number;
    error?: string | null;
  };
  generation_warning?: string | null;
  disclaimer: string;
}
