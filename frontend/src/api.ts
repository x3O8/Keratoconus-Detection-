import type { AnalysisResponse, RagAnswer, StatusResponse } from "./types";

async function parseResponse<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => {
    throw new Error(
      "The inference service returned an invalid response. Reconnect and try again.",
    );
  });
  if (!response.ok) {
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : `Request failed (${response.status}). Check your inputs and try again.`,
    );
  }
  return body as T;
}

export async function askExplanation(
  question: string,
  result: AnalysisResponse,
): Promise<RagAnswer> {
  return parseResponse<RagAnswer>(
    await fetch("/api/explain/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(60000),
      body: JSON.stringify({
        question,
        combined: result.combined,
        clinical: result.clinical,
        image: result.image,
      }),
    }),
  );
}

export async function fetchStatus(
  signal?: AbortSignal,
): Promise<StatusResponse> {
  const status = await parseResponse<StatusResponse>(
    await fetch("/api/status", {
      signal: signal
        ? AbortSignal.any([signal, AbortSignal.timeout(10000)])
        : AbortSignal.timeout(10000),
    }),
  );
  if (
    !status.clinical_model ||
    !status.image_model ||
    !Array.isArray(status.clinical_features) ||
    !Array.isArray(status.modalities)
  )
    throw new Error(
      "The inference service returned an incompatible status response.",
    );
  return status;
}

export async function runAnalysis(
  payload: FormData,
): Promise<AnalysisResponse> {
  return parseResponse<AnalysisResponse>(
    await fetch("/api/analyze", {
      method: "POST",
      body: payload,
      signal: AbortSignal.timeout(120000),
    }),
  );
}
