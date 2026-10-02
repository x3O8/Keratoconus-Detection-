import type { CSSProperties } from "react";

export function ProbabilityRing({ probability, positive, compact = false }: { probability: number; positive: boolean; compact?: boolean }) {
  const style = { "--probability": `${Math.max(0, Math.min(100, probability * 100))}%` } as CSSProperties;
  return (
    <div className={`probability-ring ${positive ? "positive" : "negative"} ${compact ? "compact" : ""}`} style={style}>
      <div><strong>{(probability * 100).toFixed(1)}%</strong><span>KC probability</span></div>
    </div>
  );
}
