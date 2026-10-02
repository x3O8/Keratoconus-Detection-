import { ArrowLeft, ScanSearch } from "lucide-react";

export function EmptyState({ onStart }: { onStart: () => void }) {
  return (
    <div className="empty-state panel">
      <span className="empty-icon"><ScanSearch size={28} /></span>
      <h2>No assessment result yet</h2>
      <p>Add available clinical values or corneal maps and run the two-model assessment first.</p>
      <button className="button primary-button inline" onClick={onStart}><ArrowLeft size={17} />Go to patient data</button>
    </div>
  );
}
