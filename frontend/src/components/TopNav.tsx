import type { AppTab } from "../types";

const navigation = [
  { id: "overview" as const, label: "Models" },
  { id: "patient" as const, label: "Assessment" },
  { id: "results" as const, label: "Results" },
  {
    id: "explainability" as const,
    label: "Explainability",
  },
];

export function TopNav({
  activeTab,
  hasResult,
  onTabChange,
}: {
  activeTab: AppTab;
  hasResult: boolean;
  onTabChange: (tab: AppTab) => void;
}) {
  return (
    <header className="workspace-header">
      <div className="workspace-header-inner">
        <nav className="workspace-nav" aria-label="Workspace navigation">
          {navigation.map(({ id, label }) => (
            <button
              key={id}
              className={`workspace-nav-link ${activeTab === id ? "active" : ""}`}
              aria-current={activeTab === id ? "page" : undefined}
              disabled={
                (id === "results" || id === "explainability") && !hasResult
              }
              title={
                (id === "results" || id === "explainability") && !hasResult
                  ? "Run an assessment to unlock this view"
                  : undefined
              }
              onClick={() => onTabChange(id)}
            >
              {label}
            </button>
          ))}
        </nav>
      </div>
    </header>
  );
}
