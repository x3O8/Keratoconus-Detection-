import { useEffect, useRef, useState } from "react";
import { LoaderCircle, RefreshCw } from "lucide-react";
import { fetchStatus } from "./api";
import { downloadEvidence, offlineSchema } from "./data/catalog";
import { Explainability } from "./components/Explainability";
import { Overview } from "./components/Overview";
import { PatientData } from "./components/PatientData";
import { Results } from "./components/Results";
import { TopNav } from "./components/TopNav";
import type { AnalysisResponse, AppTab, StatusResponse } from "./types";

const labels: Record<AppTab, string> = {
  overview: "Models",
  patient: "Assessment",
  results: "Results",
  explainability: "Explainability",
};
const getInitialTab = (): AppTab =>
  window.location.hash === "#patient" ? "patient" : "overview";
export default function App() {
  const [activeTab, setActiveTab] = useState<AppTab>(getInitialTab);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [connection, setConnection] = useState<
    "checking" | "online" | "offline"
  >("checking");
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [notification, setNotification] = useState("");
  const request = useRef<AbortController | null>(null);
  const feedbackTimer = useRef<ReturnType<typeof setTimeout>>();
  async function connect(silent = false) {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    if (!silent) setConnection("checking");
    try {
      const response = await fetchStatus(controller.signal);
      if (!controller.signal.aborted) {
        setStatus(response);
        setConnection("online");
      }
    } catch {
      if (!controller.signal.aborted) {
        setStatus(null);
        setConnection("offline");
      }
    }
  }
  useEffect(() => {
    void connect();
    return () => {
      request.current?.abort();
      clearTimeout(feedbackTimer.current);
    };
  }, []);
  useEffect(() => {
    if (connection === "checking") return;
    const timer = setInterval(
      () => void connect(connection === "online"),
      connection === "online" ? 30000 : 10000,
    );
    return () => clearInterval(timer);
  }, [connection]);
  useEffect(() => {
    document.title = labels[activeTab];
  }, [activeTab]);
  useEffect(() => {
    const onHash = () => {
      const tab = window.location.hash.slice(1) as AppTab;
      if (
        tab in labels &&
        ((tab !== "results" && tab !== "explainability") || result)
      )
        setActiveTab(tab);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [result]);
  const changeTab = (tab: AppTab) => {
    if ((tab === "results" || tab === "explainability") && !result) return;
    setActiveTab(tab);
    window.history.pushState(null, "", `#${tab}`);
    window.scrollTo({ top: 0, behavior: "instant" });
  };
  const exportEvidence = () => {
    downloadEvidence();
    setNotification("Model evaluation evidence downloaded.");
    clearTimeout(feedbackTimer.current);
    feedbackTimer.current = setTimeout(() => setNotification(""), 4000);
  };
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <TopNav
        activeTab={activeTab}
        hasResult={Boolean(result)}
        onTabChange={changeTab}
      />
      <main className="main-workspace" id="main-content">
        <div className="service-strip">
          <span
            className={`status-dot ${connection === "online" ? "online" : ""}`}
          />
          <span role="status">
            {connection === "checking"
              ? "Checking inference service…"
              : connection === "online"
                ? "Connected"
                : "Service unavailable · retrying automatically"}
          </span>
          {connection !== "online" && (
            <button
              className="service-retry"
              disabled={connection === "checking"}
              onClick={() => void connect()}
            >
              {connection === "checking" ? (
                <LoaderCircle size={12} className="spin" />
              ) : (
                <RefreshCw size={12} />
              )}
              Reconnect
            </button>
          )}
          <span className="service-research">Research use only</span>
        </div>
        {activeTab === "overview" && (
          <Overview
            status={status}
            onStart={() => changeTab("patient")}
            onExport={exportEvidence}
          />
        )}
        <div hidden={activeTab !== "patient"}>
          <PatientData
            status={status ?? offlineSchema}
            onComplete={(analysis) => {
              setResult(analysis);
              setActiveTab("results");
              window.history.pushState(null, "", "#results");
              window.scrollTo({ top: 0, behavior: "instant" });
            }}
          />
        </div>
        {activeTab === "results" && (
          <Results
            result={result}
            onStart={() => changeTab("patient")}
            clinicalModelName={String(
              status?.clinical_model.metadata?.type ?? "Clinical model",
            )}
          />
        )}
        {activeTab === "explainability" && (
          <Explainability
            result={result}
            onStart={() => changeTab("patient")}
          />
        )}
      </main>
      <div
        className={`notification ${notification ? "visible" : ""}`}
        role="status"
      >
        {notification}
      </div>
    </div>
  );
}
