import { useCallback, useEffect, useRef, useState } from "react";
import MapView from "./MapView.jsx";
import { StageProgress, ResultsPanel } from "./Panels.jsx";
import {
  getLimits,
  submitJob,
  getJob,
  getResults,
  bboxKm,
  bboxPx,
  validateBbox,
} from "./api.js";

const DEMO_AOI = [81.5247, 27.4275, 81.5753, 27.4725]; // validated 5×5 km UP box

export default function App() {
  const [limits, setLimits] = useState(null);
  const [bbox, setBbox] = useState(null);
  const [year, setYear] = useState(2021);
  const [drawMode, setDrawMode] = useState(false);
  const [basemap, setBasemap] = useState("esri");
  const [focusRequest, setFocusRequest] = useState(null);
  const [job, setJob] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [backendUp, setBackendUp] = useState(true);
  const pollRef = useRef(null);

  const stages = limits?.stages || [];
  const validity = validateBbox(bbox, limits);

  // ---- limits on mount + re-attach to a job from a previous visit ----
  useEffect(() => {
    getLimits()
      .then(setLimits)
      .catch(() => setBackendUp(false));
    const saved = localStorage.getItem("ps6.jobId");
    if (saved) attach(saved, { silent: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stopPolling = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = null;
  };

  const finishJob = useCallback(async (j) => {
    setJob(j);
    if (j.status === "done") {
      try {
        setResults(await getResults(j.job_id));
      } catch (e) {
        setError(`Could not load results: ${e.message}`);
      }
    }
  }, []);

  const startPolling = useCallback(
    (id) => {
      stopPolling();
      pollRef.current = setInterval(async () => {
        try {
          const j = await getJob(id);
          setJob(j);
          if (j.status === "done" || j.status === "failed") {
            stopPolling();
            await finishJob(j);
          }
        } catch {
          /* transient poll failure — keep trying */
        }
      }, 2000);
    },
    [finishJob]
  );

  async function attach(id, { silent } = {}) {
    try {
      const j = await getJob(id);
      setJob(j);
      if (j.bbox) setBbox(j.bbox);
      if (j.status === "done") await finishJob(j);
      else if (j.status === "queued" || j.status === "running") startPolling(id);
    } catch {
      localStorage.removeItem("ps6.jobId");
      if (!silent) setError(`Job ${id} not found on the server`);
    }
  }

  useEffect(() => stopPolling, []);

  async function run() {
    if (!validity.ok) return;
    setError(null);
    setResults(null);
    try {
      const j = await submitJob(bbox, year);
      setJob(j);
      localStorage.setItem("ps6.jobId", j.job_id);
      if (j.status === "done") await finishJob(j);
      else startPolling(j.job_id);
    } catch (e) {
      // 409 = a job is already active for this IP — offer to track it
      const m = /:\s*([0-9a-f]{12})/.exec(e.message);
      if (e.status === 409 && m) {
        setError("A job is already running for this address — tracking it.");
        localStorage.setItem("ps6.jobId", m[1]);
        attach(m[1]);
      } else {
        setError(e.message);
      }
    }
  }

  function reset() {
    stopPolling();
    setJob(null);
    setResults(null);
    setError(null);
    setBbox(null);
    localStorage.removeItem("ps6.jobId");
  }

  const busy = job && (job.status === "queued" || job.status === "running");
  const editBbox = (i, v) => {
    const b = bbox ? [...bbox] : [80, 26, 80.05, 26.05];
    b[i] = v;
    setBbox(b);
  };
  const km = bbox ? bboxKm(...bbox) : null;
  const px = bbox ? bboxPx(...bbox) : null;

  return (
    <div className="app">
      <aside className="sidebar">
        <header className="header">
          <h1>KrishiDrishti</h1>
          <p className="subtitle">
            Draw a box over northern India → Sentinel-1/2 crop map, moisture
            stress, 8-day water deficit &amp; irrigation advisory
          </p>
        </header>

        {!backendUp && (
          <div className="alert error">
            Backend unreachable — start it with{" "}
            <code>python aoi_server.py</code>
          </div>
        )}

        {!results && (
          <div className="card">
            <div className="card-title">1 · Area of interest</div>
            <div className="btn-row">
              <button
                className={drawMode ? "btn primary" : "btn"}
                onClick={() => setDrawMode(!drawMode)}
                disabled={busy}
              >
                {drawMode ? "Drag on map…" : bbox ? "Redraw box" : "Draw box"}
              </button>
              <button
                className="btn ghost"
                disabled={busy}
                onClick={() => {
                  setBbox(DEMO_AOI);
                  setFocusRequest({ bounds: DEMO_AOI, nonce: Date.now() });
                }}
              >
                Demo AOI (UP)
              </button>
            </div>
            {limits && (
              <div className="pilot-row">
                {Object.entries(limits.pilots).map(([name, bb]) => (
                  <button
                    key={name}
                    className="chip"
                    onClick={() => setFocusRequest({ bounds: bb, nonce: Date.now() })}
                  >
                    {name}
                  </button>
                ))}
              </div>
            )}

            <div className="bbox-grid">
              {["West", "South", "East", "North"].map((lbl, i) => (
                <label key={lbl}>
                  <span>{lbl}</span>
                  <input
                    type="number"
                    step="0.001"
                    value={bbox ? bbox[i].toFixed(4) : ""}
                    placeholder="—"
                    disabled={busy}
                    onChange={(e) => editBbox(i, parseFloat(e.target.value))}
                  />
                </label>
              ))}
            </div>

            {bbox && (
              <div className={validity.ok ? "bbox-status ok" : "bbox-status bad"}>
                {km[0].toFixed(1)} × {km[1].toFixed(1)} km ({px[0]}×{px[1]} px @10 m)
                {validity.ok ? (
                  validity.validated ? (
                    <span className="badge ok">VALIDATED PILOT</span>
                  ) : (
                    <span className="badge warn">EXPERIMENTAL</span>
                  )
                ) : (
                  <div className="reason">{validity.reason}</div>
                )}
              </div>
            )}

            <div className="year-row">
              <label>
                <span>Agricultural year</span>
                <input
                  type="number"
                  min={limits ? limits.years[0] : 2018}
                  max={limits ? limits.years[1] : 2025}
                  value={year}
                  disabled={busy}
                  onChange={(e) => setYear(parseInt(e.target.value || "2021", 10))}
                />
              </label>
              <span className="muted">
                {year} = Jun {year} – Apr {year + 1} (rabi {year}-
                {String((year + 1) % 100).padStart(2, "0")})
              </span>
            </div>

            <button
              className="btn primary block"
              disabled={!validity.ok || busy || !backendUp}
              onClick={run}
            >
              {busy ? "Pipeline running…" : "Run analysis"}
            </button>
            <p className="muted small">
              ~2–3 min for a fresh AOI (satellite compositing + 11-stage
              pipeline). Already-computed AOIs return instantly.
            </p>
          </div>
        )}

        {error && <div className="alert error">{error}</div>}

        {busy && <StageProgress job={job} stages={stages} />}

        {job && job.status === "failed" && (
          <div className="card">
            <div className="card-title">Job failed</div>
            <p className="muted">{job.error || "See server log."}</p>
            {job.log_tail && <pre className="log-tail">{job.log_tail.join("")}</pre>}
            <button className="btn block" onClick={run}>
              Retry (resumes finished stages)
            </button>
            <button className="btn ghost block" onClick={reset}>
              Start over
            </button>
          </div>
        )}

        {results && job && (
          <ResultsPanel job={job} results={results} onReset={reset} />
        )}

        <footer className="footer">
          <span>
            Sentinel-1/2 via GEE · FAO-56 water balance · LightGBM + LSTM
          </span>
          <button
            className="chip"
            onClick={() => setBasemap(basemap === "esri" ? "osm" : "esri")}
          >
            {basemap === "esri" ? "Map view" : "Satellite view"}
          </button>
        </footer>
      </aside>

      <main className="map-wrap">
        <MapView
          limits={limits}
          bbox={bbox}
          onBbox={setBbox}
          drawMode={drawMode && !busy}
          onDrawModeEnd={() => setDrawMode(false)}
          focusRequest={focusRequest}
          basemap={basemap}
        />
        {drawMode && (
          <div className="map-hint">Click and drag to draw the area of interest</div>
        )}
      </main>
    </div>
  );
}
