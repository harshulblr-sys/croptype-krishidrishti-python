import { useMemo, useState } from "react";
import { dashboardUrl, fileUrl } from "./api.js";

// Fixed per-crop colors (dark-surface categorical slots, validated) —
// color follows the crop, never its rank in the list.
export const CROP_COLORS = {
  Wheat: "#c98500",
  Mustard: "#d95926",
  Lentil: "#9085e9",
  "No crop/Fallow": "#898781",
  Sugarcane: "#199e70",
  Maize: "#008300",
  Rice: "#3987e5",
  Other: "#d55181",
};

const fmtPct = (x) => `${Math.round(x * 100)}%`;

function fmtVolume(m3) {
  if (m3 >= 1e6) return `${(m3 / 1e6).toFixed(2)} Mm³`;
  if (m3 >= 1e3) return `${Math.round(m3 / 1e3).toLocaleString()}k m³`;
  return `${Math.round(m3)} m³`;
}

// ---------------------------------------------------------------- progress

export function StageProgress({ job, stages }) {
  const done = job.stages_done || {};
  const nDone = Object.keys(done).length;
  const frac = stages.length ? nDone / stages.length : 0;
  return (
    <div className="card">
      <div className="row-between">
        <span className="card-title">
          {job.status === "queued" ? "Queued…" : "Running pipeline"}
        </span>
        <span className="muted">
          {job.elapsed_s != null ? `${Math.round(job.elapsed_s)}s` : ""}
        </span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${frac * 100}%` }} />
      </div>
      <ul className="stage-list">
        {stages.map((st) => {
          const isDone = st in done;
          const isCurrent = job.stage === st;
          return (
            <li
              key={st}
              className={isDone ? "stage done" : isCurrent ? "stage current" : "stage"}
            >
              <span className="stage-mark">
                {isDone ? "✓" : isCurrent ? <span className="spinner" /> : "·"}
              </span>
              <span className="stage-name">{st.replace(/_/g, " ")}</span>
              {isDone && <span className="stage-time">{Math.round(done[st])}s</span>}
            </li>
          );
        })}
      </ul>
      {job.log_tail && job.log_tail.length > 0 && (
        <pre className="log-tail">{job.log_tail.join("")}</pre>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- results

function CropMix({ crops }) {
  const rows = useMemo(
    () =>
      Object.entries(crops)
        .map(([name, c]) => ({ name, ...c }))
        .sort((a, b) => b.n - a.n),
    [crops]
  );
  const total = rows.reduce((s, r) => s + r.n, 0);
  const max = rows.length ? rows[0].n : 1;
  return (
    <div className="cropmix">
      {rows.map((r) => (
        <div key={r.name} className="cropmix-row" title={`${r.name}: ${r.n} fields`}>
          <span className="cropmix-label">{r.name}</span>
          <span className="cropmix-bar-cell">
            <span
              className="cropmix-bar"
              style={{
                width: `${Math.max(2, (r.n / max) * 100)}%`,
                background: CROP_COLORS[r.name] || "#898781",
              }}
            />
          </span>
          <span className="cropmix-n">
            {r.n.toLocaleString()}
            <span className="muted"> · {fmtPct(r.n / total)}</span>
          </span>
        </div>
      ))}
    </div>
  );
}

function DeficitSpark({ deficit }) {
  const [hover, setHover] = useState(null);
  const W = 320;
  const H = 74;
  const PAD = { l: 6, r: 6, t: 10, b: 4 };
  const vals = deficit.mean_mm;
  const maxV = Math.max(...vals, 1);
  const iMax = vals.indexOf(Math.max(...vals));
  const x = (i) => PAD.l + (i / (vals.length - 1)) * (W - PAD.l - PAD.r);
  const y = (v) => H - PAD.b - (v / maxV) * (H - PAD.t - PAD.b);
  const path = vals.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join("");
  const area = `${path}L${x(vals.length - 1)},${H - PAD.b}L${x(0)},${H - PAD.b}Z`;
  const onMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const i = Math.max(
      0,
      Math.min(
        vals.length - 1,
        Math.round(((px - PAD.l) / (W - PAD.l - PAD.r)) * (vals.length - 1))
      )
    );
    setHover(i);
  };
  const hi = hover ?? iMax;
  return (
    <div className="spark-wrap">
      <div className="row-between">
        <span className="mini-title">Unmet water demand, mm per 8-day period</span>
        <span className="spark-readout">
          {deficit.labels[hi]}: <b>{vals[hi].toFixed(1)} mm</b>
        </span>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="spark"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        <path d={area} fill="#3987e5" opacity="0.18" />
        <path d={path} fill="none" stroke="#3987e5" strokeWidth="2" />
        <line
          x1={x(hi)}
          x2={x(hi)}
          y1={PAD.t - 4}
          y2={H - PAD.b}
          stroke="#898781"
          strokeWidth="1"
          strokeDasharray="2,2"
        />
        <circle cx={x(hi)} cy={y(vals[hi])} r="3.5" fill="#3987e5" stroke="#1a1a19" strokeWidth="1.5" />
      </svg>
      <div className="spark-axis">
        <span>{deficit.labels[0]}</span>
        <span>{deficit.labels[deficit.labels.length - 1]}</span>
      </div>
    </div>
  );
}

const KEY_FILES = [
  ["advisory_summary.csv", "Advisory summary (CSV)"],
  ["deficit_fields.csv", "Per-field deficit (CSV)"],
  ["advisory_fields.csv", "Per-field advisory (CSV)"],
  ["maps/region_grid.png", "Advisory map grid (PNG)"],
  ["maps/deficit_grid.png", "Deficit map grid (PNG)"],
];

export function ResultsPanel({ job, results, onReset }) {
  const s = results.summary;
  const seasonM3 = s.deficit.total_m3.reduce((a, b) => a + b, 0);
  const peakI = s.deficit.mean_mm.indexOf(Math.max(...s.deficit.mean_mm));
  const files = results.files || [];
  const keyFiles = KEY_FILES.map(([suffix, label]) => {
    const f = files.find((p) => p.endsWith("/" + suffix));
    return f ? { url: f, label } : null;
  }).filter(Boolean);
  const geotiffs = files.filter((f) => f.includes("geotiff"));

  return (
    <div className="card">
      <div className="row-between">
        <span className="card-title">Results — {s.season}</span>
        {results.validated ? (
          <span className="badge ok">VALIDATED PILOT</span>
        ) : (
          <span className="badge warn">EXPERIMENTAL</span>
        )}
      </div>

      <div className="kpi-grid">
        <div className="kpi">
          <span className="kpi-value">{s.n_fields.toLocaleString()}</span>
          <span className="kpi-label">fields mapped</span>
        </div>
        <div className="kpi">
          <span className="kpi-value">{fmtVolume(seasonM3)}</span>
          <span className="kpi-label">season water deficit</span>
        </div>
        <div className="kpi">
          <span className="kpi-value">{s.deficit.mean_mm[peakI].toFixed(1)} mm</span>
          <span className="kpi-label">peak 8-day deficit ({s.deficit.labels[peakI]})</span>
        </div>
        <div className="kpi">
          <span className="kpi-value">{fmtPct(s.sowing.detected_frac)}</span>
          <span className="kpi-label">sowing dates observed</span>
        </div>
        <div className="kpi">
          <span className="kpi-value">{s.lstm.test.r2.toFixed(2)}</span>
          <span className="kpi-label">LSTM Ks R² vs FAO-56</span>
        </div>
        <div className="kpi">
          <span className="kpi-value">{s.lstm.test.mae.toFixed(3)}</span>
          <span className="kpi-label">LSTM Ks MAE</span>
        </div>
      </div>

      <div className="mini-title">Predicted crop mix</div>
      <CropMix crops={s.crops} />

      <DeficitSpark deficit={s.deficit} />

      <a
        className="btn primary block"
        href={dashboardUrl(job.job_id || job.id)}
        target="_blank"
        rel="noreferrer"
      >
        Open full dashboard ↗
      </a>

      {keyFiles.length > 0 && (
        <div className="filelist">
          {keyFiles.map((f) => (
            <a key={f.url} href={fileUrl(f.url)} target="_blank" rel="noreferrer">
              ⤓ {f.label}
            </a>
          ))}
        </div>
      )}
      <details className="file-details">
        <summary>
          All outputs ({files.length} files{geotiffs.length ? `, ${geotiffs.length} GeoTIFFs` : ""})
        </summary>
        <div className="filelist small">
          {files.map((f) => (
            <a key={f} href={fileUrl(f)} target="_blank" rel="noreferrer">
              {f.split("/files/")[1] || f}
            </a>
          ))}
        </div>
      </details>

      <button className="btn ghost block" onClick={onReset}>
        Analyze another area
      </button>
    </div>
  );
}
