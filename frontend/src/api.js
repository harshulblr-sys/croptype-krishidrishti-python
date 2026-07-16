// Thin client for aoi_server.py. In dev, vite proxies /api to :8000;
// in production set VITE_API_BASE to the deployed backend origin.
const BASE = import.meta.env.VITE_API_BASE || "";

async function req(path, opts) {
  const r = await fetch(BASE + path, opts);
  let body = null;
  try {
    body = await r.json();
  } catch {
    /* non-JSON (shouldn't happen on /api) */
  }
  if (!r.ok) {
    const msg = body && body.detail ? body.detail : `HTTP ${r.status}`;
    const err = new Error(msg);
    err.status = r.status;
    throw err;
  }
  return body;
}

export const getLimits = () => req("/api/limits");

export const submitJob = (bbox, year) =>
  req("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      west: bbox[0],
      south: bbox[1],
      east: bbox[2],
      north: bbox[3],
      year,
    }),
  });

export const getJob = (id) => req(`/api/jobs/${id}`);
export const getResults = (id) => req(`/api/jobs/${id}/results`);
export const dashboardUrl = (id) => `${BASE}/api/jobs/${id}/dashboard`;
export const fileUrl = (path) => BASE + path;

// Mirrors aoi_server.bbox_px: AOI size in 10 m pixels, snapped to 256.
export function bboxPx(w, s, e, n) {
  const lat = (s + n) / 2;
  const wm = (e - w) * 111320 * Math.cos((lat * Math.PI) / 180);
  const hm = (n - s) * 110540;
  return [
    Math.ceil(wm / 10 / 256) * 256,
    Math.ceil(hm / 10 / 256) * 256,
  ];
}

export function bboxKm(w, s, e, n) {
  const lat = (s + n) / 2;
  return [
    ((e - w) * 111320 * Math.cos((lat * Math.PI) / 180)) / 1000,
    ((n - s) * 110540) / 1000,
  ];
}

// Client-side mirror of the server-side gate, for live feedback while drawing.
export function validateBbox(bbox, limits) {
  if (!bbox) return { ok: false, reason: "Draw an area of interest first" };
  const [w, s, e, n] = bbox;
  if (!(w < e && s < n)) return { ok: false, reason: "Degenerate box" };
  if (limits) {
    const [zw, zs, ze, zn] = limits.supported_bbox;
    if (!(zw <= w && e <= ze && zs <= s && n <= zn))
      return {
        ok: false,
        reason:
          "Outside the supported northern-India zone — the crop model has no training data further south",
      };
    const [wpx, hpx] = bboxPx(w, s, e, n);
    if (Math.max(wpx, hpx) > limits.max_px_side)
      return {
        ok: false,
        reason: `Too large (${wpx}×${hpx} px at 10 m) — max ~${limits.max_km_side} km per side`,
      };
    const validated = Object.values(limits.pilots).some(
      ([bw, bs, be, bn]) => bw <= w && e <= be && bs <= s && n <= bn
    );
    return { ok: true, validated };
  }
  return { ok: true, validated: false };
}
