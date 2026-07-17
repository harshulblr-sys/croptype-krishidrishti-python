"""FastAPI backend around aoi_run.py — the web service for draw-a-box PS-6.

  python aoi_server.py [--port 8000] [--workers 2]

Endpoints (JSON unless noted):
  GET  /                      minimal built-in test page (submit bbox, watch
                              progress, open dashboard) — placeholder until
                              the React/MapLibre frontend exists
  GET  /api/limits            supported zone, pilot bboxes, max AOI size
  POST /api/jobs              {west,south,east,north,year} -> job (dedup by
                              rounded bbox+year: same AOI = same job/cache)
  GET  /api/jobs/{id}         status: queued|running|done|failed + stage
                              progress + log tail
  GET  /api/jobs/{id}/results summary (crop counts, deficit, sowing, LSTM)
                              + file listing
  GET  /api/jobs/{id}/dashboard        the self-contained dashboard.html
  GET  /api/jobs/{id}/files/{path}     maps / GeoTIFFs / CSVs (read-only,
                                       restricted to the job's output dir)

Protections (per the design discussion):
  - northern-India gate + AOI size cap (server-side, mirrors aoi_run.py)
  - dedupe: resubmitting a computed AOI returns the cached result instantly
    (aoi_run --resume also makes half-finished jobs cheap to retry)
  - one ACTIVE job per client IP; submission rate limit (only GEE-costing
    submissions count — cache hits are free)
  - global concurrency cap (default 2 pipeline subprocesses; rest queue)
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "aoi_runs")
DIST = os.path.join(ROOT, "frontend", "dist")   # built React app (npm run build)
SUPPORTED = (68.0, 19.0, 89.0, 32.5)          # keep in sync with aoi_run.py
PILOTS = {
    "UP": (81.13, 27.07, 82.74, 28.33), "BIHAR": (87.20, 25.27, 88.05, 25.88),
    "ODISHA": (83.00, 19.01, 83.97, 19.92), "RAJASTHAN": (76.25, 24.41, 77.31, 25.43),
}
MAX_PX = 1024                                  # ~10x10 km (matches aoi_prepare)
MAX_CONCURRENT = 2
RATE_LIMIT_N, RATE_LIMIT_WINDOW_S = 5, 600     # fresh submissions per IP
STAGES = ["prepare", "classify", "weather", "indices", "sowing",
          "water_balance", "advisory", "lstm", "advisory_maps",
          "deficit_maps", "dashboard"]

app = FastAPI(title="PS-6 AOI service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

jobs = {}                                      # id -> dict (the registry)
jobs_lock = threading.Lock()
sem = threading.Semaphore(MAX_CONCURRENT)
ip_submits = defaultdict(deque)                # ip -> recent submit times


class JobRequest(BaseModel):
    west: float
    south: float
    east: float
    north: float
    year: int = 2021


def bbox_px(w, s, e, n):
    """Approximate AOI size in 10 m pixels (mirrors aoi_prepare's UTM grid)."""
    lat = (s + n) / 2
    wm = (e - w) * 111320 * math.cos(math.radians(lat))
    hm = (n - s) * 110540
    return math.ceil(wm / 10 / 256) * 256, math.ceil(hm / 10 / 256) * 256


def job_id_for(w, s, e, n, year):
    key = f"{w:.4f},{s:.4f},{e:.4f},{n:.4f},{year}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def out_dir(job):
    return os.path.join(RUNS, f"job_{job['id']}", "moisture_stress")


def run_pipeline(job):
    """Worker thread: run aoi_run.py, stream its stdout into stage progress."""
    with sem:
        job["status"] = "running"
        job["started"] = time.time()
        ws = os.path.join(RUNS, f"job_{job['id']}")
        os.makedirs(ws, exist_ok=True)
        log_path = os.path.join(ws, "run.log")
        cmd = [sys.executable, "-u", os.path.join(ROOT, "aoi_run.py"),
               "--bbox", str(job["bbox"][0]), str(job["bbox"][1]),
               str(job["bbox"][2]), str(job["bbox"][3]),
               "--year", str(job["year"]), "--name", f"job_{job['id']}",
               "--resume"]
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        try:
            with open(log_path, "a", encoding="utf-8") as log:
                p = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace")
                for line in p.stdout:
                    log.write(line)
                    m = re.match(r"== (\w+) ==$", line.strip())
                    if m and m.group(1) in STAGES:
                        job["stage"] = m.group(1)
                    m = re.match(r"== (\w+) done in ([\d.]+)s", line.strip())
                    if m and m.group(1) in STAGES:
                        job["stages_done"][m.group(1)] = float(m.group(2))
                p.wait()
            if p.returncode == 0 and os.path.exists(
                    os.path.join(out_dir(job), "dashboard.html")):
                job["status"] = "done"
            else:
                job["status"] = "failed"
                job["error"] = f"pipeline exit {p.returncode} (see log)"
        except Exception as e:                          # noqa: BLE001
            job["status"] = "failed"
            job["error"] = str(e)[:300]
        job["finished"] = time.time()
        job["stage"] = None
        with open(os.path.join(ws, "job.json"), "w") as f:
            json.dump({k: v for k, v in job.items() if k != "thread"}, f,
                      indent=1, default=str)


def load_existing_jobs():
    """Recover finished jobs from disk after a server restart."""
    if not os.path.isdir(RUNS):
        return
    for d in os.listdir(RUNS):
        jp = os.path.join(RUNS, d, "job.json")
        if d.startswith("job_") and os.path.exists(jp):
            try:
                with open(jp) as f:
                    j = json.load(f)
                if j.get("status") == "done":
                    jobs[j["id"]] = j
            except Exception:                           # noqa: BLE001
                pass
    if jobs:
        print(f"recovered {len(jobs)} finished job(s) from {RUNS}")


@app.get("/api/limits")
def limits():
    return dict(supported_bbox=SUPPORTED, pilots=PILOTS,
                max_px_side=MAX_PX, max_km_side=MAX_PX / 100,
                years=[2018, 2025], stages=STAGES,
                note="crop map validated only inside pilot bboxes; "
                     "supported zone = northern India")


@app.post("/api/jobs")
def submit(req: JobRequest, request: Request):
    w, s, e, n = req.west, req.south, req.east, req.north
    if not (w < e and s < n):
        raise HTTPException(400, "bbox must have west<east and south<north")
    if not (SUPPORTED[0] <= w and e <= SUPPORTED[2]
            and SUPPORTED[1] <= s and n <= SUPPORTED[3]):
        raise HTTPException(400, "AOI outside the supported northern-India "
                                 f"zone {SUPPORTED}: the crop model has no "
                                 "training data further south.")
    wpx, hpx = bbox_px(w, s, e, n)
    if max(wpx, hpx) > MAX_PX:
        raise HTTPException(400, f"AOI too large ({wpx}x{hpx} px at 10 m); "
                                 f"max ~{MAX_PX // 100} km per side.")
    if not (2017 <= req.year <= 2100):
        raise HTTPException(400, "year out of range (agricultural year, "
                                 "e.g. 2021 = Jun 2021 - Apr 2022)")

    jid = job_id_for(w, s, e, n, req.year)
    ip = request.client.host if request.client else "unknown"
    with jobs_lock:
        j = jobs.get(jid)
        if j and j["status"] in ("queued", "running", "done"):
            return status_payload(j, cached=True)
        # fresh (or retry-after-failure) submission -> rate limits apply
        now = time.time()
        q = ip_submits[ip]
        while q and now - q[0] > RATE_LIMIT_WINDOW_S:
            q.popleft()
        if len(q) >= RATE_LIMIT_N:
            raise HTTPException(429, "rate limit: too many new AOIs from "
                                     "this address; try again later")
        active = [x for x in jobs.values()
                  if x.get("ip") == ip and x["status"] in ("queued", "running")]
        if active:
            raise HTTPException(409, f"a job is already {active[0]['status']} "
                                     f"for this address: {active[0]['id']}")
        q.append(now)
        validated = any(bw <= w and e <= be and bs <= s and n <= bn
                        for bw, bs, be, bn in PILOTS.values())
        j = dict(id=jid, bbox=[w, s, e, n], year=req.year, ip=ip,
                 status="queued", stage=None, stages_done={},
                 validated=validated, submitted=time.time())
        jobs[jid] = j
        t = threading.Thread(target=run_pipeline, args=(j,), daemon=True)
        j["thread"] = t
        t.start()
    return status_payload(j, cached=False)


def status_payload(j, cached=None):
    out = dict(job_id=j["id"], status=j["status"], bbox=j["bbox"],
               year=j["year"], validated=j.get("validated"),
               stage=j.get("stage"), stages_done=j.get("stages_done", {}),
               n_stages=len(STAGES), error=j.get("error"))
    if cached is not None:
        out["cached"] = cached
    if j.get("started"):
        end = j.get("finished") or time.time()
        out["elapsed_s"] = round(end - j["started"], 1)
    return out


@app.get("/api/jobs/{jid}")
def status(jid: str):
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "unknown job")
    out = status_payload(j)
    log = os.path.join(RUNS, f"job_{jid}", "run.log")
    if os.path.exists(log):
        with open(log, encoding="utf-8", errors="replace") as f:
            out["log_tail"] = f.readlines()[-8:]
    return out


@app.get("/api/jobs/{jid}/results")
def results(jid: str):
    j = jobs.get(jid)
    if not j or j["status"] != "done":
        raise HTTPException(404, "job not done")
    od = out_dir(j)
    with open(os.path.join(od, "dashboard_data.json")) as f:
        data = json.load(f)
    files = []
    for base, _, names in os.walk(od):
        rel = os.path.relpath(base, od)
        for nm in names:
            files.append(os.path.join(rel, nm).replace("\\", "/").lstrip("./"))
    return dict(job_id=jid, validated=j.get("validated"),
                summary=dict(n_fields=data["n_fields"], crops=data["crops"],
                             deficit=data["deficit"], sowing=data["sowing"],
                             lstm=data["lstm"], peak_dekad=data["peak_dekad"],
                             season=data["season"]),
                dashboard=f"/api/jobs/{jid}/dashboard",
                files=[f"/api/jobs/{jid}/files/{p}" for p in sorted(files)])


@app.get("/api/jobs/{jid}/dashboard")
def dashboard(jid: str):
    j = jobs.get(jid)
    if not j or j["status"] != "done":
        raise HTTPException(404, "job not done")
    return FileResponse(os.path.join(out_dir(j), "dashboard.html"),
                        media_type="text/html")


@app.get("/api/jobs/{jid}/files/{path:path}")
def files(jid: str, path: str):
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "unknown job")
    od = os.path.realpath(out_dir(j))
    full = os.path.realpath(os.path.join(od, path))
    if not full.startswith(od + os.sep) or not os.path.isfile(full):
        raise HTTPException(404, "no such file")
    return FileResponse(full)


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the built React frontend (frontend/dist); dev test page fallback."""
    idx = os.path.join(DIST, "index.html")
    if os.path.exists(idx):
        with open(idx, encoding="utf-8") as f:
            return f.read()
    return """<!doctype html><meta charset="utf-8">
<title>PS-6 AOI service</title>
<style>body{font-family:system-ui;max-width:640px;margin:40px auto;padding:0 16px}
input{width:7em;margin:2px}button{padding:6px 14px}#st{white-space:pre-wrap;
background:#f4f4f0;border-radius:8px;padding:10px;font-size:13px}</style>
<h2>PS-6 AOI service <small style="color:#888">(dev test page)</small></h2>
<p>bbox (supported zone: northern India, max ~10&times;10 km):</p>
W <input id=w value=81.5247> S <input id=s value=27.4275>
E <input id=e value=81.5753> N <input id=n value=27.4725>
year <input id=y value=2021 style="width:4em">
<button onclick=go()>Run</button>
<p id=st>idle</p><div id=link></div>
<script>
let t;
async function go(){
  clearInterval(t);
  const b={west:+w.value,south:+s.value,east:+e.value,north:+n.value,year:+y.value};
  const r=await fetch('/api/jobs',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const j=await r.json();
  if(!r.ok){st.textContent='ERROR: '+(j.detail||r.status);return}
  st.textContent='job '+j.job_id+' '+j.status+(j.cached?' (cached)':'');
  t=setInterval(async()=>{
    const s2=await (await fetch('/api/jobs/'+j.job_id)).json();
    st.textContent='job '+s2.job_id+' — '+s2.status+
      (s2.stage?('  stage: '+s2.stage):'')+
      '  ['+Object.keys(s2.stages_done).length+'/'+s2.n_stages+' stages]'+
      (s2.elapsed_s?('  '+s2.elapsed_s+'s'):'')+
      (s2.error?('\\n'+s2.error):'')+
      '\\n'+(s2.log_tail||[]).join('');
    if(s2.status==='done'){clearInterval(t);
      link.innerHTML='<a href="/api/jobs/'+s2.job_id+'/dashboard" '+
        'target=_blank>open dashboard</a> &middot; <a href="/api/jobs/'+
        s2.job_id+'/results" target=_blank>results JSON</a>'}
    if(s2.status==='failed')clearInterval(t);
  },2000)}
</script>"""


if os.path.isdir(os.path.join(DIST, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST, "assets")),
              name="assets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--workers", type=int, default=MAX_CONCURRENT,
                    help="max concurrent pipeline runs")
    args = ap.parse_args()
    global sem
    sem = threading.Semaphore(args.workers)
    load_existing_jobs()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
