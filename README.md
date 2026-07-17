# KrishiDrishti 🌾🛰️

**Draw a box anywhere in northern India → per-field crop map, moisture stress, 8-day crop-water deficit and an irrigation advisory — in ~2–3 minutes.**

A full-stack AI + remote-sensing system built for ISRO Problem Statement 6 (AI-driven crop type, moisture-stress detection & irrigation advisory from optical + microwave satellite data), extended into a deployable web service.

![Field-level crop-type map](docs/images/crop_map.png)
*Field-level (10 m) crop-type map for a 9×8 km AOI in Uttar Pradesh — wheat (amber), mustard (magenta), sugarcane (teal), maize (green).*

## What it does

| Capability | How |
|---|---|
| **8-class crop map** (per field, 10 m) | LightGBM on 1,305-dim Sentinel-1 + Sentinel-2 time-series features, + TempCNN rice-rescue |
| **Sowing dates** | Per-field NDVI green-up detection (87% observed directly) |
| **Moisture stress** | Stage-aware FAO-56 water balance × spectral/SAR stress indicators |
| **8-day water deficit** | Rainfed FAO-56 bucket per (weather cell × crop × sowing bin), mm and m³ |
| **Irrigation advisory** | 5-level, fixed auditable rules — physics and observation must agree |
| **Satellite-only stress** | Causal LSTM Ks emulator (R² 0.96 vs FAO-56 on held-out chips) for weather-sparse operation |
| **Web service** | Draw an AOI → GEE fetch (~40 s) → 11-stage pipeline → interactive dashboard + GeoTIFF/CSV downloads |

**Accuracy (honest numbers):** test OA 0.78 / macro-F1 0.55 / κ 0.68; 5-fold chip-level CV OA ~0.73. Validated on the four AgriFieldNet pilot regions; everywhere else is explicitly badged *experimental*.

📄 **Full write-up:** [PROJECT_REPORT.md](PROJECT_REPORT.md) · [PDF](docs/KrishiDrishti_Technical_Report.pdf)

## Run it yourself (on your own machine)

Anyone can run the full web app locally — you don't need the training data,
just the code, the trained models, and your own (free) Earth Engine access.

**Prerequisites**
1. **Python 3.10+** and **Node 18+**.
2. **The trained models** — download `models.zip` from this repo's
   [Releases](../../releases) and unzip it into the project root so you have
   `runs/final_classifier_rebuilt/`, `runs/tempcnn_rebuilt/`,
   `runs/stress_lstm/` (these are ~11 MB, not stored in git).
3. **Your own Earth Engine account** (free for non-commercial use):
   sign up at [earthengine.google.com](https://earthengine.google.com), which
   gives you a Google Cloud project id.

**Setup**
```bash
pip install -r requirements.txt
earthengine authenticate                     # one-time browser login
export GEE_PROJECT=your-cloud-project-id     # Windows: set GEE_PROJECT=...

cd frontend && npm install && npm run build && cd ..
python aoi_server.py                          # → http://127.0.0.1:8000
```

On Windows, `start_servers.cmd` runs the server in its own window.
For frontend development with hot reload: `cd frontend && npm run dev`
(Vite on :5173, proxies `/api` to :8000).

> **Reproducing the models / dataset** (not needed just to run the app):
> the AgriFieldNet chips and satellite tiles aren't in this repo — see
> [handoff.md](handoff.md) and run `rebuild_metadata.py` →
> `finalize_classifier.py`. Deployment guides (VM, Cloudflare Tunnel) are
> in [deploy/](deploy/).

## Repository layout

```
├── README.md / PROJECT_REPORT.md / handoff.md   docs & development log
├── docs/                       report PDF, figures, problem statement
├── deploy/                     systemd unit · Caddyfile · hosting guide
├── Dockerfile                  container build (VM / Hugging Face Spaces)
├── frontend/                   React + MapLibre draw-AOI web UI
│
│   — web service —
├── aoi_server.py               FastAPI service (jobs API + static frontend)
├── aoi_run.py                  orchestrator: bbox → full pipeline
├── aoi_prepare.py / aoi_classify.py             AOI fetch + classification
├── gee_auth.py                 EE auth (service account / personal login)
├── gee_export_s2.py            GEE compositing recipes (shared train/serve)
├── spike_gee_fetch.py          GEE latency benchmark
│
│   — model training —
├── rebuild_metadata.py / make_splits.py / build_features.py
├── train_field_tempcnn.py / finalize_classifier.py
│
│   — stress → advisory → dashboard —
├── stress_common.py            season/crop constants (env-parameterized)
├── weather_et0.py / stress_indices.py / sowing_detect.py
├── water_balance.py / advisory.py / stress_lstm.py
├── advisory_maps.py / deficit_maps.py / dashboard_data.py
├── dashboard_template.html     self-contained dashboard shell
│
├── data_prep/                  one-time training-dataset construction
└── experiments/                research studies & negative results
                                (the evidence behind the honest 0.73 CV)
```

## License & data

Code © Harshul Aggarwal. AgriFieldNet ground reference data © Radiant Earth Foundation / IDinsight (CC-BY-4.0). Contains modified Copernicus Sentinel data. Weather from NASA POWER.
