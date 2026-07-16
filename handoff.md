# Crop Classification + Moisture Stress Pipeline — Handoff

> Rebuilt 2026-07-13 after the original scripts/handoff were lost (probable
> OneDrive sync event — only datasets and `runs/` artifacts survived). The
> classifier is rebuilt and validated at parity (§4). 2026-07-14: data-source
> scouting done (§5b) — no free OA lever remains — AND the **moisture-stress +
> irrigation-advisory half is now BUILT end-to-end (§6)**, so both halves of
> PS-6 exist. **Back this folder's `.py`/`.md`/`.json` files up under git.**

## 1. Problem

ISRO PS-6 (now a personal project): AI-driven crop-type classification,
moisture-stress detection, and irrigation advisory from Sentinel-1/2
time series. See `isro_hackathon.docx`. Classification half is built
(§3–5); moisture-stress + irrigation-advisory half is now built too (§6).

## 2. Data (all intact)

- `Extracted_dataset_gee/chips/` — 1151 npz chips, 256×256 @10m, T=11
  monthly composites Jun 2021–Apr 2022 (matches AgriFieldNet label season).
  Regions: UP+UPNEW (537, region_id 0), Bihar (176, id 1), Odisha (200, id 2,
  **descending-only S1**), Rajasthan (238, id 3).
  Keys: `awifs` [11,H,W,2]=(NDVI,NDMI) — S2-derived, name is legacy;
  `s1_asc` [11,H,W,3]=(VV,VH,ratio dB/linear); `redge` (NDRE,CIre);
  `opt2` (EVI,gNDWI); `redge2` (RE-NDVI,CIre07/05); `flower` (NDYI, unused);
  `season_feat`/`flood_feat` (dekadal, tested = noise, unused); masks;
  `s2` static [H,W,4]; `crop_id`/`field_id`/`label_mask`/`region_id`.
- `Satellite_data/agrifieldnet_s2_2021` — 55-band tifs (11 mo × B02/03/04/08/11),
  `agrifieldnet_s2re_2021` — 44-band (B05/06/07/8A). Raw-band features are read
  from these at build time (not stored in npz). Ignore `godavari_*` (abandoned
  Andhra supplement — didn't transfer).
- Labels: AgriFieldNet 2021-22 (`agrifieldnet_data/`, `Pilot_Area_Labels_*`).
  5771 fields, 13 raw crops; avg field ~26 px (0.26 ha).

## 3. Pipeline scripts (rebuilt 2026-07-13)

Run in order:

1. `rebuild_metadata.py` → `Extracted_dataset_gee/metadata.json`
   (crop maps, field counts, chip index).
2. `make_splits.py` → `splits.json`: chip-level stratified 80/10/10
   (stratum = region × rarest-present crop). 927/112/112 chips.
   **Original split was lost — current OA numbers are on the new split and
   not directly comparable to pre-loss single-split numbers.**
3. `build_features.py` → `field_table.npz` (X[n,1305]) + `field_seq.npz`
   (TempCNN sequences). ~15 min. 1305 = 261 per-pixel feats × 5 field stats
   (mean/std/median/min/max). Per-pixel 261 = base 82 (NDVI/NDMI 22 +
   opt-mask 11 + region-normalized VV/VH/ratio 33 + SAR-mask 11 + static S2 4 +
   region 1) + red-edge 22 + re-mask 11 + indices2 44 + raw bands 99 +
   kharif SAR flooding 3. S1 region-norm fit on train only.
4. `train_field_tempcnn.py` → `runs/tempcnn_rebuilt/best_model.pt`.
   13-class field-level TempCNN (7-ch monthly seq + 8 static), used only as
   the rice-rescue specialist.
5. `finalize_classifier.py` → `runs/final_classifier_rebuilt/`:
   8-class LightGBM (500 trees/31 leaves/lr .05/balanced) + rice-rescue
   (LGB-predicted Fallow flipped to Rice where P_tempcnn(rice) > 0.70).

8-class scheme: Wheat, Mustard, Lentil, No crop/Fallow, Sugarcane, Maize,
Rice, Other.

## 4. Where the accuracy stands

**Rebuilt model (2026-07-13, new split) — validated at parity with pre-loss:**
- Test **OA 0.7805 / macroF1 0.548 / kappa 0.680** (no rescue).
  Rescue flips 1 Fallow→Rice: OA 0.7787 / macroF1 0.542 / Rice F1 0.40.
  Artifacts: `runs/final_classifier_rebuilt/{lgb.joblib,report.txt,meta.json}`
  + `runs/tempcnn_rebuilt/best_model.pt` (val macroF1 0.346).
- LGB hyperparameter sweep (`lgb_variant_sweep.py`): the original
  500/31/.05/balanced config wins on val; unweighted / 1200-tree / 63-leaf
  variants all ≤ it → 5th independent confirmation of the ~0.73 wall.

Pre-loss reference (different, now-unrecoverable split): OA 0.780 / mF1 0.528
/ kappa 0.689 (`runs/final_classifier/report.txt`). 5-fold chip-level CV
**OA ~0.730** remains the honest generalization number.

## 5. What was tried and is DEAD (do not retry)

Exhaustively benchmarked, all flat at the ~0.73 CV wall: phenology feats,
GLCM texture, red-edge indices (marginal, kept), S1 RVI/entropy, monthly+
dekadal NDYI flowering (0σ), dekadal timing feats, field-core erosion,
SAR flooding for trees, label-noise cleaning (fields genuinely ambiguous),
TempCNN prob stacking, soft-vote ensembles, focal loss, deep spatial models
(hybrid TempCNN-UNet 0.52, pretrained UNet 0.37 — signal is temporal),
Godavari/AP rice supplement (didn't transfer). Raw 9-band time series =
only keeper (macroF1 +0.013 CV).

Root cause of the wall: Wheat↔Mustard ~1σ spectral/phenology overlap at 10m
on 0.26 ha mixed fields (410 swapped fields ≈ 7 OA points), plus rare-class
scarcity. Levers that would genuinely move OA: more labeled W/M fields,
hyperspectral (PRISMA/EnMAP), or reporting consolidated schemes.

Static-s2 block ablation (2026-07-16, investigating whether the AOI
distribution-match approximation caps OA): the 4 static bands are AgriFieldNet
source tiles stored as **uint8 0-255** (verified: source B02-B08 dtype uint8,
range 34-82 = the npz values exactly) — a byte-scaled display composite, NOT
reflectance, NOT the GEE SR /10000 scale, and NOT related to the S2_SR DN+1000
harmonization shift (that's on the temporal features, and gee_export already
used S2_SR_HARMONIZED — verified no boundary discontinuity in exported
series). The block's 20 features carry 2.9% of LGB split-importance (best
column ranks #4/1305); permuting them costs 2.3 test-OA points, BUT retraining
without them changes test OA by +0.0017 (0.7805→0.7822) — i.e. informative but
fully REDUNDANT with the 99 raw temporal bands. So static-s2 is not the wall
and not a hidden OA cap, and the AOI approximation of it is low-risk (the same
signal is reproduced cleanly in the raw bands). Note HARMONIZED normalizes all
seasons to the pre-2022 baseline, so --year 2024 AOIs stay train-consistent.

## 5b. Data-source scouting (2026-07-14) — no free OA lever remains

Investigated whether new data could break the wall. Scripts:
`rdm_india_crops.py`, `hyperspectral_coverage_check.py` (+ JSON outputs).

- **AgriFieldNet — EXHAUSTED (proven).** source=1217 tiles, extracted=1151,
  labeled chip-ids=1151 (exact match), labeled-but-not-extracted=0. The 66
  leftover tiles are the competition **test holdout** — no crop labels were
  ever released (no `_labels_test_` files exist, none downloadable). Nothing
  more to extract.
- **WorldCereal RDM — negligible for India.** Live public REST query
  (`ewoc-rdm-api.iiasa.ac.at`, no auth): only 12 collections intersect the
  India bbox, mostly neighbours/global. One genuinely-Indian collection =
  `2018_ind_cgiargardian` (762 **wheat-only** points, Haryana, 2017-18, 0
  mustard/rice). Best W/M asset nearby = `2023_pak_adbrabi` (24.9k wheat +
  2.8k rapeseed/mustard **points**, Pakistan Punjab, 2023) — a transfer/
  pretraining candidate only, not India and not 2021-22. EWOC codes: wheat =
  prefix `110101`, rice `110108`, oilseed mustard/rapeseed `110600003x`/
  `110600008x` (`1103080130` mustard_greens is a vegetable, NOT the class).
  Extraction path verified: `GET /collections/{id}/download` → text URL →
  Azure-blob GeoParquet (Point geom; cols crop/ewoc_code/valid_time/
  quality_score_ct). **Verdict: won't get India to 0.80.**
- **Hyperspectral — viable only for a NEW season.** EnMAP has an open STAC
  (`geoservice.dlr.de/eoc/ogc/stac/v1`, `ENMAP_HSI_L2A`). Its archive starts
  **2022-04-27** (launch 2022-04-01) → **zero coverage of the 2021-22
  labels**. But it *does* cover all four pilot areas with clear (0% cloud)
  rabi scenes from FY2023 on: Rajasthan 8 clear rabi dates (best;
  2024-02-11/15/19/23), UP+UPNEW 5, Bihar 7, Odisha 4. So a hyperspectral
  Wheat↔Mustard study is physically viable **only if paired with new
  ground-truth for a recent rabi season (2023-24+)**. PRISMA (2019) and
  HySIS (2018) are the only season-matched options but are **login-gated**
  (`prismauserregistration.asi.it` / `bhoonidhi.nrsc.gov.in`) — draw the
  pilot bbox and filter Nov 2021–Apr 2022 manually. Pilot bboxes (W,S,E,N):
  UP+UPNEW 81.13/27.07/82.74/28.33, RAJ 76.25/24.41/77.31/25.43,
  BIHAR 87.20/25.27/88.05/25.88, ODISHA 83.00/19.01/83.97/19.92.

**Net:** no free lever left to lift the 2021-22 OA. Real options are
(a) class consolidation for the reported number, (b) a manual PRISMA/HySIS
2021-22 check (coin-flip on sparse tasking; whole new static-spectral
pipeline), or (c) a new-season EnMAP + new-labels study (highest rigor).
Recommendation: proceed to §6.

## 6. Moisture-stress + irrigation advisory (BUILT 2026-07-14)

Demonstrator region = **UP + UPNEW** (region_id 0, 537 chips / 2992 fields),
rabi 2021-22 (Nov 2021–Apr 2022) on 18 ten-day dekads. All outputs land in
`moisture_stress/`. Shared constants/crop params/FAO-56 stage model live in
`stress_common.py`. Run the scripts in this order:

> **2026-07-14 (later): upgraded end-to-end (§6b)** — per-field sowing dates
> from NDVI green-up (caveat C fixed), water balance per (cell × crop ×
> sow-bin) with 8-day deficit reporting, deficit maps/GeoTIFFs, an LSTM
> satellite-only Ks emulator, and a rebuilt dashboard. New run order:
> stress_crop_map → weather_et0 → stress_indices → **sowing_detect** →
> water_balance → advisory → **stress_lstm** → advisory_maps →
> **deficit_maps** → dashboard_data (now also assembles dashboard.html).

1. `stress_crop_map.py` → `crop_map.npz` (every region-0 field classified by
   `runs/final_classifier_rebuilt/lgb.joblib`; split membership kept — this is
   the product map, not an eval) + `chip_index.json` (per-chip CRS / affine /
   centroid lon-lat, the georef spine for every later stage). Predicted crop
   mix: Wheat 1721, Mustard 539, Fallow 404, Sugarcane 173, Lentil 85,
   Other 58, Maize 12, Rice 0. Label agreement on held-out chips 0.70 (val) /
   0.74 (test) — consistent with the classifier's own numbers.
2. `weather_et0.py` → `weather_daily.csv` + `weather_cells.json`. Pulls NASA
   POWER daily met (T/RH/wind/solar/rain) for a 3×3 grid (0.5°) over the UP
   bbox, Oct 15 2021–Apr 30 2022, computes **FAO-56 Penman-Monteith ET0** per
   day/cell. Cached raw JSON in `weather/` (skip-existing). Seasonal ET0
   ~550–800 mm, rain ~180–290 mm across cells. NOTE the NE grid cells report
   inflated POWER elevations (790–1919 m — SRTM edge/Himalayan foothills bleed)
   which nudges their ET0; low-lying cells (120–400 m) are the ones actually
   used by the chips.
3. `stress_indices.py` → `field_timeseries.npz`. Per-field dekadal NDVI/NDRE
   (from `agrifieldnet_season_2022` SEASON tiles, linear gap-fill), monthly
   NDMI/VV/VH (from chip npz), an S1 **soil-moisture proxy** (VV scaled between
   each field's seasonal dry/wet dB refs), and single-season peer statistics:
   NDVI anomaly z + **VCI** (percentile vs same-crop fields — no multi-year
   climatology exists, peers substitute), NDMI z. Valid fractions ndvi 0.67 /
   ndmi 0.79 / vv 0.98.
4. `water_balance.py` → `water_balance.json` + `chip_cell.json`. Stage-aware
   FAO-56 daily root-zone bucket per (weather cell × crop): ETc = Kc(stage)×ET0,
   effective rainfall, depletion vs RAW=p·TAW trigger. Two runs: **scheduled**
   (refills at trigger → the advised irrigation calendar) and **rainfed** (Ks
   stress coefficient if nobody irrigates). Kc curves + sowing calendars for
   Wheat/Mustard/Lentil/Sugarcane + a generic rabi fallback in
   `stress_common.CROP_PARAMS`. Wheat comes out needing ~2–3 irrigations
   (~155–235 mm), mostly late Feb–Mar as ET0 climbs and rain stops.
5. `advisory.py` → `advisory.npz` + `advisory_summary.csv` +
   `advisory_fields.csv`. **Transparent rule fusion** (not a classifier) of the
   water-balance state (depletion, Ks, stage sensitivity, trigger) with observed
   spectral stress (VCI, NDVI-z, NDMI-z, SSM) into a 5-level advisory per field
   per dekad: −1 out-of-season, 0 no stress, 1 watch, 2 irrigation-advised,
   3 severe. Kharif/bare fields with no green rabi signal → out-of-season.
   Peak stress dekad = **2022-02 d3** (733 fields ≥ advised).
6. `advisory_maps.py` → `maps/`: 18 per-dekad region PNGs + `region_grid.png`
   overview, **537 georeferenced field-level advisory GeoTIFFs** (`maps/geotiff/`,
   uint8 with colormap, GIS-ready), and example field maps over the NDVI
   backdrop for the busiest chips (`maps/chips_11/`).
7. `dashboard_data.py` → `dashboard_data.json` (77 KB compact payload).
   `dashboard_template.html` + inline JSON → **`dashboard.html`** (self-contained,
   no server; dataviz status palette; region timeline, ET0-vs-rain, per-crop
   water-need bars, an interactive dekad-driven field map, and per-field
   drill-downs with NDVI/NDMI/VCI/SSM series + advisory ribbon). Published as a
   claude.ai Artifact 2026-07-14.

Design choices worth remembering: single-season VCI uses same-crop **peers**
because there's no historical baseline; the advisory is deliberately
rule-based/interpretable (this is an advisory product, not another ML model);
crop identity flows straight from the §3–4 classifier so the two halves of
PS-6 are one pipeline.

## 6b. Upgrades built 2026-07-14 (later session)

1. **Per-field sowing from NDVI green-up** (`sowing_detect.py` →
   `sowing.npz` + `sowing_summary.json`). Kharif-trough-aware: fields green
   at Nov d1 must dip below NDVI 0.30 and RE-cross before counting (else the
   green signal was standing kharif). Sowing = crossing date − GREENUP_LAG
   (Wheat 30 d, Mustard 25, Lentil 30, generic 28 — in `stress_common`).
   87% of fields detected (2278 greenup + 315 stayed-green-early); fallbacks:
   crop-median (226), literature (0). Wheat p10/50/90 = Oct 20/Nov 21/Dec 13
   — the ±3-week spread the fixed calendar (Nov 15) ignored. CAUSAL: needs
   only 1 dekad of confirmation past the crossing.
2. **Water balance per (cell × crop × sow-bin)** (`water_balance.py`,
   257 buckets). New outputs per bucket: `deficit_mm` = ETc − Ks·ETc
   (rainfed unmet demand), `eta_mm`, and **8-day** parallels of everything
   (`*_8d`, 23 windows Nov 1–Apr 30, `stress_common.PERIODS_8D`) — the PS-6
   "weekly (8-day)" deficit. Sowing date dominates demand: modal-cell wheat
   needs ~153 mm irrigation sown Oct 26 vs ~495 mm sown Dec 15.
3. **Advisory** now reads the field's own bucket (stage timing per field);
   peak stress moved to 2022-03 d1 (1296 fields ≥ advised). advisory.npz
   gains `deficit_8d/irrig_8d/ks_8d/sow_ord/sow_bin`; new
   `deficit_fields.csv`.
4. **Deficit maps** (`deficit_maps.py`): `maps/deficit_grid.png` (23
   panels), peak-period PNG (Mar 25, mean 26.6 mm/8d), 537 multi-band
   GeoTIFFs (`maps/deficit_geotiff/<chip>.tif`, 23 bands = mm per 8-day
   period, nodata −1), 4 seasonal-total examples.
5. **LSTM Ks emulator** (`stress_lstm.py` → `runs/stress_lstm/`,
   `lstm_eval.json`, `lstm_ks.npz`). 2-layer causal LSTM (h=64, 19 feats)
   maps the spectral/SAR dekadal series (NO weather inputs) to the FAO-56
   rainfed Ks, chip-level splits from splits.json. Test (348 fields): MAE
   0.036, R² 0.963, stress-flag (Ks<0.85) P 0.94 / R 0.95. Point: satellite-
   only stress estimation where weather is missing/late/coarse — the PS-6
   "deep learning (LSTM)" box, built as a physics emulator because no stress
   ground truth exists (a supervised stress classifier would be circular).
   Honest note: high R² is expected — it's a distillation of a smooth target.
6. **Dashboard** rebuilt (`dashboard_data.py` now also writes
   `dashboard.html` from the template + `<meta charset>`): new cards for
   8-day deficit (region + per-field), sowing histogram, LSTM-vs-FAO Ks in
   the drill-down, 6 KPIs (1.3 Mm³ season deficit over 793 ha, 87% sowing
   observed, LSTM R² 0.96). Republished to the SAME artifact URL
   (claude.ai/code/artifact/6026b651-...).

Remaining caveats: weather grid still 0.5° POWER (7 cells); GREENUP_LAG and
Kc curves still literature (but now offsets on observed events, not absolute
calendars); still a retrospective season (though every detection step is now
causal, so pointing the pipeline at a live season is a data-plumbing task,
not a methods change).

## 6c. Full-stack web service — GEE fetch spike (2026-07-15)

Plan: website where users draw an AOI in India → backend fetches S1/S2 via
GEE → pipeline runs → crop map + stress/advisory/deficit deliverables.
Stack decision: FastAPI + job queue backend, React + MapLibre frontend.

`spike_gee_fetch.py` (run `python spike_gee_fetch.py [workers] [size_px]`)
measured the critical unknown — on-demand GEE compositing latency. Uses the
user's existing GEE login (`~/.config/earthengine/credentials`) + Cloud
project **`crop-identification-501611`** (found in gee_export_s2.py; the
credentials file itself has no project bound). Fetches the exact production
payload: 11 monthly S2 (9 bands, SCL-masked median) + 11 monthly S1 (VV/VH
dB) + 18 dekadal SEASON (B04/B05/B08) via parallel `ee.data.computePixels`.

**Results (2026-07-15, 10 workers): 5×5 km (500 px): 40/40 composites,
175 MB, 39.3 s wall. 10×10 km (1000 px): 40/40, 700 MB, 39.6 s wall** —
wall time is bounded by the slowest single composite (~33 s), NOT by area,
until the ~50 MB/request computePixels cap forces tiling (>1000 px at 9
bands). Verdict: a 10×10 km AOI job returns in ~1 min + pipeline time;
"couple of minutes" budget is comfortably met; 25×25 km would need
per-composite tiling (16 requests each) — still likely ~2-3 min.
Each job ≈ 40 requests, so a global concurrency cap (~3 jobs) + composite
cache keyed by (bbox, month, product) protects the per-project quota.

## 6d. AOI refactor — draw-a-box → full pipeline (BUILT 2026-07-15)

`python aoi_run.py --bbox W S E N [--year 2021] [--name x] [--resume]` runs
the ENTIRE PS-6 pipeline (crop map → indices → sowing → water balance →
advisory → LSTM → maps → dashboard) for an arbitrary bbox, in a workspace
under `aoi_runs/<name>/`. **Validated end-to-end 2026-07-15: 5×5 km UP box,
142 s total** (prepare 53 s incl. GEE fetch, classify 20 s, LSTM infer 38 s,
rest seconds). Crop mix Wheat 52% / Mustard 28% / Fallow 11% / Sugarcane 8%
— consistent with the UP demonstrator map — and LSTM-vs-FAO-56 agreement on
the AOI was MAE 0.034 / R² 0.967. Test workspace: `aoi_runs/up_test_5km/`.

How it works (three new scripts + env-parameterization of the old ones):

- **`stress_common.py`** derives ALL season constants from
  `AGRI_SEASON_YEAR` (Y → months Jun Y–Apr Y+1) and paths/prefixes from
  `AGRI_DATA_DIR / AGRI_SEASON_DIR / AGRI_OUT_DIR / AGRI_CHIP_PREFIXES /
  AGRI_REGION_NAME`. Defaults reproduce the UP demonstrator bit-for-bit.
  `weather_et0.py` snaps a POWER-lattice grid to `AGRI_WEATHER_BBOX`
  (reproduces the legacy UP grid exactly); `build_features.py` honors
  `AGRI_S2_DIR/AGRI_S2RE_DIR`.
- **`aoi_prepare.py`** fetches the season from GEE with the SAME compositing
  functions the training tiles used (imports `gee_export_s2._s2_monthly/
  _s1_monthly/_s2_dekadal`: s2cloudless<40 + SCL, median, /10000, unmask(0),
  S1 IW ascending dB) via parallel computePixels, slices into 256×256
  pipeline-shaped tiles (`AOI_r{r}c{c}`): chip npz + raw S2/S2RE tifs +
  SEASON tifs. **Pseudo-fields**: felzenszwalb on (season max NDVI, peak
  timing), min 8 px, NDVI_max<0.15 excluded; crop_id/label_mask are
  placeholders so `build_features.chip_fields` runs verbatim. Derived-index
  formulas REVERSE-VERIFIED exact vs training chips: opt2=[EVI,
  gNDWI=(B03−B08)/(B03+B08)], redge2=[(B8A−B05)/(B8A+B05),
  clip(B07/B05−1,0,15)]. Static `s2` block (irreproducible uint8-ish scale)
  → Dec–Feb median distribution-matched to UP chip stats (mu
  [40.5,40.0,42.2,62.7], sd [2.8,4.2,7.2,5.8]); 20 of 1305 features. Max
  AOI 1024 px side (~10×10 km, computePixels 48 MB cap).
- **`aoi_classify.py`** fits AOI-local S1 norm stats (per-region z-scoring
  with AOI as its own region; region-id feature = 0/UP), builds the 1305-dim
  features via `chip_fields` unchanged, predicts with the final LGB →
  `crop_map.npz` (split="aoi", true8=−1) + `chip_index.json`.
- **`stress_lstm.py --infer`**: no training/splits; loads the UP-trained
  emulator, predicts Ks per AOI field, reports agreement with the AOI's own
  FAO-56 bucket in `lstm_eval.json`.
- **`aoi_run.py`** orchestrates via subprocess + env; `--resume` skips done
  stages. **Northern-India gate**: bbox must lie in (68,19,89,32.5) —
  Indo-Gangetic belt + pilots — unless `--experimental`; south of ~19°N the
  classifier has zero training support. Pilot bboxes badge "VALIDATED", the
  rest of the zone "experimental accuracy".

Caveats: pseudo-fields ≠ cadastral fields (felzenszwalb over-segments;
a field-delineation model is the upgrade); Rice still never predicted
(inherited); accuracy outside the four pilot bboxes is unmeasured; static-s2
substitution is an approximation.

## 6e. FastAPI backend (BUILT 2026-07-15)

`python aoi_server.py [--port 8000] [--workers 2]` — single-file FastAPI
wrapper around aoi_run.py (`aoi_server.py`; fastapi+uvicorn pip-installed).
Endpoints: `GET /api/limits` (zone/pilots/size caps for the future draw
tool), `POST /api/jobs` {west,south,east,north,year}, `GET /api/jobs/{id}`
(status + per-stage progress parsed from aoi_run stdout + log tail),
`/results` (summary from dashboard_data.json + file list), `/dashboard`,
`/files/{path}` (traversal-guarded). `GET /` is a minimal built-in test page
(submit bbox → live progress → dashboard link) until the React UI exists.

Protections, all VERIFIED live 2026-07-15: northern-India gate + size cap
(400s), dedupe by rounded-bbox+year hash → job_id (resubmit while running
returns same job; after done returns cached instantly), one active job per
IP (409), 5 fresh submissions/10 min per IP (429), global semaphore
(default 2 concurrent pipelines), `--resume` makes retries cheap, finished
jobs recovered from `aoi_runs/job_*/job.json` on restart. Full test through
the API: same 5×5 km UP bbox → done in 160.6 s, 11/11 stages tracked,
results identical to the direct aoi_run.py run, dashboard 200/48 KB.

Still to build for the website: React + MapLibre draw frontend; composite
cache keyed (bbox,month) shared ACROSS jobs (currently cache = whole-job
dedupe only); service-account GEE auth for deployment (server currently
uses the personal login + project crop-identification-501611).

## 7. Possible extensions (not built)

Assimilate the SSM proxy into the bucket instead of only cross-checking it;
extend the advisory to a second region (Rajasthan has the most/cleanest
weather + 238 chips); canal command-area rollup of the per-field irrigation
volumes (deficit_fields.csv already has per-field m³-ready mm × area);
finer weather (IMD 0.25° / ERA5-Land 9 km) to fix caveat B; live-season NRT
run (all steps causal now).
