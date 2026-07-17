# Experiments & studies (not part of the runtime pipeline)

Research scripts kept as the project's evidence trail — several of these are
the reason the reported accuracy is honest. None are imported by the live
pipeline; they share their own small framework (`data_loader.py`,
`field_level_models.py`) and still run from inside this folder.

| Script | What it established |
|---|---|
| `cv_verify.py`, `cv_perclass.py` | 5-fold chip-level cross-validation — the honest OA ~0.73 generalization number and per-class breakdown |
| `lgb_variant_sweep.py` | LightGBM hyperparameter sweep — confirmed the 500/31/0.05/balanced config is at the ceiling |
| `rice_rescue_sweep.py` | TempCNN rice-rescue threshold selection (P > 0.70) |
| `compute_spri.py` | SPRI spectral index experiment — no accuracy gain (abandoned) |
| `inspect_spatial_profiles.py` | Spatial autocorrelation diagnostics behind the chip-level split decision |
| `hyperspectral_coverage_check.py` (+ json) | EnMAP archive coverage scouting: no scenes for the 2021-22 label season; viable from 2023 rabi onward |
| `rdm_india_crops.py` (+ json) | WorldCereal RDM scouting: no usable India wheat/mustard reference data for 2021-22 |
| `data_loader.py`, `field_level_models.py` | Shared loaders/baselines used by the studies above |
