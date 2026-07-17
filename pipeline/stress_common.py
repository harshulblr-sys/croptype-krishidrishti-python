"""Shared constants/helpers for the moisture-stress + irrigation-advisory pipeline.

Default configuration = the original demonstrator: UP + UPNEW (region_id 0),
rabi season 2021-22 (Nov 2021 - Apr 2022), matching the SEASON dekadal export
and the AgriFieldNet label season. Crop map comes from the finalized 8-class LGB.

AOI mode (aoi_run.py) retargets everything through environment variables, set
BEFORE this module is imported:
  AGRI_SEASON_YEAR   base agricultural year Y -> months Jun Y .. Apr Y+1,
                     rabi dekads Nov Y .. Apr Y+1 (default 2021)
  AGRI_DATA_DIR      workspace with chips/ (default Extracted_dataset_gee)
  AGRI_SEASON_DIR    dir of SEASON_<chip>.tif dekadal tiles
  AGRI_OUT_DIR       output dir (default <root>/moisture_stress)
  AGRI_CHIP_PREFIXES comma-separated chip-id prefixes (default "UP_,UPNEW_")
  AGRI_REGION_NAME   display name for the dashboard
"""
import calendar
import datetime as dt
import os

import numpy as np

PKG_DIR = os.path.dirname(os.path.abspath(__file__))         # pipeline/
ROOT = os.path.dirname(PKG_DIR)                              # repo root
DATA_DIR = os.environ.get("AGRI_DATA_DIR", os.path.join(ROOT, "Extracted_dataset_gee"))
CHIP_DIR = os.path.join(DATA_DIR, "chips")
SAT_DIR = os.path.join(ROOT, "Satellite_data")
SEASON_DIR = os.environ.get("AGRI_SEASON_DIR",
                            os.path.join(SAT_DIR, "agrifieldnet_season_2022"))
OUT_DIR = os.environ.get("AGRI_OUT_DIR", os.path.join(ROOT, "moisture_stress"))

Y = int(os.environ.get("AGRI_SEASON_YEAR", "2021"))   # agri year: Jun Y .. Apr Y+1
REGION_NAME = os.environ.get("AGRI_REGION_NAME",
                             "Uttar Pradesh (UP + UPNEW) demonstrator")
SEASON_LABEL = f"Rabi {Y}-{(Y + 1) % 100:02d} (Nov {Y} - Apr {Y + 1})"

REGION_ID = 0                      # UP + UPNEW
REGION_PREFIXES = tuple(os.environ.get("AGRI_CHIP_PREFIXES", "UP_,UPNEW_").split(","))

SCHEME = ["Wheat", "Mustard", "Lentil", "No crop/Fallow", "Sugarcane", "Maize", "Rice", "Other"]

# ---------------------------------------------------------------- dekads ----
# 18 dekads Nov Y .. Apr Y+1 (d1=1-10, d2=11-20, d3=21-end), matching the
# SEASON tile band order B04/B05/B08 x dekad.
RABI_MONTHS = [(Y, 11), (Y, 12), (Y + 1, 1), (Y + 1, 2), (Y + 1, 3), (Y + 1, 4)]


def dekad_list():
    out = []
    for (y, m) in RABI_MONTHS:
        last = calendar.monthrange(y, m)[1]
        out += [(y, m, 1, dt.date(y, m, 1), dt.date(y, m, 10)),
                (y, m, 2, dt.date(y, m, 11), dt.date(y, m, 20)),
                (y, m, 3, dt.date(y, m, 21), dt.date(y, m, last))]
    return out

DEKADS = dekad_list()              # list of (year, month, dekad#, start, end)
N_DEKADS = len(DEKADS)             # 18
DEKAD_LABELS = [f"{y}-{m:02d} d{d}" for (y, m, d, _, _) in DEKADS]
DEKAD_MID = [s + (e - s) / 2 for (_, _, _, s, e) in DEKADS]

# npz monthly timeline (T=11): Jun Y .. Apr Y+1
MONTHS = [(Y, m) for m in range(6, 13)] + [(Y + 1, m) for m in range(1, 5)]
RABI_T = [5, 6, 7, 8, 9, 10]       # Nov..Apr indices into the monthly axis

# ----------------------------------------------- 8-day reporting periods ----
# PS-6 asks for the crop-water deficit on a "weekly (8-day)" cadence (MODIS
# compositing convention). The bucket runs daily, so both the dekadal view
# (satellite-composite native) and this 8-day view are exact aggregations.
def periods_8day():
    out, s = [], dt.date(Y, 11, 1)
    end = dt.date(Y + 1, 4, 30)
    while s <= end:
        e = min(s + dt.timedelta(days=7), end)
        out.append((s, e))
        s = e + dt.timedelta(days=1)
    return out

PERIODS_8D = periods_8day()        # 23 windows Nov 1 2021 .. Apr 30 2022
N_8D = len(PERIODS_8D)
LABELS_8D = [s.strftime("%b %d") for s, _ in PERIODS_8D]

# ----------------------------------------------------- sowing-date bins -----
# Per-field sowing dates (from NDVI green-up, sowing_detect.py) are binned to
# dekads so the water balance runs one bucket per (cell, crop, sow_bin).
def sow_bin_list():
    out = []
    for (y, m) in [(Y, 10), (Y, 11), (Y, 12), (Y + 1, 1)]:
        last = calendar.monthrange(y, m)[1]
        out += [(dt.date(y, m, 1), dt.date(y, m, 10)),
                (dt.date(y, m, 11), dt.date(y, m, 20)),
                (dt.date(y, m, 21), dt.date(y, m, last))]
    return out

SOW_BINS = sow_bin_list()          # 12 dekadal bins Oct 2021 .. Jan 2022
SOW_BIN_KEYS = [s.strftime("%Y-%m-%d") for s, _ in SOW_BINS]

def sow_bin_of(date):
    """Bin key for a sowing date (clipped into the Oct-Jan window)."""
    date = max(SOW_BINS[0][0], min(SOW_BINS[-1][1], date))
    for (s, e), k in zip(SOW_BINS, SOW_BIN_KEYS):
        if s <= date <= e:
            return k
    return SOW_BIN_KEYS[-1]

def sow_bin_mid(key):
    """Representative sowing date for a bin = its middle day."""
    i = SOW_BIN_KEYS.index(key)
    s, e = SOW_BINS[i]
    return s + (e - s) // 2

# ------------------------------------------------------- crop parameters ----
# FAO-56 style parameters per 8-class crop for the rabi water balance.
# kc = (ini, mid, end); stages = (L_ini, L_dev, L_mid, L_late) days;
# root = max rooting depth (m); p = soil-water depletion fraction (RAW = p*TAW);
# sow = sowing date. Kharif crops (Rice/Maize) and Fallow get no rabi calendar:
# they are advised only if observed green (double-cropped), via the generic curve.
CROP_PARAMS = {
    "Wheat":          dict(sow=dt.date(Y, 11, 15), stages=(30, 40, 40, 30),
                           kc=(0.40, 1.15, 0.30), root=1.00, p=0.55),
    "Mustard":        dict(sow=dt.date(Y, 10, 25), stages=(25, 35, 45, 25),
                           kc=(0.35, 1.10, 0.35), root=1.00, p=0.60),
    "Lentil":         dict(sow=dt.date(Y, 11, 10), stages=(25, 35, 50, 20),
                           kc=(0.40, 1.10, 0.30), root=0.80, p=0.50),
    "Sugarcane":      dict(sow=None, stages=None,      # standing cane all rabi
                           kc=(1.05, 1.05, 1.05), root=1.20, p=0.65),
    "_generic_rabi":  dict(sow=dt.date(Y, 11, 5), stages=(28, 35, 45, 27),
                           kc=(0.40, 1.10, 0.50), root=0.90, p=0.55),
}
# Days from sowing to the field-mean NDVI crossing GREENUP_NDVI (0.30) at
# 10 m on small mixed fields. This physiological offset converts an OBSERVED
# green-up date into a per-field sowing date (sowing_detect.py) — replacing
# the fixed literature sowing calendar. The `sow` dates above remain only as
# the last-resort fallback for fields with no usable NDVI series.
GREENUP_NDVI = 0.30
GREENUP_LAG = {"Wheat": 30, "Mustard": 25, "Lentil": 30, "Sugarcane": 0,
               "_generic_rabi": 28}
# plausible rabi sowing window (detections outside are clipped)
SOW_WINDOW = (dt.date(Y, 10, 1), dt.date(Y + 1, 1, 31))


def curve_with_sowing(name, sow_date):
    """CROP_PARAMS entry for `name` with the sowing date overridden."""
    p = dict(crop_curve(name))
    if p["sow"] is not None and sow_date is not None:
        p["sow"] = sow_date
    return p


# stage names used for stage-aware reporting/sensitivity
STAGE_NAMES = ["initial", "development", "mid-season", "late-season"]
# irrigation-criticality weight per stage (wheat CRI ~ start of dev is critical;
# mid-season flowering/grain-fill critical; late season winds down)
STAGE_SENSITIVITY = {"initial": 0.8, "development": 1.0, "mid-season": 1.0,
                     "late-season": 0.5, "off-season": 0.0}

TAW_PER_M = 140.0                  # mm available water per m root depth (loam)


def crop_curve(name):
    """Return CROP_PARAMS entry for an 8-class name (generic for the rest)."""
    if name in CROP_PARAMS:
        return CROP_PARAMS[name]
    return CROP_PARAMS["_generic_rabi"]


def stage_on(date, params):
    """FAO-56 growth stage + Kc on a date. Returns (stage_name, kc, frac_root)."""
    if params["sow"] is None:                      # season-long (sugarcane)
        return "mid-season", params["kc"][1], 1.0
    das = (date - params["sow"]).days
    L = params["stages"]
    kci, kcm, kce = params["kc"]
    if das < 0 or das > sum(L):
        return "off-season", 0.0, 0.0
    if das <= L[0]:
        return "initial", kci, max(0.15, das / max(1, L[0]) * 0.4) / params["root"]
    if das <= L[0] + L[1]:
        f = (das - L[0]) / L[1]
        return "development", kci + f * (kcm - kci), (0.4 + 0.6 * f)
    if das <= L[0] + L[1] + L[2]:
        return "mid-season", kcm, 1.0
    f = (das - L[0] - L[1] - L[2]) / L[3]
    return "late-season", kcm + f * (kce - kcm), 1.0


def chip_ids_region0():
    """Sorted chip ids (e.g. 'UP_004a2') for the demonstrator region."""
    ids = [f[:-4] for f in os.listdir(CHIP_DIR)
           if f.endswith(".npz") and f.startswith(REGION_PREFIXES)]
    return sorted(ids)


def season_tile(chip_id):
    return os.path.join(SEASON_DIR, f"SEASON_{chip_id}.tif")


def chip_npz(chip_id):
    return os.path.join(CHIP_DIR, f"{chip_id}.npz")


def valid_mean(vals, valid):
    """Mean of vals where valid, else nan."""
    v = valid & np.isfinite(vals)
    return float(vals[v].mean()) if v.any() else np.nan
