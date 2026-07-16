"""
Feature Extraction for the TempCNN + U-Net Hybrid
=================================================================
Builds one aligned data package per labelled chip, with SEPARATE per-sensor
temporal branches (one TempCNN each for AWiFS / S1-ascending / S1-descending),
plus the static Sentinel-2 stream for the U-Net encoder and the crop-type target.

Everything is already collocated onto the same 256x256, 10 m, EPSG:32644 grid
(see sentinel1_collocation.py / spatial_collocation_actual.py), so extraction is
a pure stack-and-index step — no resampling.

PER-CHIP OUTPUT  (Extracted_dataset/chips/chip_<id>.npz)
--------------------------------------------------------
  awifs        float32 [T_awifs, H, W, C_awifs]   temporal branch 1 (NDVI, NDMI)
  awifs_mask   uint8   [T_awifs, H, W]            1 = valid pixel at that date
  s1_asc       float32 [T_asc,  H, W, 3]          temporal branch 2 (VV, VH, VV/VH)
  s1_asc_mask  uint8   [T_asc,  H, W]
  s1_desc      float32 [T_desc, H, W, 3]          temporal branch 3 (VV, VH, VV/VH)
  s1_desc_mask uint8   [T_desc, H, W]
  s2           float32 [H, W, 4]                  static stream (B02,B03,B04,B08)
  crop_id      uint16  [H, W]                      raw crop label (0 = unlabelled)
  label_mask   uint8   [H, W]                      1 = labelled pixel (crop_id > 0)
  field_id     uint16  [H, W]                      field id (for reference)

GLOBAL METADATA  (Extracted_dataset/metadata.json)
--------------------------------------------------
  chronological date lists per sensor, channel names, grid size, the
  crop-class -> contiguous-index mapping (classes are non-contiguous: 1..16
  with gaps), and the chip list. The data loader reads this to know shapes and
  to remap labels for the softmax head.

Normalisation is intentionally NOT applied here — it belongs in the data loader
so you can tune it per channel without re-extracting.
"""

import os
import re
import csv
import glob
import json
import numpy as np
import rasterio


# ============================================================
# Configuration
# ============================================================
BASE = r"C:\Users\harsh\OneDrive\Desktop\Harshul\ISRO_Hackathon"

AWIFS_DIR = os.path.join(BASE, "Collocated_awifs")
S1_DIR = os.path.join(BASE, "Collocated_s1")
S2_DIR = os.path.join(BASE, "Collocated_ground_data", "source_labels_aligned")
LABELS_DIR = os.path.join(BASE, "Collocated_ground_data", "train_labels_aligned")
SAT_DIR = os.path.join(BASE, "Satellite_data")   # for S1 orbit detection

OUT_DIR = os.path.join(BASE, "Extracted_dataset")
OUT_CHIPS = os.path.join(OUT_DIR, "chips")

H = W = 256
NODATA = 0.0

# Channel definitions (documented in metadata.json)
AWIFS_CHANNELS = ["NDVI", "NDMI"]            # derived from BAND2/3/4/5
S1_CHANNELS = ["VV", "VH", "VV_VH_ratio"]    # dB, dB, linear
S2_BANDS = ["B02", "B03", "B04", "B08"]      # static high-res stream

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


# ============================================================
# Date + orbit helpers
# ============================================================
def parse_tag_date(tag):
    """'04dec2025' -> (year, month, day) sortable tuple."""
    m = re.match(r"(\d{2})([a-z]{3})(\d{4})", tag)
    d, mon, y = int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3))
    return (y, mon, d)


def detect_s1_orbits():
    """Map each S1 date tag -> 'ASC'/'DESC' from the acquisition-time in the
    original measurement filename (Sentinel-1 over India: descending ~00:00 UTC,
    ascending ~12:00 UTC). Falls back to VH file when VV was renamed."""
    orbits = {}
    for d in sorted(glob.glob(os.path.join(SAT_DIR, "*_s1"))):
        tag = os.path.basename(d).replace("_s1", "")
        files = (glob.glob(os.path.join(d, "measurement", "*vv*.tiff")) +
                 glob.glob(os.path.join(d, "measurement", "*vh*.tiff")))
        hh = None
        for f in files:
            m = re.search(r"t(\d{2})\d{4}-", os.path.basename(f))
            if m:
                hh = int(m.group(1))
                break
        if hh is None:
            raise RuntimeError(f"Could not detect orbit time for {tag}")
        orbits[tag] = "DESC" if hh < 6 else "ASC"
    return orbits


def s1_date_tags():
    return [os.path.basename(d).replace("s1_", "").replace("_col", "")
            for d in glob.glob(os.path.join(S1_DIR, "s1_*_col"))]


def awifs_date_tags():
    return [os.path.basename(d).replace("awifs_", "").replace("_col", "")
            for d in glob.glob(os.path.join(AWIFS_DIR, "awifs_*_col"))]


# ============================================================
# Raster IO
# ============================================================
def read_band(path):
    with rasterio.open(path) as r:
        return r.read(1).astype(np.float32)


def labelled_chip_ids():
    ids = []
    for f in glob.glob(os.path.join(LABELS_DIR, "aligned_*labels_train_*.tif")):
        if "field_ids" in f:
            continue
        ids.append(re.search(r"labels_train_([0-9a-f]+)\.tif", os.path.basename(f)).group(1))
    return sorted(ids)


# ============================================================
# Per-sensor cube builders
# ============================================================
def build_awifs_cube(chip_id, awifs_dates):
    """[T, H, W, C] NDVI/NDMI cube + [T, H, W] validity mask."""
    T = len(awifs_dates)
    cube = np.zeros((T, H, W, len(AWIFS_CHANNELS)), np.float32)
    mask = np.zeros((T, H, W), np.uint8)
    for t, tag in enumerate(awifs_dates):
        cdir = os.path.join(AWIFS_DIR, f"awifs_{tag}_col", f"chip_{chip_id}")
        green = read_band(os.path.join(cdir, "aligned_BAND2.tif"))
        red = read_band(os.path.join(cdir, "aligned_BAND3.tif"))
        nir = read_band(os.path.join(cdir, "aligned_BAND4.tif"))
        swir = read_band(os.path.join(cdir, "aligned_BAND5.tif"))
        valid = (nir > 0) & (red > 0) & (green > 0) & (swir > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = np.where((nir + red) > 0, (nir - red) / (nir + red), 0.0)
            ndmi = np.where((nir + swir) > 0, (nir - swir) / (nir + swir), 0.0)
        cube[t, ..., 0] = np.where(valid, ndvi, 0.0)
        cube[t, ..., 1] = np.where(valid, ndmi, 0.0)
        mask[t] = valid.astype(np.uint8)
    return cube, mask


def build_s1_cube(chip_id, s1_dates):
    """[T, H, W, 3] VV/VH/ratio cube + [T, H, W] validity mask for one orbit."""
    T = len(s1_dates)
    cube = np.zeros((T, H, W, len(S1_CHANNELS)), np.float32)
    mask = np.zeros((T, H, W), np.uint8)
    for t, tag in enumerate(s1_dates):
        cdir = os.path.join(S1_DIR, f"s1_{tag}_col", f"chip_{chip_id}")
        vv = read_band(os.path.join(cdir, "aligned_VV.tif"))
        vh = read_band(os.path.join(cdir, "aligned_VH.tif"))
        ratio = read_band(os.path.join(cdir, "aligned_VV_VH_ratio.tif"))
        valid = vv != 0                      # 0 dB is the nodata sentinel here
        cube[t, ..., 0] = np.where(valid, vv, 0.0)
        cube[t, ..., 1] = np.where(valid, vh, 0.0)
        cube[t, ..., 2] = np.where(valid, ratio, 0.0)
        mask[t] = valid.astype(np.uint8)
    return cube, mask


def build_s2_static(chip_id):
    """[H, W, 4] static Sentinel-2 stream (B02,B03,B04,B08)."""
    static = np.zeros((H, W, len(S2_BANDS)), np.float32)
    src = os.path.join(S2_DIR, f"ref_agrifieldnet_competition_v1_source_{chip_id}")
    for i, b in enumerate(S2_BANDS):
        p = glob.glob(os.path.join(src, f"*_{b}_10m.tif"))[0]
        static[..., i] = read_band(p)
    return static


def read_targets(chip_id):
    """crop_id, label_mask, field_id."""
    crop_p = os.path.join(
        LABELS_DIR,
        f"aligned_ref_agrifieldnet_competition_v1_labels_train_{chip_id}.tif")
    fid_p = os.path.join(
        LABELS_DIR,
        f"aligned_ref_agrifieldnet_competition_v1_labels_train_{chip_id}_field_ids.tif")
    with rasterio.open(crop_p) as r:
        crop = r.read(1).astype(np.uint16)
    with rasterio.open(fid_p) as r:
        fid = r.read(1).astype(np.uint16)
    label_mask = (crop > 0).astype(np.uint8)
    return crop, label_mask, fid


# ============================================================
# Driver
# ============================================================
def extract(chip_ids=None, limit=None):
    os.makedirs(OUT_CHIPS, exist_ok=True)

    orbits = detect_s1_orbits()
    s1_tags = s1_date_tags()
    asc_dates = sorted([t for t in s1_tags if orbits[t] == "ASC"], key=parse_tag_date)
    desc_dates = sorted([t for t in s1_tags if orbits[t] == "DESC"], key=parse_tag_date)
    awifs_dates = sorted(awifs_date_tags(), key=parse_tag_date)

    all_labelled = labelled_chip_ids()
    if chip_ids is None:
        chip_ids = all_labelled
    if limit:
        chip_ids = chip_ids[:limit]

    # crop-class -> contiguous index (classes are non-contiguous with gaps)
    classes = set()
    for cid in all_labelled:
        crop, _, _ = read_targets(cid)
        classes |= set(np.unique(crop).tolist())
    classes.discard(0)
    class_list = sorted(classes)
    class_to_index = {c: i for i, c in enumerate(class_list)}

    print(f"AWiFS dates ({len(awifs_dates)}): {awifs_dates}")
    print(f"S1 ASC dates ({len(asc_dates)}): {asc_dates}")
    print(f"S1 DESC dates ({len(desc_dates)}): {desc_dates}")
    print(f"Crop classes ({len(class_list)}): {class_list}")
    print(f"Extracting {len(chip_ids)} chip(s)...\n")

    manifest = []
    for i, cid in enumerate(chip_ids, 1):
        awifs, awifs_mask = build_awifs_cube(cid, awifs_dates)
        s1_asc, s1_asc_mask = build_s1_cube(cid, asc_dates)
        s1_desc, s1_desc_mask = build_s1_cube(cid, desc_dates)
        s2 = build_s2_static(cid)
        crop, label_mask, field_id = read_targets(cid)

        np.savez_compressed(
            os.path.join(OUT_CHIPS, f"chip_{cid}.npz"),
            awifs=awifs, awifs_mask=awifs_mask,
            s1_asc=s1_asc, s1_asc_mask=s1_asc_mask,
            s1_desc=s1_desc, s1_desc_mask=s1_desc_mask,
            s2=s2, crop_id=crop, label_mask=label_mask, field_id=field_id,
        )

        n_lab = int(label_mask.sum())
        asc_cov = float(s1_asc_mask.mean())
        manifest.append({
            "chip_id": cid, "labelled_px": n_lab,
            "n_fields": int(len(np.unique(field_id[label_mask == 1]))),
            "asc_coverage": round(asc_cov, 4),
            "desc_coverage": round(float(s1_desc_mask.mean()), 4),
        })
        print(f"  [{i}/{len(chip_ids)}] chip_{cid}: "
              f"labelled_px={n_lab}, asc_cov={asc_cov:.2f}")

    # global metadata
    meta = {
        "grid": {"height": H, "width": W, "crs": "EPSG:32644", "res_m": 10},
        "awifs": {"dates": awifs_dates, "channels": AWIFS_CHANNELS},
        "s1_asc": {"dates": asc_dates, "channels": S1_CHANNELS},
        "s1_desc": {"dates": desc_dates, "channels": S1_CHANNELS},
        "s2_static": {"bands": S2_BANDS},
        "crop_classes": class_list,
        "class_to_index": {str(k): v for k, v in class_to_index.items()},
        "num_classes": len(class_list),
        "normalization": "NOT applied here — do per-channel in the data loader",
        "n_chips_extracted": len(chip_ids),
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        wr.writeheader()
        wr.writerows(manifest)

    print(f"\nWrote {len(chip_ids)} chip package(s) to {OUT_CHIPS}")
    print(f"Metadata -> {os.path.join(OUT_DIR, 'metadata.json')}")


if __name__ == "__main__":
    # Test run on a few chips (include a chip that is missing ascending coverage).
    TEST_CHIPS = ["02160", "02694", "05c70"]
    labelled = set(labelled_chip_ids())
    test = [c for c in TEST_CHIPS if c in labelled] or labelled_chip_ids()[:3]
    extract(chip_ids=test)
