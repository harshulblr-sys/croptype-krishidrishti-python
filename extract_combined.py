"""
Combined UP + Bihar Feature Extraction (calendar-harmonised)
=================================================================
Extends extract_dataset.py to BOTH regions at once, solving the cross-region
temporal-axis problem: UP and Bihar are separate scenes on different acquisition
dates, so stacking by raw acquisition index would make timestep t mean different
calendar months per region. Here every acquisition is composited into a shared
MONTHLY grid (union of months across both regions), so index t == same calendar
month everywhere. Per-region gaps in a month become mask = 0.

Still three separate per-sensor temporal branches (AWiFS / S1-asc / S1-desc),
plus the static Sentinel-2 stream, plus a REGION feature so the model can learn
a region-specific correction if it needs one.

PER-CHIP OUTPUT  (Extracted_dataset_combined/chips/<REGION>_<id>.npz)
--------------------------------------------------------------------
  awifs        float32 [T_awifs, H, W, 2]   NDVI, NDMI   (monthly composited)
  awifs_mask   uint8   [T_awifs, H, W]
  s1_asc       float32 [T_asc,  H, W, 3]   VV, VH, VV/VH (monthly composited)
  s1_asc_mask  uint8   [T_asc,  H, W]
  s2           float32 [H, W, 4]           B02,B03,B04,B08 (static)
  crop_id      uint16  [H, W]              RAW crop id (0 = unlabelled)
  label_mask   uint8   [H, W]
  field_id     uint16  [H, W]
  region_id    uint8   scalar              0 = UP, 1 = BIHAR   <-- region feature

Descending S1 is DROPPED (too sparse; ascending-only run) — set
INCLUDE_DESCENDING=True to restore it. Crop classes with <= MERGE_TAIL_THRESHOLD
fields (over both regions) are merged into a single "Others" class; the raw
crop_id is kept and the merge lives in metadata's class_to_index, so re-mapping
needs no re-extraction. Grouping key for the leakage-free split = (region_id,
chip_id). Normalisation stays in the data loader.
"""

import os
import re
import csv
import glob
import json
from collections import Counter
import numpy as np
import rasterio

BASE = r"C:\Users\harsh\OneDrive\Desktop\Harshul\ISRO_Hackathon"
GT = os.path.join(BASE, "Collocated_ground_data")

OUT_DIR = os.path.join(BASE, "Extracted_dataset_combined")
OUT_CHIPS = os.path.join(OUT_DIR, "chips")

H = W = 256
NODATA = 0.0
AWIFS_CHANNELS = ["NDVI", "NDMI"]
S1_CHANNELS = ["VV", "VH", "VV_VH_ratio"]
S2_BANDS = ["B02", "B03", "B04", "B08"]

# Ascending-only S1: the descending branch is too sparse (UP 2 months, Bihar 1)
# to work as a temporal stream, so it is dropped. Flip to True to re-enable.
INCLUDE_DESCENDING = False

# Crop classes with <= this many fields (counted over BOTH regions) are merged
# into a single "Others" class — too few fields to learn or to split cleanly.
MERGE_TAIL_THRESHOLD = 3

CROP_NAMES = {1: "Wheat", 2: "Mustard", 3: "Lentil", 4: "No crop/Fallow",
              5: "Green pea", 6: "Sugarcane", 8: "Garlic", 9: "Maize",
              13: "Gram", 14: "Coriander", 15: "Potato", 16: "Bersem", 36: "Rice"}

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

# ------------------------------------------------------------------
# Region configuration. S1 orbit lists are pinned explicitly (UP raw
# S1 is no longer on disk; orbit is a fixed known property of each scene).
# ------------------------------------------------------------------
REGIONS = {
    "UP": {
        "id": 0,
        "awifs_dir": os.path.join(BASE, "Collocated_awifs"),
        "s1_dir": os.path.join(BASE, "Collocated_s1"),
        "source_dir": os.path.join(GT, "source_labels_aligned"),
        "labels_dir": os.path.join(GT, "train_labels_aligned"),
        "s1_asc": ["07jul2025", "19jul2025", "31jul2025", "12aug2025",
                   "05sep2025", "17sep2025", "29sep2025"],
        "s1_desc": ["03aug2025", "27aug2025", "08sep2025"],
    },
    "BIHAR": {
        "id": 1,
        "awifs_dir": os.path.join(BASE, "Collocated_awifs_new"),
        "s1_dir": os.path.join(BASE, "Collocated_s1_new"),
        "source_dir": os.path.join(BASE, "Pilot_Area_Labels_New", "source_labels_new"),
        "labels_dir": os.path.join(GT, "train_labels_new_aligned"),
        "s1_asc": ["04jul25", "16jul25", "28jul25", "09aug25",
                   "21aug25", "02sep25", "14sep25", "26sep25"],
        "s1_desc": ["12aug25"],
    },
    # Expansion regions (GEE-only path; awifs/s1 keys unused by extract_gee).
    # Source is the full 1217-chip download (grid-identical to the collocated
    # sources); membership is set by the region label folder.
    "UPNEW": {           # more UP chips — same region as UP -> id 0 (region feature)
        "id": 0,
        "source_dir": os.path.join(BASE, "agrifieldnet_data", "source"),
        "labels_dir": os.path.join(BASE, "Pilot_Area_Labels_UPnew", "train_labels_upnew"),
        "awifs_dir": None, "s1_dir": None, "s1_asc": [], "s1_desc": [],
    },
    "ODISHA": {          # rice belt — new region -> id 2
        "id": 2,
        "source_dir": os.path.join(BASE, "agrifieldnet_data", "source"),
        "labels_dir": os.path.join(BASE, "Pilot_Area_Labels_Odisha", "train_labels_odisha"),
        "awifs_dir": None, "s1_dir": None, "s1_asc": [], "s1_desc": [],
    },
    "RAJASTHAN": {       # western cluster -> id 3
        "id": 3,
        "source_dir": os.path.join(BASE, "agrifieldnet_data", "source"),
        "labels_dir": os.path.join(BASE, "Pilot_Area_Labels_Rajasthan", "train_labels_rajasthan"),
        "awifs_dir": None, "s1_dir": None, "s1_asc": [], "s1_desc": [],
    },
}


# ============================================================
# Date helpers
# ============================================================
def parse_month(tag):
    """'04dec2025' or '04jul25' -> (year, month). Handles 2- or 4-digit year."""
    m = re.match(r"(\d{2})([a-z]{3})(\d{2,4})", tag)
    mon = _MONTHS[m.group(2)]
    y = int(m.group(3))
    if y < 100:
        y += 2000
    return (y, mon)


def month_label(mk):
    return f"{mk[0]}-{mk[1]:02d}"


def awifs_folder_tag(folder):
    """'awifs_04dec2025_col' / 'awifs_new_01oct2025_col' -> date tag."""
    t = folder[:-4] if folder.endswith("_col") else folder
    for pre in ("awifs_new_", "awifs_"):
        if t.startswith(pre):
            return t[len(pre):]
    return t


def region_awifs_folders(reg):
    """{month_key: folder_name} for a region's AWiFS collocated folders."""
    out = {}
    for f in sorted(os.listdir(reg["awifs_dir"])):
        if os.path.isdir(os.path.join(reg["awifs_dir"], f)):
            out[parse_month(awifs_folder_tag(f))] = f
    return out


def region_s1_folders(reg, orbit):
    """{month_key: [folder_names]} for a region's S1 scenes of one orbit."""
    out = {}
    for tag in reg[orbit]:
        out.setdefault(parse_month(tag), []).append(f"s1_{tag}_col")
    return out


# ============================================================
# Raster IO + compositing
# ============================================================
def read_band(path):
    with rasterio.open(path) as r:
        return r.read(1).astype(np.float32)


def composite(frames, masks):
    """Per-pixel mean over a month's valid frames.

    frames : list of [H, W, C] float32 ; masks : list of [H, W] uint8/bool
    Returns (composited [H, W, C], mask [H, W] uint8).
    """
    C = frames[0].shape[-1]
    acc = np.zeros((H, W, C), np.float64)
    cnt = np.zeros((H, W), np.float64)
    for fr, mk in zip(frames, masks):
        m = mk.astype(bool)
        acc[m] += fr[m]
        cnt[m] += 1
    valid = cnt > 0
    out = np.zeros((H, W, C), np.float32)
    out[valid] = (acc[valid] / cnt[valid, None]).astype(np.float32)
    return out, valid.astype(np.uint8)


def awifs_frame(reg, folder, cid):
    """One AWiFS acquisition -> ([H,W,2] NDVI/NDMI, [H,W] mask)."""
    d = os.path.join(reg["awifs_dir"], folder, f"chip_{cid}")
    green = read_band(os.path.join(d, "aligned_BAND2.tif"))
    red = read_band(os.path.join(d, "aligned_BAND3.tif"))
    nir = read_band(os.path.join(d, "aligned_BAND4.tif"))
    swir = read_band(os.path.join(d, "aligned_BAND5.tif"))
    valid = (nir > 0) & (red > 0) & (green > 0) & (swir > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where((nir + red) > 0, (nir - red) / (nir + red), 0.0)
        ndmi = np.where((nir + swir) > 0, (nir - swir) / (nir + swir), 0.0)
    fr = np.stack([np.where(valid, ndvi, 0.0), np.where(valid, ndmi, 0.0)], -1).astype(np.float32)
    return fr, valid


def s1_frame(reg, folder, cid):
    """One S1 acquisition -> ([H,W,3] VV/VH/ratio, [H,W] mask)."""
    d = os.path.join(reg["s1_dir"], folder, f"chip_{cid}")
    vv = read_band(os.path.join(d, "aligned_VV.tif"))
    vh = read_band(os.path.join(d, "aligned_VH.tif"))
    ratio = read_band(os.path.join(d, "aligned_VV_VH_ratio.tif"))
    valid = vv != 0
    fr = np.stack([np.where(valid, vv, 0.0), np.where(valid, vh, 0.0),
                   np.where(valid, ratio, 0.0)], -1).astype(np.float32)
    return fr, valid


def build_temporal(reg, cid, month_grid, folder_map, frame_fn, n_ch):
    """Composite a region's acquisitions onto the shared month grid.

    month_grid : list of month_keys (the common calendar axis).
    folder_map : {month_key: folder or [folders]} for this region.
    Returns (cube [T,H,W,n_ch], mask [T,H,W]).
    """
    T = len(month_grid)
    cube = np.zeros((T, H, W, n_ch), np.float32)
    mask = np.zeros((T, H, W), np.uint8)
    for t, mk in enumerate(month_grid):
        entry = folder_map.get(mk)
        if not entry:
            continue                       # region has no acquisition this month
        folders = entry if isinstance(entry, list) else [entry]
        frames, masks = [], []
        for fo in folders:
            fr, m = frame_fn(reg, fo, cid)
            frames.append(fr); masks.append(m)
        cube[t], mask[t] = composite(frames, masks)
    return cube, mask


def build_s2(reg, cid):
    static = np.zeros((H, W, len(S2_BANDS)), np.float32)
    src = os.path.join(reg["source_dir"], f"ref_agrifieldnet_competition_v1_source_{cid}")
    for i, b in enumerate(S2_BANDS):
        static[..., i] = read_band(glob.glob(os.path.join(src, f"*_{b}_10m.tif"))[0])
    return static


def read_targets(reg, cid):
    base = os.path.join(reg["labels_dir"],
                        f"aligned_ref_agrifieldnet_competition_v1_labels_train_{cid}")
    with rasterio.open(base + ".tif") as r:
        crop = r.read(1).astype(np.uint16)
    with rasterio.open(base + "_field_ids.tif") as r:
        fid = r.read(1).astype(np.uint16)
    return crop, (crop > 0).astype(np.uint8), fid


def labelled_chip_ids(reg):
    ids = []
    for f in glob.glob(os.path.join(reg["labels_dir"], "aligned_*labels_train_*.tif")):
        if "field_ids" in f:
            continue
        ids.append(re.search(r"labels_train_([0-9a-fA-F]+)\.tif", os.path.basename(f)).group(1))
    return sorted(ids)


# ============================================================
# Driver
# ============================================================
def build_common_grids():
    """Union of month keys across both regions, per sensor (sorted).
    Descending grid is empty unless INCLUDE_DESCENDING is set."""
    aw, asc, desc = set(), set(), set()
    for reg in REGIONS.values():
        aw |= set(region_awifs_folders(reg))
        asc |= set(region_s1_folders(reg, "s1_asc"))
        if INCLUDE_DESCENDING:
            desc |= set(region_s1_folders(reg, "s1_desc"))
    return sorted(aw), sorted(asc), sorted(desc)


def build_class_map():
    """Count fields per crop class over BOTH regions, then build the label map,
    merging classes with <= MERGE_TAIL_THRESHOLD fields into a single 'Others'.

    Returns (field_counts, trainable_ids, tail_ids, class_to_index, others_index).
    class_to_index maps raw crop_id -> contiguous training index; every tail id
    maps to the shared 'Others' index (last).
    """
    counts = Counter()
    for reg in REGIONS.values():
        for cid in labelled_chip_ids(reg):
            crop, mask, fid = read_targets(reg, cid)
            m = mask.astype(bool)
            for f in np.unique(fid[m]):
                fm = (fid == f) & m
                counts[int(np.bincount(crop[fm]).argmax())] += 1
    trainable = sorted([c for c, n in counts.items() if n > MERGE_TAIL_THRESHOLD])
    tail = sorted([c for c, n in counts.items() if n <= MERGE_TAIL_THRESHOLD])
    class_to_index = {c: i for i, c in enumerate(trainable)}
    others_index = None
    if tail:
        others_index = len(trainable)
        for c in tail:
            class_to_index[c] = others_index
    return counts, trainable, tail, class_to_index, others_index


def extract(limit_per_region=None):
    os.makedirs(OUT_CHIPS, exist_ok=True)
    aw_grid, asc_grid, desc_grid = build_common_grids()
    print("Common AWiFS months (%d): %s" % (len(aw_grid), [month_label(m) for m in aw_grid]))
    print("Common S1-ASC months (%d): %s" % (len(asc_grid), [month_label(m) for m in asc_grid]))
    print("Descending branch: %s" % ("ON (%d months)" % len(desc_grid) if INCLUDE_DESCENDING else "DROPPED"))

    # Label space: merge the <=MERGE_TAIL_THRESHOLD-field tail into "Others".
    counts, trainable, tail, class_to_index, others_index = build_class_map()
    index_to_name = {i: CROP_NAMES.get(c, str(c)) for c, i in class_to_index.items() if i != others_index}
    if others_index is not None:
        index_to_name[others_index] = "Others (" + ", ".join(CROP_NAMES.get(c, str(c)) for c in tail) + ")"
    num_classes = len(trainable) + (1 if tail else 0)
    print("Trainable classes (%d): %s" % (len(trainable),
          [f"{CROP_NAMES[c]}={counts[c]}f" for c in trainable]))
    print("Merged into 'Others' (idx %s): %s\n" % (
          others_index, [f"{CROP_NAMES[c]}={counts[c]}f" for c in tail]))

    manifest = []
    for rname, reg in REGIONS.items():
        aw_folders = region_awifs_folders(reg)
        asc_folders = region_s1_folders(reg, "s1_asc")
        cids = labelled_chip_ids(reg)
        if limit_per_region:
            cids = cids[:limit_per_region]
        print(f"--- {rname}: {len(cids)} chips ---")
        for i, cid in enumerate(cids, 1):
            awifs, awifs_mask = build_temporal(reg, cid, aw_grid, aw_folders, awifs_frame, 2)
            s1_asc, s1_asc_mask = build_temporal(reg, cid, asc_grid, asc_folders, s1_frame, 3)
            s2 = build_s2(reg, cid)
            crop, label_mask, field_id = read_targets(reg, cid)

            np.savez_compressed(
                os.path.join(OUT_CHIPS, f"{rname}_{cid}.npz"),
                awifs=awifs, awifs_mask=awifs_mask,
                s1_asc=s1_asc, s1_asc_mask=s1_asc_mask,
                s2=s2, crop_id=crop, label_mask=label_mask, field_id=field_id,
                region_id=np.uint8(reg["id"]),
            )
            manifest.append({
                "file": f"{rname}_{cid}.npz", "region": rname, "region_id": reg["id"],
                "chip_id": cid, "labelled_px": int(label_mask.sum()),
                "asc_months_present": int((s1_asc_mask.reshape(len(asc_grid), -1).any(1)).sum()),
            })
            if i % 50 == 0:
                print(f"  {rname} {i}/{len(cids)}")

    meta = {
        "grid": {"height": H, "width": W, "res_m": 10,
                 "note": "UP=EPSG:32644, BIHAR=EPSG:32645; per-chip"},
        "temporal_harmonization": "monthly composites on a shared calendar grid",
        "awifs": {"months": [month_label(m) for m in aw_grid], "channels": AWIFS_CHANNELS},
        "s1_asc": {"months": [month_label(m) for m in asc_grid], "channels": S1_CHANNELS},
        "s1_descending": "DROPPED (ascending-only run)" if not INCLUDE_DESCENDING else
                         {"months": [month_label(m) for m in desc_grid], "channels": S1_CHANNELS},
        "s2_static": {"bands": S2_BANDS},
        "region_map": {r: cfg["id"] for r, cfg in REGIONS.items()},
        "region_feature": "region_id scalar per chip (0=UP,1=BIHAR); broadcast/embed in loader",
        "class_to_index": {str(k): v for k, v in class_to_index.items()},
        "index_to_name": {str(k): v for k, v in sorted(index_to_name.items())},
        "num_classes": num_classes,
        "field_counts": {str(k): v for k, v in sorted(counts.items())},
        "merge_tail_threshold_fields": MERGE_TAIL_THRESHOLD,
        "others_index": others_index,
        "others_source_ids": tail,
        "group_split_key": "(region_id, chip_id)",
        "normalization": "NOT applied — do per-channel in the data loader",
        "n_chips": len(manifest),
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        wr.writeheader(); wr.writerows(manifest)

    print(f"\nWrote {len(manifest)} chips to {OUT_CHIPS}")
    print(f"Shapes: awifs[{len(aw_grid)},H,W,2] s1_asc[{len(asc_grid)},H,W,3] s2[H,W,4] | {num_classes} classes")


if __name__ == "__main__":
    import sys
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    extract(limit_per_region=lim)
