"""
GEE-tile extractor adapter (season-matched 2021-22 dataset)
=================================================================
Reads the chip-aligned GEE exports
    Satellite_data/agrifieldnet_s2_2021/S2_<REGION>_<id>.tif   (55 bands: B02..B11 x 11 months)
    Satellite_data/agrifieldnet_s1_2021/S1_<REGION>_<id>.tif   (22 bands: VV,VH x 11 months)
and emits per-chip packages in the SAME format as extract_combined.py, so
data_loader.py / model_hybrid.py / train_hybrid.py run unchanged — just point
the loader at the new dataset:   set AGRI_DATA_DIR=<...>/Extracted_dataset_gee

PER-CHIP OUTPUT  (Extracted_dataset_gee/chips/<REGION>_<id>.npz)
----------------------------------------------------------------
  awifs        float32 [11, H, W, 2]   NDVI, NDMI  <-- NOW FROM 10 m SENTINEL-2
               (key name kept as "awifs" for loader compatibility; this is the
                optical temporal branch, no longer AWiFS)
  awifs_mask   uint8   [11, H, W]      per-month validity (cloud-masked months -> 0)
  s1_asc       float32 [11, H, W, 3]   VV dB, VH dB, VV/VH linear ratio
  s1_asc_mask  uint8   [11, H, W]
  s2           float32 [H, W, 4]       static B02,B03,B04,B08 from the ORIGINAL
                                       AgriFieldNet source chips (contemporaneous
                                       with the labels; sharp 10 m edges)
  crop_id / label_mask / field_id / region_id   as before

Notes
-----
* Months (Jun 2021 .. Apr 2022) are parsed from the tif band descriptions, so a
  regenerated export with different months adapts automatically.
* VV/VH ratio: GEE S1 is in dB, so linear ratio = 10^((VV_dB - VH_dB)/10).
* Only chips with BOTH tiles AND train labels are packaged; missing tiles are
  reported (exports/downloads may still be in flight — re-run as they arrive,
  optionally with --skip-existing).
* Stale caches (splits.json / norm_stats.json) in the output dir are deleted on
  every run, since the chip set may have changed.
"""

import os
import re
import csv
import json
import argparse
import numpy as np
import rasterio

from extract_combined import (REGIONS, CROP_NAMES, MERGE_TAIL_THRESHOLD,
                              build_class_map, labelled_chip_ids, read_targets,
                              build_s2, H, W)

BASE = r"C:\Users\harsh\OneDrive\Desktop\Harshul\ISRO_Hackathon"
S2_DIR = os.path.join(BASE, "Satellite_data", "agrifieldnet_s2_2021")
S1_DIR = os.path.join(BASE, "Satellite_data", "agrifieldnet_s1_2021")
S2RE_DIR = os.path.join(BASE, "Satellite_data", "agrifieldnet_s2re_2021")  # red-edge (optional)
OUT_DIR = os.path.join(BASE, "Extracted_dataset_gee")
OUT_CHIPS = os.path.join(OUT_DIR, "chips")


# ============================================================
# GEE tile reading
# ============================================================
def read_tile_months(path):
    """Read a GEE tile into {month: {band: 2D float32}} using band descriptions
    like 'B04_2021_12' / 'VV_2021_06'. Returns (per_month, sorted_month_list)."""
    per_month = {}
    with rasterio.open(path) as r:
        if r.width != W or r.height != H:
            raise ValueError(f"{path}: {r.width}x{r.height}, expected {W}x{H}")
        for i, desc in enumerate(r.descriptions, start=1):
            m = re.match(r"([A-Za-z0-9]+)_(\d{4})_(\d{2})$", desc or "")
            if not m:
                raise ValueError(f"{path}: unparseable band description {desc!r}")
            band, y, mo = m.group(1), int(m.group(2)), int(m.group(3))
            per_month.setdefault((y, mo), {})[band] = r.read(i)
    return per_month, sorted(per_month)


def verify_grid(tile_path, reg, cid):
    ref = os.path.join(reg["source_dir"],
                       f"ref_agrifieldnet_competition_v1_source_{cid}",
                       f"ref_agrifieldnet_competition_v1_source_{cid}_B04_10m.tif")
    with rasterio.open(ref) as a, rasterio.open(tile_path) as b:
        return str(a.crs) == str(b.crs) and a.transform == b.transform


def build_optical_cube(tile_path, months):
    """S2 tile -> ([T,H,W,2] NDVI/NDMI, [T,H,W] mask)."""
    per_month, tile_months = read_tile_months(tile_path)
    assert tile_months == months, f"{tile_path}: month set differs"
    cube = np.zeros((len(months), H, W, 2), np.float32)
    mask = np.zeros((len(months), H, W), np.uint8)
    for t, mk in enumerate(months):
        b = per_month[mk]
        red, nir, swir = b["B04"], b["B08"], b["B11"]
        valid = (red != 0) & (nir != 0) & (swir != 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = np.where(valid, (nir - red) / np.maximum(nir + red, 1e-9), 0.0)
            ndmi = np.where(valid, (nir - swir) / np.maximum(nir + swir, 1e-9), 0.0)
        cube[t, ..., 0] = ndvi
        cube[t, ..., 1] = ndmi
        mask[t] = valid.astype(np.uint8)
    return cube, mask


def build_redge_cube(s2_tile, s2re_tile, months):
    """Red-edge indices from B08 (base tile) + B05/B06 (red-edge tile):
    NDRE=(B08-B05)/(B08+B05) and CIre=B08/B05-1. -> ([T,H,W,2], [T,H,W] mask).
    These separate Wheat<->Mustard far better than plain NDVI."""
    base_pm, _ = read_tile_months(s2_tile)
    re_pm, re_months = read_tile_months(s2re_tile)
    assert re_months == months, f"{s2re_tile}: month set differs"
    cube = np.zeros((len(months), H, W, 2), np.float32)
    mask = np.zeros((len(months), H, W), np.uint8)
    for t, mk in enumerate(months):
        b08 = base_pm[mk]["B08"]
        b05 = re_pm[mk]["B05"]
        valid = (b08 != 0) & (b05 != 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ndre = np.where(valid, (b08 - b05) / np.maximum(b08 + b05, 1e-9), 0.0)
            cire = np.clip(np.where(valid, b08 / np.maximum(b05, 1e-4) - 1.0, 0.0), 0, 15)
        cube[t, ..., 0] = ndre
        cube[t, ..., 1] = cire
        mask[t] = valid.astype(np.uint8)
    return cube, mask


def build_sar_cube(tile_path, months):
    """S1 tile (dB) -> ([T,H,W,3] VV,VH,linear ratio, [T,H,W] mask)."""
    per_month, tile_months = read_tile_months(tile_path)
    assert tile_months == months, f"{tile_path}: month set differs"
    cube = np.zeros((len(months), H, W, 3), np.float32)
    mask = np.zeros((len(months), H, W), np.uint8)
    for t, mk in enumerate(months):
        b = per_month[mk]
        vv, vh = b["VV"], b["VH"]
        valid = (vv != 0) & (vh != 0)
        ratio = np.where(valid, 10.0 ** ((vv - vh) / 10.0), 0.0)  # dB diff -> linear
        cube[t, ..., 0] = np.where(valid, vv, 0.0)
        cube[t, ..., 1] = np.where(valid, vh, 0.0)
        cube[t, ..., 2] = ratio
        mask[t] = valid.astype(np.uint8)
    return cube, mask


# ============================================================
# Driver
# ============================================================
def extract(skip_existing=False):
    os.makedirs(OUT_CHIPS, exist_ok=True)

    # Month grid from the first available S2 tile (authoritative for this dataset)
    sample = sorted(f for f in os.listdir(S2_DIR) if f.endswith(".tif"))
    if not sample:
        raise SystemExit(f"No S2 tiles in {S2_DIR}")
    _, months = read_tile_months(os.path.join(S2_DIR, sample[0]))
    print(f"Month grid ({len(months)}): {[f'{y}-{m:02d}' for y, m in months]}")

    counts, trainable, tail, class_to_index, others_index = build_class_map()
    index_to_name = {i: CROP_NAMES.get(c, str(c))
                     for c, i in class_to_index.items() if i != others_index}
    if others_index is not None:
        index_to_name[others_index] = ("Others (" +
            ", ".join(CROP_NAMES.get(c, str(c)) for c in tail) + ")")
    num_classes = len(trainable) + (1 if tail else 0)

    manifest, missing = [], {"S2": [], "S1": []}
    done = skipped = 0
    for rname, reg in REGIONS.items():
        cids = labelled_chip_ids(reg)
        print(f"--- {rname}: {len(cids)} labelled chips ---")
        for cid in cids:
            s2_tile = os.path.join(S2_DIR, f"S2_{rname}_{cid}.tif")
            s1_tile = os.path.join(S1_DIR, f"S1_{rname}_{cid}.tif")
            if not os.path.exists(s2_tile):
                missing["S2"].append(f"{rname}_{cid}")
                continue
            if not os.path.exists(s1_tile):
                missing["S1"].append(f"{rname}_{cid}")
                continue
            out_path = os.path.join(OUT_CHIPS, f"{rname}_{cid}.npz")
            if skip_existing and os.path.exists(out_path):
                skipped += 1
                continue
            if not (verify_grid(s2_tile, reg, cid) and verify_grid(s1_tile, reg, cid)):
                print(f"  GRID MISMATCH, skipping {rname}_{cid}")
                continue

            optical, optical_mask = build_optical_cube(s2_tile, months)
            sar, sar_mask = build_sar_cube(s1_tile, months)
            s2_static = build_s2(reg, cid)
            crop, label_mask, field_id = read_targets(reg, cid)

            arrays = dict(
                awifs=optical, awifs_mask=optical_mask,   # optical branch (S2-derived)
                s1_asc=sar, s1_asc_mask=sar_mask,
                s2=s2_static, crop_id=crop, label_mask=label_mask,
                field_id=field_id, region_id=np.uint8(reg["id"]),
            )
            # Red-edge indices (NDRE, CIre) if that tile was exported/downloaded.
            s2re_tile = os.path.join(S2RE_DIR, f"S2RE_{rname}_{cid}.tif")
            if os.path.exists(s2re_tile) and verify_grid(s2re_tile, reg, cid):
                redge, redge_mask = build_redge_cube(s2_tile, s2re_tile, months)
                arrays["redge"] = redge
                arrays["redge_mask"] = redge_mask
            np.savez_compressed(out_path, **arrays)
            manifest.append({
                "file": f"{rname}_{cid}.npz", "region": rname,
                "region_id": reg["id"], "chip_id": cid,
                "labelled_px": int(label_mask.sum()),
                "optical_months": int(optical_mask.reshape(len(months), -1).any(1).sum()),
                "sar_months": int(sar_mask.reshape(len(months), -1).any(1).sum()),
            })
            done += 1
            if done % 50 == 0:
                print(f"  ... {done} packaged")

    meta = {
        "source": "GEE season-matched 2021-22 (S2_SR_HARMONIZED + s2cloudless, S1_GRD asc)",
        "grid": {"height": H, "width": W, "res_m": 10,
                 "note": "UP=EPSG:32644, BIHAR=EPSG:32645; per-chip"},
        "temporal_harmonization": "monthly composites, shared calendar grid",
        "awifs": {"months": [f"{y}-{m:02d}" for y, m in months],
                  "channels": ["NDVI", "NDMI"],
                  "note": "key kept for loader compat — data is 10 m Sentinel-2"},
        "s1_asc": {"months": [f"{y}-{m:02d}" for y, m in months],
                   "channels": ["VV", "VH", "VV_VH_ratio"],
                   "note": "GEE S1_GRD dB; ratio linear from dB difference"},
        "s2_static": {"bands": ["B02", "B03", "B04", "B08"],
                      "note": "original AgriFieldNet source chips (label season)"},
        "region_map": {r: cfg["id"] for r, cfg in REGIONS.items()},
        "class_to_index": {str(k): v for k, v in class_to_index.items()},
        "index_to_name": {str(k): v for k, v in sorted(index_to_name.items())},
        "num_classes": num_classes,
        "field_counts": {str(k): v for k, v in sorted(counts.items())},
        "merge_tail_threshold_fields": MERGE_TAIL_THRESHOLD,
        "others_index": others_index,
        "others_source_ids": tail,
        "group_split_key": "(region_id, chip_id)",
        "normalization": "NOT applied — data loader does per-channel",
        "n_chips": done + skipped,
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if manifest:
        with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            wr.writeheader(); wr.writerows(manifest)

    # dataset changed -> loader caches are stale
    for stale in ("splits.json", "norm_stats.json"):
        p = os.path.join(OUT_DIR, stale)
        if os.path.exists(p):
            os.remove(p)
            print(f"removed stale {stale}")

    print(f"\nPackaged {done} chips ({skipped} skipped existing) -> {OUT_CHIPS}")
    print(f"Missing S2 tiles: {len(missing['S2'])} | Missing S1 tiles: {len(missing['S1'])}")
    if missing["S2"][:5]:
        print("  e.g. S2:", missing["S2"][:5])
    if missing["S1"][:5]:
        print("  e.g. S1:", missing["S1"][:5])
    print("\nTrain on this dataset with:")
    print("  set AGRI_DATA_DIR=" + OUT_DIR)
    print("  python train_hybrid.py --run-name hybrid_gee_v1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-existing", action="store_true",
                    help="don't re-package chips already in the output dir")
    args = ap.parse_args()
    extract(skip_existing=args.skip_existing)
