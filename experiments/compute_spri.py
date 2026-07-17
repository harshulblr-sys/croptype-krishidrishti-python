"""
SPRI (SAR-based Paddy Rice Index) — per-field, from existing tiles
=================================================================
Implements the SPRI method (Sentinel-1 VH, kharif transplant->growth window):
  SPRI = f(D)*f(V)*f(W),  in [0,1], ->1 means very likely rice.
    p1 = min VH over kharif (transplant/flood, water-like low)
    p2 = max VH over kharif (vegetative peak, vegetation-like high)
    D  = p2 - p1                              (rice swings more than other crops)
    v  = V-line: high-percentile max-VH over vegetation pixels (NDVI>0.4)
    w  = W-line: low-percentile  min-VH over water pixels     (NDWI>0)
    f(D)=sigmoid(D-(v-w)/2) ; f(W)=1-((p1-w)/(v-w))^2 ; f(V)=1-((v-p2)/(v-w))^2
Reference lines v,w are computed PER REGION (region_id) from all chip pixels.
Training-free. Outputs Extracted_dataset_gee/spri_by_field.json  {"rid|file|fld": spri}.

Run:  set AGRI_DATA_DIR=...  &&  python compute_spri.py
"""
import os
import re
import glob
import json
import numpy as np
import rasterio

from data_loader import CHIPS_DIR, DATA_DIR

BASE = r"C:\Users\harsh\OneDrive\Desktop\Harshul\ISRO_Hackathon"
S2_DIR = os.path.join(BASE, "Satellite_data", "agrifieldnet_s2_2021")
KHARIF = [0, 1, 2, 3, 4]                 # Jun-Oct 2021 (transplant -> growth)
V_PCT, W_PCT = 90, 10                    # V-line / W-line percentiles


def field_key(rid, fn, fld):
    return f"{rid}|{fn}|{fld}"


def kharif_ndwi_max(region, cid):
    """max-over-kharif NDWI=(B03-B08)/(B03+B08) from the S2 tile. [H,W]."""
    tile = os.path.join(S2_DIR, f"S2_{region}_{cid}.tif")
    if not os.path.exists(tile):
        return None
    with rasterio.open(tile) as r:
        idx = {d: i + 1 for i, d in enumerate(r.descriptions)}
        vals = []
        for m in (6, 7, 8, 9, 10):
            b03, b08 = f"B03_2021_{m:02d}", f"B08_2021_{m:02d}"
            if b03 not in idx or b08 not in idx:
                continue
            g = r.read(idx[b03]).astype(np.float32)
            n = r.read(idx[b08]).astype(np.float32)
            valid = (g != 0) & (n != 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                ndwi = np.where(valid, (g - n) / np.maximum(g + n, 1e-6), np.nan)
            vals.append(ndwi)
    with np.errstate(invalid="ignore"):
        return np.nanmax(np.stack(vals), 0) if vals else None


def chip_arrays(f):
    """Return (region, cid, region_id, vh_min, vh_max, ndvi_max, ndwi_max) for kharif."""
    d = np.load(f)
    rid = int(d["region_id"])
    fn = os.path.basename(f)
    region, cid = fn.split("_", 1)[0], fn.split("_")[1].replace(".npz", "")
    vh = d["s1_asc"][..., 1]; vhm = d["s1_asc_mask"].astype(bool)     # [T,H,W]
    ndvi = d["awifs"][..., 0]; awm = d["awifs_mask"].astype(bool)
    vhk = np.where(vhm[KHARIF], vh[KHARIF], np.nan)
    ndk = np.where(awm[KHARIF], ndvi[KHARIF], np.nan)
    with np.errstate(invalid="ignore"):
        vh_min = np.nanmin(vhk, 0); vh_max = np.nanmax(vhk, 0)
        ndvi_max = np.nanmax(ndk, 0)
    ndwi_max = kharif_ndwi_max(region, cid)
    return fn, rid, vh_min, vh_max, ndvi_max, ndwi_max, d


def main():
    files = sorted(glob.glob(os.path.join(CHIPS_DIR, "*.npz")))
    print(f"Pass 1: V/W reference lines per region over {len(files)} chips...")
    veg_vh, water_vh = {}, {}
    cache = {}
    for f in files:
        fn, rid, vh_min, vh_max, ndvi_max, ndwi_max, d = chip_arrays(f)
        cache[f] = (fn, rid)
        veg = (ndvi_max > 0.4) & np.isfinite(vh_max)
        if veg.any():
            veg_vh.setdefault(rid, []).append(vh_max[veg])
        if ndwi_max is not None:
            water = (ndwi_max > 0) & np.isfinite(vh_min)
            if water.any():
                water_vh.setdefault(rid, []).append(vh_min[water])

    vw = {}
    for rid in sorted(veg_vh):
        v = float(np.percentile(np.concatenate(veg_vh[rid]), V_PCT))
        w = float(np.percentile(np.concatenate(water_vh[rid]), W_PCT)) if rid in water_vh else v - 10
        vw[rid] = (v, w)
        print(f"  region {rid}: V-line={v:.2f} dB  W-line={w:.2f} dB")

    print("Pass 2: per-field SPRI...")
    spri_by_field, rice_spri, other_spri = {}, [], []
    for f in files:
        d = np.load(f); rid = int(d["region_id"])
        if rid not in vw:            # region has no kharif S1 (e.g. Odisha) -> no SPRI
            continue
        v, w = vw[rid]
        vh = d["s1_asc"][..., 1]; vhm = d["s1_asc_mask"].astype(bool)
        vhk = np.where(vhm[KHARIF], vh[KHARIF], np.nan)
        with np.errstate(invalid="ignore"):
            p1 = np.nanmin(vhk, 0); p2 = np.nanmax(vhk, 0); D = p2 - p1
            Wv = np.clip((p1 - w) / (v - w), 0, 1); fW = 1 - Wv ** 2
            Vv = np.clip((v - p2) / (v - w), 0, 1); fV = 1 - Vv ** 2
            fD = 1.0 / (1.0 + np.exp((v - w) / 2.0 - D))
        spri = fD * fV * fW
        fid = d["field_id"]; crop = d["crop_id"]
        fn = os.path.basename(f)
        for fld in np.unique(fid[crop > 0]):
            m = (fid == fld) & (crop > 0) & np.isfinite(spri)
            if m.sum() == 0:
                continue
            s = float(np.nanmean(spri[m]))
            spri_by_field[field_key(rid, fn, int(fld))] = s
            (rice_spri if int(np.bincount(crop[m]).argmax()) == 36 else other_spri).append(s)

    with open(os.path.join(DATA_DIR, "spri_by_field.json"), "w") as fo:
        json.dump(spri_by_field, fo)
    print(f"\nWrote {len(spri_by_field)} field SPRI values -> {DATA_DIR}/spri_by_field.json")
    # sanity: rice should score higher than non-rice
    print(f"SANITY  mean SPRI  rice={np.mean(rice_spri):.3f} (n={len(rice_spri)})  "
          f"non-rice={np.mean(other_spri):.3f} (n={len(other_spri)})")
    r, o = np.array(rice_spri), np.array(other_spri)
    for t in (0.5, 0.6, 0.7, 0.8):
        print(f"  SPRI>{t}: rice {100*(r>t).mean():.0f}% vs non-rice {100*(o>t).mean():.0f}%")


if __name__ == "__main__":
    main()
