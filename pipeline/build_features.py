"""Field-level feature builder (rebuild of the lost baseline_rf_xgb.py core).

One pass over Extracted_dataset_gee/chips builds two caches:

  field_table.npz — X[n_fields, 1305] for the tree models:
      261 per-pixel features x 5 stats (mean/std/median/min/max) per field.
      Per-pixel layout (matches the final pre-deletion config
      USE_REDGE + USE_INDICES2 + USE_RAW_BANDS + USE_SAR_FLOODING + region-norm):
        ndvi/ndmi           11x2 = 22
        optical valid mask    11
        VV/VH/ratio (region-normalized)  11x3 = 33
        SAR valid mask        11
        static S2 B02/B03/B04/B08         4
        region id scalar                  1     -> base 82
        red-edge NDRE/CIre  11x2 = 22
        red-edge valid mask   11
        opt2 EVI/gNDWI      11x2 = 22
        redge2 RE-NDVI/CIre 11x2 = 22
        raw bands B02,B03,B04,B08,B11,B05,B06,B07,B8A  11x9 = 99
        SAR flooding (kharif minVV, minVH, riseVV)      3   -> 261 total

  field_seq.npz — per-field monthly sequences for the TempCNN:
      seq[n_fields, 11, 7]  = field-mean (ndvi, ndmi, vv, vh, ratio) + valid
                              fractions (optical, sar)
      static[n_fields, 8]   = region one-hot(4) + field-mean static S2(4)

S1 per-region normalization is fit on TRAIN chips only (splits.json).
"""
import glob
import json
import os
import time

import numpy as np
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (script now in pipeline/)
DATA_DIR = os.environ.get("AGRI_DATA_DIR", os.path.join(ROOT, "Extracted_dataset_gee"))
CHIP_DIR = os.path.join(DATA_DIR, "chips")
SAT_DIR = os.path.join(ROOT, "Satellite_data")
S2_DIR = os.environ.get("AGRI_S2_DIR", os.path.join(SAT_DIR, "agrifieldnet_s2_2021"))
S2RE_DIR = os.environ.get("AGRI_S2RE_DIR", os.path.join(SAT_DIR, "agrifieldnet_s2re_2021"))

T = 11
N_REGIONS = 4
KHARIF = slice(0, 5)  # Jun..Oct 2021

CLASSES_13 = [1, 2, 3, 4, 5, 6, 8, 9, 13, 14, 15, 16, 36]
CROP_TO_13 = {c: i for i, c in enumerate(CLASSES_13)}
CROP_TO_8 = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 9: 5, 36: 6}  # rest -> 7


def _f32(a):
    return np.nan_to_num(np.asarray(a, np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def load_raw_bands(name):
    """[T, H, W, 9] reflectance: B02,B03,B04,B08,B11 then B05,B06,B07,B8A."""
    out = np.zeros((T, 256, 256, 9), np.float32)
    for path, nb, off in ((os.path.join(S2_DIR, f"S2_{name}.tif"), 5, 0),
                          (os.path.join(S2RE_DIR, f"S2RE_{name}.tif"), 4, 5)):
        if not os.path.exists(path):
            continue
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)  # [T*nb, H, W]
        if arr.max() > 10:  # stored as DN
            arr /= 10000.0
        arr = arr.reshape(T, nb, arr.shape[1], arr.shape[2]).transpose(0, 2, 3, 1)
        out[:, :, :, off:off + nb] = _f32(arr)
    return out


def fit_s1_region_stats(train_chips, sample_px=4000):
    """Per-region mean/std for VV/VH/ratio over valid labeled pixels (train only)."""
    acc = {r: [] for r in range(N_REGIONS)}
    for name in train_chips:
        d = np.load(os.path.join(CHIP_DIR, f"{name}.npz"))
        lm = d["label_mask"].astype(bool) & (d["field_id"] > 0)
        if not lm.any():
            continue
        r = int(d["region_id"])
        s1 = _f32(d["s1_asc"])[:, lm, :]          # [T, npx, 3]
        m = d["s1_asc_mask"][:, lm].astype(bool)  # [T, npx]
        vals = s1[m]                               # [nvalid, 3]
        if len(vals):
            k = min(len(vals), sample_px)
            acc[r].append(vals[np.random.default_rng(0).choice(len(vals), k, replace=False)])
    stats = {}
    for r, chunks in acc.items():
        v = np.concatenate(chunks) if chunks else np.zeros((1, 3), np.float32)
        stats[r] = (v.mean(0), v.std(0) + 1e-6)
    return stats


def chip_fields(name, s1_stats):
    """Per-field rows for one chip.

    Returns list of (crop_id, row1305, seq[T,7], static8) tuples plus ids.
    """
    d = np.load(os.path.join(CHIP_DIR, f"{name}.npz"))
    lm = d["label_mask"].astype(bool) & (d["field_id"] > 0)
    if not lm.any():
        return []
    region = int(d["region_id"])
    ys, xs = np.where(lm)

    awifs = _f32(d["awifs"])[:, ys, xs, :]        # [T,n,2]
    amask = d["awifs_mask"][:, ys, xs].astype(np.float32)
    s1 = _f32(d["s1_asc"])[:, ys, xs, :]          # [T,n,3]
    smask = d["s1_asc_mask"][:, ys, xs].astype(np.float32)
    s2 = _f32(d["s2"])[ys, xs, :]                 # [n,4]
    redge = _f32(d["redge"])[:, ys, xs, :]
    rmask = d["redge_mask"][:, ys, xs].astype(np.float32)
    opt2 = _f32(d["opt2"])[:, ys, xs, :]
    redge2 = _f32(d["redge2"])[:, ys, xs, :]
    raw = load_raw_bands(name)[:, ys, xs, :]      # [T,n,9]

    mu, sd = s1_stats[region]
    s1n = (s1 - mu) / sd
    s1n *= smask[..., None]  # zero out invalid months after norm

    n = len(ys)
    # flooding from *unnormalized* VV/VH dB over kharif, valid months only
    vv, vh = s1[..., 0], s1[..., 1]
    big = 1e4
    vv_k = np.where(smask[KHARIF] > 0, vv[KHARIF], big)
    vh_k = np.where(smask[KHARIF] > 0, vh[KHARIF], big)
    min_vv = vv_k.min(0)
    min_vh = vh_k.min(0)
    vv_post = np.where(smask[2:8] > 0, vv[2:8], -big).max(0)
    rise = vv_post - min_vv
    novalid = min_vv >= big
    min_vv[novalid] = 0.0
    min_vh[min_vh >= big] = 0.0
    rise[novalid | (vv_post <= -big)] = 0.0
    flood = np.stack([min_vv, min_vh, rise], 1)   # [n,3]

    def tflat(a):  # [T,n,C] -> [n, T*C]
        return a.transpose(1, 0, 2).reshape(n, -1)

    feats = np.concatenate([
        tflat(awifs), amask.T,
        tflat(s1n), smask.T,
        s2, np.full((n, 1), region, np.float32),
        tflat(redge), rmask.T,
        tflat(opt2), tflat(redge2),
        tflat(raw), flood,
    ], 1)  # [n, 261]

    fid = d["field_id"][ys, xs]
    crop = d["crop_id"][ys, xs]
    out = []
    for f in np.unique(fid):
        sel = fid == f
        cids, cnt = np.unique(crop[sel], return_counts=True)
        cid = int(cids[np.argmax(cnt)])
        if cid == 0:
            continue
        fx = feats[sel]
        row = np.concatenate([fx.mean(0), fx.std(0), np.median(fx, 0), fx.min(0), fx.max(0)])
        # TempCNN sequence: masked field-mean per month
        seq = np.zeros((T, 7), np.float32)
        am, sm = amask[:, sel], smask[:, sel]
        aw, s1v = awifs[:, sel, :], s1[:, sel, :]
        for t in range(T):
            av, sv = am[t] > 0, sm[t] > 0
            if av.any():
                seq[t, 0:2] = aw[t, av].mean(0)
            if sv.any():
                seq[t, 2:5] = s1v[t, sv].mean(0)
            seq[t, 5] = av.mean()
            seq[t, 6] = sv.mean()
        static = np.zeros(8, np.float32)
        static[region] = 1.0
        static[4:8] = s2[sel].mean(0)
        out.append((cid, int(f), row.astype(np.float32), seq, static))
    return out


def build():
    with open(os.path.join(DATA_DIR, "splits.json")) as f:
        splits = json.load(f)
    all_chips = sorted(splits["train"] + splits["val"] + splits["test"])
    print("fitting per-region S1 stats on train chips...")
    s1_stats = fit_s1_region_stats(splits["train"])
    for r, (mu, sd) in s1_stats.items():
        print(f"  region {r}: mu={np.round(mu,2)} sd={np.round(sd,2)}")

    X, y13, y8, seqs, stats_, regions, chips, fids = [], [], [], [], [], [], [], []
    t0 = time.time()
    for i, name in enumerate(all_chips):
        for cid, f, row, seq, static in chip_fields(name, s1_stats):
            X.append(row)
            y13.append(CROP_TO_13.get(cid, -1))
            y8.append(CROP_TO_8.get(cid, 7))
            seqs.append(seq)
            stats_.append(static)
            regions.append(int(name.split("_")[0] != "UP"))  # placeholder, fixed below
            chips.append(name)
            fids.append(f)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_chips)} chips, {len(X)} fields, {time.time()-t0:.0f}s")

    X = np.stack(X)
    seqs = np.stack(seqs)
    stats_ = np.stack(stats_)
    regions = stats_[:, :4].argmax(1).astype(np.int8)
    print(f"table: {X.shape}, seq: {seqs.shape}")

    np.savez_compressed(
        os.path.join(DATA_DIR, "field_table.npz"),
        X=X, y13=np.array(y13, np.int16), y8=np.array(y8, np.int16),
        region=regions, chip=np.array(chips), fid=np.array(fids, np.int32),
        s1_mu=np.stack([s1_stats[r][0] for r in range(N_REGIONS)]),
        s1_sd=np.stack([s1_stats[r][1] for r in range(N_REGIONS)]),
    )
    np.savez_compressed(
        os.path.join(DATA_DIR, "field_seq.npz"),
        seq=seqs, static=stats_, y13=np.array(y13, np.int16), y8=np.array(y8, np.int16),
        region=regions, chip=np.array(chips), fid=np.array(fids, np.int32),
    )
    print("wrote field_table.npz + field_seq.npz in", DATA_DIR)


if __name__ == "__main__":
    build()
