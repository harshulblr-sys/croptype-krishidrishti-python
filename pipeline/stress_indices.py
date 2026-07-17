"""Step 3: per-field stress-index time series for the demonstrator region.

For every field in UP+UPNEW (crop map from stress_crop_map.py):
  - dekadal NDVI + NDRE (18 dekads Nov 2021 - Apr 2022) from the SEASON tiles,
    gap-filled by linear interpolation (validity mask kept),
  - monthly NDMI (canopy water) from the chip npz (Jun 2021 - Apr 2022),
  - monthly S1 VV/VH + a change-detection soil-moisture proxy
    SSM = (VV - VV_dry) / (VV_wet - VV_dry) per field over the season
    (TU-Wien style; VV in dB, min = dry reference, max = wet reference),
and cross-field, per-predicted-crop statistics per time step:
  - NDVI anomaly z-score vs same-crop regional median/std,
  - single-season VCI analogue: percentile rank among same-crop fields
    (climatology is impossible with one season; peers replace history),
  - NDMI z-score vs same-crop peers.

Output: moisture_stress/field_timeseries.npz
"""
import os

import numpy as np
import rasterio

import stress_common as sc

MIN_VALID_PX = 3          # field-dekad needs >= this many valid pixels


def season_indices(chip_id):
    """Read SEASON tile -> (ndvi[18,H,W], ndre[18,H,W], valid[18,H,W])."""
    with rasterio.open(sc.season_tile(chip_id)) as src:
        arr = src.read().astype(np.float32)          # [54,H,W] B04,B05,B08 x dekad
    b04, b05, b08 = arr[0::3], arr[1::3], arr[2::3]  # each [18,H,W]
    valid = (b04 > 0) & (b08 > 0) & np.isfinite(b04) & np.isfinite(b08)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (b08 - b04) / (b08 + b04)
        ndre = (b08 - b05) / (b08 + b05)
    ndvi[~valid] = np.nan
    ndre[~(valid & (b05 > 0))] = np.nan
    return ndvi, ndre


def field_means(cube, fmask):
    """cube [T,H,W] (nan-invalid), fmask [H,W] bool -> mean[T], nvalid[T]."""
    px = cube[:, fmask]                              # [T, npx]
    nvalid = np.isfinite(px).sum(1)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(px, axis=1)
    mu[nvalid < MIN_VALID_PX] = np.nan
    return mu.astype(np.float32), nvalid


def interp_gaps(v):
    """Linear-interp nan gaps along axis 1 (no extrapolation)."""
    out = v.copy()
    x = np.arange(v.shape[1])
    for i in range(v.shape[0]):
        ok = np.isfinite(v[i])
        if 1 < ok.sum() < v.shape[1]:
            inner = (x >= x[ok][0]) & (x <= x[ok][-1])
            fill = inner & ~ok
            out[i, fill] = np.interp(x[fill], x[ok], v[i, ok])
    return out


def peer_stats(vals, crop8):
    """vals [nf,T] -> (z[nf,T], pct[nf,T]) vs same-crop peers per time step."""
    z = np.full_like(vals, np.nan)
    pct = np.full_like(vals, np.nan)
    for c in range(8):
        m = crop8 == c
        if m.sum() < 8:
            continue
        v = vals[m]                                   # [nc,T]
        med = np.nanmedian(v, axis=0)
        sd = np.nanstd(v, axis=0)
        sd[sd < 0.02] = 0.02                          # z-score floor
        z[m] = (v - med) / sd
        # percentile rank among finite peers, per column
        for t in range(v.shape[1]):
            col = v[:, t]
            ok = np.isfinite(col)
            if ok.sum() < 8:
                continue
            order = col[ok].argsort().argsort().astype(np.float32)
            r = np.full(len(col), np.nan, dtype=np.float32)
            r[ok] = order / max(1, ok.sum() - 1) * 100.0
            sub = pct[m]
            sub[:, t] = r
            pct[m] = sub
    return z, pct


def main():
    cm = np.load(os.path.join(sc.OUT_DIR, "crop_map.npz"), allow_pickle=True)
    order = {}                                        # (chip,fid) -> row
    for i, (c, f) in enumerate(zip(cm["chip"], cm["fid"])):
        order[(str(c), int(f))] = i
    nf = len(cm["fid"])
    crop8 = cm["pred8"].astype(int)

    ndvi = np.full((nf, sc.N_DEKADS), np.nan, np.float32)
    ndre = np.full((nf, sc.N_DEKADS), np.nan, np.float32)
    ndmi = np.full((nf, 11), np.nan, np.float32)
    vv = np.full((nf, 11), np.nan, np.float32)
    vh = np.full((nf, 11), np.nan, np.float32)
    npx = np.zeros(nf, np.int32)

    chips = sc.chip_ids_region0()
    for ci, cid in enumerate(chips):
        d = np.load(sc.chip_npz(cid))
        fids = d["field_id"]
        present = np.unique(fids[fids > 0])
        if not len(present):
            continue
        nd, nr = season_indices(cid)
        aw, awm = d["awifs"], d["awifs_mask"] > 0     # [11,H,W,2], [11,H,W]
        ndmi_cube = np.where(awm, aw[..., 1], np.nan)
        s1, s1m = d["s1_asc"], d["s1_asc_mask"] > 0
        vv_cube = np.where(s1m, s1[..., 0], np.nan)
        vh_cube = np.where(s1m, s1[..., 1], np.nan)
        for f in present:
            key = (cid, int(f))
            if key not in order:
                continue                              # field not in table (<20px etc.)
            i = order[key]
            fm = fids == f
            npx[i] = fm.sum()
            ndvi[i], _ = field_means(nd, fm)
            ndre[i], _ = field_means(nr, fm)
            ndmi[i], _ = field_means(ndmi_cube, fm)
            vv[i], _ = field_means(vv_cube, fm)
            vh[i], _ = field_means(vh_cube, fm)
        if (ci + 1) % 100 == 0:
            print(f"  {ci + 1}/{len(chips)} chips")

    print("valid fractions: ndvi %.2f ndmi %.2f vv %.2f" % (
        np.isfinite(ndvi).mean(), np.isfinite(ndmi).mean(), np.isfinite(vv).mean()))

    ndvi_f = interp_gaps(ndvi)
    ndre_f = interp_gaps(ndre)

    # SSM change-detection proxy on the field VV series (dB): scale between
    # season min (dry) and max (wet); needs a reasonable dynamic range.
    vmin = np.nanmin(vv, axis=1, keepdims=True)
    vmax = np.nanmax(vv, axis=1, keepdims=True)
    rng = vmax - vmin
    with np.errstate(invalid="ignore"):
        ssm = (vv - vmin) / np.where(rng > 1.5, rng, np.nan)   # <1.5 dB = no signal

    ndvi_z, vci = peer_stats(ndvi_f, crop8)
    ndmi_z, ndmi_pct = peer_stats(ndmi, crop8)

    np.savez_compressed(
        os.path.join(sc.OUT_DIR, "field_timeseries.npz"),
        chip=cm["chip"], fid=cm["fid"], crop8=crop8.astype(np.int16), npx=npx,
        ndvi=ndvi, ndvi_filled=ndvi_f, ndre=ndre_f,
        ndmi=ndmi, vv=vv, vh=vh, ssm=ssm.astype(np.float32),
        ndvi_z=ndvi_z, vci=vci, ndmi_z=ndmi_z, ndmi_pct=ndmi_pct,
        dekad_labels=np.array(sc.DEKAD_LABELS),
        months=np.array([f"{y}-{m:02d}" for y, m in sc.MONTHS]))
    print("wrote field_timeseries.npz for", nf, "fields")


if __name__ == "__main__":
    main()
