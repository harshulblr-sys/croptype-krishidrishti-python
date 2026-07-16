"""AOI step 1: fetch one season of S1/S2 from GEE for a user bbox and lay it
out as pipeline-compatible 256x256 "chips" + raw-band tiles + SEASON tiles.

Compositing parity: imports the EXACT builder functions the training tiles
were exported with (gee_export_s2._s2_monthly/_s1_monthly/_s2_dekadal —
s2cloudless<40 + SCL mask, monthly/dekadal median, /10000 reflectance,
unmask(0) so invalid px == 0, S1 IW ascending dB). Derived npz indices use
the formulas verified against the training chips (EVI, gNDWI=(B03-B08)/(..),
NDRE, CIre=B08/B05-1, RE-NDVI=(B8A-B05)/(..), CIre07/05=B07/B05-1).

Pseudo-fields: no AgriFieldNet boundaries exist for arbitrary AOIs, so fields
are segmented (felzenszwalb) on the season NDVI stack; never-vegetated pixels
(season max NDVI < 0.15: water/urban/rock) are excluded. crop_id/label_mask
are placeholders (=field presence) so build_features.chip_fields runs as-is;
real crops come from the classifier in aoi_classify.py.

Static 's2' block: the original chips carry uint8-scaled single-date bands
(scale irreproducible); the AOI substitute is a Dec-Feb median composite
distribution-matched per band to the UP chip statistics (20/1305 LGB
features — minor block, matched to stay in-distribution).

Usage:
  python aoi_prepare.py --bbox W S E N --year 2021 --workspace aoi_runs/demo
Max AOI ~10x10 km (1024 px side cap = computePixels 48 MB limit at 9 bands).
"""
import argparse
import calendar
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio

import gee_export_s2 as gx

TILE = 256
MAX_PX = 1024
NEVER_GREEN_NDVI = 0.15
SEG_SCALE, SEG_SIGMA, SEG_MIN_PX = 30.0, 0.8, 8
# per-band target stats of the original chips' static s2 (measured over 60 UP
# chips): B02,B03,B04,B08
S2_STATIC_MU = np.array([40.53, 39.98, 42.23, 62.70], np.float32)
S2_STATIC_SD = np.array([2.79, 4.15, 7.24, 5.75], np.float32)

S2_ALL_GEE = ["B2", "B3", "B4", "B8", "B11", "B5", "B6", "B7", "B8A"]
S2_ALL_OUT = ["B02", "B03", "B04", "B08", "B11", "B05", "B06", "B07", "B8A"]


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def season_dekads(y):
    out = []
    for (yy, m) in [(y, 11), (y, 12), (y + 1, 1), (y + 1, 2), (y + 1, 3), (y + 1, 4)]:
        out += [(yy, m, 1, 11), (yy, m, 11, 21), (yy, m, 21, 0)]
    return out


def fetch_all(ee, geom, grid, year, workers):
    """All composites for the AOI via parallel computePixels. Returns dict of
    numpy [bands, H, W] float32 keyed by composite name."""
    months = [(year, m) for m in range(6, 13)] + [(year + 1, m) for m in range(1, 5)]
    jobs = {}
    for (y, m) in months:
        jobs[f"S2_{y}_{m:02d}"] = gx._s2_monthly(ee, geom, y, m, S2_ALL_GEE, S2_ALL_OUT)
        jobs[f"S1_{y}_{m:02d}"] = gx._s1_monthly(ee, geom, y, m)
    for (y, m, d0, d1) in season_dekads(year):
        tag = {1: "d1", 11: "d2", 21: "d3"}[d0]
        jobs[f"SEASON_{y}_{m:02d}_{tag}"] = gx._s2_dekadal(
            ee, geom, y, m, d0, d1, tag, gx.SEASON_BANDS_GEE, gx.SEASON_BANDS_OUT)
    # static: Dec-Feb cloud-masked median (B02,B03,B04,B08)
    static = (gx._s2_masked_col(ee, geom, ee.Date.fromYMD(year, 12, 1),
                                ee.Date.fromYMD(year + 1, 3, 1))
              .select(["B2", "B3", "B4", "B8"])
              .map(lambda im: im.toFloat()).median()
              .divide(10000).rename(["B02", "B03", "B04", "B08"])
              .unmask(0).toFloat())
    jobs["STATIC"] = static

    def fetch(name, img):
        arr = ee.data.computePixels({"expression": img,
                                     "fileFormat": "NUMPY_NDARRAY", "grid": grid})
        return name, np.stack([arr[f] for f in arr.dtype.names]).astype(np.float32)

    out, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, k, im): k for k, im in jobs.items()}
        for fu in as_completed(futs):
            name, a = fu.result()
            out[name] = a
            if len(out) % 10 == 0:
                print(f"  fetched {len(out)}/{len(jobs)} ({time.time()-t0:.0f}s)")
    print(f"[fetch] {len(out)} composites, "
          f"{sum(a.nbytes for a in out.values())/1e6:.0f} MB, {time.time()-t0:.0f}s")
    return out, months


def segment_fields(ndvi_max, ndvi_peaktime):
    from skimage.segmentation import felzenszwalb
    img = np.stack([np.clip(ndvi_max, 0, 1),
                    np.clip(ndvi_peaktime, 0, 1),
                    np.clip(ndvi_max, 0, 1)], -1)
    seg = felzenszwalb(img, scale=SEG_SCALE, sigma=SEG_SIGMA, min_size=SEG_MIN_PX)
    seg = seg.astype(np.int32) + 1
    seg[ndvi_max < NEVER_GREEN_NDVI] = 0
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("W", "S", "E", "N"))
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    w, s, e, n = args.bbox
    ws = args.workspace
    for sub in ("chips", "s2raw", "s2reraw", "season"):
        os.makedirs(os.path.join(ws, sub), exist_ok=True)

    import ee
    ee.Initialize(project=gx.PROJECT)

    # ---- pixel grid: UTM, 10 m, padded to a multiple of 256 ----
    lon0, lat0 = (w + e) / 2, (s + n) / 2
    epsg = utm_epsg(lon0, lat0)
    crs = f"EPSG:{epsg}"
    p0 = ee.Geometry.Point([w, s]).transform(crs, 1).coordinates().getInfo()
    p1 = ee.Geometry.Point([e, n]).transform(crs, 1).coordinates().getInfo()
    xmin, ymin = min(p0[0], p1[0]), min(p0[1], p1[1])
    xmax, ymax = max(p0[0], p1[0]), max(p0[1], p1[1])
    xmin, ymax = math.floor(xmin / 10) * 10, math.ceil(ymax / 10) * 10
    wpx = math.ceil((xmax - xmin) / 10 / TILE) * TILE
    hpx = math.ceil((ymax - ymin) / 10 / TILE) * TILE
    if max(wpx, hpx) > MAX_PX:
        raise SystemExit(f"AOI too large: {wpx}x{hpx} px (max {MAX_PX} = "
                         f"~{MAX_PX//100} km side). Draw a smaller box.")
    grid = {"dimensions": {"width": wpx, "height": hpx},
            "affineTransform": {"scaleX": 10, "shearX": 0, "translateX": xmin,
                                "shearY": 0, "scaleY": -10, "translateY": ymax},
            "crsCode": crs}
    geom = ee.Geometry.Rectangle([xmin, ymin, xmin + wpx * 10, ymax], proj=crs,
                                 geodesic=False)
    print(f"[grid] {wpx}x{hpx} px @10m {crs}, "
          f"{wpx//TILE}x{hpx//TILE} tiles, year {args.year}")

    cubes, months = fetch_all(ee, geom, grid, args.year, args.workers)

    # ---- whole-AOI derived stacks ----
    dk_keys = [k for k in cubes if k.startswith("SEASON_")]
    dk_keys.sort(key=lambda k: (int(k.split("_")[1]), int(k.split("_")[2]),
                                k.split("_")[3]))
    dk_keys = sorted(dk_keys, key=lambda k: ((int(k.split("_")[1]) - args.year) * 12
                                             + int(k.split("_")[2]),
                                             k.split("_")[3]))
    ndvi_dk = []
    for k in dk_keys:
        b04, b05, b08 = cubes[k]
        v = (b04 > 0) & (b08 > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            nd = np.where(v, (b08 - b04) / np.maximum(b08 + b04, 1e-9), np.nan)
        ndvi_dk.append(nd)
    ndvi_dk = np.stack(ndvi_dk)                       # [18, H, W]
    ndvi_max = np.nanmax(np.where(np.isfinite(ndvi_dk), ndvi_dk, -1), 0)
    peak_t = np.argmax(np.where(np.isfinite(ndvi_dk), ndvi_dk, -1), 0) / 18.0

    print("[segment] felzenszwalb pseudo-fields...")
    seg = segment_fields(ndvi_max, peak_t)
    print(f"  {len(np.unique(seg)) - 1} segments, "
          f"{(seg > 0).mean():.0%} of AOI is candidate cropland")

    # static distribution-matched to UP chip stats
    st = cubes["STATIC"]
    st_valid = (st > 0).all(0)
    s2_static = np.zeros_like(st)
    for b in range(4):
        v = st[b][st_valid]
        mu, sd = (v.mean(), v.std() + 1e-6) if v.size else (0, 1)
        s2_static[b] = np.where(st_valid, (st[b] - mu) / sd * S2_STATIC_SD[b]
                                + S2_STATIC_MU[b], S2_STATIC_MU[b])

    # ---- slice into 256x256 tiles, write chips + raw tifs ----
    n_tiles = 0
    for r in range(hpx // TILE):
        for c in range(wpx // TILE):
            name = f"AOI_r{r}c{c}"
            sl = (slice(r * TILE, (r + 1) * TILE), slice(c * TILE, (c + 1) * TILE))
            tx = xmin + c * TILE * 10
            ty = ymax - r * TILE * 10
            transform = rasterio.Affine(10, 0, tx, 0, -10, ty)

            def wtif(path, bands, descs):
                with rasterio.open(path, "w", driver="GTiff", width=TILE,
                                   height=TILE, count=len(bands), dtype="float32",
                                   crs=crs, transform=transform,
                                   compress="deflate") as dst:
                    for i, (b, d) in enumerate(zip(bands, descs), 1):
                        dst.write(b[sl].astype(np.float32), i)
                        dst.set_band_description(i, d)

            # raw monthly tifs (feed build_features.load_raw_bands)
            s2b, s2d, reb, red = [], [], [], []
            for (y, m) in months:
                a = cubes[f"S2_{y}_{m:02d}"]
                for i in range(5):
                    s2b.append(a[i]); s2d.append(f"{S2_ALL_OUT[i]}_{y}_{m:02d}")
                for i in range(5, 9):
                    reb.append(a[i]); red.append(f"{S2_ALL_OUT[i]}_{y}_{m:02d}")
            wtif(os.path.join(ws, "s2raw", f"S2_{name}.tif"), s2b, s2d)
            wtif(os.path.join(ws, "s2reraw", f"S2RE_{name}.tif"), reb, red)

            # SEASON tile (54 bands, dekad-major B04,B05,B08)
            ssb, ssd = [], []
            for k in dk_keys:
                y_, m_, tag = k.split("_")[1:]
                for i, bn in enumerate(["B04", "B05", "B08"]):
                    ssb.append(cubes[k][i]); ssd.append(f"{bn}_{y_}_{m_}_{tag}")
            wtif(os.path.join(ws, "season", f"SEASON_{name}.tif"), ssb, ssd)

            # chip npz with derived indices (formulas verified vs training chips)
            T = len(months)
            awifs = np.zeros((T, TILE, TILE, 2), np.float32)
            amask = np.zeros((T, TILE, TILE), np.uint8)
            s1 = np.zeros((T, TILE, TILE, 3), np.float32)
            smask = np.zeros((T, TILE, TILE), np.uint8)
            redge = np.zeros((T, TILE, TILE, 2), np.float32)
            rmask = np.zeros((T, TILE, TILE), np.uint8)
            opt2 = np.zeros((T, TILE, TILE, 2), np.float32)
            redge2 = np.zeros((T, TILE, TILE, 2), np.float32)
            for t, (y, m) in enumerate(months):
                B = cubes[f"S2_{y}_{m:02d}"][(slice(None),) + sl]
                B02, B03, B04, B08, B11, B05, B06, B07, B8A = B
                v = (B04 > 0) & (B08 > 0) & (B11 > 0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    awifs[t, ..., 0] = np.where(v, (B08 - B04) / np.maximum(B08 + B04, 1e-9), 0)
                    awifs[t, ..., 1] = np.where(v, (B08 - B11) / np.maximum(B08 + B11, 1e-9), 0)
                    amask[t] = v
                    vr = (B08 > 0) & (B05 > 0)
                    redge[t, ..., 0] = np.where(vr, (B08 - B05) / np.maximum(B08 + B05, 1e-9), 0)
                    redge[t, ..., 1] = np.clip(np.where(vr, B08 / np.maximum(B05, 1e-4) - 1, 0), 0, 15)
                    rmask[t] = vr
                    vo = v & (B02 > 0) & (B03 > 0)
                    opt2[t, ..., 0] = np.where(vo, 2.5 * (B08 - B04)
                                               / np.maximum(B08 + 6 * B04 - 7.5 * B02 + 1, 1e-6), 0)
                    opt2[t, ..., 1] = np.where(vo, (B03 - B08) / np.maximum(B03 + B08, 1e-9), 0)
                    v2 = vr & (B8A > 0) & (B07 > 0)
                    redge2[t, ..., 0] = np.where(v2, (B8A - B05) / np.maximum(B8A + B05, 1e-9), 0)
                    redge2[t, ..., 1] = np.clip(np.where(v2, B07 / np.maximum(B05, 1e-4) - 1, 0), 0, 15)
                S = cubes[f"S1_{y}_{m:02d}"][(slice(None),) + sl]
                vv, vh = S
                vs = (vv != 0) & (vh != 0)
                s1[t, ..., 0] = np.where(vs, vv, 0)
                s1[t, ..., 1] = np.where(vs, vh, 0)
                s1[t, ..., 2] = np.where(vs, 10.0 ** ((vv - vh) / 10.0), 0)
                smask[t] = vs

            fid = seg[sl]
            # remap to compact per-tile ids (uint16)
            u = np.unique(fid[fid > 0])
            remap = np.zeros(seg.max() + 1, np.uint16)
            remap[u] = np.arange(1, len(u) + 1, dtype=np.uint16)
            fid16 = remap[fid]
            np.savez_compressed(
                os.path.join(ws, "chips", f"{name}.npz"),
                awifs=awifs, awifs_mask=amask, s1_asc=s1, s1_asc_mask=smask,
                s2=s2_static[(slice(None),) + sl].transpose(1, 2, 0),
                redge=redge, redge_mask=rmask, opt2=opt2, redge2=redge2,
                field_id=fid16,
                crop_id=(fid16 > 0).astype(np.uint16),      # placeholder
                label_mask=(fid16 > 0).astype(np.uint8),    # placeholder
                region_id=np.uint8(0))
            n_tiles += 1

    with open(os.path.join(ws, "aoi_meta.json"), "w") as f:
        json.dump(dict(bbox=[w, s, e, n], year=args.year, crs=crs,
                       width_px=wpx, height_px=hpx,
                       tiles=[f"AOI_r{r}c{c}" for r in range(hpx // TILE)
                              for c in range(wpx // TILE)],
                       n_segments=int(len(np.unique(seg)) - 1)), f, indent=1)
    print(f"[done] {n_tiles} tiles -> {ws} (chips/, s2raw/, s2reraw/, season/)")


if __name__ == "__main__":
    main()
