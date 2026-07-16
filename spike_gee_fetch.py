"""SPIKE: time on-demand GEE fetch of one season for a web-service AOI.

Measures the single riskiest unknown of the full-stack plan: how long GEE
takes to composite + download everything the pipeline needs for one job.

AOI: 5x5 km (500x500 px @10 m, EPSG:32644) inside the UP pilot region.
Fetched (mirrors the training data layout):
  - 11 monthly S2 median composites, Jun 2021 - Apr 2022, SCL cloud-masked,
    bands B02,B03,B04,B08,B11 + B05,B06,B07,B8A          (9 bands)
  - 11 monthly S1 IW GRD composites, VV/VH dB mean       (2 bands)
  - 18 dekadal SEASON composites Nov 2021 - Apr 2022, B04/B05/B08
Download via ee.data.computePixels (NUMPY_NDARRAY), ThreadPoolExecutor.

Run:  python spike_gee_fetch.py [n_workers] [size_px]
      (size_px 500 = 5x5 km default; 1000 = 10x10 km, still under the
       ~50 MB computePixels cap at 9 bands float32)
"""
import calendar
import datetime as dt
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

PROJECT = "crop-identification-501611"
CENTER_LON, CENTER_LAT = 81.55, 27.45          # inside the UP pilot bbox
SIZE_PX = int(sys.argv[2]) if len(sys.argv) > 2 else 500
SCALE_M = 10                                   # 500 px = 5 x 5 km
CRS = "EPSG:32644"                             # UTM 44N
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B5", "B6", "B7", "B8A"]
MONTHS = [(2021, m) for m in range(6, 13)] + [(2022, m) for m in range(1, 5)]
N_WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def dekads():
    out = []
    for (y, m) in [(2021, 11), (2021, 12), (2022, 1), (2022, 2), (2022, 3), (2022, 4)]:
        last = calendar.monthrange(y, m)[1]
        out += [(dt.date(y, m, 1), dt.date(y, m, 11)),
                (dt.date(y, m, 11), dt.date(y, m, 21)),
                (dt.date(y, m, 21), dt.date(y, m, last) + dt.timedelta(days=1))]
    return out


def main():
    import ee
    t_init = time.time()
    ee.Initialize(project=PROJECT)
    print(f"[init] {time.time() - t_init:.1f}s")

    # ---- AOI grid in UTM ----
    import math
    # quick lon/lat -> UTM 44N (central meridian 81E)
    geom = ee.Geometry.Point([CENTER_LON, CENTER_LAT]).buffer(SIZE_PX * SCALE_M / 2 * 1.05).bounds()
    # affine transform: get UTM coords of the center via EE (one small call)
    t0 = time.time()
    utm = ee.Geometry.Point([CENTER_LON, CENTER_LAT]).transform(CRS, 1).coordinates().getInfo()
    cx, cy = utm
    half = SIZE_PX * SCALE_M / 2
    xmin, ymax = cx - half, cy + half
    grid = {"dimensions": {"width": SIZE_PX, "height": SIZE_PX},
            "affineTransform": {"scaleX": SCALE_M, "shearX": 0, "translateX": xmin,
                                "shearY": 0, "scaleY": -SCALE_M, "translateY": ymax},
            "crsCode": CRS}
    print(f"[grid] center UTM ({cx:.0f},{cy:.0f}) in {time.time() - t0:.1f}s")

    # ---- lazy expressions ----
    t0 = time.time()

    def s2_month(y, m):
        d0 = ee.Date.fromYMD(y, m, 1)
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(geom).filterDate(d0, d0.advance(1, "month"))
               .map(lambda im: im.updateMask(
                   im.select("SCL").remap([1, 3, 8, 9, 10, 11],
                                          [0, 0, 0, 0, 0, 0], 1))))
        return col.select(S2_BANDS).median().toFloat()

    def s1_month(y, m):
        d0 = ee.Date.fromYMD(y, m, 1)
        col = (ee.ImageCollection("COPERNICUS/S1_GRD")
               .filterBounds(geom).filterDate(d0, d0.advance(1, "month"))
               .filter(ee.Filter.eq("instrumentMode", "IW"))
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
        return col.select(["VV", "VH"]).mean().toFloat()

    def season_dekad(d0, d1):
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(geom)
               .filterDate(d0.isoformat(), d1.isoformat())
               .map(lambda im: im.updateMask(
                   im.select("SCL").remap([1, 3, 8, 9, 10, 11],
                                          [0, 0, 0, 0, 0, 0], 1))))
        return col.select(["B4", "B5", "B8"]).median().toFloat()

    jobs = {}
    for (y, m) in MONTHS:
        jobs[f"S2 {y}-{m:02d}"] = s2_month(y, m)
        jobs[f"S1 {y}-{m:02d}"] = s1_month(y, m)
    for i, (d0, d1) in enumerate(dekads()):
        jobs[f"SEASON {d0.isoformat()}"] = season_dekad(d0, d1)
    print(f"[expressions] {len(jobs)} composites defined in {time.time() - t0:.2f}s (lazy)")

    # ---- warm-up: one sequential request to separate latency from throughput ----
    def fetch(name, img):
        t = time.time()
        arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY",
                                     "grid": grid})
        a = np.stack([arr[f] for f in arr.dtype.names])   # [bands, H, W]
        valid = float((a != 0).any(0).mean())
        return name, a, time.time() - t, valid

    first_key = f"S2 2022-01"
    t0 = time.time()
    _, a0, dt0, v0 = fetch(first_key, jobs[first_key])
    print(f"[warm-up] {first_key}: {a0.shape}, {a0.nbytes/1e6:.1f} MB, "
          f"valid {v0:.0%}, {dt0:.1f}s")

    # ---- the real measurement: everything in parallel ----
    t_all = time.time()
    results, times, fails = {}, {}, []
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(fetch, k, im): k for k, im in jobs.items()}
        for fu in as_completed(futs):
            k = futs[fu]
            try:
                name, a, dtx, valid = fu.result()
                results[name] = (a.shape, valid)
                times[name] = dtx
                total_bytes += a.nbytes
            except Exception as e:
                fails.append((k, str(e)[:120]))
    wall = time.time() - t_all

    km = SIZE_PX * SCALE_M / 1000
    print(f"\n=== RESULT ({km:g}x{km:g} km, {SIZE_PX}x{SIZE_PX} px, {N_WORKERS} workers) ===")
    print(f"composites: {len(results)} ok, {len(fails)} failed")
    for k, err in fails:
        print(f"  FAIL {k}: {err}")
    tv = sorted(times.values())
    print(f"per-request s: min {tv[0]:.1f} / median {tv[len(tv)//2]:.1f} / max {tv[-1]:.1f}")
    print(f"downloaded: {total_bytes/1e6:.0f} MB")
    print(f"WALL TIME (parallel fetch): {wall:.1f}s")
    print(f"projected {km*2:g}x{km*2:g} km (4x pixels via tiling): "
          f"~{wall*2/60:.1f}-{wall*4/60:.1f} min")
    lows = [k for k, (_, v) in results.items() if v < 0.5]
    if lows:
        print(f"low-valid (<50%) composites (monsoon clouds expected): {sorted(lows)}")


if __name__ == "__main__":
    main()
