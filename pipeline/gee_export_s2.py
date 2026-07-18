"""
Google Earth Engine export: season-matched Sentinel-2 for the AgriFieldNet chips
================================================================================
Pulls monthly, cloud-masked Sentinel-2 SURFACE REFLECTANCE composites over the
2021-22 agricultural year (the season the AgriFieldNet labels come from) and
exports ONE GeoTIFF per chip, pixel-aligned to that chip's exact 10 m grid.

Why this matters (see earlier analysis):
  * S2_SR_HARMONIZED  -> fixes the +1000 DN baseline shift after 2022-01-25
                         (your window Jun-2021..Apr-2022 crosses that boundary).
  * 10 m native        -> resolves the small (~0.26 ha) fields that 56 m AWiFS
                         blurred together (the Wheat-vs-Mustard failure).
  * 2021-22 season     -> matches the labels, removing crop-rotation label noise.

Because each export uses the chip's own CRS + crsTransform + 256x256 dimensions,
the output lands on EXACTLY the existing chip grid (UP=EPSG:32644, BIHAR=32645) —
no local collocation/resampling needed; it feeds the extractor directly.

PREREQUISITES (run on YOUR machine, not here — GEE needs your account):
  pip install earthengine-api
  earthengine authenticate                # one-time, opens browser
  # then set PROJECT below to your Cloud project id
Run:
  python gee_export_s2.py --dry-run              # validate chip grids, submit nothing
  python gee_export_s2.py --sensor s2 --limit 5  # smoke-test 5 S2 tiles
  python gee_export_s2.py --sensor both          # submit S2 + S1 for all chips
Monitor tasks at https://code.earthengine.google.com/tasks

Outputs (Google Drive):
  Drive/agrifieldnet_s2_2021/S2_<REGION>_<chipid>.tif
       bands {B02,B03,B04,B08,B11}_{YYYY}_{MM}  (surface reflectance, /10000)
       masking: s2cloudless (S2_CLOUD_PROBABILITY) + SCL, monthly median
  Drive/agrifieldnet_s1_2021/S1_<REGION>_<chipid>.tif
       bands {VV,VH}_{YYYY}_{MM}  (sigma0 dB, ascending; monthly median = despeckled)

Both are chip-grid-aligned (crs+crsTransform+256x256), so they land directly on
the existing chip grid — no local collocation. Use --sensor {s2,s1,both}.
"""

import os
import re
import glob
import time
import argparse
import rasterio

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
PROJECT = "crop-identification-501611"          # <-- set to your Earth Engine Cloud project
GDRIVE_S2 = "agrifieldnet_s2_2021"    # Drive folder for Sentinel-2 exports
GDRIVE_S1 = "agrifieldnet_s1_2021"    # Drive folder for Sentinel-1 exports
GDRIVE_S2RE = "agrifieldnet_s2re_2021"  # red-edge bands (separate, additive export)
GDRIVE_BLOOM = "agrifieldnet_bloom_2022"  # DEKADAL bloom-window optical (additive)

BASE = r"C:\Users\harsh\OneDrive\Desktop\Harshul\ISRO_Hackathon"
_SRC_ALL = os.path.join(BASE, "agrifieldnet_data", "source")   # full 1217-chip download
SOURCES = [
    ("UP",    os.path.join(BASE, "Collocated_ground_data", "source_labels_aligned"),
              os.path.join(BASE, "Collocated_ground_data", "train_labels_aligned")),
    ("BIHAR", os.path.join(BASE, "Pilot_Area_Labels_New", "source_labels_new"),
              os.path.join(BASE, "Collocated_ground_data", "train_labels_new_aligned")),
    # Expansion regions (source = full download; membership via label folder).
    # Existing UP/BIHAR tiles are skipped automatically (already downloaded).
    ("UPNEW",  _SRC_ALL,
               os.path.join(BASE, "Pilot_Area_Labels_UPnew", "train_labels_upnew")),
    ("ODISHA", _SRC_ALL,
               os.path.join(BASE, "Pilot_Area_Labels_Odisha", "train_labels_odisha")),
    ("RAJASTHAN", _SRC_ALL,
               os.path.join(BASE, "Pilot_Area_Labels_Rajasthan", "train_labels_rajasthan")),
]

# Skip a chip if its tile is already downloaded locally (avoids re-exporting the
# 403 UP+Bihar tiles you already have).
LOCAL_TILE_DIRS = {"S2": os.path.join(BASE, "Satellite_data", "agrifieldnet_s2_2021"),
                   "S1": os.path.join(BASE, "Satellite_data", "agrifieldnet_s1_2021"),
                   "S2RE": os.path.join(BASE, "Satellite_data", "agrifieldnet_s2re_2021"),
                   "BLOOM": os.path.join(BASE, "Satellite_data", "agrifieldnet_bloom_2022"),
                   "SEASON": os.path.join(BASE, "Satellite_data", "agrifieldnet_season_2022"),
                   "S1DEKAD": os.path.join(BASE, "Satellite_data", "agrifieldnet_s1dekad_2021")}

# ---- Dekadal bloom-window optical (additive; keeps the monthly stack untouched) ----
# Monthly median washes out mustard's ~2-week Jan-Feb bloom; dekadal (10-day) composites
# over Dec 2021 - Mar 2022 preserve it. Visible B02/B03 for NDYI (yellowness) + B05/B08
# for NDRE. Downstream keeps the PEAK NDYI over these dekads, not the median.
BLOOM_MONTHS = [(2021, 12), (2022, 1), (2022, 2), (2022, 3)]
BLOOM_DEKADS = [(1, 11), (11, 21), (21, 0)]        # day ranges; 0 = end of month
BLOOM_BANDS_GEE = ["B2", "B3", "B5", "B8"]
BLOOM_BANDS_OUT = ["B02", "B03", "B05", "B08"]

# ---- Dekadal FULL rabi-season optical (Wheat<->Mustard timing) ----
# The W/M discriminator is senescence/green-up TIMING at the season SHOULDERS
# (Dec green-up, Mar senescence), smeared by monthly composites. 10-day composites
# over the whole rabi window (Nov 2021 - Apr 2022) resolve the slopes. NDVI (B04,B08)
# + NDRE (B05,B08) — the two indices that carry the timing. Additive; keeps monthly.
GDRIVE_SEASON = "agrifieldnet_season_2022"
SEASON_MONTHS = [(2021, 11), (2021, 12), (2022, 1), (2022, 2), (2022, 3), (2022, 4)]
SEASON_DEKADS = [(1, 11), (11, 21), (21, 0)]
SEASON_BANDS_GEE = ["B4", "B5", "B8"]
SEASON_BANDS_OUT = ["B04", "B05", "B08"]

# ---- Dekadal kharif S1 (rice transplant-flood; sharpens SPRI + TempCNN rice) ----
# Rice's SPRI signal is the ~1-2 week transplant FLOOD (VH crashes very low over smooth
# water), washed out by monthly-median VH. Dekadal VV/VH over kharif (Jun-Oct 2021)
# preserves the flood dip; downstream min-VH over dekads is a far sharper p1 than min
# over monthly medians. Also feeds the TempCNN a crisper flood in the SAR sequence.
# Odisha is DESCENDING-only: run  --sensor s1dekad --orbit DESCENDING --regions ODISHA
# separately from the ascending regions (same split as the main S1 export).
GDRIVE_S1DEKAD = "agrifieldnet_s1dekad_2021"
KHARIF_MONTHS = [(2021, 6), (2021, 7), (2021, 8), (2021, 9), (2021, 10)]
KHARIF_DEKADS = [(1, 11), (11, 21), (21, 0)]        # S1_BANDS (VV,VH) reused

LABELED_ONLY = True                   # export only chips that have train labels (403)

# ---- Sentinel-2 (optical) ----
# S2_SR_HARMONIZED names bands B2,B3,B4,B8 (no zero-pad); B11 is already 2-digit.
# We select those, but rename the OUTPUT to the padded AgriFieldNet convention.
BANDS_GEE = ["B2", "B3", "B4", "B8", "B11"]        # blue,green,red,NIR,SWIR1 in GEE
BANDS_OUT = ["B02", "B03", "B04", "B08", "B11"]    # padded names on the export
# Red-edge bands (separate additive export) — purpose-built for crop
# discrimination (NDRE separates Wheat<->Mustard). GEE names B5,B6,B7,B8A.
BANDS_RE_GEE = ["B5", "B6", "B7", "B8A"]
BANDS_RE_OUT = ["B05", "B06", "B07", "B8A"]
# 2021-22 agricultural year: kharif (Rice/Maize) + rabi (Wheat/Mustard/...)
MONTHS = [(2021, m) for m in (6, 7, 8, 9, 10, 11, 12)] + [(2022, m) for m in (1, 2, 3, 4)]
MAX_CLOUD_PCT = 60                    # scene-level pre-filter (tightened from 80)
CLD_PROB_THRESH = 40                  # s2cloudless: mask pixels with prob % above this

# ---- Sentinel-1 (SAR) ----
# COPERNICUS/S1_GRD is analysis-ready: thermal-noise removed, calibrated to sigma0,
# terrain-corrected, values in dB. Monthly median further suppresses speckle, so the
# manual .SAFE -> Refined Lee -> geocode pipeline is not needed here.
S1_MONTHS = MONTHS                    # full year; set to Jun-Oct for monsoon-only (~1 GB)
S1_ORBIT = "ASCENDING"                # match the ascending-only design
S1_BANDS = ["VV", "VH"]              # VV/VH ratio is derived in the extractor

THROTTLE_S = 0.3                      # pause between task submissions


# ------------------------------------------------------------------
# Chip grids (read locally — authoritative geometry, no GEE needed)
# ------------------------------------------------------------------
def chip_specs():
    """Yield (region, chip_id, crs, crsTransform[6], (W,H)) for each chip,
    read from its B04 tif so exports align pixel-perfectly to the chip grid."""
    specs = []
    for region, src_dir, lab_dir in SOURCES:
        if not os.path.isdir(src_dir):
            print(f"  (skipping {region}: source dir not downloaded yet — {src_dir})")
            continue
        for d in sorted(os.listdir(src_dir)):
            m = re.search(r"source_([0-9a-fA-F]+)$", d)
            if not m:
                continue
            cid = m.group(1)
            if LABELED_ONLY:
                lab = os.path.join(
                    lab_dir,
                    f"aligned_ref_agrifieldnet_competition_v1_labels_train_{cid}.tif")
                if not os.path.exists(lab):
                    continue
            b04 = glob.glob(os.path.join(src_dir, d, "*_B04_10m.tif"))
            if not b04:
                continue
            with rasterio.open(b04[0]) as r:
                t = r.transform
                specs.append((region, cid, str(r.crs),
                              [t.a, t.b, t.c, t.d, t.e, t.f], (r.width, r.height)))
    return specs


# ------------------------------------------------------------------
# GEE composites
# ------------------------------------------------------------------
def _masked_template(ee, band_names):
    """Fully-masked image carrying the given band names. Merging it into a
    monthly collection guarantees the median always has these bands (empty/cloudy
    months would otherwise give a 0-band image); being masked it never affects
    real observations."""
    return (ee.Image.constant([0] * len(band_names)).rename(band_names)
            .toFloat().updateMask(ee.Image.constant(0)))


def _geom(ee, transform, crs):
    xmin, ymax = transform[2], transform[5]
    xmax, ymin = xmin + transform[0] * 256, ymax + transform[4] * 256
    return ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], proj=crs, geodesic=False)


# ---- Sentinel-2 with s2cloudless + SCL masking ----
def _s2_masked_col(ee, geom, start, end):
    """S2_SR joined with S2_CLOUD_PROBABILITY (s2cloudless); mask by cloud
    probability AND the SCL shadow/cirrus/snow classes."""
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(geom).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PCT)))
    cld = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
           .filterBounds(geom).filterDate(start, end))
    joined = ee.ImageCollection(ee.Join.saveFirst("cldprob").apply(
        primary=s2, secondary=cld,
        condition=ee.Filter.equals(leftField="system:index",
                                   rightField="system:index")))

    def _mask(img):
        prob = ee.Image(img.get("cldprob")).select("probability")
        scl = img.select("SCL")
        bad_scl = (scl.eq(1).Or(scl.eq(3)).Or(scl.eq(8)).Or(scl.eq(9))
                   .Or(scl.eq(10)).Or(scl.eq(11)))
        clear = prob.lt(CLD_PROB_THRESH).And(bad_scl.Not())
        return img.updateMask(clear)

    return joined.map(_mask)


def _s2_monthly(ee, geom, year, month, bands_gee, bands_out):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    col = (_s2_masked_col(ee, geom, start, end)
           .select(bands_gee).map(lambda im: im.toFloat()))
    med = col.merge(ee.ImageCollection([_masked_template(ee, bands_gee)])).median()
    names = [f"{b}_{year}_{month:02d}" for b in bands_out]
    # /10000 -> surface reflectance; unmask(0) -> nodata months become 0.
    return med.divide(10000).rename(names).unmask(0).toFloat()


def build_s2_image(ee, crs, transform):
    geom = _geom(ee, transform, crs)
    return ee.Image.cat([_s2_monthly(ee, geom, y, m, BANDS_GEE, BANDS_OUT) for y, m in MONTHS])


def build_s2re_image(ee, crs, transform):
    """Red-edge composites (B05,B06,B07,B8A) — same cloud masking, separate file."""
    geom = _geom(ee, transform, crs)
    return ee.Image.cat([_s2_monthly(ee, geom, y, m, BANDS_RE_GEE, BANDS_RE_OUT) for y, m in MONTHS])


# ---- Sentinel-1 GRD (analysis-ready sigma0 dB) ----
def _s1_monthly(ee, geom, year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(geom).filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .filter(ee.Filter.eq("orbitProperties_pass", S1_ORBIT))
           .select(S1_BANDS).map(lambda im: im.toFloat()))
    med = col.merge(ee.ImageCollection([_masked_template(ee, S1_BANDS)])).median()
    names = [f"{b}_{year}_{month:02d}" for b in S1_BANDS]
    # values already in dB (S1_GRD); no /10000. Monthly median suppresses speckle.
    return med.rename(names).unmask(0).toFloat()


def build_s1_image(ee, crs, transform):
    geom = _geom(ee, transform, crs)
    return ee.Image.cat([_s1_monthly(ee, geom, y, m) for y, m in S1_MONTHS])


# ---- Dekadal kharif S1 (VV/VH dB, 10-day windows) ----
def _s1_dekadal(ee, geom, year, month, d0, d1, tag):
    start = ee.Date.fromYMD(year, month, d0)
    end = (ee.Date.fromYMD(year, month, 1).advance(1, "month")
           if d1 == 0 else ee.Date.fromYMD(year, month, d1))
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(geom).filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .filter(ee.Filter.eq("orbitProperties_pass", S1_ORBIT))
           .select(S1_BANDS).map(lambda im: im.toFloat()))
    med = col.merge(ee.ImageCollection([_masked_template(ee, S1_BANDS)])).median()
    names = [f"{b}_{year}_{month:02d}_{tag}" for b in S1_BANDS]      # dB; no /10000
    return med.rename(names).unmask(0).toFloat()


def build_s1dekad_image(ee, crs, transform):
    geom = _geom(ee, transform, crs)
    imgs = []
    for y, m in KHARIF_MONTHS:
        for i, (d0, d1) in enumerate(KHARIF_DEKADS, 1):
            imgs.append(_s1_dekadal(ee, geom, y, m, d0, d1, f"d{i}"))
    return ee.Image.cat(imgs)        # 15 dekads x 2 = 30 bands


# ---- Dekadal (10-day) composites (shared by bloom + full-season) ----
def _s2_dekadal(ee, geom, year, month, d0, d1, tag, bands_gee, bands_out):
    start = ee.Date.fromYMD(year, month, d0)
    end = (ee.Date.fromYMD(year, month, 1).advance(1, "month")
           if d1 == 0 else ee.Date.fromYMD(year, month, d1))
    col = (_s2_masked_col(ee, geom, start, end)
           .select(bands_gee).map(lambda im: im.toFloat()))
    med = col.merge(ee.ImageCollection([_masked_template(ee, bands_gee)])).median()
    names = [f"{b}_{year}_{month:02d}_{tag}" for b in bands_out]
    return med.divide(10000).rename(names).unmask(0).toFloat()


def _build_dekadal(ee, crs, transform, months, dekads, bands_gee, bands_out):
    geom = _geom(ee, transform, crs)
    imgs = []
    for y, m in months:
        for i, (d0, d1) in enumerate(dekads, 1):
            imgs.append(_s2_dekadal(ee, geom, y, m, d0, d1, f"d{i}", bands_gee, bands_out))
    return ee.Image.cat(imgs)


def build_bloom_image(ee, crs, transform):
    return _build_dekadal(ee, crs, transform, BLOOM_MONTHS, BLOOM_DEKADS,
                          BLOOM_BANDS_GEE, BLOOM_BANDS_OUT)     # 12 dekads x 4 = 48 bands


def build_season_image(ee, crs, transform):
    return _build_dekadal(ee, crs, transform, SEASON_MONTHS, SEASON_DEKADS,
                          SEASON_BANDS_GEE, SEASON_BANDS_OUT)   # 18 dekads x 3 = 54 bands


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="validate chip grids and print tasks; submit nothing")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sensor",
                    choices=["s2", "s1", "s2re", "bloom", "season", "s1dekad", "both"], default="both",
                    help="s2/s1/both, s2re red-edge, bloom dekadal bloom-window, "
                         "season dekadal rabi (Wheat<->Mustard timing), "
                         "s1dekad dekadal kharif S1 (rice transplant-flood / SPRI)")
    ap.add_argument("--orbit", choices=["ASCENDING", "DESCENDING"], default=None,
                    help="override S1 orbit pass (e.g. DESCENDING for Odisha)")
    ap.add_argument("--regions", default=None,
                    help="comma-separated region filter, e.g. ODISHA,RAJASTHAN")
    args = ap.parse_args()

    if args.orbit:
        global S1_ORBIT
        S1_ORBIT = args.orbit
        print(f"S1 orbit override: {S1_ORBIT}")

    specs = chip_specs()
    if args.regions:
        keep = {r.strip().upper() for r in args.regions.split(",")}
        specs = [s for s in specs if s[0] in keep]
        print(f"Region filter {keep}: {len(specs)} chips")
    if args.limit:
        specs = specs[:args.limit]

    # (tag, drive folder, image-builder, months) per requested sensor
    jobs = []
    if args.sensor in ("s2", "both"):
        jobs.append(("S2", GDRIVE_S2, build_s2_image, MONTHS, BANDS_OUT))
    if args.sensor in ("s1", "both"):
        jobs.append(("S1", GDRIVE_S1, build_s1_image, S1_MONTHS, S1_BANDS))
    if args.sensor == "s2re":
        jobs.append(("S2RE", GDRIVE_S2RE, build_s2re_image, MONTHS, BANDS_RE_OUT))
    if args.sensor == "bloom":
        dekad_months = [(y, m) for y, m in BLOOM_MONTHS for _ in BLOOM_DEKADS]
        jobs.append(("BLOOM", GDRIVE_BLOOM, build_bloom_image, dekad_months, BLOOM_BANDS_OUT))
    if args.sensor == "season":
        dekad_months = [(y, m) for y, m in SEASON_MONTHS for _ in SEASON_DEKADS]
        jobs.append(("SEASON", GDRIVE_SEASON, build_season_image, dekad_months, SEASON_BANDS_OUT))
    if args.sensor == "s1dekad":
        dekad_months = [(y, m) for y, m in KHARIF_MONTHS for _ in KHARIF_DEKADS]
        jobs.append(("S1DEKAD", GDRIVE_S1DEKAD, build_s1dekad_image, dekad_months, S1_BANDS))

    for tag, folder, _b, months, bands in jobs:
        print(f"{tag}: {len(specs)} chips x {len(months)} months x {len(bands)} bands "
              f"= {len(months) * len(bands)} bands/chip -> Drive/{folder}")
        print(f"   months: {[f'{y}-{m:02d}' for y, m in months]}")

    if args.dry_run:
        for region, cid, crs, tr, wh in specs[:5]:
            print(f"  {region}_{cid}: crs={crs} dims={wh[0]}x{wh[1]} crsTransform={tr}")
        print("DRY RUN — no GEE calls made. Drop --dry-run to submit.")
        return

    import ee
    import gee_auth
    print("EE auth:", gee_auth.init(PROJECT))
    submitted, skipped = 0, 0
    total = len(specs) * len(jobs)
    for region, cid, crs, tr, (w, h) in specs:
        for tag, folder, builder, _months, _bands in jobs:
            local = os.path.join(LOCAL_TILE_DIRS[tag], f"{tag}_{region}_{cid}.tif")
            if os.path.exists(local):
                skipped += 1
                continue
            task = ee.batch.Export.image.toDrive(
                image=builder(ee, crs, tr),
                description=f"{tag}_{region}_{cid}",
                folder=folder,
                fileNamePrefix=f"{tag}_{region}_{cid}",
                crs=crs, crsTransform=tr, dimensions=f"{w}x{h}",
                maxPixels=int(1e9), fileFormat="GeoTIFF",
            )
            task.start()
            submitted += 1
            time.sleep(THROTTLE_S)
        if submitted % 50 == 0:
            print(f"  submitted {submitted}/{total}")
    print(f"Submitted {submitted} export tasks ({skipped} skipped — already local). "
          f"Monitor: https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
