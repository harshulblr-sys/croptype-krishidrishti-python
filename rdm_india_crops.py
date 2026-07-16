"""Query the ESA WorldCereal Reference Data Module (RDM) public REST API and
count how many in-situ reference COLLECTIONS (and how many field/point
samples) exist for Wheat / Mustard / Rice over India.

No login needed for public collections. Only depends on `requests`
(+ pandas for the pretty table; falls back to plain print if absent).

API base:   https://ewoc-rdm-api.iiasa.ac.at
Endpoints used (verified live 2026-07):
  GET /collections/search?Bbox=W&Bbox=S&Bbox=E&Bbox=N[&ValidityTime.Start=..]
  GET /collections/{id}/items/codestats      -> per-crop (EWOC) sample counts
Legend (EWOC code -> crop name):
  https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/
      legend/WorldCereal_LC_CT_legend_latest.csv

Run:  python rdm_india_crops.py
Out:  console table + rdm_india_crops.json
"""
import csv
import io
import json
import os
import sys

import requests

RDM = "https://ewoc-rdm-api.iiasa.ac.at"
LEGEND_URL = ("https://artifactory.vgt.vito.be/artifactory/auxdata-public/"
              "worldcereal/legend/WorldCereal_LC_CT_legend_latest.csv")

# Bounding boxes: (west, south, east, north)
BBOXES = {
    "India (national)": (68.0, 8.0, 97.5, 37.0),
    # the four AgriFieldNet states, roughly:
    "AgriFieldNet states (UP/Raj/Odisha/Bihar)": (73.0, 17.5, 88.5, 31.0),
}
BBOX = BBOXES["India (national)"]

# Optional season filter — set to None to include every year.
VALIDITY = None  # e.g. ("2020-11-01", "2022-04-30")

# ---- EWOC crop-code sets (dashes stripped, as the API returns them) -------
def _pref(code, p):        # code integer, p 6-digit string prefix
    return str(code).startswith(p)

CROP_MATCHERS = {
    "Wheat":   lambda c: _pref(c, "110101"),                 # all wheat cultivars
    "Rice":    lambda c: _pref(c, "110108"),                 # 1101080000
    "Mustard/Rapeseed (oilseed)": lambda c: (
        _pref(c, "110600003") or _pref(c, "110600008")),      # rapeseed_rape + mustard
}
# reported separately (leaf vegetable, likely NOT the AgriFieldNet 'Mustard'):
MUSTARD_GREENS = 1103080130


def get_json(url, **params):
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def load_legend():
    """EWOC integer code -> label_full (best-effort; empty dict on failure)."""
    try:
        r = requests.get(LEGEND_URL, timeout=60)
        r.raise_for_status()
        text = r.content.decode("utf-8-sig")
        out = {}
        for row in csv.DictReader(io.StringIO(text), delimiter=";"):
            raw = (row.get("ewoc_code") or "").replace("-", "").strip()
            if raw.isdigit():
                out[int(raw)] = row.get("label_full") or row.get("sampling_label") or ""
        return out
    except Exception as e:                                    # noqa: BLE001
        print(f"  (legend fetch failed: {e})", file=sys.stderr)
        return {}


def search_collections(bbox, validity=None):
    params = [("Bbox", bbox[0]), ("Bbox", bbox[1]), ("Bbox", bbox[2]), ("Bbox", bbox[3])]
    if validity:
        params += [("ValidityTime.Start", f"{validity[0]}T00:00:00Z"),
                   ("ValidityTime.End", f"{validity[1]}T00:00:00Z")]
    r = requests.get(f"{RDM}/collections/search", params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else data.get("items", [])


def codestats(cid):
    try:
        d = get_json(f"{RDM}/collections/{cid}/items/codestats")
        return {int(s["code"]): int(s["count"]) for s in d.get("ewocStats", [])}
    except Exception as e:                                    # noqa: BLE001
        print(f"  (codestats failed for {cid}: {e})", file=sys.stderr)
        return {}


def bbox_within_india(cb, india=(68.0, 8.0, 97.5, 37.0)):
    w, s, e, n = cb
    return w >= india[0] and s >= india[1] and e <= india[2] and n <= india[3]


def main():
    print(f"BBox (W,S,E,N) = {BBOX}   validity = {VALIDITY}\n")
    legend = load_legend()
    cols = search_collections(BBOX, VALIDITY)
    print(f"{len(cols)} collections intersect the bounding box.\n")

    results, crop_collection_hits = [], {k: 0 for k in CROP_MATCHERS}
    crop_sample_totals_india = {k: 0 for k in CROP_MATCHERS}
    crop_sample_totals_all = {k: 0 for k in CROP_MATCHERS}
    greens_total = 0

    for c in sorted(cols, key=lambda x: -x.get("featureCount", 0)):
        cid = c["collectionId"]
        cb = c["extent"]["spatial"]["bbox"][0]
        india_only = ("_ind_" in cid) or bbox_within_india(cb)
        stats = codestats(cid)
        per_crop = {}
        for crop, match in CROP_MATCHERS.items():
            n = sum(cnt for code, cnt in stats.items() if match(code))
            per_crop[crop] = n
            if n > 0:
                crop_collection_hits[crop] += 1
                crop_sample_totals_all[crop] += n
                if india_only:
                    crop_sample_totals_india[crop] += n
        greens_total += stats.get(MUSTARD_GREENS, 0)
        results.append({
            "collectionId": cid, "featureCount": c.get("featureCount"),
            "type": c.get("type"), "accessType": c.get("accessType"),
            "india_specific": india_only, "bbox": [round(x, 2) for x in cb],
            "crop_counts": per_crop,
        })

    # ---- console report ----
    hdr = f"{'collection':<40}{'India?':<7}{'total':>8}{'Wheat':>8}{'Mustard':>9}{'Rice':>7}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        pc = r["crop_counts"]
        if sum(pc.values()) == 0:
            continue
        print(f"{r['collectionId']:<40}{'yes' if r['india_specific'] else 'no':<7}"
              f"{r['featureCount']:>8}{pc['Wheat']:>8}"
              f"{pc['Mustard/Rapeseed (oilseed)']:>9}{pc['Rice']:>7}")

    print("\n=== NUMBER OF COLLECTIONS containing each crop (bbox-intersecting) ===")
    for crop, n in crop_collection_hits.items():
        print(f"  {crop:<28}: {n} collections")

    print("\n=== SAMPLE COUNTS (India-specific collections only) ===")
    for crop, n in crop_sample_totals_india.items():
        print(f"  {crop:<28}: {n:,} samples")
    print("\n=== SAMPLE COUNTS (all bbox-intersecting collections; incl. regional/global) ===")
    for crop, n in crop_sample_totals_all.items():
        print(f"  {crop:<28}: {n:,} samples")
    print(f"\n  (mustard_greens vegetable, reported separately): {greens_total:,} samples")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rdm_india_crops.json")
    with open(out, "w") as f:
        json.dump({"bbox": BBOX, "validity": VALIDITY,
                   "n_collections_intersecting": len(cols),
                   "collections_per_crop": crop_collection_hits,
                   "samples_india_specific": crop_sample_totals_india,
                   "samples_all_intersecting": crop_sample_totals_all,
                   "mustard_greens": greens_total,
                   "collections": results}, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
