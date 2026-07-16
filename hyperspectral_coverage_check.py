"""Hyperspectral coverage check over the pilot areas — EnMAP / PRISMA / HySIS.

Only EnMAP exposes an open, unauthenticated catalog (DLR EOC Geoservice
STAC). PRISMA (ASI) and HySIS (ISRO Bhoonidhi) are login-gated and must be
checked manually — see NOTES at the bottom.

EnMAP STAC:  https://geoservice.dlr.de/eoc/ogc/stac/v1
  collection ENMAP_HSI_L2A, temporal coverage 2022-04-27 -> present
  => cannot cover the 2021-22 AgriFieldNet label season (EnMAP launched
     2022-04-01). Useful only for a NEW rabi season paired with new labels.

Run:  python hyperspectral_coverage_check.py
"""
import json

import requests

STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1/search"
COLLECTION = "ENMAP_HSI_L2A"

# pilot-area lat/lon bboxes (W,S,E,N), computed from the Satellite_data tiles
REGIONS = {
    "UP+UPNEW (wheat/mustard)": (81.13, 27.07, 82.74, 28.33),
    "RAJASTHAN (wheat/mustard)": (76.25, 24.41, 77.31, 25.43),
    "BIHAR (wheat/mustard)":     (87.20, 25.27, 88.05, 25.88),
    "ODISHA (rice)":             (83.00, 19.01, 83.97, 19.92),
}
RABI_MONTHS = {11, 12, 1, 2, 3}   # wheat/mustard growing season
CLOUD_MAX = 15.0


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 999.0


def search(bbox):
    r = requests.get(STAC, params={"collections": COLLECTION,
                                    "bbox": ",".join(map(str, bbox)),
                                    "limit": 300}, timeout=60)
    r.raise_for_status()
    return r.json().get("features", [])


def main():
    print(f"EnMAP {COLLECTION} coverage over pilot areas "
          f"(rabi = Nov-Mar, cloud <= {CLOUD_MAX:.0f}%)\n")
    summary = {}
    for name, bbox in REGIONS.items():
        feats = search(bbox)
        dates = {}
        for f in feats:
            dt = f["properties"]["datetime"][:10]
            cc = num(f["properties"].get("eo:cloud_cover"))
            dates.setdefault(dt, 999.0)
            dates[dt] = min(dates[dt], cc)
        rabi_clear = sorted((dt, cc) for dt, cc in dates.items()
                            if int(dt[5:7]) in RABI_MONTHS and cc <= CLOUD_MAX)
        summary[name] = {"total_dates": len(dates),
                         "rabi_clear": [[dt, cc] for dt, cc in rabi_clear]}
        print(f"{name}")
        print(f"  {len(dates)} distinct acquisition dates; "
              f"{len(rabi_clear)} clear rabi dates:")
        for dt, cc in rabi_clear:
            print(f"     {dt}  cloud {cc:.0f}%")
        print()
    with open("hyperspectral_coverage.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote hyperspectral_coverage.json")

    print("""
NOTES — season-matched (2021-22) hyperspectral must be checked MANUALLY:
  PRISMA (ASI):  register https://prismauserregistration.asi.it/ -> catalog
                 https://prisma.asi.it -> draw pilot bbox, filter Nov2021-Apr2022.
                 PRISMA operates since 2019 so 2021-22 scenes MAY exist (tasking-
                 based, sparse). Products are HDF5 (.he5), 30 m, 240 bands.
  HySIS (ISRO):  https://bhoonidhi.nrsc.gov.in -> login -> Browse & Order ->
                 sensor HySIS, draw pilot bbox, season 2021-22. 30 m, VNIR 60 +
                 SWIR 256 bands. Availability sparse; India-owned (on-theme).
""")


if __name__ == "__main__":
    main()
