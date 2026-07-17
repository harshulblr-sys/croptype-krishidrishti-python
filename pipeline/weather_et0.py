"""Step 2: weather grid + FAO-56 Penman-Monteith reference ET0.

Fetches NASA POWER daily point data (AG community, 0.5 deg x 0.625 deg cells)
on a 3x3 grid covering the UP+UPNEW bbox (81.13-82.74E, 27.07-28.33N) for
2021-10-15 .. 2022-04-30 (pre-season spin-up + full rabi), computes daily
FAO-56 ET0 per cell, and writes:

  moisture_stress/weather/power_<lat>_<lon>.json   raw API cache (skip-existing)
  moisture_stress/weather_daily.csv                cell,date,et0,rain,t2m,rh,ws,rs
  moisture_stress/weather_cells.json               cell -> lat/lon/elevation

Chips are mapped to their nearest cell downstream (chip_index.json has
centroid lon/lat).
"""
import datetime as dt
import json
import math
import os

import numpy as np
import requests

import stress_common as sc

W_DIR = os.path.join(sc.OUT_DIR, "weather")
# POWER native lattice: lat = 0.25 + 0.5k, lon = 0.625k. The legacy UP grid
# (27.25/27.75/28.25 x 81.25/81.875/82.5) is exactly this lattice snapped to
# the UP bbox. AOI mode (AGRI_WEATHER_BBOX="W,S,E,N") derives the grid the
# same way for any AOI.
def _snap_grid(w, s, e, n):
    """Lattice points inside the bbox (legacy-UP-compatible); if the AOI is
    smaller than a cell, fall back to the single nearest lattice point."""
    lat0 = math.ceil((s - 0.25) / 0.5) * 0.5 + 0.25
    lon0 = math.ceil(w / 0.625) * 0.625
    lats = [round(lat0 + 0.5 * k, 3) for k in range(max(0, int((n - lat0) / 0.5)) + 1)
            if lat0 + 0.5 * k <= n]
    lons = [round(lon0 + 0.625 * k, 3) for k in range(max(0, int((e - lon0) / 0.625)) + 1)
            if lon0 + 0.625 * k <= e]
    if not lats:
        lats = [round(round(((s + n) / 2 - 0.25) / 0.5) * 0.5 + 0.25, 3)]
    if not lons:
        lons = [round(round((w + e) / 2 / 0.625) * 0.625, 3)]
    return lats, lons

if os.environ.get("AGRI_WEATHER_BBOX"):
    _w, _s, _e, _n = map(float, os.environ["AGRI_WEATHER_BBOX"].split(","))
    LATS, LONS = _snap_grid(_w, _s, _e, _n)
else:
    LATS = [27.25, 27.75, 28.25]
    LONS = [81.25, 81.875, 82.5]
START = f"{sc.Y}1015"
END = f"{sc.Y + 1}0430"
PARAMS = "T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,ALLSKY_SFC_SW_DWN,PRECTOTCORR"
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
FILL = -999.0


def fetch(lat, lon):
    cache = os.path.join(W_DIR, f"power_{lat}_{lon}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    r = requests.get(URL, params=dict(parameters=PARAMS, community="AG",
                                      longitude=lon, latitude=lat,
                                      start=START, end=END, format="JSON"),
                     timeout=120)
    r.raise_for_status()
    j = r.json()
    with open(cache, "w") as f:
        json.dump(j, f)
    return j


def et0_fao56(doy, lat_rad, z, tmax, tmin, tmean, rh, ws, rs):
    """Daily FAO-56 Penman-Monteith ET0 (mm/day). rs in MJ/m2/day."""
    e0 = lambda t: 0.6108 * math.exp(17.27 * t / (t + 237.3))
    es = (e0(tmax) + e0(tmin)) / 2.0
    ea = rh / 100.0 * e0(tmean)
    delta = 4098 * e0(tmean) / (tmean + 237.3) ** 2
    P = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26
    gamma = 0.000665 * P
    # extraterrestrial radiation Ra
    dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * doy)
    dec = 0.409 * math.sin(2 * math.pi / 365 * doy - 1.39)
    ws_angle = math.acos(max(-1, min(1, -math.tan(lat_rad) * math.tan(dec))))
    Ra = (24 * 60 / math.pi) * 0.0820 * dr * (
        ws_angle * math.sin(lat_rad) * math.sin(dec)
        + math.cos(lat_rad) * math.cos(dec) * math.sin(ws_angle))
    Rso = (0.75 + 2e-5 * z) * Ra
    Rns = 0.77 * rs
    sigma = 4.903e-9
    Rnl = sigma * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2 \
        * (0.34 - 0.14 * math.sqrt(max(ea, 0.001))) \
        * max(0.05, min(1.0, 1.35 * rs / max(Rso, 0.1) - 0.35))
    Rn = Rns - Rnl
    num = 0.408 * delta * Rn + gamma * 900 / (tmean + 273) * ws * (es - ea)
    den = delta + gamma * (1 + 0.34 * ws)
    return max(0.0, num / den)


def main():
    os.makedirs(W_DIR, exist_ok=True)
    cells, rows = {}, []
    for lat in LATS:
        for lon in LONS:
            j = fetch(lat, lon)
            z = j["geometry"]["coordinates"][2]
            p = j["properties"]["parameter"]
            cell = f"{lat}_{lon}"
            cells[cell] = dict(lat=lat, lon=lon, elev=z)
            dates = sorted(p["T2M"].keys())
            n_fill = 0
            for ds in dates:
                vals = {k: p[k][ds] for k in p}
                if any(v == FILL for v in vals.values()):
                    n_fill += 1
                    continue
                d = dt.datetime.strptime(ds, "%Y%m%d").date()
                rs = vals["ALLSKY_SFC_SW_DWN"]        # MJ/m2/day (AG community)
                et0 = et0_fao56(d.timetuple().tm_yday, math.radians(lat), z,
                                vals["T2M_MAX"], vals["T2M_MIN"], vals["T2M"],
                                vals["RH2M"], vals["WS2M"], rs)
                rows.append((cell, d.isoformat(), round(et0, 3),
                             vals["PRECTOTCORR"], vals["T2M"], vals["RH2M"],
                             vals["WS2M"], rs))
            print(f"cell {cell}: elev {z} m, {len(dates)} days, {n_fill} fill-skipped")

    with open(os.path.join(sc.OUT_DIR, "weather_cells.json"), "w") as f:
        json.dump(cells, f, indent=1)
    import csv
    with open(os.path.join(sc.OUT_DIR, "weather_daily.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "date", "et0", "rain", "t2m", "rh2m", "ws2m", "rs"])
        w.writerows(rows)
    # quick sanity: seasonal ET0 + rain per cell
    arr = np.array([(r[0], r[2], r[3]) for r in rows], dtype=object)
    for cell in cells:
        m = arr[:, 0] == cell
        print(f"  {cell}: total ET0 {sum(arr[m, 1]):.0f} mm, rain {sum(arr[m, 2]):.0f} mm"
              f" over {m.sum()} days")
    print("wrote weather_daily.csv + weather_cells.json")


if __name__ == "__main__":
    main()
