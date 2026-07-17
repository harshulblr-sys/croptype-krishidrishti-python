"""Step 4: stage-aware FAO-56 crop-water balance per (cell x crop x sow bin).

Daily root-zone bucket, Oct 15 spin-up -> Apr 30:
  ETc = Kc(stage) * ET0          (Kc curve from stress_common; stage timing
                                  anchored on the PER-FIELD sowing dates from
                                  sowing_detect.py, binned to dekads)
  Pe  = rain - runoff            (runoff = 50% of rain beyond 25 mm/day)
  Dr += ETc - Pe                 (depletion; drainage clips at 0)
Two runs per combo:
  scheduled : when Dr > RAW = p*TAW -> irrigation event refills to FC
              (the ADVISORY schedule: dates + mm)
  rainfed   : no irrigation; Ks = (TAW-Dr)/((1-p)*TAW) clipped [0,1];
              ETa = Ks*ETc, and the CROP WATER DEFICIT = ETc - ETa
TAW grows with the root fraction from the stage model. Initial profile at
90% FC (post-monsoon). Fields inherit their chip's nearest weather cell.

Reported on two grids: 18 dekads (satellite-native) and 23 8-day periods
(PS-6 "weekly (8-day)" requirement) — both exact aggregations of the daily run.

Output: moisture_stress/water_balance.json
  {cell: {crop: {sow_bin: {et0, rain, etc, pe, depl_frac, ks_rainfed,
                           irrig_mm, deficit_mm, eta_mm (all [18]), stage [18],
                           *_8d (all [23]), stage_8d [23], events, sow}}}}
  + chip_cell.json {chip: cell}
Sugarcane has the single bin "ALL" (season-long standing cane).
"""
import csv
import datetime as dt
import json
import os
from collections import defaultdict

import numpy as np

import stress_common as sc

CROPS = ["Wheat", "Mustard", "Lentil", "Sugarcane", "_generic_rabi"]
INIT_DEPL = 0.10          # start at 90% of field capacity (post-monsoon)


def dekad_index(date):
    for i, (_, _, _, s, e) in enumerate(sc.DEKADS):
        if s <= date <= e:
            return i
    return None


def p8_index(date):
    for i, (s, e) in enumerate(sc.PERIODS_8D):
        if s <= date <= e:
            return i
    return None


def load_weather():
    daily = defaultdict(list)
    with open(os.path.join(sc.OUT_DIR, "weather_daily.csv")) as f:
        for r in csv.DictReader(f):
            daily[r["cell"]].append((dt.date.fromisoformat(r["date"]),
                                     float(r["et0"]), float(r["rain"])))
    for c in daily:
        daily[c].sort()
    return daily


def run_bucket(days, params):
    """days = [(date, et0, rain)] sorted. Returns dekadal + 8-day dict."""
    taw_max = sc.TAW_PER_M * params["root"]
    keys = ("et0", "rain", "etc", "pe", "irrig_mm", "deficit_mm", "eta_mm")
    dk = {k: np.zeros(sc.N_DEKADS) for k in keys}
    p8 = {k: np.zeros(sc.N_8D) for k in keys}
    depl_end = np.full(sc.N_DEKADS, np.nan)
    ks_min = np.ones(sc.N_DEKADS)
    stage_lab = ["off-season"] * sc.N_DEKADS
    depl_end8 = np.full(sc.N_8D, np.nan)
    ks_min8 = np.ones(sc.N_8D)
    stage_lab8 = ["off-season"] * sc.N_8D
    events = []

    dr_s = INIT_DEPL * taw_max     # scheduled-irrigation bucket
    dr_r = INIT_DEPL * taw_max     # rainfed bucket
    for date, et0, rain in days:
        stage, kc, froot = sc.stage_on(date, params)
        taw = max(0.15, froot) * taw_max if stage != "off-season" else taw_max
        raw = params["p"] * taw
        etc = kc * et0
        runoff = max(0.0, rain - 25.0) * 0.5
        pe = rain - runoff

        # scheduled bucket
        dr_s = max(0.0, dr_s + etc - pe)
        if stage != "off-season" and dr_s > raw:
            events.append((date.isoformat(), round(dr_s, 1)))
            dr_s = 0.0
            irr = events[-1][1]
        else:
            irr = 0.0
        # rainfed bucket (ETc scales down when stressed)
        ks = 1.0 if dr_r <= raw else max(0.0, (taw - dr_r) / max(1e-6, (1 - params["p"]) * taw))
        dr_r = min(taw, max(0.0, dr_r + ks * etc - pe))
        eta = ks * etc
        deficit = etc - eta

        for idx, out, dep, ksm, stg in ((dekad_index(date), dk, depl_end, ks_min, stage_lab),
                                        (p8_index(date), p8, depl_end8, ks_min8, stage_lab8)):
            if idx is None:
                continue
            out["et0"][idx] += et0
            out["rain"][idx] += rain
            out["etc"][idx] += etc
            out["pe"][idx] += pe
            out["irrig_mm"][idx] += irr
            out["deficit_mm"][idx] += deficit
            out["eta_mm"][idx] += eta
            dep[idx] = dr_s / taw
            ksm[idx] = min(ksm[idx], ks)
            stg[idx] = stage if stage != "off-season" else stg[idx]

    res = {k: dk[k].round(1).tolist() for k in keys}
    res.update(depl_frac=np.nan_to_num(depl_end, nan=0.0).round(3).tolist(),
               ks_rainfed=ks_min.round(3).tolist(), stage=stage_lab, events=events)
    res.update({k + "_8d": p8[k].round(1).tolist() for k in keys})
    res.update(depl_frac_8d=np.nan_to_num(depl_end8, nan=0.0).round(3).tolist(),
               ks_rainfed_8d=ks_min8.round(3).tolist(), stage_8d=stage_lab8)
    return res


def main():
    daily = load_weather()
    with open(os.path.join(sc.OUT_DIR, "weather_cells.json")) as f:
        cells = json.load(f)
    with open(os.path.join(sc.OUT_DIR, "chip_index.json")) as f:
        chip_index = json.load(f)
    sow = np.load(os.path.join(sc.OUT_DIR, "sowing.npz"), allow_pickle=True)

    # chip -> nearest cell
    chip_cell = {}
    for cid, info in chip_index.items():
        best = min(cells, key=lambda c: (cells[c]["lat"] - info["lat"]) ** 2
                   + (cells[c]["lon"] - info["lon"]) ** 2)
        chip_cell[cid] = best
    used = sorted(set(chip_cell.values()))
    print("cells in use:", {c: sum(1 for v in chip_cell.values() if v == c) for c in used})

    # (cell, curve, bin) combos actually present among the fields. Kharif/bare
    # crops map to _generic_rabi (advisory only uses them if observed green).
    CURVE_OF = {"Wheat": "Wheat", "Mustard": "Mustard", "Lentil": "Lentil",
                "Sugarcane": "Sugarcane"}
    combos = defaultdict(int)
    for i in range(len(sow["fid"])):
        cell = chip_cell[str(sow["chip"][i])]
        curve = CURVE_OF.get(sc.SCHEME[int(sow["crop8"][i])], "_generic_rabi")
        combos[(cell, curve, str(sow["sow_bin"][i]))] += 1
    print(f"{len(combos)} (cell x crop x sow-bin) buckets for "
          f"{len(sow['fid'])} fields")

    wb = {}
    n_run = 0
    for (cell, curve, bin_key), n in sorted(combos.items()):
        sow_date = None if bin_key == "ALL" else sc.sow_bin_mid(bin_key)
        params = sc.curve_with_sowing(curve, sow_date)
        b = run_bucket(daily[cell], params)
        b["sow"] = sow_date.isoformat() if sow_date else None
        b["n_fields"] = n
        wb.setdefault(cell, {}).setdefault(curve, {})[bin_key] = b
        n_run += 1

    # headline: wheat irrigation need across sowing bins in the modal cell
    from collections import Counter
    modal = Counter(chip_cell.values()).most_common(1)[0][0]
    if "Wheat" in wb.get(modal, {}):
        for bin_key, b in sorted(wb[modal]["Wheat"].items()):
            ev = b["events"]
            print(f"  {modal} wheat sown {b['sow']}: {len(ev)} irrigations "
                  f"{sum(e[1] for e in ev):.0f} mm, season deficit "
                  f"{sum(b['deficit_mm']):.0f} mm (rainfed)")

    with open(os.path.join(sc.OUT_DIR, "water_balance.json"), "w") as f:
        json.dump(wb, f, indent=1)
    with open(os.path.join(sc.OUT_DIR, "chip_cell.json"), "w") as f:
        json.dump(chip_cell, f, indent=1)
    print(f"wrote water_balance.json ({n_run} buckets) + chip_cell.json")


if __name__ == "__main__":
    main()
