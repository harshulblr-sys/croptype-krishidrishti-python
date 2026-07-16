"""Export a compact JSON payload for the advisory dashboard (self-contained,
no server). Bundles: region timeline, per-crop breakdown, weather+ET0 dekadal
series, water-balance schedules, chip centroids with per-dekad worst level, and
a sample of representative field drill-downs (time series + advisory).

Output: moisture_stress/dashboard_data.json
"""
import json
import os

import numpy as np

import stress_common as sc

LEVEL_KEYS = [-1, 0, 1, 2, 3]
DEKAD_MONTH = [sc.MONTHS.index((y, m)) for (y, m, _, _, _) in sc.DEKADS]


def daily_to_dekad_weather():
    import csv
    from collections import defaultdict
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0])   # et0, rain, t2m, ndays
    with open(os.path.join(sc.OUT_DIR, "weather_daily.csv")) as f:
        import datetime as dt
        for r in csv.DictReader(f):
            d = dt.date.fromisoformat(r["date"])
            di = next((i for i, (_, _, _, s, e) in enumerate(sc.DEKADS) if s <= d <= e), None)
            if di is None:
                continue
            a = agg[di]
            a[0] += float(r["et0"]); a[1] += float(r["rain"])
            a[2] += float(r["t2m"]); a[3] += 1
    et0 = [round(agg[i][0], 1) if i in agg else 0 for i in range(sc.N_DEKADS)]
    rain = [round(agg[i][1], 1) if i in agg else 0 for i in range(sc.N_DEKADS)]
    tmean = [round(agg[i][2] / agg[i][3], 1) if i in agg and agg[i][3] else None
             for i in range(sc.N_DEKADS)]
    return et0, rain, tmean


def main():
    import datetime as dtt
    adv = np.load(os.path.join(sc.OUT_DIR, "advisory.npz"), allow_pickle=True)
    ts = np.load(os.path.join(sc.OUT_DIR, "field_timeseries.npz"), allow_pickle=True)
    sow = np.load(os.path.join(sc.OUT_DIR, "sowing.npz"), allow_pickle=True)
    lstm = np.load(os.path.join(sc.OUT_DIR, "lstm_ks.npz"), allow_pickle=True)
    with open(os.path.join(sc.OUT_DIR, "lstm_eval.json")) as f:
        lstm_eval = json.load(f)
    with open(os.path.join(sc.OUT_DIR, "chip_index.json")) as f:
        cindex = json.load(f)
    with open(os.path.join(sc.OUT_DIR, "chip_cell.json")) as f:
        chip_cell = json.load(f)
    with open(os.path.join(sc.OUT_DIR, "water_balance.json")) as f:
        wb = json.load(f)

    chip = adv["chip"].astype(str)
    fid = adv["fid"].astype(int)
    crop8 = adv["crop8"].astype(int)
    level = adv["level"]
    deficit8 = adv["deficit_8d"]
    irrig8 = adv["irrig_8d"]
    nf = len(fid)

    # ---- region timeline (stacked counts per dekad) ----
    timeline = {sc.SCHEME[c] if False else "": 0}  # placeholder to keep linter calm
    timeline = []
    for t in range(sc.N_DEKADS):
        timeline.append({str(k): int((level[:, t] == k).sum()) for k in LEVEL_KEYS})

    # ---- per-crop: field count + fraction ever >=2 (irrigation advised) ----
    crops = {}
    for c in range(8):
        m = crop8 == c
        if not m.any():
            continue
        ever = (level[m] >= 2).any(1).mean()
        sev = (level[m] == 3).any(1).mean()
        # per-dekad share advised for this crop
        share = [int((level[m, t] >= 2).sum()) for t in range(sc.N_DEKADS)]
        crops[sc.SCHEME[c]] = dict(n=int(m.sum()), ever_advised=round(float(ever), 3),
                                   ever_severe=round(float(sev), 3), advised_per_dekad=share)

    # ---- weather / ET0 ----
    et0, rain, tmean = daily_to_dekad_weather()

    # ---- 8-day crop-water deficit (PS-6 weekly deficit layer) ----
    in_season = deficit8.sum(1) > 0
    area_ha = ts["npx"].astype(float) * 0.01          # 10 m px -> ha
    # regional mean over cropped fields + total volume (mm*ha -> m^3 = *10)
    deficit_mean = deficit8[in_season].mean(0)
    deficit_m3 = (deficit8 * area_ha[:, None]).sum(0) * 10.0
    irrig_m3_total = float((irrig8 * area_ha[:, None]).sum() * 10.0)
    per_crop_deficit = {}
    for c in range(8):
        m = (crop8 == c) & in_season
        if m.sum() < 5:
            continue
        per_crop_deficit[sc.SCHEME[c]] = [round(float(v), 1)
                                          for v in deficit8[m].mean(0)]
    deficit_block = dict(
        labels=sc.LABELS_8D,
        mean_mm=[round(float(v), 1) for v in deficit_mean],
        total_m3=[round(float(v)) for v in deficit_m3],
        per_crop=per_crop_deficit,
        season_mean_mm=round(float(deficit8[in_season].sum(1).mean()), 1),
        season_total_m3=round(float(deficit_m3.sum())),
        season_irrig_m3=round(irrig_m3_total),
        cropped_area_ha=round(float(area_ha[in_season].sum()), 1),
        peak=int(deficit_mean.argmax()))

    # ---- sowing dates from NDVI green-up ----
    method = sow["method"]
    sow_bin = sow["sow_bin"].astype(str)
    sowing_block = dict(
        bins=[k[:10] for k in sc.SOW_BIN_KEYS],
        detected_frac=round(float((method <= 1).mean()), 3),
        per_crop={})
    for c in range(8):
        m = crop8 == c
        det = m & (method <= 1)
        if det.sum() < 10:
            continue
        hist = [int(((sow_bin == k) & det).sum()) for k in sc.SOW_BIN_KEYS]
        med = dtt.date.fromordinal(int(np.median(sow["sow_ord"][det]))).isoformat()
        cal = sc.crop_curve(sc.SCHEME[c])["sow"]
        sowing_block["per_crop"][sc.SCHEME[c]] = dict(
            hist=hist, n_detected=int(det.sum()), median=med,
            calendar=cal.isoformat() if cal else None)

    # ---- LSTM Ks emulator ----
    lstm_block = dict(test=lstm_eval["test"], val=lstm_eval["val"],
                      threshold=lstm_eval["stress_ks_threshold"])

    # ---- mosaic map products (AOI-gridded runs; advisory_maps writes them) ----
    mosaics = None
    mos_path = os.path.join(sc.OUT_DIR, "maps", "mosaics.json")
    if os.path.exists(mos_path):
        with open(mos_path) as f:
            mosaics = json.load(f)
    # fixed per-crop colors (same as advisory_maps.CROP_COLORS / the web UI)
    crop_colors = {sc.SCHEME[i]: c for i, c in enumerate(
        ["#c98500", "#d95926", "#9085e9", "#898781",
         "#199e70", "#008300", "#3987e5", "#d55181"])}

    # ---- representative water-balance schedule (median cell, per crop) ----
    from collections import Counter
    modal_cell = Counter(chip_cell.values()).most_common(1)[0][0]
    schedules = {}
    for crop in ["Wheat", "Mustard", "Lentil", "Sugarcane", "_generic_rabi"]:
        bins = wb[modal_cell].get(crop)
        if not bins:
            continue
        # modal sowing bin = the one covering the most fields
        b = max(bins.values(), key=lambda v: v.get("n_fields", 0))
        schedules[crop] = dict(etc=b["etc"], depl_frac=b["depl_frac"],
                               ks_rainfed=b["ks_rainfed"], irrig_mm=b["irrig_mm"],
                               stage=b["stage"], events=b["events"], sow=b["sow"])

    # ---- chips: centroid + per-dekad worst in-season level ----
    chips_u = sorted(set(chip.tolist()))
    chips = {}
    for c in chips_u:
        m = chip == c
        worst = []
        for t in range(sc.N_DEKADS):
            lv = level[m, t]
            ins = lv[lv >= 0]
            worst.append(int(ins.max()) if len(ins) else -1)
        chips[c] = dict(lon=cindex[c]["lon"], lat=cindex[c]["lat"],
                        n=int(m.sum()), cell=chip_cell[c], worst=worst)

    # ---- representative field drill-downs ----
    # pick a few per crop: one clearly-stressed (max severe dekads), one healthy
    ndvi = ts["ndvi_filled"]; ndmi = ts["ndmi"]; vci = ts["vci"]; ssm = ts["ssm"]
    ndmi_dk = ndmi[:, DEKAD_MONTH]; ssm_dk = ssm[:, DEKAD_MONTH]
    fields = []

    ks_pred, ks_fao, ks_season = lstm["ks_pred"], lstm["ks_fao"], lstm["season"]

    def add_field(i, tag):
        fields.append(dict(
            id=f"{chip[i]}#{fid[i]}", chip=str(chip[i]), fid=int(fid[i]),
            crop=sc.SCHEME[crop8[i]], cell=chip_cell[str(chip[i])], tag=tag,
            npx=int(ts["npx"][i]),
            sow=(dtt.date.fromordinal(int(sow["sow_ord"][i])).isoformat()
                 if sow["sow_ord"][i] > 0 else None),
            sow_method=int(sow["method"][i]),
            ndvi=[round(float(v), 3) if np.isfinite(v) else None for v in ndvi[i]],
            ndmi=[round(float(v), 3) if np.isfinite(v) else None for v in ndmi_dk[i]],
            vci=[round(float(v), 1) if np.isfinite(v) else None for v in vci[i]],
            ssm=[round(float(v), 2) if np.isfinite(v) else None for v in ssm_dk[i]],
            deficit8=[round(float(v), 1) for v in deficit8[i]],
            ks_fao=[round(float(v), 3) if ks_season[i, t] else None
                    for t, v in enumerate(ks_fao[i])],
            ks_lstm=[round(float(v), 3) if ks_season[i, t] else None
                     for t, v in enumerate(ks_pred[i])],
            level=[int(v) for v in level[i]]))

    seen = set()
    for c in range(8):
        m = np.where(crop8 == c)[0]
        if len(m) < 5:
            continue
        sevcount = (level[m] == 3).sum(1) + (level[m] >= 2).sum(1) * 0.1
        stressed = m[sevcount.argmax()]
        healthy = m[((level[m] >= 2).sum(1)).argmin()]
        for i, tag in [(stressed, "high stress"), (healthy, "well watered")]:
            if i not in seen:
                add_field(i, tag); seen.add(i)

    payload = dict(
        region=sc.REGION_NAME,
        season=sc.SEASON_LABEL,
        dekad_labels=sc.DEKAD_LABELS,
        level_names=["Out of season", "No stress", "Watch",
                     "Irrigation advised", "Severe stress"],
        n_fields=nf, n_chips=len(chips_u),
        timeline=timeline, crops=crops,
        weather=dict(et0=et0, rain=rain, tmean=tmean, cell=modal_cell),
        deficit=deficit_block, sowing=sowing_block, lstm=lstm_block,
        schedules=schedules, chips=chips, fields=fields,
        mosaics=mosaics, crop_colors=crop_colors,
        peak_dekad=int((level >= 2).sum(0).argmax()))

    out = os.path.join(sc.OUT_DIR, "dashboard_data.json")
    with open(out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"wrote {out} ({os.path.getsize(out)//1024} KB); "
          f"{len(fields)} drill-down fields, {len(chips_u)} chips")

    # assemble the self-contained dashboard: template + inline JSON
    with open(os.path.join(sc.ROOT, "dashboard_template.html"), encoding="utf-8") as f:
        tpl = f.read()
    html = tpl.replace("__DATA__", json.dumps(payload, separators=(",", ":"))
                       .replace("</", "<\\/"))
    outh = os.path.join(sc.OUT_DIR, "dashboard.html")
    with open(outh, "w", encoding="utf-8") as f:
        f.write('<meta charset="utf-8">\n'
                "<title>Moisture-Stress & Irrigation Advisory</title>\n" + html)
    print(f"wrote {outh} ({os.path.getsize(outh)//1024} KB)")


if __name__ == "__main__":
    main()
