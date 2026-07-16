"""Step 3b: per-field sowing-date estimation from the NDVI green-up dekad.

Replaces the fixed literature sowing calendar (handoff caveat C): for every
field, find the dekad where the field-mean NDVI first crosses GREENUP_NDVI
(0.30) rising, interpolate the crossing date inside the dekad, and subtract a
crop-specific emergence lag (GREENUP_LAG, days from sowing to that NDVI at
10 m) to get the sowing date. The only remaining assumption is the lag — a
physiological offset applied to an OBSERVED event, not a calendar.

The detection is CAUSAL: the crossing at dekad t uses only NDVI up to t+1
(one dekad of confirmation), so in an operational season the sowing date is
known ~1 dekad after green-up — early enough, since irrigation demand peaks
months later.

Method flags per field:
  0 greenup          crossing observed inside the season
  1 greenup-early    already green at the first dekad (crossing clipped)
  2 crop-median      no crossing (NDVI never reaches 0.30 / too few obs);
                     median detected sowing of same-crop peers
  3 default          literature calendar (last resort)
Sugarcane (season-long standing cane, sow=None) gets flag 4 / bin "ALL".

Outputs: moisture_stress/sowing.npz  (chip, fid, crop8, sow_ord, greenup_ord,
         method, sow_bin), moisture_stress/sowing_summary.json
"""
import datetime as dt
import json
import os

import numpy as np

import stress_common as sc

METHODS = ["greenup", "greenup-early", "crop-median", "default", "perennial"]
SUSTAIN_TOL = 0.03        # next dekad may dip this far below threshold


def detect_crossing(v):
    """Rabi green-up: rising crossing of GREENUP_NDVI in a dekadal series.

    Fields that START green (kharif crop still standing in early Nov) are not
    taken at face value: we wait for the harvest trough (NDVI dropping below
    the threshold) and use the subsequent rising RE-crossing as the rabi
    green-up. Only a field that stays green all season keeps the
    early flag. Returns (crossing_date, early) or None. Causal: the decision
    at dekad t uses v[:t+2] only.
    """
    thr = sc.GREENUP_NDVI
    fin = np.isfinite(v)
    if fin.sum() < 3 or np.nanmax(v) < thr:
        return None
    first = int(np.argmax(fin))
    below_seen = v[first] < thr       # armed once we've been below threshold
    for t in range(first + 1, sc.N_DEKADS):
        if not (np.isfinite(v[t]) and np.isfinite(v[t - 1])):
            continue
        if v[t] < thr:
            below_seen = True
            continue
        if below_seen and v[t - 1] < thr:          # rising crossing
            nxt = v[t + 1] if t + 1 < sc.N_DEKADS else np.nan
            if np.isfinite(nxt) and nxt < thr - SUSTAIN_TOL:
                continue              # unconfirmed spike
            f = (thr - v[t - 1]) / max(1e-6, v[t] - v[t - 1])
            span = (sc.DEKAD_MID[t] - sc.DEKAD_MID[t - 1]).days
            return sc.DEKAD_MID[t - 1] + dt.timedelta(days=round(f * span)), False
    if v[first] >= thr and not below_seen:         # green the whole season
        return sc.DEKAD_MID[first], True
    if v[first] >= thr:               # went below but never re-crossed: the
        return None                   # green signal was kharif, no rabi crop
    return None


def main():
    ts = np.load(os.path.join(sc.OUT_DIR, "field_timeseries.npz"), allow_pickle=True)
    chip, fid, crop8 = ts["chip"], ts["fid"], ts["crop8"].astype(int)
    ndvi = ts["ndvi_filled"]
    nf = len(fid)

    sow_ord = np.zeros(nf, np.int64)
    green_ord = np.zeros(nf, np.int64)
    method = np.full(nf, 3, np.int8)

    # pass 1: direct green-up detections
    for i in range(nf):
        name = sc.SCHEME[crop8[i]]
        if name == "Sugarcane":
            method[i] = 4
            continue
        hit = detect_crossing(ndvi[i])
        if hit is None:
            continue
        gdate, early = hit
        lag = sc.GREENUP_LAG.get(name, sc.GREENUP_LAG["_generic_rabi"])
        sow = gdate - dt.timedelta(days=lag)
        sow = max(sc.SOW_WINDOW[0], min(sc.SOW_WINDOW[1], sow))
        sow_ord[i] = sow.toordinal()
        green_ord[i] = gdate.toordinal()
        method[i] = 1 if early else 0

    # pass 2: crop-median fallback for undetected fields
    med_of = {}
    for c in range(8):
        m = (crop8 == c) & (method <= 1)
        if m.sum() >= 20:
            med_of[c] = int(np.median(sow_ord[m]))
    for i in range(nf):
        if method[i] in (0, 1, 4):
            continue
        if crop8[i] in med_of:
            sow_ord[i] = med_of[crop8[i]]
            method[i] = 2
        else:                          # literature default, generic if kharif
            name = sc.SCHEME[crop8[i]]
            p = sc.crop_curve(name)
            sow_ord[i] = (p["sow"] or sc.CROP_PARAMS["_generic_rabi"]["sow"]).toordinal()
            method[i] = 3

    sow_bin = np.array(["ALL" if method[i] == 4 else
                        sc.sow_bin_of(dt.date.fromordinal(int(sow_ord[i])))
                        for i in range(nf)])

    np.savez_compressed(os.path.join(sc.OUT_DIR, "sowing.npz"),
                        chip=chip, fid=fid, crop8=crop8.astype(np.int16),
                        sow_ord=sow_ord, greenup_ord=green_ord,
                        method=method, sow_bin=sow_bin)

    # summary: per-crop method counts, sowing spread, bin histogram
    summary = {"methods": METHODS, "crops": {}}
    for c in range(8):
        m = crop8 == c
        if not m.any():
            continue
        det = m & (method <= 1)
        ent = {"n": int(m.sum()),
               "method_counts": {METHODS[k]: int((method[m] == k).sum())
                                 for k in range(5) if (method[m] == k).any()}}
        if det.any():
            q = np.percentile(sow_ord[det], [10, 50, 90]).astype(int)
            ent["sow_p10_p50_p90"] = [dt.date.fromordinal(int(v)).isoformat() for v in q]
            ent["default_calendar"] = (sc.crop_curve(sc.SCHEME[c])["sow"] or "").isoformat() \
                if sc.crop_curve(sc.SCHEME[c])["sow"] else None
        summary["crops"][sc.SCHEME[c]] = ent
    summary["bin_histogram"] = {k: int((sow_bin == k).sum()) for k in sc.SOW_BIN_KEYS
                                if (sow_bin == k).any()}
    summary["bin_histogram"]["ALL"] = int((sow_bin == "ALL").sum())
    with open(os.path.join(sc.OUT_DIR, "sowing_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    n_det = int((method <= 1).sum())
    print(f"sowing detected from green-up for {n_det}/{nf} fields "
          f"({n_det / nf:.0%}); methods:",
          {METHODS[k]: int((method == k).sum()) for k in range(5)})
    for cname, ent in summary["crops"].items():
        if "sow_p10_p50_p90" in ent:
            print(f"  {cname:14s} n={ent['n']:5d} sow p10/50/90 = "
                  f"{'/'.join(ent['sow_p10_p50_p90'])} (default {ent['default_calendar']})")
    print("wrote sowing.npz + sowing_summary.json")


if __name__ == "__main__":
    main()
