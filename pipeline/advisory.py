"""Step 5: fuse water balance + observed stress into per-field advisories.

For each field and dekad (Nov 2021 - Apr 2022), combines
  - the FAO-56 bucket for the field's (weather cell, crop, SOW BIN) — stage
    timing anchored on the field's own NDVI-green-up sowing date
    (sowing_detect.py), not a fixed calendar:
    scheduled depletion fraction, rainfed stress coefficient Ks,
    stage + stage sensitivity, scheduled irrigation events;
  - observed spectral condition: single-season VCI (percentile vs same-crop
    peers), NDVI z-score, NDMI z-score (canopy water), S1 SSM proxy;
into a 5-level advisory:

  -1 GRAY   out of season / no rabi crop detected
   0 GREEN  no stress
   1 YELLOW watch — depletion approaching trigger or mild anomaly
   2 ORANGE irrigation advised — trigger crossed / moderate stress corroborated
   3 RED    severe — rainfed Ks < 0.85 and spectra confirm, sensitive stage

Rules are deliberately transparent (advisory product, not a classifier).

Also carries the field's 8-day crop-water-deficit series (PS-6 "weekly
(8-day)" deficit) from its water-balance bucket into the per-field outputs.

Outputs: moisture_stress/advisory.npz  (adds deficit_8d, irrig_8d, ks_8d),
         advisory_summary.csv, advisory_fields.csv,
         deficit_fields.csv (per field: 8-day deficit series, mm)
"""
import csv
import datetime as dt
import json
import os

import numpy as np

import stress_common as sc

LEVELS = {-1: "Out of season", 0: "No stress", 1: "Watch",
          2: "Irrigation advised", 3: "Severe stress"}
CURVE_OF = {"Wheat": "Wheat", "Mustard": "Mustard", "Lentil": "Lentil",
            "Sugarcane": "Sugarcane"}          # others -> _generic_rabi
KHARIF_OR_BARE = {"Rice", "Maize", "No crop/Fallow"}
GREEN_NDVI = 0.35                              # rabi-crop-present threshold
# dekad index -> month index in the 11-month npz axis
DEKAD_MONTH = [sc.MONTHS.index((y, m)) for (y, m, _, _, _) in sc.DEKADS]


def month_to_dekad(arr):
    """[nf,11] monthly -> [nf,18] broadcast to each month's dekads."""
    return arr[:, DEKAD_MONTH]


def main():
    ts = np.load(os.path.join(sc.OUT_DIR, "field_timeseries.npz"), allow_pickle=True)
    sow = np.load(os.path.join(sc.OUT_DIR, "sowing.npz"), allow_pickle=True)
    with open(os.path.join(sc.OUT_DIR, "water_balance.json")) as f:
        wb = json.load(f)
    with open(os.path.join(sc.OUT_DIR, "chip_cell.json")) as f:
        chip_cell = json.load(f)

    chip, fid = ts["chip"], ts["fid"]
    crop8 = ts["crop8"]
    nf = len(fid)
    ndvi = ts["ndvi_filled"]
    vci, ndvi_z = ts["vci"], ts["ndvi_z"]
    ndmi_z = month_to_dekad(ts["ndmi_z"])
    ssm = month_to_dekad(ts["ssm"])

    # per-field sowing bin (sowing.npz row order == crop_map order == ts order)
    assert (sow["chip"] == chip).all() and (sow["fid"] == fid).all()
    sow_bin = sow["sow_bin"].astype(str)
    sow_ord = sow["sow_ord"]

    level = np.full((nf, sc.N_DEKADS), 0, np.int8)
    score = np.zeros((nf, sc.N_DEKADS), np.float32)
    deficit_8d = np.zeros((nf, sc.N_8D), np.float32)
    irrig_8d = np.zeros((nf, sc.N_8D), np.float32)
    ks_8d = np.ones((nf, sc.N_8D), np.float32)

    # spectral stress components in [0,1] (nan -> 0 contribution)
    s_vci = np.clip((40.0 - vci) / 40.0, 0, 1)
    s_nz = np.clip(-ndvi_z / 2.0, 0, 1)
    s_mz = np.clip(-ndmi_z / 2.0, 0, 1)
    s_ssm = np.clip((0.40 - ssm) / 0.40, 0, 1)
    comp = np.stack([s_vci, s_nz, s_mz], -1)
    with np.errstate(invalid="ignore"):
        s_spec = np.nanmean(comp, axis=-1)
    s_spec = np.nan_to_num(s_spec, nan=0.0)
    spec_known = np.isfinite(vci)

    for i in range(nf):
        cell = chip_cell[str(chip[i])]
        cname = CURVE_OF.get(sc.SCHEME[crop8[i]], "_generic_rabi")
        b = wb[cell][cname][sow_bin[i]]
        depl = np.array(b["depl_frac"])
        ks = np.array(b["ks_rainfed"])
        irr = np.array(b["irrig_mm"])
        stages = b["stage"]
        p = sc.crop_curve(cname)["p"]

        crop_name = sc.SCHEME[crop8[i]]
        season_ndvi = ndvi[i, 3:15]            # Dec d1 .. Mar d3
        no_rabi = (crop_name in KHARIF_OR_BARE and
                   (np.nanmax(season_ndvi) if np.isfinite(season_ndvi).any()
                    else 0.0) < GREEN_NDVI)
        if not no_rabi:
            deficit_8d[i] = b["deficit_mm_8d"]
            irrig_8d[i] = b["irrig_mm_8d"]
            ks_8d[i] = b["ks_rainfed_8d"]

        for t in range(sc.N_DEKADS):
            sens = sc.STAGE_SENSITIVITY[stages[t]]
            if no_rabi or stages[t] == "off-season":
                level[i, t] = -1
                continue
            s_wb = np.clip(depl[t] / max(p, 1e-6), 0, 1.5) / 1.5
            score[i, t] = sens * (0.5 * s_wb * 1.5 + 0.5 * s_spec[i, t])

            trigger = (irr[t] > 0) or (depl[t] >= p)
            near = depl[t] >= 0.8 * p
            dry_ssm = np.isfinite(ssm[i, t]) and ssm[i, t] < 0.30
            spec_mild = spec_known[i, t] and s_spec[i, t] >= 0.30
            spec_mod = spec_known[i, t] and s_spec[i, t] >= 0.50
            spec_sev = spec_known[i, t] and (s_spec[i, t] >= 0.65 or
                                             (np.isfinite(vci[i, t]) and vci[i, t] < 15))

            if ks[t] < 0.85 and spec_sev and sens >= 1.0:
                level[i, t] = 3
            elif (trigger and sens >= 0.8) or (spec_mod and dry_ssm) or \
                 (ks[t] < 0.85 and spec_mod):
                level[i, t] = 2
            elif near or spec_mild or ks[t] < 1.0:
                level[i, t] = 1
            else:
                level[i, t] = 0

    np.savez_compressed(os.path.join(sc.OUT_DIR, "advisory.npz"),
                        chip=chip, fid=fid, crop8=crop8, level=level,
                        score=score, s_spec=s_spec.astype(np.float32),
                        deficit_8d=deficit_8d, irrig_8d=irrig_8d, ks_8d=ks_8d,
                        sow_ord=sow_ord, sow_bin=sow_bin,
                        labels_8d=np.array(sc.LABELS_8D),
                        dekad_labels=np.array(sc.DEKAD_LABELS))

    # summaries
    with open(os.path.join(sc.OUT_DIR, "advisory_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dekad"] + [LEVELS[k] for k in (-1, 0, 1, 2, 3)])
        for t in range(sc.N_DEKADS):
            w.writerow([sc.DEKAD_LABELS[t]] +
                       [int((level[:, t] == k).sum()) for k in (-1, 0, 1, 2, 3)])
    with open(os.path.join(sc.OUT_DIR, "advisory_fields.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chip", "fid", "crop", "cell", "levels_nov_to_apr"])
        for i in range(nf):
            w.writerow([chip[i], fid[i], sc.SCHEME[crop8[i]],
                        chip_cell[str(chip[i])],
                        "".join(str(v) if v >= 0 else "." for v in level[i])])

    with open(os.path.join(sc.OUT_DIR, "deficit_fields.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chip", "fid", "crop", "cell", "sow_date", "season_deficit_mm",
                    "season_irrig_req_mm"] + [f"deficit_{l}" for l in sc.LABELS_8D])
        for i in range(nf):
            w.writerow([chip[i], fid[i], sc.SCHEME[crop8[i]],
                        chip_cell[str(chip[i])],
                        dt.date.fromordinal(int(sow_ord[i])).isoformat()
                        if sow_ord[i] > 0 else "",
                        round(float(deficit_8d[i].sum()), 1),
                        round(float(irrig_8d[i].sum()), 1)] +
                       [round(float(v), 1) for v in deficit_8d[i]])

    print("advisory level counts per dekad:")
    for t in range(sc.N_DEKADS):
        c = {LEVELS[k]: int((level[:, t] == k).sum()) for k in (-1, 0, 1, 2, 3)}
        print(f"  {sc.DEKAD_LABELS[t]}: {c}")
    print("wrote advisory.npz + advisory_summary.csv + advisory_fields.csv "
          "+ deficit_fields.csv")


if __name__ == "__main__":
    main()
