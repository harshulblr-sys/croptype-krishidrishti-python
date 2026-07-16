"""Step 6b: render the 8-day crop-water-deficit maps (PS-6 deficit layer).

Deficit = ETc - ETa under the rainfed FAO-56 bucket for the field's
(cell, crop, sowing-bin) — the unmet crop water demand per 8-day period,
in mm. Zero for out-of-season / no-rabi fields.

Outputs (moisture_stress/maps/):
  deficit_grid.png            all 23 8-day periods on one sheet (chip means)
  deficit_<pp>.png            standalone region map at the peak-deficit period
  deficit_geotiff/<chip>.tif  float32 GeoTIFF per chip, 23 bands = deficit mm
                              per 8-day period per field pixel (nodata -1),
                              band descriptions = period start dates
  deficit_season_<chip>.png   example seasonal-total field maps (busiest chips)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

import stress_common as sc

MAPS = os.path.join(sc.OUT_DIR, "maps")
N_EXAMPLE_CHIPS = 4
CMAP = plt.get_cmap("YlOrRd")


def main():
    os.makedirs(MAPS, exist_ok=True)
    gt_dir = os.path.join(MAPS, "deficit_geotiff")
    os.makedirs(gt_dir, exist_ok=True)
    adv = np.load(os.path.join(sc.OUT_DIR, "advisory.npz"), allow_pickle=True)
    with open(os.path.join(sc.OUT_DIR, "chip_index.json")) as f:
        cindex = json.load(f)

    chip = adv["chip"].astype(str)
    fid = adv["fid"].astype(int)
    deficit = adv["deficit_8d"]                    # [nf, 23] mm
    chips_u = sorted(set(chip.tolist()))

    # peak-deficit 8-day period (field mean)
    mean_d = deficit.mean(0)
    peak = int(mean_d.argmax())
    print(f"peak 8-day deficit period: {sc.LABELS_8D[peak]} "
          f"(mean {mean_d[peak]:.1f} mm/field)")

    # ---- chip means per period, for the region maps ----
    chip_mean = {}
    for c in chips_u:
        chip_mean[c] = deficit[chip == c].mean(0)  # [23]
    vmax = max(10.0, np.percentile([m.max() for m in chip_mean.values()], 98))

    lons = {c: cindex[c]["lon"] for c in chips_u}
    lats = {c: cindex[c]["lat"] for c in chips_u}
    fig_g, axes = plt.subplots(4, 6, figsize=(26, 15), sharex=True, sharey=True)
    for t in range(sc.N_8D):
        vals = np.array([chip_mean[c][t] for c in chips_u])
        for ax, standalone in ((axes.flat[t], False), (None, True)):
            if standalone:
                if t != peak:
                    continue
                f2, ax = plt.subplots(figsize=(8, 6.5))
            sca = ax.scatter([lons[c] for c in chips_u], [lats[c] for c in chips_u],
                             c=vals, cmap=CMAP, vmin=0, vmax=vmax,
                             s=(12 if not standalone else 30), edgecolors="none")
            ax.set_title(sc.LABELS_8D[t], fontsize=(9 if not standalone else 13))
            ax.set_aspect("equal")
            if standalone:
                ax.set_xlabel("lon"); ax.set_ylabel("lat")
                f2.colorbar(sca, ax=ax, label="crop water deficit (mm / 8 days)")
                f2.tight_layout()
                f2.savefig(os.path.join(MAPS, f"deficit_{t:02d}.png"), dpi=110)
                plt.close(f2)
    for t in range(sc.N_8D, axes.size):
        axes.flat[t].axis("off")
    fig_g.suptitle(f"{sc.REGION_NAME} — 8-day crop water deficit (chip mean, mm), "
                   f"{sc.SEASON_LABEL}", fontsize=15)
    fig_g.colorbar(sca, ax=axes, shrink=0.5, label="mm / 8 days")
    fig_g.savefig(os.path.join(MAPS, "deficit_grid.png"), dpi=100)
    plt.close(fig_g)

    # ---- per-chip multi-band GeoTIFFs (field-level, all 23 periods) ----
    dmap = {}
    for i in range(len(chip)):
        dmap[(chip[i], fid[i])] = deficit[i]
    for c in chips_u:
        d = np.load(sc.chip_npz(c))
        fids = d["field_id"]
        ras = np.full((sc.N_8D,) + fids.shape, -1.0, np.float32)
        for f in np.unique(fids[fids > 0]):
            v = dmap.get((c, int(f)))
            if v is not None:
                ras[:, fids == f] = v[:, None]
        info = cindex[c]
        tr = rasterio.Affine(*info["transform"])
        with rasterio.open(os.path.join(gt_dir, f"{c}.tif"), "w",
                           driver="GTiff", width=ras.shape[2], height=ras.shape[1],
                           count=sc.N_8D, dtype="float32", crs=info["crs"],
                           transform=tr, nodata=-1.0, compress="deflate") as dst:
            dst.write(ras)
            for b, (s, _) in enumerate(sc.PERIODS_8D, 1):
                dst.set_band_description(b, f"deficit mm {s.isoformat()}")

    # ---- example seasonal-total field maps ----
    season = {c: deficit[chip == c].sum(1) for c in chips_u}
    busy = sorted(chips_u, key=lambda c: -season[c].mean())[:N_EXAMPLE_CHIPS]
    for c in busy:
        d = np.load(sc.chip_npz(c))
        fids = d["field_id"]
        ras = np.full(fids.shape, np.nan, np.float32)
        for f in np.unique(fids[fids > 0]):
            v = dmap.get((c, int(f)))
            if v is not None:
                ras[fids == f] = v.sum()
        fig, ax = plt.subplots(figsize=(6.5, 6))
        bg = d["awifs"][9, :, :, 0]
        ax.imshow(bg, cmap="Greys_r", vmin=-0.1, vmax=0.9)
        im = ax.imshow(ras, cmap=CMAP, vmin=0)
        fig.colorbar(im, ax=ax, label="seasonal crop water deficit (mm)")
        ax.set_title(f"{c} — rabi 2021-22 total deficit", fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(MAPS, f"deficit_season_{c}.png"), dpi=110)
        plt.close(fig)

    print(f"wrote deficit_grid.png, deficit_{peak:02d}.png, "
          f"{len(chips_u)} multi-band GeoTIFFs, {len(busy)} seasonal examples")


if __name__ == "__main__":
    main()
