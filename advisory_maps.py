"""Step 6: render the colour-coded advisory maps.

Outputs (moisture_stress/maps/):
  region_<dekad>.png      chip-level overview (centroid colored by worst
                          in-season advisory among the chip's fields), 18x
  region_grid.png         all 18 dekads on one sheet
  geotiff/<chip>_<peak>.tif  field-level advisory raster (uint8: 255 bg,
                          0..3 levels, 4 = out of season) at the peak-stress
                          dekad, georeferenced to the chip grid — GIS-ready
  chips_<peak>/<chip>.png example field-level maps for the busiest chips
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
LEVEL_COLORS = {-1: "#b8b3ab", 0: "#2e7d32", 1: "#f9a825", 2: "#ef6c00", 3: "#c62828"}
LEVEL_NAMES = {-1: "Out of season", 0: "No stress", 1: "Watch",
               2: "Irrigation advised", 3: "Severe stress"}
N_EXAMPLE_CHIPS = 6


def worst_per_chip(chips_u, chip_arr, level_t):
    """Worst in-season level per chip at one dekad (-1 if nothing in season)."""
    out = {}
    for c in chips_u:
        lv = level_t[chip_arr == c]
        ins = lv[lv >= 0]
        out[c] = int(ins.max()) if len(ins) else -1
    return out


def main():
    os.makedirs(MAPS, exist_ok=True)
    adv = np.load(os.path.join(sc.OUT_DIR, "advisory.npz"), allow_pickle=True)
    with open(os.path.join(sc.OUT_DIR, "chip_index.json")) as f:
        cindex = json.load(f)
    chip, level = adv["chip"].astype(str), adv["level"]
    chips_u = sorted(set(chip.tolist()))
    lons = {c: cindex[c]["lon"] for c in chips_u}
    lats = {c: cindex[c]["lat"] for c in chips_u}

    # peak-stress dekad = most level>=2 fields
    need = (level >= 2).sum(0)
    peak = int(need.argmax())
    print(f"peak-stress dekad: {sc.DEKAD_LABELS[peak]} ({need[peak]} fields >= orange)")

    # ---- region overview PNGs ----
    fig_g, axes = plt.subplots(3, 6, figsize=(26, 12), sharex=True, sharey=True)
    for t in range(sc.N_DEKADS):
        wc = worst_per_chip(chips_u, chip, level[:, t])
        xs = [lons[c] for c in chips_u]
        ys = [lats[c] for c in chips_u]
        cs = [LEVEL_COLORS[wc[c]] for c in chips_u]
        for ax, standalone in ((axes.flat[t], False), (None, True)):
            if standalone:
                f2, ax = plt.subplots(figsize=(8, 6.5))
            ax.scatter(xs, ys, c=cs, s=(12 if not standalone else 28),
                       edgecolors="none")
            ax.set_title(sc.DEKAD_LABELS[t], fontsize=(9 if not standalone else 13))
            ax.set_aspect("equal")
            if standalone:
                ax.set_xlabel("lon"); ax.set_ylabel("lat")
                handles = [plt.Line2D([], [], marker="o", ls="", color=LEVEL_COLORS[k],
                                      label=LEVEL_NAMES[k]) for k in (-1, 0, 1, 2, 3)]
                ax.legend(handles=handles, loc="lower right", fontsize=8)
                f2.tight_layout()
                f2.savefig(os.path.join(MAPS, f"region_{t:02d}.png"), dpi=110)
                plt.close(f2)
    fig_g.suptitle(f"{sc.REGION_NAME} — worst field advisory per chip, "
                   f"{sc.SEASON_LABEL}", fontsize=15)
    fig_g.tight_layout()
    fig_g.savefig(os.path.join(MAPS, "region_grid.png"), dpi=100)
    plt.close(fig_g)

    # ---- field-level GeoTIFFs + example PNGs at peak dekad ----
    gt_dir = os.path.join(MAPS, "geotiff")
    ex_dir = os.path.join(MAPS, f"chips_{peak:02d}")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(ex_dir, exist_ok=True)

    # busiest chips = most orange/red fields at peak
    busy = sorted(chips_u,
                  key=lambda c: -(level[chip == c, peak] >= 2).sum())[:N_EXAMPLE_CHIPS]

    cmap = np.zeros((256, 4), np.uint8)
    for k, hx in LEVEL_COLORS.items():
        rgb = tuple(int(hx[j:j + 2], 16) for j in (1, 3, 5))
        cmap[k if k >= 0 else 4] = (*rgb, 255)
    cmap[255] = (0, 0, 0, 0)

    lv_of = {}
    for i in range(len(chip)):
        lv_of[(chip[i], int(adv["fid"][i]))] = int(level[i, peak])

    for c in chips_u:
        d = np.load(sc.chip_npz(c))
        fids = d["field_id"]
        ras = np.full(fids.shape, 255, np.uint8)
        for f in np.unique(fids[fids > 0]):
            lv = lv_of.get((c, int(f)))
            if lv is not None:
                ras[fids == f] = lv if lv >= 0 else 4
        info = cindex[c]
        tr = rasterio.Affine(*info["transform"])
        with rasterio.open(os.path.join(gt_dir, f"{c}_{peak:02d}.tif"), "w",
                           driver="GTiff", width=ras.shape[1], height=ras.shape[0],
                           count=1, dtype="uint8", crs=info["crs"], transform=tr,
                           nodata=255, compress="deflate") as dst:
            dst.write(ras, 1)
            dst.write_colormap(1, {int(k): tuple(int(v) for v in cmap[k])
                                   for k in [0, 1, 2, 3, 4, 255]})
        if c in busy:
            rgba = cmap[ras]
            bg = d["awifs"][9, :, :, 0]        # Mar NDVI backdrop
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(bg, cmap="Greys_r", vmin=-0.1, vmax=0.9)
            ax.imshow(rgba)
            ax.set_title(f"{c} — {sc.DEKAD_LABELS[peak]}", fontsize=11)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(ex_dir, f"{c}.png"), dpi=110)
            plt.close(fig)

    print(f"wrote maps -> {MAPS} (region_grid, 18 region PNGs, "
          f"{len(chips_u)} GeoTIFFs, {len(busy)} example chips)")


if __name__ == "__main__":
    main()
