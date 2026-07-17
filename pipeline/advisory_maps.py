"""Step 6: render the colour-coded advisory maps.

Outputs (moisture_stress/maps/):
  region_<dekad>.png      chip-level overview (centroid colored by worst
                          in-season advisory among the chip's fields), 18x
  region_grid.png         all 18 dekads on one sheet
  geotiff/<chip>_<peak>.tif  field-level advisory raster (uint8: 255 bg,
                          0..3 levels, 4 = out of season) at the peak-stress
                          dekad, georeferenced to the chip grid — GIS-ready
  crop_geotiff/<chip>.tif field-level crop-type raster (uint8 class codes)
  chips_<peak>/<chip>.png example field-level maps for the busiest chips
  crop_map.png            (AOI-gridded runs) full-AOI field-level crop map
  advisory_mosaic_<t>.png (AOI-gridded runs) per-dekad advisory levels as
                          translucent colors over the NDVI backdrop
  mosaics.json            index of the mosaic products for the dashboard
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image

import stress_common as sc

MAPS = os.path.join(sc.OUT_DIR, "maps")
LEVEL_COLORS = {-1: "#b8b3ab", 0: "#2e7d32", 1: "#f9a825", 2: "#ef6c00", 3: "#c62828"}
LEVEL_NAMES = {-1: "Out of season", 0: "No stress", 1: "Watch",
               2: "Irrigation advised", 3: "Severe stress"}
# fixed per-crop colors, index = sc.SCHEME position (matches the web UI).
# Mustard is magenta so it separates cleanly from Wheat's amber on the map;
# the rare "Other" class takes the orange slot instead.
CROP_COLORS = ["#c98500", "#d55181", "#9085e9", "#898781",
               "#199e70", "#008300", "#3987e5", "#d95926"]
N_EXAMPLE_CHIPS = 6
MOSAIC_MAX_PX = 1024          # downsample mosaics beyond this for the PNGs


def hex_rgb(hx):
    return tuple(int(hx[j:j + 2], 16) for j in (1, 3, 5))


def aoi_mosaics(chips_u, cindex, chip_arr, fid_arr, crop8, level):
    """AOI-gridded runs (chips AOI_r{r}c{c}): stitch the chips and render the
    field-level crop-type map plus per-dekad advisory overlays, both drawn as
    translucent colors over the March-NDVI backdrop."""
    grid = {}
    for c in chips_u:
        m = re.fullmatch(r"AOI_r(\d+)c(\d+)", c)
        if not m:
            return None                      # scattered tiles — keep dot map
        grid[c] = (int(m.group(1)), int(m.group(2)))
    R = max(r for r, _ in grid.values()) + 1
    C = max(c for _, c in grid.values()) + 1

    side = np.load(sc.chip_npz(chips_u[0]))["field_id"].shape[0]
    H, W = R * side, C * side
    bg = np.zeros((H, W), np.float32)
    crop_ras = np.full((H, W), 255, np.uint8)
    lvl_ras = np.full((sc.N_DEKADS, H, W), 255, np.uint8)

    crop_of, lvl_of = {}, {}
    for i in range(len(chip_arr)):
        key = (str(chip_arr[i]), int(fid_arr[i]))
        crop_of[key] = int(crop8[i])
        lvl_of[key] = level[i]

    for c in chips_u:
        r, cc = grid[c]
        ys, xs = slice(r * side, (r + 1) * side), slice(cc * side, (cc + 1) * side)
        d = np.load(sc.chip_npz(c))
        bg[ys, xs] = d["awifs"][9, :, :, 0]              # March NDVI backdrop
        fids = d["field_id"]
        for f in np.unique(fids[fids > 0]):
            key = (c, int(f))
            if key not in crop_of:
                continue
            fm = fids == f
            crop_ras[ys, xs][fm] = crop_of[key]
            lv = lvl_of[key]
            for t in range(sc.N_DEKADS):
                lvl_ras[t, ys, xs][fm] = lv[t] if lv[t] >= 0 else 4

    # grayscale backdrop -> RGB
    g = np.clip((bg + 0.1) / 1.0, 0, 1)
    bg_rgb = (np.stack([g, g, g], -1) * 210 + 20).astype(np.uint8)

    def compose(ras, colors, alpha):
        rgba = np.zeros((256, 4), np.uint8)
        for code, hx in colors.items():
            rgba[code] = (*hex_rgb(hx), alpha)
        ov = rgba[ras]
        a = ov[..., 3:4].astype(np.float32) / 255.0
        out = (bg_rgb * (1 - a) + ov[..., :3] * a).astype(np.uint8)
        im = Image.fromarray(out)
        if max(im.size) > MOSAIC_MAX_PX:
            im.thumbnail((MOSAIC_MAX_PX, MOSAIC_MAX_PX), Image.NEAREST)
        return im

    crop_cols = {i: hx for i, hx in enumerate(CROP_COLORS)}
    compose(crop_ras, crop_cols, 235).save(os.path.join(MAPS, "crop_map.png"))

    lvl_cols = {0: LEVEL_COLORS[0], 1: LEVEL_COLORS[1], 2: LEVEL_COLORS[2],
                3: LEVEL_COLORS[3], 4: LEVEL_COLORS[-1]}
    adv_files = []
    for t in range(sc.N_DEKADS):
        fn = f"advisory_mosaic_{t:02d}.png"
        compose(lvl_ras[t], lvl_cols, 175).save(os.path.join(MAPS, fn))
        adv_files.append(f"maps/{fn}")

    lons = [cindex[c]["lon"] for c in chips_u]
    lats = [cindex[c]["lat"] for c in chips_u]
    meta = dict(px=[W, H], crop_map="maps/crop_map.png", advisory=adv_files,
                approx_center=[round(float(np.mean(lons)), 4),
                               round(float(np.mean(lats)), 4)],
                km=[round(W * 10 / 1000, 1), round(H * 10 / 1000, 1)])
    with open(os.path.join(MAPS, "mosaics.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote AOI mosaics: crop_map.png + {sc.N_DEKADS} advisory overlays "
          f"({W}x{H} px)")
    return meta


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

    aoi_mosaics(chips_u, cindex, chip, adv["fid"], adv["crop8"], level)

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

    cg_dir = os.path.join(MAPS, "crop_geotiff")
    os.makedirs(cg_dir, exist_ok=True)
    crop_cmap = {i: (*hex_rgb(hx), 255) for i, hx in enumerate(CROP_COLORS)}
    crop_cmap[255] = (0, 0, 0, 0)

    lv_of, cr_of = {}, {}
    for i in range(len(chip)):
        lv_of[(chip[i], int(adv["fid"][i]))] = int(level[i, peak])
        cr_of[(chip[i], int(adv["fid"][i]))] = int(adv["crop8"][i])

    for c in chips_u:
        d = np.load(sc.chip_npz(c))
        fids = d["field_id"]
        ras = np.full(fids.shape, 255, np.uint8)
        crop_ras = np.full(fids.shape, 255, np.uint8)
        for f in np.unique(fids[fids > 0]):
            lv = lv_of.get((c, int(f)))
            if lv is not None:
                ras[fids == f] = lv if lv >= 0 else 4
                crop_ras[fids == f] = cr_of[(c, int(f))]
        info = cindex[c]
        tr = rasterio.Affine(*info["transform"])
        with rasterio.open(os.path.join(gt_dir, f"{c}_{peak:02d}.tif"), "w",
                           driver="GTiff", width=ras.shape[1], height=ras.shape[0],
                           count=1, dtype="uint8", crs=info["crs"], transform=tr,
                           nodata=255, compress="deflate") as dst:
            dst.write(ras, 1)
            dst.write_colormap(1, {int(k): tuple(int(v) for v in cmap[k])
                                   for k in [0, 1, 2, 3, 4, 255]})
        with rasterio.open(os.path.join(cg_dir, f"{c}.tif"), "w",
                           driver="GTiff", width=crop_ras.shape[1],
                           height=crop_ras.shape[0], count=1, dtype="uint8",
                           crs=info["crs"], transform=tr, nodata=255,
                           compress="deflate") as dst:
            dst.write(crop_ras, 1)
            dst.write_colormap(1, crop_cmap)
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
