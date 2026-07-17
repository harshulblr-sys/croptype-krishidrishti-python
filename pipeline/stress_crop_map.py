"""Step 1 of the moisture-stress pipeline: wall-to-wall crop map for the
demonstrator region (UP+UPNEW) from the finalized 8-class LightGBM.

Predicts a crop for EVERY field in region 0 (train fields included — this is
the product map, not an evaluation; split membership is recorded so honest
accuracy statements remain possible). Also builds a per-chip georeference
index (CRS, affine transform, centroid lon/lat) used by every later stage.

Outputs (moisture_stress/):
  crop_map.npz    chip, fid, pred8, conf, true8, split
  chip_index.json {chip: {crs, transform, lon, lat, n_fields}}
"""
import json
import os

import joblib
import numpy as np
import rasterio
from pyproj import Transformer

import stress_common as sc

MODEL = os.path.join(sc.ROOT, "runs", "final_classifier_rebuilt", "lgb.joblib")


def main():
    os.makedirs(sc.OUT_DIR, exist_ok=True)

    d = np.load(os.path.join(sc.DATA_DIR, "field_table.npz"), allow_pickle=True)
    with open(os.path.join(sc.DATA_DIR, "splits.json")) as f:
        splits = json.load(f)
    split = np.full(len(d["chip"]), "train", dtype="U5")
    for s in ("val", "test"):
        split[np.isin(d["chip"], splits[s])] = s

    sel = d["region"] == sc.REGION_ID
    X, chip, fid = d["X"][sel], d["chip"][sel], d["fid"][sel]
    true8, spl = d["y8"][sel].astype(int), split[sel]
    print(f"region-0 fields: {sel.sum()} over {len(set(chip.tolist()))} chips")

    lgb = joblib.load(MODEL)
    prob = lgb.predict_proba(X)
    pred8 = prob.argmax(1).astype(np.int16)
    conf = prob.max(1).astype(np.float32)

    counts = {sc.SCHEME[i]: int((pred8 == i).sum()) for i in range(8)}
    print("predicted crop counts:", counts)
    acc = (pred8 == true8)
    for s in ("train", "val", "test"):
        m = spl == s
        print(f"  agreement with labels [{s}]: {acc[m].mean():.3f} (n={m.sum()})")

    np.savez_compressed(os.path.join(sc.OUT_DIR, "crop_map.npz"),
                        chip=chip, fid=fid.astype(np.int32), pred8=pred8,
                        conf=conf, true8=true8.astype(np.int16), split=spl)

    # ---- per-chip georeference index ----
    index = {}
    for cid in sc.chip_ids_region0():
        with rasterio.open(sc.season_tile(cid)) as src:
            crs = src.crs.to_string()
            tr = src.transform
            cx, cy = src.xy(src.height // 2, src.width // 2)
        lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(cx, cy)
        index[cid] = dict(crs=crs, transform=[tr.a, tr.b, tr.c, tr.d, tr.e, tr.f],
                          lon=round(lon, 5), lat=round(lat, 5),
                          n_fields=int((chip == cid).sum()))
    with open(os.path.join(sc.OUT_DIR, "chip_index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f"wrote crop_map.npz + chip_index.json ({len(index)} chips) -> {sc.OUT_DIR}")


if __name__ == "__main__":
    main()
