"""AOI step 2: classify every pseudo-field with the finalized 8-class LGB.

Reuses build_features.chip_fields VERBATIM (same 261x5 feature assembly the
model was trained on) — the AOI chips carry placeholder crop_id/label_mask =
field presence, so the labeled-pixel path just selects all field pixels.
S1 normalization stats are fit on the AOI itself (the per-region z-scoring
the training pipeline applies, computed for this AOI as its own "region");
the region-id feature is 0 (UP) — the service is gated to northern India
where UP is the nearest training domain.

Requires env (set by aoi_run.py): AGRI_DATA_DIR=<workspace>,
AGRI_S2_DIR=<workspace>/s2raw, AGRI_S2RE_DIR=<workspace>/s2reraw,
AGRI_SEASON_DIR=<workspace>/season, AGRI_OUT_DIR, AGRI_CHIP_PREFIXES=AOI_.

Outputs (AGRI_OUT_DIR): crop_map.npz (split="aoi", true8=-1),
chip_index.json — the same interface stress_crop_map.py provides for UP.
"""
import json
import os

import joblib
import numpy as np
import rasterio
from pyproj import Transformer

import stress_common as sc
import build_features as bf

MODEL = os.path.join(sc.ROOT, "runs", "final_classifier_rebuilt", "lgb.joblib")


def main():
    os.makedirs(sc.OUT_DIR, exist_ok=True)
    tiles = sc.chip_ids_region0()
    if not tiles:
        raise SystemExit(f"no AOI chips under {sc.CHIP_DIR}")
    print(f"AOI tiles: {len(tiles)}")

    s1_stats = bf.fit_s1_region_stats(tiles)
    mu, sd = s1_stats[0]
    print(f"AOI S1 stats: mu={np.round(mu, 2)} sd={np.round(sd, 2)}")

    X, chips, fids = [], [], []
    for name in tiles:
        for cid, f, row, seq, static in bf.chip_fields(name, s1_stats):
            X.append(row)
            chips.append(name)
            fids.append(f)
    X = np.stack(X)
    print(f"fields: {len(X)}, features: {X.shape[1]}")

    lgb = joblib.load(MODEL)
    prob = lgb.predict_proba(X)
    pred8 = prob.argmax(1).astype(np.int16)
    conf = prob.max(1).astype(np.float32)
    print("predicted crop counts:",
          {sc.SCHEME[i]: int((pred8 == i).sum()) for i in range(8)
           if (pred8 == i).any()})
    print(f"mean confidence: {conf.mean():.3f}")

    np.savez_compressed(os.path.join(sc.OUT_DIR, "crop_map.npz"),
                        chip=np.array(chips), fid=np.array(fids, np.int32),
                        pred8=pred8, conf=conf,
                        true8=np.full(len(X), -1, np.int16),
                        split=np.full(len(X), "aoi", dtype="U5"))

    chips_arr = np.array(chips)
    index = {}
    for cid in tiles:
        with rasterio.open(sc.season_tile(cid)) as src:
            crs = src.crs.to_string()
            tr = src.transform
            cx, cy = src.xy(src.height // 2, src.width // 2)
        lon, lat = Transformer.from_crs(crs, "EPSG:4326",
                                        always_xy=True).transform(cx, cy)
        index[cid] = dict(crs=crs, transform=[tr.a, tr.b, tr.c, tr.d, tr.e, tr.f],
                          lon=round(lon, 5), lat=round(lat, 5),
                          n_fields=int((chips_arr == cid).sum()))
    with open(os.path.join(sc.OUT_DIR, "chip_index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f"wrote crop_map.npz + chip_index.json ({len(index)} tiles) -> {sc.OUT_DIR}")


if __name__ == "__main__":
    main()
