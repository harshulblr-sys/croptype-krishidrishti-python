"""Step 5b: LSTM moisture-stress model — emulates the FAO-56 stress
coefficient Ks from SATELLITE FEATURES ONLY (PS-6 "deep learning models
(LSTM)" for moisture-stress detection).

Why an emulator and not a supervised stress classifier: no moisture-stress
ground truth exists (nobody labeled field stress in 2021-22), so the FAO-56
rainfed water balance — anchored on per-field green-up sowing dates — is the
physically-based reference, and the LSTM learns to reproduce its dekadal Ks
from the spectral/SAR time series alone. Because the inputs contain NO
weather variables, the trained model estimates crop water stress where
meteorological data is missing, late, or too coarse — the satellite series
substitutes for the weather-driven bucket.

Inputs per field per dekad (18 steps, Nov 2021 - Apr 2022):
  NDVI, NDRE (dekadal), NDMI, VV, SSM proxy (monthly -> dekad), VCI/100,
  NDVI-z, validity flags, days-after-sowing, crop one-hot (8).
Target: Ks_rainfed [18] from the field's (cell, crop, sow-bin) bucket,
        loss-masked to in-season dekads.
Split: chip-level train/val/test from Extracted_dataset_gee/splits.json
       (same protocol as the classifier — no chip leaks across splits).
Model: 2-layer unidirectional LSTM (hidden 64) -> sigmoid head. CAUSAL:
       Ks(t) uses dekads <= t only, so it can run mid-season.

Outputs: runs/stress_lstm/best_model.pt, moisture_stress/lstm_eval.json,
         moisture_stress/lstm_ks.npz (predicted Ks for every field)
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

import stress_common as sc

RUN_DIR = os.path.join(sc.ROOT, "runs", "stress_lstm")
SEED = 42
HIDDEN, LAYERS, DROPOUT = 64, 2, 0.2
EPOCHS, PATIENCE, BATCH, LR = 200, 20, 128, 1e-3
STRESS_KS = 0.85          # binary stress flag threshold (matches advisory)
DEKAD_MONTH = [sc.MONTHS.index((y, m)) for (y, m, _, _, _) in sc.DEKADS]


def build_dataset():
    ts = np.load(os.path.join(sc.OUT_DIR, "field_timeseries.npz"), allow_pickle=True)
    sow = np.load(os.path.join(sc.OUT_DIR, "sowing.npz"), allow_pickle=True)
    with open(os.path.join(sc.OUT_DIR, "water_balance.json")) as f:
        wb = json.load(f)
    with open(os.path.join(sc.OUT_DIR, "chip_cell.json")) as f:
        chip_cell = json.load(f)

    chip = ts["chip"].astype(str)
    crop8 = ts["crop8"].astype(int)
    nf = len(chip)

    def dk(a):                          # monthly [nf,11] -> dekadal [nf,18]
        return a[:, DEKAD_MONTH]

    ndvi, ndre = ts["ndvi_filled"], ts["ndre"]
    ndmi, vv, ssm = dk(ts["ndmi"]), dk(ts["vv"]), dk(ts["ssm"])
    vci, nz = ts["vci"], ts["ndvi_z"]

    CURVE_OF = {"Wheat": "Wheat", "Mustard": "Mustard", "Lentil": "Lentil",
                "Sugarcane": "Sugarcane"}
    ks = np.zeros((nf, sc.N_DEKADS), np.float32)
    season = np.zeros((nf, sc.N_DEKADS), bool)
    das = np.zeros((nf, sc.N_DEKADS), np.float32)
    for i in range(nf):
        cell = chip_cell[chip[i]]
        curve = CURVE_OF.get(sc.SCHEME[crop8[i]], "_generic_rabi")
        b = wb[cell][curve][str(sow["sow_bin"][i])]
        ks[i] = b["ks_rainfed"]
        season[i] = np.array(b["stage"]) != "off-season"
        if sow["sow_ord"][i] > 0:
            das[i] = [(m.toordinal() - sow["sow_ord"][i]) / 150.0
                      for m in sc.DEKAD_MID]

    def clean(a, lo, hi, scale=1.0, shift=0.0):
        v = np.clip((a + shift) * scale, lo, hi)
        return np.nan_to_num(v, nan=0.0).astype(np.float32)

    feats = [clean(ndvi, -0.2, 1.0), clean(ndre, -0.2, 1.0),
             clean(ndmi, -1.0, 1.0), clean(vv, -1.5, 1.5, 1 / 10.0, 15.0),
             clean(ssm, 0.0, 1.0), clean(vci, 0.0, 1.0, 1 / 100.0),
             clean(nz, -1.5, 1.5, 1 / 2.0),
             np.isfinite(ndvi).astype(np.float32),
             np.isfinite(ndmi).astype(np.float32),
             np.isfinite(vv).astype(np.float32),
             np.clip(das, -1.0, 2.0)]
    X = np.stack(feats, -1)                        # [nf, 18, 11]
    onehot = np.zeros((nf, sc.N_DEKADS, 8), np.float32)
    onehot[np.arange(nf), :, crop8] = 1.0
    X = np.concatenate([X, onehot], -1)            # [nf, 18, 19]

    part = np.full(nf, -1, np.int8)
    splits_path = os.path.join(sc.DATA_DIR, "splits.json")
    if os.path.exists(splits_path):               # absent in AOI workspaces
        with open(splits_path) as f:
            sp = json.load(f)
        for k, v in (("train", 0), ("val", 1), ("test", 2)):
            part[np.isin(chip, sp[k])] = v
    return X, ks, season, part, chip, ts["fid"], crop8


class KsLSTM(nn.Module):
    def __init__(self, nfeat):
        super().__init__()
        self.lstm = nn.LSTM(nfeat, HIDDEN, LAYERS, batch_first=True,
                            dropout=DROPOUT)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return torch.sigmoid(self.head(h)).squeeze(-1)   # [B, T] in [0,1]


def masked_mse(pred, y, m):
    d = (pred - y) ** 2 * m
    return d.sum() / m.sum().clamp(min=1)


def infer():
    """AOI mode: no training, no splits. Load the UP-trained emulator, predict
    Ks for every field from its satellite series, and report agreement with
    the AOI's own FAO-56 water balance as a consistency diagnostic."""
    X, ks, season, part, chip, fid, crop8 = build_dataset()
    model = KsLSTM(X.shape[-1])
    model.load_state_dict(torch.load(os.path.join(RUN_DIR, "best_model.pt")))
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X)).numpy()

    m = season
    p, y = pred[m], ks[m]
    mae = float(np.abs(p - y).mean())
    r2 = 1.0 - float(((p - y) ** 2).sum() / max(1e-9, ((y - y.mean()) ** 2).sum()))
    sp_, sy = p < STRESS_KS, y < STRESS_KS
    ent = dict(n_dekads=int(m.sum()), mae=round(mae, 4), r2=round(r2, 4),
               stress_precision=round(float((sp_ & sy).sum() / max(1, sp_.sum())), 3),
               stress_recall=round(float((sp_ & sy).sum() / max(1, sy.sum())), 3),
               stress_base_rate=round(float(sy.mean()), 3))
    print(f"AOI inference ({len(chip)} fields): MAE {mae:.4f} R2 {r2:.4f} "
          f"| stress P {ent['stress_precision']:.2f} R {ent['stress_recall']:.2f}")
    ev = {"mode": "inference (UP-trained model applied to this AOI)",
          "stress_ks_threshold": STRESS_KS, "val": ent, "test": ent}
    with open(os.path.join(sc.OUT_DIR, "lstm_eval.json"), "w") as f:
        json.dump(ev, f, indent=1)
    np.savez_compressed(os.path.join(sc.OUT_DIR, "lstm_ks.npz"),
                        chip=chip, fid=fid, crop8=crop8.astype(np.int16),
                        ks_pred=pred.astype(np.float32),
                        ks_fao=ks, season=season, part=part)
    print("wrote lstm_eval.json + lstm_ks.npz (inference mode)")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(RUN_DIR, exist_ok=True)
    X, ks, season, part, chip, fid, crop8 = build_dataset()
    print(f"dataset: X{X.shape}, in-season target dekads "
          f"{season.sum()}/{season.size}")
    tX = torch.tensor(X)
    tY = torch.tensor(ks)
    tM = torch.tensor(season.astype(np.float32))

    idx = {k: np.where(part == v)[0] for k, v in
           (("train", 0), ("val", 1), ("test", 2))}
    print("fields per split:", {k: len(v) for k, v in idx.items()})

    model = KsLSTM(X.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, best_ep, bad = 1e9, -1, 0
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(idx["train"])
        tot, nb = 0.0, 0
        for b0 in range(0, len(perm), BATCH):
            b = perm[b0:b0 + BATCH]
            opt.zero_grad()
            loss = masked_mse(model(tX[b]), tY[b], tM[b])
            loss.backward()
            opt.step()
            tot += float(loss); nb += 1
        model.eval()
        with torch.no_grad():
            vl = float(masked_mse(model(tX[idx["val"]]), tY[idx["val"]],
                                  tM[idx["val"]]))
        if vl < best_val - 1e-5:
            best_val, best_ep, bad = vl, ep, 0
            torch.save(model.state_dict(), os.path.join(RUN_DIR, "best_model.pt"))
        else:
            bad += 1
        if ep % 10 == 0 or bad == 0:
            print(f"  ep {ep:3d} train {tot / nb:.5f} val {vl:.5f}"
                  + (" *" if bad == 0 else ""))
        if bad >= PATIENCE:
            break
    print(f"best val MSE {best_val:.5f} @ epoch {best_ep}")

    model.load_state_dict(torch.load(os.path.join(RUN_DIR, "best_model.pt")))
    model.eval()
    with torch.no_grad():
        pred = model(tX).numpy()

    ev = {"val_mse": round(best_val, 5), "epoch": best_ep,
          "features": int(X.shape[-1]), "stress_ks_threshold": STRESS_KS}
    for k in ("val", "test"):
        m = season[idx[k]]
        p, y = pred[idx[k]][m], ks[idx[k]][m]
        mae = float(np.abs(p - y).mean())
        ss = 1.0 - float(((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        sp_, sy = p < STRESS_KS, y < STRESS_KS
        prec = float((sp_ & sy).sum() / max(1, sp_.sum()))
        rec = float((sp_ & sy).sum() / max(1, sy.sum()))
        ev[k] = dict(n_dekads=int(m.sum()), mae=round(mae, 4), r2=round(ss, 4),
                     stress_precision=round(prec, 3), stress_recall=round(rec, 3),
                     stress_base_rate=round(float(sy.mean()), 3))
        print(f"{k}: MAE {mae:.4f} R2 {ss:.4f} | stress(Ks<{STRESS_KS}) "
              f"P {prec:.2f} R {rec:.2f} (base {sy.mean():.2f})")

    with open(os.path.join(sc.OUT_DIR, "lstm_eval.json"), "w") as f:
        json.dump(ev, f, indent=1)
    np.savez_compressed(os.path.join(sc.OUT_DIR, "lstm_ks.npz"),
                        chip=chip, fid=fid, crop8=crop8.astype(np.int16),
                        ks_pred=pred.astype(np.float32),
                        ks_fao=ks, season=season, part=part)
    print("wrote lstm_eval.json + lstm_ks.npz + runs/stress_lstm/best_model.pt")


if __name__ == "__main__":
    import sys
    if "--infer" in sys.argv:
        infer()
    else:
        main()
