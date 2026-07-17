"""
Field-level classification + 3-model ensemble
=================================================================
Two ideas we hadn't tried, both aimed at lifting field-level OA:

1. FIELD-LEVEL modelling. Instead of classifying pixels then majority-voting,
   aggregate each field's pixels into ONE feature vector (mean/std/median/min/max
   of every base feature) and classify the field directly. This matches the
   evaluation unit, denoises boundary/speckle pixels, and trains on the true
   ~independent samples (fields, not correlated pixels).

2. ENSEMBLE. Soft-vote (average predicted probabilities) of RandomForest +
   XGBoost + LightGBM. Different learners make different errors; blending them
   reliably adds a point or two.

Same chip-level split as everything else (splits.json) so fields never leak
across train/test. Base 82 features (phenology/GLCM stay off — they didn't help).

Run:  set AGRI_DATA_DIR=...  &&  python field_level_models.py
Outputs runs/field_level/report.txt
"""

import os
import numpy as np
import torch
from collections import defaultdict

from data_loader import load_metadata, make_splits, build_label_lut, CHIPS_DIR, IGNORE_INDEX
import baseline_rf_xgb as base           # reuses chip_features
from train_hybrid import metrics_from_cm, report

BASE = r"C:\Users\harsh\OneDrive\Desktop\Harshul\ISRO_Hackathon"
base.USE_SAR_FLOODING = True             # rice-targeted SAR flooding features ON
base.USE_REDGE = True                    # red-edge NDRE/CIre ON (used if present in npz)


def tempcnn_test_probs(meta, C):
    """Load the trained field-level TempCNN and return {field_key: prob[C]} on test."""
    import torch
    import train_field_tempcnn as T
    lut = build_label_lut(meta); splits = make_splits(meta)
    cdir = os.path.dirname(CHIPS_DIR)
    Xs, Xt, _, _ = T.build_field_data(splits["train"], lut, os.path.join(cdir, "fieldseq_train.npz"))
    Ts, Tt, _, kte = T.build_field_data(splits["test"], lut, os.path.join(cdir, "fieldseq_test.npz"))
    sm, ss = Xs.reshape(-1, T.SEQ_CH).mean(0), Xs.reshape(-1, T.SEQ_CH).std(0) + 1e-6
    tm, ts = Xt.mean(0), Xt.std(0) + 1e-6
    Tsn = torch.tensor((Ts - sm) / ss).float(); Ttn = torch.tensor((Tt - tm) / ts).float()
    model = T.FieldTempCNN(C)
    model.load_state_dict(torch.load(os.path.join(BASE, "runs", "tempcnn_v1", "best_model.pt"),
                                     map_location="cpu", weights_only=False)["model"])
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(Tsn, Ttn), 1).numpy()
    return {tuple(k): prob[i] for i, k in enumerate(kte)}

# per-field aggregation statistics applied to every base feature
_AGG = ["mean", "std", "median", "min", "max"]


def _aggregate(Xpix):
    return np.concatenate([
        Xpix.mean(0), Xpix.std(0), np.median(Xpix, 0),
        Xpix.min(0), Xpix.max(0)]).astype(np.float32)


def field_rows(files, lut):
    """Aggregate labelled pixels into one row per field.
    Returns Xf [n_fields, 82*5], yf [n_fields], keys [(region,chip,field)]."""
    Xf, yf, keys = [], [], []
    for fn in files:
        Xpix, ypix, pkeys = base.chip_features(fn, lut)
        groups = defaultdict(list)
        for i, k in enumerate(pkeys):
            groups[k].append(i)
        for k, idx in groups.items():
            idx = np.array(idx)
            Xf.append(_aggregate(Xpix[idx]))
            yf.append(int(np.bincount(ypix[idx]).argmax()))
            keys.append(k)
    return np.vstack(Xf), np.array(yf), keys


def aligned_proba(model, X, C):
    """Model probabilities mapped to the full 0..C-1 class axis."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), C), np.float64)
    out[:, model.classes_] = p
    return out


def cm_of(y_true, y_pred, C):
    cm = torch.zeros(C, C, dtype=torch.long)
    idx = torch.from_numpy(y_true.astype(np.int64) * C + y_pred.astype(np.int64))
    cm += torch.bincount(idx, minlength=C * C).reshape(C, C)
    return cm


def evaluate(name, y_true, y_pred, meta, out_lines):
    C = meta["num_classes"]
    cm = cm_of(y_true, y_pred, C)
    m = metrics_from_cm(cm)
    out_lines.append(f"\n===== {name} (field-level) =====\n" + report(m, meta, cm))
    return m


def main():
    meta = load_metadata()
    lut = build_label_lut(meta)
    splits = make_splits(meta)
    C = meta["num_classes"]
    out_dir = os.path.join(BASE, "runs", "field_level")
    os.makedirs(out_dir, exist_ok=True)

    print("Aggregating fields (train/test) with SAR flooding features...")
    Xtr, ytr, _ = field_rows(splits["train"], lut)
    Xte, yte, kte = field_rows(splits["test"], lut)
    print(f"train fields={len(ytr)}  test fields={len(yte)}  features={Xtr.shape[1]}")

    # inverse-frequency sample weights (for models w/o class_weight)
    cls, cnt = np.unique(ytr, return_counts=True)
    freq = dict(zip(cls, cnt))
    sw = np.array([len(ytr) / (len(cls) * freq[t]) for t in ytr])

    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    import lightgbm as lgb

    models = {}
    print("Training RandomForest...");
    models["RF"] = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42).fit(Xtr, ytr)
    print("Training XGBoost...")
    models["XGB"] = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", num_class=C,
        objective="multi:softprob", n_jobs=-1, random_state=42).fit(Xtr, ytr, sample_weight=sw)
    print("Training LightGBM...")
    models["LGB"] = lgb.LGBMClassifier(
        n_estimators=500, num_leaves=31, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, class_weight="balanced", n_jobs=-1,
        random_state=42, verbose=-1).fit(Xtr, ytr)

    lines, summary = [], []
    probs = {}
    for name, mdl in models.items():
        probs[name] = aligned_proba(mdl, Xte, C)
        m = evaluate(name, yte, probs[name].argmax(1), meta, lines)
        summary.append((name, m))

    # 3-way soft-vote ensemble
    ens3 = sum(probs.values()) / len(probs)
    m = evaluate("ENSEMBLE-3 (RF+XGB+LGB)", yte, ens3.argmax(1), meta, lines)
    summary.append(("ENS-3", m))

    # 4-way: add the field-level TempCNN (aligned by field key; uniform fallback)
    try:
        tc = tempcnn_test_probs(meta, C)
        uni = np.ones(C) / C
        tc_prob = np.stack([tc.get(tuple(k), uni) for k in kte])
        ens4 = (sum(probs.values()) + tc_prob) / (len(probs) + 1)
        m = evaluate("ENSEMBLE-4 (RF+XGB+LGB+TempCNN)", yte, ens4.argmax(1), meta, lines)
        summary.append(("ENS-4", m))
    except Exception as e:
        print(f"TempCNN not added to ensemble: {e}")

    header = "Field-level models on the full dataset (same chip split)\n" \
             f"train fields={len(ytr)}  test fields={len(yte)}\n"
    table = ["\n" + "=" * 60,
             f"{'model':10} {'field_OA':>9} {'macroF1':>9} {'kappa':>8}"]
    for name, m in summary:
        table.append(f"{name:10} {m['oa']:9.3f} {m['macro_f1']:9.3f} {m['kappa']:8.3f}")
    txt = header + "\n".join(lines) + "\n".join(table)
    print("\n".join(table))
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(txt + "\n")
    print(f"\nArtifacts -> {out_dir}")


if __name__ == "__main__":
    main()
