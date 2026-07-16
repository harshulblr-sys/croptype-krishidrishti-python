"""Final crop classifier (rebuild of the lost finalize_classifier.py).

8-class LightGBM (n_est=500, leaves=31, lr=0.05, class_weight=balanced) on
the 1305-dim field table, plus the confidence-gated TempCNN rice-rescue:
fields the LGB predicts as Fallow are flipped to Rice where
P_tempcnn(rice) > RESCUE_FIXED_THRESH (0.70, fixed mode — no val veto).

Outputs runs/final_classifier/ {lgb.joblib, report.txt, meta.json}.
Run build_features.py (and optionally train_field_tempcnn.py) first.
"""
import json
import os

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("AGRI_DATA_DIR", os.path.join(ROOT, "Extracted_dataset_gee"))
OUT_DIR = os.path.join(ROOT, "runs", "final_classifier_rebuilt")

SCHEME = ["Wheat", "Mustard", "Lentil", "No crop/Fallow", "Sugarcane", "Maize", "Rice", "Other"]
FALLOW, RICE = 3, 6
RESCUE_FIXED_THRESH = 0.70


def load_table():
    d = np.load(os.path.join(DATA_DIR, "field_table.npz"), allow_pickle=True)
    with open(os.path.join(DATA_DIR, "splits.json")) as f:
        splits = json.load(f)
    idx = {s: np.isin(d["chip"], splits[s]) for s in ("train", "val", "test")}
    return d, idx


def report(y, p, title):
    oa = (y == p).mean()
    kap = cohen_kappa_score(y, p)
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=range(8), zero_division=0)
    lines = [f"===== {title} =====",
             f"OA={oa:.4f}  macroF1={f1.mean():.4f}  kappa={kap:.4f}",
             f"{'idx':>3} {'class':<20} {'prec':>6} {'rec':>6} {'f1':>6} {'n':>5}"]
    for i, name in enumerate(SCHEME):
        lines.append(f"{i:>3} {name:<20} {pr[i]:6.3f} {rc[i]:6.3f} {f1[i]:6.3f} {sup[i]:>5}")
    lines.append("Confusion (rows=true):")
    lines.append(str(confusion_matrix(y, p, labels=range(8))))
    txt = "\n".join(lines)
    print(txt)
    return txt, dict(OA=oa, macroF1=float(f1.mean()), kappa=kap,
                     rice_F1=float(f1[RICE]), rice_rec=float(rc[RICE]), rice_prec=float(pr[RICE]))


def main():
    d, idx = load_table()
    X, y = d["X"], d["y8"].astype(int)
    Xtr, ytr = X[idx["train"]], y[idx["train"]]
    Xva, yva = X[idx["val"]], y[idx["val"]]
    Xte, yte = X[idx["test"]], y[idx["test"]]
    print(f"train/val/test fields = {len(ytr)}/{len(yva)}/{len(yte)}")

    lgb = LGBMClassifier(n_estimators=500, num_leaves=31, learning_rate=0.05,
                         class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
    lgb.fit(Xtr, ytr)
    pv = lgb.predict(Xva)
    pt = lgb.predict(Xte)
    txt_v, _ = report(yva, pv, "VAL, no rescue")
    txt_base, m_base = report(yte, pt, "TEST, no rescue")

    # ---- rice rescue ----
    txt_res, m_res, p_rescued = "", None, None
    try:
        import train_field_tempcnn as tc
        model, norm = tc.load_trained()
        ds = np.load(os.path.join(DATA_DIR, "field_seq.npz"), allow_pickle=True)
        p_rice = tc.rice_probs(model, norm, ds["seq"][idx["test"]], ds["static"][idx["test"]])
        p_rescued = pt.copy()
        flip = (p_rescued == FALLOW) & (p_rice > RESCUE_FIXED_THRESH)
        p_rescued[flip] = RICE
        print(f"\nrescue: {flip.sum()} Fallow fields flipped to Rice (t={RESCUE_FIXED_THRESH})")
        txt_res, m_res = report(yte, p_rescued, f"TEST, rice-rescue t={RESCUE_FIXED_THRESH}")
    except FileNotFoundError:
        print("no TempCNN checkpoint (runs/tempcnn_rebuilt) — skipping rescue")

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(lgb, os.path.join(OUT_DIR, "lgb.joblib"))
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
        f.write("FINAL CLASSIFIER (rebuilt) — 8-class LightGBM + TempCNN rice-rescue\n\n")
        f.write(txt_v + "\n\n" + txt_base + "\n\n" + txt_res + "\n")
    meta = {"scheme": SCHEME, "model": "LightGBM 500/31/0.05 balanced",
            "features": int(X.shape[1]), "base": m_base}
    if m_res:
        meta["rescue"] = {"thresh": RESCUE_FIXED_THRESH, "mode": "fixed-fallow", **m_res}
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("\nwrote", OUT_DIR)


if __name__ == "__main__":
    main()
