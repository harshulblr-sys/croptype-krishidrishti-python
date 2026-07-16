"""
Per-crop recall/precision pooled over 5-fold CV (leakage-free, current best config)
=================================================================
Single-split per-class numbers are noisy on the small classes (Rice=11, Lentil=10
test fields). This pools the 8-class confusion matrix over all 5 chip-level folds
(every field predicted exactly once, out-of-fold) for a stable per-crop picture, and
prints the top confusions per class so you can see WHERE each crop's misses go.

Run:  set AGRI_DATA_DIR=...  &&  python cv_perclass.py
"""
import os
import glob
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb

from data_loader import load_metadata, CHIPS_DIR
import baseline_rf_xgb as base
base.USE_SAR_FLOODING = True
base.USE_REDGE = True
base.USE_S1_REGION_NORM = True
base.USE_INDICES2 = True                 # current best config
from field_level_models import field_rows, aligned_proba
from consolidate_eval import lut8, C, NAMES

N_SPLITS = 5


def main():
    load_metadata()
    lut = lut8()
    files = np.array(sorted(os.path.basename(f)
                            for f in glob.glob(os.path.join(CHIPS_DIR, "*.npz"))
                            if "GODAVARI" not in f))
    cm = np.zeros((C, C), np.int64)                      # pooled, rows=true cols=pred
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for tr_i, te_i in kf.split(files):
        tr_files, te_files = list(files[tr_i]), list(files[te_i])
        base.S1_REGION_STATS = base.fit_s1_region_stats(tr_files)
        Xtr, ytr, _ = field_rows(tr_files, lut)
        Xte, yte, _ = field_rows(te_files, lut)
        m = lgb.LGBMClassifier(n_estimators=500, num_leaves=31, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                n_jobs=-1, random_state=42, verbose=-1).fit(Xtr, ytr)
        pred = aligned_proba(m, Xte, C).argmax(1)
        for t, p in zip(yte, pred):
            cm[t, p] += 1

    supp = cm.sum(1)
    recall = np.divide(np.diag(cm), np.maximum(supp, 1))
    prec = np.divide(np.diag(cm), np.maximum(cm.sum(0), 1))
    f1 = np.divide(2 * prec * recall, np.maximum(prec + recall, 1e-9))
    oa = np.trace(cm) / cm.sum()

    print(f"Pooled {N_SPLITS}-fold CV — per-crop (all {int(cm.sum())} fields, "
          f"pooled OA {oa:.3f})\n")
    print(f"{'crop':16} {'recall':>7} {'prec':>6} {'F1':>6} {'support':>8}   top confusions")
    order = np.argsort(recall)                            # worst recall first
    for i in order:
        row = cm[i].copy(); row[i] = 0
        conf = ", ".join(f"{NAMES[j]}:{row[j]}" for j in np.argsort(row)[::-1][:3] if row[j] > 0)
        print(f"{NAMES[i]:16} {recall[i]:7.3f} {prec[i]:6.3f} {f1[i]:6.3f} {supp[i]:8d}   {conf}")
    print(f"\nmacro-recall {recall.mean():.3f}  macro-F1 {f1.mean():.3f}")


if __name__ == "__main__":
    main()
