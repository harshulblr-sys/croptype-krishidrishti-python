"""
5-fold chip-level cross-validation of the finalized crop classifier
=================================================================
Cross-verifies that the single-split OA (~0.77) is stable, not an artefact of one
lucky test set. Folds are cut at the CHIP level (each chip's fields go wholly to
train OR test) so a field never leaks across the boundary — the same leakage-free
rule the fixed split uses. Per fold we:
  1. fit per-region S1 normalisation on the fold's TRAIN chips (orbit alignment),
  2. aggregate field features (red-edge + flooding + SPRI column),
  3. train the finalized default LightGBM,
  4. score OA / macro-F1 / kappa / Rice-F1.
Reports mean +/- std across the 5 folds.

NOTE: this CVs the MAIN classifier (LightGBM). The TempCNN rice-rescue is a fixed
post-hoc specialist (retraining it per fold is out of scope), so these are the
PRE-rescue numbers — the honest lower bound the rescue lifts on top of.

Run:  set AGRI_DATA_DIR=...  &&  python cv_verify.py
"""
import os
import json
import glob
import numpy as np
import torch
from sklearn.model_selection import KFold
import lightgbm as lgb

from data_loader import load_metadata, CHIPS_DIR, DATA_DIR
import baseline_rf_xgb as base
base.USE_SAR_FLOODING = True
base.USE_REDGE = True
base.USE_S1_REGION_NORM = True            # same orbit alignment as finalize
from field_level_models import field_rows, aligned_proba
from consolidate_eval import lut8, C, NAMES
from train_hybrid import metrics_from_cm

RICE8 = NAMES.index("Rice")
USE_SPRI = False        # match the finalized model (SPRI dropped)
N_SPLITS = 5


def field_metrics(y, pred):
    cm = torch.zeros(C, C, dtype=torch.long)
    idx = torch.from_numpy(y.astype(np.int64) * C + pred.astype(np.int64))
    cm += torch.bincount(idx, minlength=C * C).reshape(C, C)
    return metrics_from_cm(cm)


# Configs to benchmark: (label, USE_INDICES2, USE_SMOOTH). Region-norm + red-edge +
# flooding stay ON for all (they're in the current best); this isolates the Tier-1 add.
CONFIGS = [
    ("base (current)", False, False),
    ("+indices2",      True,  False),
    ("+smooth",        False, True),
    ("+indices2+smooth", True, True),
]


def run_cv(files, lut, indices2, smooth):
    base.USE_INDICES2, base.USE_SMOOTH = indices2, smooth
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    rows = []
    for tr_i, te_i in kf.split(files):
        tr_files, te_files = list(files[tr_i]), list(files[te_i])
        base.S1_REGION_STATS = base.fit_s1_region_stats(tr_files)   # fit per fold
        Xtr, ytr, _ = field_rows(tr_files, lut)
        Xte, yte, _ = field_rows(te_files, lut)
        m = lgb.LGBMClassifier(n_estimators=500, num_leaves=31, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                n_jobs=-1, random_state=42, verbose=-1).fit(Xtr, ytr)
        pred = aligned_proba(m, Xte, C).argmax(1)
        mt = field_metrics(yte, pred)
        rows.append((mt["oa"], mt["macro_f1"], mt["kappa"], mt["f1"][RICE8]))
    return np.array(rows)


def main():
    load_metadata()
    lut = lut8()
    files = np.array(sorted(os.path.basename(f)
                            for f in glob.glob(os.path.join(CHIPS_DIR, "*.npz"))
                            if "GODAVARI" not in f))
    print(f"{N_SPLITS}-fold chip-level CV over {len(files)} chips (region-norm ON, "
          f"Tier-1 feature pass benchmark)\n")
    results = {}
    for label, i2, sm in CONFIGS:
        # skip configs that need opt2 if not present
        a = run_cv(files, lut, i2, sm)
        results[label] = a
        print(f"{label:20} OA {a[:,0].mean():.3f}±{a[:,0].std():.3f}  "
              f"mF1 {a[:,1].mean():.3f}±{a[:,1].std():.3f}  "
              f"kappa {a[:,2].mean():.3f}  RiceF1 {a[:,3].mean():.3f}")

    base_oa = results["base (current)"][:, 0].mean()
    print("\n" + "=" * 68)
    print(f"{'config':20} {'OA':>7} {'dOA':>7} {'macroF1':>8} {'kappa':>7} {'RiceF1':>7}")
    for label, _, _ in CONFIGS:
        a = results[label]
        print(f"{label:20} {a[:,0].mean():7.3f} {a[:,0].mean()-base_oa:+7.3f} "
              f"{a[:,1].mean():8.3f} {a[:,2].mean():7.3f} {a[:,3].mean():7.3f}")
    print("\n(ΔOA vs base; anything under ~+0.005 is within fold noise)")


if __name__ == "__main__":
    main()
