"""
Rice-rescue threshold sweep — the precision/recall trade-off curve
=================================================================
Shows, for the Fallow-targeted TempCNN rice rescue, how test OA / macro-F1 / rice
recall / rice precision move as the P(rice) threshold varies. Lets you pick the
operating point deliberately instead of guessing 0.30. Val columns shown too so the
choice isn't pure test-peeking.

Run:  set AGRI_DATA_DIR=...  &&  python rice_rescue_sweep.py
"""
import numpy as np
from data_loader import make_splits, load_metadata
import baseline_rf_xgb as base
base.USE_SAR_FLOODING = True
base.USE_REDGE = True
base.USE_S1_REGION_NORM = True
base.USE_INDICES2 = True
from field_level_models import field_rows, aligned_proba
from consolidate_eval import lut8, C, NAMES
import lightgbm as lgb
import finalize_classifier as FC

RICE8, FALLOW8 = NAMES.index("Rice"), NAMES.index("No crop/Fallow")


def m_of(y, p):
    return FC.field_metrics(y, p)[0]


def main():
    meta13 = load_metadata()
    splits = make_splits(meta13)
    lut = lut8()
    base.S1_REGION_STATS = base.fit_s1_region_stats(splits["train"])
    Xtr, ytr, _ = field_rows(splits["train"], lut)
    Xva, yva, kva = field_rows(splits["val"], lut)
    Xte, yte, kte = field_rows(splits["test"], lut)
    lgbm = lgb.LGBMClassifier(n_estimators=500, num_leaves=31, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
              n_jobs=-1, random_state=42, verbose=-1).fit(Xtr, ytr)
    Pva, Pte = aligned_proba(lgbm, Xva, C), aligned_proba(lgbm, Xte, C)
    tc = FC.tempcnn_rice_probs(splits, meta13)
    rva = np.array([tc["val"].get(tuple(k), 0.0) for k in kva])
    rte = np.array([tc["test"].get(tuple(k), 0.0) for k in kte])

    def rescue(pred, r, t):
        out = pred.copy()
        out[(r > t) & (pred == FALLOW8)] = RICE8
        return out

    base_te = m_of(yte, Pte.argmax(1))
    print(f"no rescue:  test OA {base_te['oa']:.3f}  mF1 {base_te['macro_f1']:.3f}  "
          f"RiceRec {base_te['recall'][RICE8]:.3f}  RicePrec {base_te['precision'][RICE8]:.3f}\n")
    print(f"{'thresh':>7} {'flips':>6} {'caught':>7} {'FP':>4} | "
          f"{'test_OA':>8} {'test_mF1':>9} {'RiceRec':>8} {'RicePrec':>9} | {'val_RiceRec':>11}")
    for t in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        tf = (Pte.argmax(1) == FALLOW8) & (rte > t)
        caught = int(((yte == RICE8) & tf).sum()); fp = int(((yte != RICE8) & tf).sum())
        mt = m_of(yte, rescue(Pte.argmax(1), rte, t))
        mv = m_of(yva, rescue(Pva.argmax(1), rva, t))
        print(f"{t:7.2f} {int(tf.sum()):6d} {caught:7d} {fp:4d} | "
              f"{mt['oa']:8.3f} {mt['macro_f1']:9.3f} {mt['recall'][RICE8]:8.3f} "
              f"{mt['precision'][RICE8]:9.3f} | {mv['recall'][RICE8]:11.3f}")
    print("\n(caught = true rice recovered, FP = Fallow wrongly flipped to Rice)")


if __name__ == "__main__":
    main()
