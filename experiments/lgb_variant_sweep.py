"""Small LGB variant sweep for the final classifier — select on VAL OA,
report TEST once. Variants: class weighting on/off, more trees + lower lr,
deeper leaves."""
import json
import os

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import cohen_kappa_score, f1_score

from finalize_classifier import load_table

VARIANTS = {
    "balanced_500_31_.05": dict(n_estimators=500, num_leaves=31, learning_rate=0.05, class_weight="balanced"),
    "none_500_31_.05": dict(n_estimators=500, num_leaves=31, learning_rate=0.05, class_weight=None),
    "balanced_1200_31_.03": dict(n_estimators=1200, num_leaves=31, learning_rate=0.03, class_weight="balanced"),
    "none_1200_31_.03": dict(n_estimators=1200, num_leaves=31, learning_rate=0.03, class_weight=None),
    "balanced_800_63_.03": dict(n_estimators=800, num_leaves=63, learning_rate=0.03, class_weight="balanced"),
}


def main():
    d, idx = load_table()
    X, y = d["X"], d["y8"].astype(int)
    Xtr, ytr = X[idx["train"]], y[idx["train"]]
    Xva, yva = X[idx["val"]], y[idx["val"]]
    Xte, yte = X[idx["test"]], y[idx["test"]]
    rows = []
    for name, kw in VARIANTS.items():
        m = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **kw)
        m.fit(Xtr, ytr)
        pv, pt = m.predict(Xva), m.predict(Xte)
        row = dict(
            name=name,
            val_OA=(yva == pv).mean(), val_mF1=f1_score(yva, pv, average="macro", zero_division=0),
            test_OA=(yte == pt).mean(), test_mF1=f1_score(yte, pt, average="macro", zero_division=0),
            test_kappa=cohen_kappa_score(yte, pt),
        )
        rows.append(row)
        print(f"{name:26s} val OA {row['val_OA']:.4f} mF1 {row['val_mF1']:.3f} | "
              f"test OA {row['test_OA']:.4f} mF1 {row['test_mF1']:.3f} k {row['test_kappa']:.3f}")
    best = max(rows, key=lambda r: r["val_OA"])
    print("\nbest by val OA:", best["name"])


if __name__ == "__main__":
    main()
