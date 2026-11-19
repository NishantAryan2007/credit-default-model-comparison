import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
N_BOOT = 1000
PAIRS = [
    ("XGBoost", "Random Forest"),
    ("XGBoost", "Logistic Regression"),
    ("Random Forest", "Logistic Regression"),
    ("XGBoost", "Decision Tree"),
    ("Logistic Regression", "Decision Tree"),
]

if __name__ == "__main__":
    d = pd.read_csv("artifacts/test_predictions.csv")
    y = d["default"].values
    P = {m: d[m].values for m in MODELS}
    rng = np.random.default_rng(42)
    n = len(y)

    aucs = {m: [] for m in MODELS}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)          # same resampled customers for every model
        if y[idx].min() == y[idx].max():
            continue
        for m in MODELS:
            aucs[m].append(roc_auc_score(y[idx], P[m][idx]))

    out = {"n_boot": N_BOOT, "auc_ci": {}, "diff_ci": {}}
    print("AUC with 95% interval")
    for m in MODELS:
        lo, hi = np.percentile(aucs[m], [2.5, 97.5])
        out["auc_ci"][m] = [round(float(roc_auc_score(y, P[m])), 4), round(float(lo), 4), round(float(hi), 4)]
        print(f"  {m:20s} {out['auc_ci'][m][0]:.4f}  [{lo:.4f}, {hi:.4f}]")

    print("\nDifference in AUC (first minus second)")
    for a, b in PAIRS:
        diff = np.array(aucs[a]) - np.array(aucs[b])
        lo, hi = np.percentile(diff, [2.5, 97.5])
        mean = float(diff.mean())
        verdict = "tie (interval includes 0)" if lo <= 0 <= hi else "gap looks real"
        out["diff_ci"][f"{a} - {b}"] = [round(mean, 4), round(float(lo), 4), round(float(hi), 4)]
        print(f"  {a} - {b}: {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")

    with open("artifacts/bootstrap_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved artifacts/bootstrap_results.json")