import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)

from src.features import split_xy
from src.models import build_dummy
from src.tune import load_tuned_models

COST_FN, COST_FP = 5, 1                      # same assumption as the cutoff notebook
THRESHOLD = COST_FP / (COST_FP + COST_FN)    # 1/6, about 0.167
MODEL_DIR = Path("artifacts/models")


def evaluate(y, p, t):
    """All scores for one model: ranking/probability scores plus cutoff-based ones."""
    pred = p >= t
    missed = ((~pred) & (y == 1)).sum()
    false_alarm = (pred & (y == 0)).sum()
    return {
        "roc_auc": roc_auc_score(y, p),
        "average_precision": average_precision_score(y, p),
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "cost": (COST_FN * missed + COST_FP * false_alarm) / len(y),
    }


if __name__ == "__main__":
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)
    y = y_test.values

    models = {"Dummy": build_dummy(), **load_tuned_models()}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results, preds = {}, {"default": y}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_s = time.time() - t0

        t0 = time.time()
        p = model.predict_proba(X_test)[:, 1]
        pred_s = time.time() - t0

        results[name] = {k: round(float(v), 4) for k, v in evaluate(y, p, THRESHOLD).items()}
        results[name]["fit_seconds"] = round(fit_s, 2)
        results[name]["predict_seconds"] = round(pred_s, 3)
        preds[name] = p

        if name != "Dummy":
            path = MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib")
            joblib.dump(model, path, compress=3)
            results[name]["file_mb"] = round(path.stat().st_size / 1e6, 2)

        print(f"\n{name}")
        for k, v in results[name].items():
            print(f"  {k:18s} {v}")

    with open("artifacts/test_results.json", "w") as f:
        json.dump({"threshold": THRESHOLD, "results": results}, f, indent=2)
    pd.DataFrame(preds).to_csv("artifacts/test_predictions.csv", index=False)
    print("\nSaved test_results.json, test_predictions.csv and models/")