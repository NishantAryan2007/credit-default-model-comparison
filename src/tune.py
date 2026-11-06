import json
import time

import pandas as pd
from sklearn.model_selection import (
    RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_validate,
)

from src.features import split_xy
from src.models import SEARCH_SPACES

N_ITER = 20
METRICS = ["roc_auc", "average_precision", "neg_log_loss", "neg_brier_score"]


def plain(v):
    """Turn numpy numbers into normal Python numbers so JSON can save them."""
    return v.item() if hasattr(v, "item") else v


if __name__ == "__main__":
    train = pd.read_csv("data/processed/train.csv")
    X, y = split_xy(train)

    tune_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    check_cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 2, random_state = 7)

    results = {}
    for name, (builder, space) in SEARCH_SPACES.items():
        start = time.time()
        search = RandomizedSearchCV(
            builder(), space, n_iter = N_ITER, scoring = "roc_auc",
            cv=tune_cv, random_state=42,
        )
        search.fit(X, y)

        check = cross_validate(search.best_estimator_, X, y, cv = check_cv, scoring = METRICS)
        results[name] = {
            "best_params": {k: plain(v) for k, v in search.best_params_.items()},
            "search_auc": round(search.best_score_, 4),
            "check": {m: [round(check[f"test_{m}"].mean(), 4),
                          round(check[f"test_{m}"].std(), 4)] for m in METRICS},
        }
        print(f"\n{name}  ({time.time() - start:.0f}s)")
        print("  best settings:", results[name]["best_params"])
        print("  search AUC:", results[name]["search_auc"])
        for m in METRICS:
            mean, std = results[name]["check"][m]
            print(f"  check {m:18s} {mean:.4f} +/- {std:.4f}")

    with open("artifacts/tuning_results.json", "w") as f:
        json.dump(results, f, indent = 2)
    print("\nSaved artifacts/tuning_results.json")