import numpy as np
import pandas as pd
import joblib

from src.features import split_xy

MODELS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}

if __name__ == "__main__":
    X, _ = split_xy(pd.read_csv("data/processed/test.csv"))
    X0, X2 = X.copy(), X.copy()
    X0["PAY_0"] = 0
    X2["PAY_0"] = 2

    jumps = {}
    for name, f in MODELS.items():
        m = joblib.load(f"artifacts/models/{f}.joblib")
        jumps[name] = m.predict_proba(X2)[:, 1] - m.predict_proba(X0)[:, 1]
    J = pd.DataFrame(jumps)

    print("Rise in predicted risk when PAY_0 goes from 0 to 2 (all 6,000 test customers)")
    print(J.describe().loc[["mean", "50%"]].round(3).to_string())
    smallest = J.idxmin(axis=1).value_counts(normalize=True).round(3)
    print("\nShare of customers where this model rises LEAST:")
    print(smallest.to_string())