import json

import pandas as pd

MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
THRESHOLD = 1 / 6
COST_FN, COST_FP = 5, 1

if __name__ == "__main__":
    d = pd.read_csv("artifacts/test_predictions.csv")
    y = d["default"]
    flags = d[MODELS] >= THRESHOLD
    split = flags.any(axis = 1) & ~flags.all(axis = 1)
    fs, ys = flags[split], y[split]

    print(f"Split cases: {len(fs)} of {len(d)}")
    print(f"Real default rate among split cases: {ys.mean():.3f} (all customers: {y.mean():.3f})\n")

    out = {"n_split": int(len(fs)), "default_rate_split": round(float(ys.mean()), 4), "models": {}}
    print(f"{'Model':20s} {'accuracy':>9s} {'cost':>7s}   {'alone':>5s} {'alone & right':>14s}")
    for m in MODELS:
        pred = fs[m]
        acc = float((pred == (ys == 1)).mean())
        missed = int(((~pred) & (ys == 1)).sum())
        false_alarm = int((pred & (ys == 0)).sum())
        cost = (COST_FN * missed + COST_FP * false_alarm) / len(ys)

        others = fs.drop(columns=m)
        alone = (others.nunique(axis=1) == 1) & (fs[m] != others.iloc[:, 0])
        n_alone = int(alone.sum())
        right = int((pred[alone] == (ys[alone] == 1)).sum())
        out["models"][m] = {"accuracy": round(acc, 4), "cost": round(float(cost), 4),
                            "alone": n_alone, "alone_and_right": right}
        print(f"{m:20s} {acc:9.3f} {cost:7.3f}   {n_alone:5d} {right:14d}")

    with open("artifacts/disagreement_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved artifacts/disagreement_results.json")