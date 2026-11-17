import json

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve

from common import MODEL_FILES, ROOT, THRESHOLD

NAMES = list(MODEL_FILES)
COLOR = alt.Scale(domain=NAMES, range=["#1F4E79", "#E67E22", "#27AE60", "#C0392B"])


@st.cache_data
def load_results():
    with open(ROOT / "artifacts" / "test_results.json") as f:
        test = json.load(f)["results"]
    with open(ROOT / "artifacts" / "tuning_results.json") as f:
        tune = json.load(f)
    preds = pd.read_csv(ROOT / "artifacts" / "test_predictions.csv")
    return test, tune, preds


def scores_tab(test, tune):
    rows = []
    for name in ["Dummy"] + NAMES:
        r = test[name]
        cv = tune.get(name, {}).get("check", {}).get("roc_auc")
        rows.append({
            "Model": name,
            "Test ROC-AUC": r["roc_auc"],
            "CV ROC-AUC": f"{cv[0]:.4f} ± {cv[1]:.4f}" if cv else "-",
            "Avg precision": r["average_precision"],
            "Log-loss": r["log_loss"],
            "Brier": r["brier"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1": r["f1"],
            "Accuracy": r["accuracy"],
            "Cost": r["cost"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index = True, width = "stretch")
    st.caption(
        f"Precision, recall, F1, accuracy and cost use the {THRESHOLD:.1%} cutoff. "
        "For log-loss, Brier and cost, lower is better. The test set has 6,000 customers, "
        "so AUC differences under about 0.01 are within noise."
    )
    eng = pd.DataFrame([{
        "Model": n,
        "Training time (s)": test[n]["fit_seconds"],
        "Prediction time, 6,000 rows (s)": test[n]["predict_seconds"],
        "File size (MB)": test[n].get("file_mb", 0),
    } for n in NAMES])
    st.dataframe(eng, hide_index = True, width = "stretch")


def roc_tab(preds):
    y = preds["default"].values
    curves, dots = [], []
    for name in NAMES:
        fpr, tpr, _ = roc_curve(y, preds[name])
        curves.append(pd.DataFrame({"FPR": fpr, "TPR": tpr, "Model": name}))
        flag = preds[name].values >= THRESHOLD
        dots.append({
            "FPR": (flag & (y == 0)).sum() / (y == 0).sum(),
            "TPR": (flag & (y == 1)).sum() / (y == 1).sum(),
            "Model": name,
        })
    lines = alt.Chart(pd.concat(curves)).mark_line().encode(
        x=alt.X("FPR:Q", title = "False alarm rate (innocent customers flagged)"),
        y=alt.Y("TPR:Q", title = "Recall (defaulters caught)"),
        color=alt.Color("Model:N", scale = COLOR, legend = alt.Legend(title = None, orient = "bottom")),
    )
    diag = alt.Chart(pd.DataFrame({"FPR": [0, 1], "TPR": [0, 1]})).mark_line(
        color = "gray", strokeDash = [4, 4]).encode(x = "FPR:Q", y = "TPR:Q")
    points = alt.Chart(pd.DataFrame(dots)).mark_point(filled = True, size = 90).encode(
        x = "FPR:Q", y = "TPR:Q", color = alt.Color("Model:N", scale = COLOR, legend = None))
    st.altair_chart((diag + lines + points).properties(height = 380), width = "stretch")
    st.caption("Dashed diagonal = no skill. Dots show where each model lands at the 16.7% cutoff.")


def calibration_tab(preds):
    y = preds["default"].values
    parts = []
    for name in NAMES:
        true_rate, pred_rate = calibration_curve(y, preds[name], n_bins=10, strategy="quantile")
        parts.append(pd.DataFrame({"Predicted": pred_rate, "Actual": true_rate, "Model": name}))
    lines = alt.Chart(pd.concat(parts)).mark_line(point=True).encode(
        x = alt.X("Predicted:Q", title = "Predicted risk", scale = alt.Scale(domain = [0, 0.8])),
        y = alt.Y("Actual:Q", title = "Actual default rate", scale = alt.Scale(domain = [0, 0.8])),
        color=alt.Color("Model:N", scale = COLOR, legend = alt.Legend(title = None, orient = "bottom")),
    )
    diag = alt.Chart(pd.DataFrame({"Predicted": [0, 0.8], "Actual": [0, 0.8]})).mark_line(
        color = "gray", strokeDash=[4, 4]).encode(x = "Predicted:Q", y = "Actual:Q")
    st.altair_chart((diag + lines).properties(height = 380), width = "stretch")
    st.caption("Test set, 10 equal-sized groups (about 600 customers each). "
               "Closer to the dashed line means the probabilities are more honest.")


def confusion_tab(preds):
    y = preds["default"].values
    name = st.selectbox("Model", NAMES, key = "perf_model")
    flag = preds[name].values >= THRESHOLD
    tn, fp, fn, tp = confusion_matrix(y, flag).ravel()
    a, b = st.columns([2, 3], gap = "large")
    with a:
        m = pd.DataFrame(
            [[tp, fn], [fp, tn]],
            index=["Really defaults", "Really does not"],
            columns=["Flagged", "Not flagged"],
        )
        st.dataframe(m, width="stretch")
    with b:
        c1, c2 = st.columns(2)
        c1.metric("Defaulters caught", f"{tp:,}", f"{tp / (tp + fn):.0%} of all defaulters", delta_color="off")
        c2.metric("Defaulters missed", f"{fn:,}", f"{fn / (tp + fn):.0%} of all defaulters", delta_color="off")
        c3, c4 = st.columns(2)
        c3.metric("False alarms", f"{fp:,}", f"{fp / (fp + tn):.0%} of innocent customers", delta_color="off")
        c4.metric("Correctly cleared", f"{tn:,}")
    st.caption(f"{len(y):,} test customers at the {THRESHOLD:.1%} cutoff.")


def render():
    test, tune, preds = load_results()
    st.header("Model performance")
    st.caption("All numbers are from the 6,000-customer test set, which was used once, after tuning.")
    t1, t2, t3, t4 = st.tabs(["Scores", "ROC curve", "Calibration", "Confusion matrix"])
    with t1:
        scores_tab(test, tune)
    with t2:
        roc_tab(preds)
    with t3:
        calibration_tab(preds)
    with t4:
        confusion_tab(preds)