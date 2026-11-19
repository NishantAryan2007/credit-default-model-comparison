import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import joblib
import pandas as pd
import streamlit as st

from src.features import split_xy

THRESHOLD = 1 / 6  # same cutoff as the final test (5 per miss, 1 per false alarm)
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}
EDU = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
FEATURES = ["LIMIT_BAL", "EDUCATION"] + PAY_COLS + BILL_COLS + PAYAMT_COLS

# Readable names. The data's own naming is odd (PAY_0 then PAY_2), so we map by position.
MONTH = ["Latest", "1 mo ago", "2 mo ago", "3 mo ago", "4 mo ago", "5 mo ago"]
LABELS = {"LIMIT_BAL": "Credit limit (NT$)", "EDUCATION": "Education"}
LABELS.update({c: f"Repayment status, {m.lower()}" for c, m in zip(PAY_COLS, MONTH)})
LABELS.update({c: f"Bill, {m.lower()} (NT$)" for c, m in zip(BILL_COLS, MONTH)})
LABELS.update({c: f"Payment, {m.lower()} (NT$)" for c, m in zip(PAYAMT_COLS, MONTH)})

# Allowed range for each input. Loaded customers are clamped into these.
LIMITS = {"LIMIT_BAL": (10_000, 1_000_000), "EDUCATION": (1, 4)}
LIMITS.update({c: (-2, 8) for c in PAY_COLS})
LIMITS.update({c: (-400_000, 1_000_000) for c in BILL_COLS})
LIMITS.update({c: (0, 2_000_000) for c in PAYAMT_COLS})

DEFAULTS = {"LIMIT_BAL": 140_000, "EDUCATION": 2}
DEFAULTS.update({c: 0 for c in PAY_COLS})
DEFAULTS.update({c: 20_000 for c in BILL_COLS})
DEFAULTS.update({c: 2_000 for c in PAYAMT_COLS})


@st.cache_resource
def load_models():
    return {n: joblib.load(ROOT / "artifacts" / "models" / f) for n, f in MODEL_FILES.items()}


@st.cache_data
def load_test():
    """Test customers, used only for the demo button. Never for training or tuning."""
    return split_xy(pd.read_csv(ROOT / "data" / "processed" / "test.csv"))


@st.cache_data
def load_ranges():
    """What is 'typical', measured on the training set only."""
    X, _ = split_xy(pd.read_csv(ROOT / "data" / "processed" / "train.csv"))
    money = ["LIMIT_BAL"] + BILL_COLS + PAYAMT_COLS
    ranges = {c: (float(X[c].quantile(0.01)), float(X[c].quantile(0.99))) for c in money}
    common_vals = {}
    for c in PAY_COLS:
        counts = X[c].value_counts()
        common_vals[c] = {int(v) for v in counts[counts >= 100].index}
    return ranges, common_vals


def range_warnings(row):
    """List of plain-English notes for inputs the models rarely saw in training."""
    ranges, common_vals = load_ranges()
    notes = []
    for c, (lo, hi) in ranges.items():
        if not lo <= row[c] <= hi:
            notes.append(f"{LABELS[c]} is {row[c]:,}, outside the typical range "
                         f"({lo:,.0f} to {hi:,.0f}).")
    for c in PAY_COLS:
        if row[c] not in common_vals[c]:
            notes.append(f"{LABELS[c]} is {row[c]}, a value that is rare in the training data.")
    return notes


def init_state():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)


def load_random_customer():
    """Button callback: copy one real test customer into the input boxes."""
    X, y = load_test()
    i = random.randrange(len(X))
    row = {k: int(X.iloc[i][k]) for k in FEATURES}
    for k, (lo, hi) in LIMITS.items():
        row[k] = min(max(row[k], lo), hi)
    st.session_state.update(row)
    st.session_state["loaded"] = {"row": row, "actual": int(y.iloc[i]), "index": i}


def current_row():
    return {k: int(st.session_state[k]) for k in FEATURES}


@st.cache_data
def load_disagreements():
    """Test customers where at least one model flags and at least one does not.
    Sorted by probability spread (biggest gap first)."""
    d = pd.read_csv(ROOT / "artifacts" / "test_predictions.csv")
    m = d[list(MODEL_FILES)]
    f = m >= THRESHOLD
    mask = f.any(axis=1) & ~f.all(axis=1)
    out = m[mask].copy()
    out["spread"] = out.max(axis=1) - out.min(axis=1)
    return out.sort_values("spread", ascending=False)


def load_customer(i):
    """Copy test customer number i into the input boxes."""
    X, y = load_test()
    row = {k: int(X.iloc[i][k]) for k in FEATURES}
    for k, (lo, hi) in LIMITS.items():
        row[k] = min(max(row[k], lo), hi)
    st.session_state.update(row)
    st.session_state["loaded"] = {"row": row, "actual": int(y.iloc[i]), "index": i}


def load_disagreement_customer(top_n=300):
    """Button callback: pick a random customer from the top_n biggest disagreements."""
    dis = load_disagreements().head(top_n)
    load_customer(int(random.choice(list(dis.index))))