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