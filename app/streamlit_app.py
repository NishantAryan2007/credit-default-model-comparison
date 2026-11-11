import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import joblib
import pandas as pd
import streamlit as st

THRESHOLD = 1 / 6  # same cutoff as the final test (5 per miss, 1 per false alarm)
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}
EDU = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}
PAY_HELP = "-2 = no usage, -1 = paid in full, 0 = paid minimum, 1+ = months late"


@st.cache_resource
def load_models():
    return {n: joblib.load(ROOT / "artifacts" / "models" / f) for n, f in MODEL_FILES.items()}


models = load_models()

st.title("Credit default risk: make a prediction")
st.write("Enter one customer's details and see the predicted chance of defaulting next month.")

model_name = st.selectbox("Model", list(MODEL_FILES))

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Profile and repayment")
    limit_bal = st.number_input("Credit limit (NT$)", 10_000, 1_000_000, 140_000, step=10_000)
    education = st.selectbox("Education", list(EDU), format_func=EDU.get, index=1)
    pay = {}
    for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        pay[col] = st.number_input(col, -2, 8, 0, help=PAY_HELP)
with c2:
    st.subheader("Bill amounts (NT$)")
    bills = {f"BILL_AMT{i}": st.number_input(f"BILL_AMT{i}", -200_000, 1_000_000, 20_000, step=1000)
             for i in range(1, 7)}
with c3:
    st.subheader("Payment amounts (NT$)")
    pays = {f"PAY_AMT{i}": st.number_input(f"PAY_AMT{i}", 0, 1_000_000, 2_000, step=500)
            for i in range(1, 7)}

row = {"LIMIT_BAL": limit_bal, "EDUCATION": education, **pay, **bills, **pays}
X_new = pd.DataFrame([row])

if st.button("Predict", type="primary"):
    p = models[model_name].predict_proba(X_new)[0, 1]
    st.metric("Predicted default risk", f"{p:.1%}")
    if p >= THRESHOLD:
        st.error(f"FLAG: risk is at or above the {THRESHOLD:.1%} cutoff.")
    else:
        st.success(f"Not flagged: risk is below the {THRESHOLD:.1%} cutoff.")