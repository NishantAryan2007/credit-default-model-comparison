import pandas as pd
import streamlit as st

from common import (BILL_COLS, EDU, LIMITS, MODEL_FILES, PAY_COLS, PAYAMT_COLS,
                    THRESHOLD, current_row, init_state, load_models,
                    load_random_customer)


def render():
    init_state()
    models = load_models()

    st.header("Predict default risk")
    st.caption(
        "Four models, one question: will a credit-card customer miss next month's payment? "
        "Portfolio demo on 2005 Taiwan data, not lending advice."
    )
    with st.expander("About this app and what the inputs mean"):
        st.markdown(
            "This app compares logistic regression, a decision tree, a random forest and XGBoost on "
            "30,000 credit-card customers from Taiwan in 2005. Enter a customer, or load a real test "
            "customer, and the result updates as you type. Use the sidebar to compare models and see "
            "how they were evaluated.\n\n"
            "- **Credit limit**: the card limit in New Taiwan dollars.\n"
            "- **Repayment status**: one value per month, latest first. -2 means no card use, -1 paid "
            "in full, 0 paid the minimum, 1 or more means that many months late.\n"
            "- **Bill amounts**: the statement balance each month.\n"
            "- **Payment amounts**: how much the customer paid each month.\n"
            "- Sex, age and marital status are not used by the models."
        )

    left, right = st.columns([3, 2], gap="large")

    with left:
        tab1, tab2, tab3 = st.tabs(["Profile and repayment", "Bill amounts", "Payment amounts"])
        with tab1:
            a, b = st.columns(2)
            a.number_input("Credit limit (NT$)", *LIMITS["LIMIT_BAL"], step=10_000, key="LIMIT_BAL")
            b.selectbox("Education", list(EDU), format_func=EDU.get, key="EDUCATION")
            st.markdown("**Repayment status by month** (latest first)")
            st.caption("-2 = no usage, -1 = paid in full, 0 = paid minimum, 1 or more = months late")
            cols = st.columns(6)
            for c, name in zip(cols, PAY_COLS):
                c.number_input(name, *LIMITS[name], key=name)
        with tab2:
            cols = st.columns(3)
            for i, name in enumerate(BILL_COLS):
                cols[i % 3].number_input(name, *LIMITS[name], step=1000, key=name)
        with tab3:
            cols = st.columns(3)
            for i, name in enumerate(PAYAMT_COLS):
                cols[i % 3].number_input(name, *LIMITS[name], step=500, key=name)

    with right:
        with st.container(border=True):
            model_name = st.selectbox("Model", list(MODEL_FILES), key="model_choice")
            st.button("Load a random test customer", on_click=load_random_customer, width="stretch")

            row = current_row()
            p = models[model_name].predict_proba(pd.DataFrame([row]))[0, 1]

            st.metric("Predicted default risk", f"{p:.1%}")
            st.progress(min(float(p), 1.0))
            if p >= THRESHOLD:
                st.error(f"FLAGGED: risk is at or above the {THRESHOLD:.1%} cutoff")
            else:
                st.success(f"NOT FLAGGED: risk is below the {THRESHOLD:.1%} cutoff")

            loaded = st.session_state.get("loaded")
            if loaded and loaded["row"] == row:
                outcome = "defaulted" if loaded["actual"] else "did not default"
                st.info(f"Real test customer #{loaded['index']}: they actually **{outcome}**.")
            st.caption("Cutoff assumes a missed defaulter costs 5 times a false alarm.")