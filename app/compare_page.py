import altair as alt
import pandas as pd
import streamlit as st

from common import (BILL_COLS, EDU, LABELS, LIMITS, MODEL_FILES, MONTH, PAY_COLS,
                    PAYAMT_COLS, THRESHOLD, current_row, init_state,
                    load_disagreement_customer, load_models, load_random_customer,
                    range_warnings)


def render():
    init_state()
    models = load_models()

    st.header("Compare models on one customer")
    st.caption("The same inputs go to all four models. Change them here or load a real customer.")

    row = current_row()
    X_new = pd.DataFrame([row])
    probs = {n: float(m.predict_proba(X_new)[0, 1]) for n, m in models.items()}
    flags = {n: p >= THRESHOLD for n, p in probs.items()}

    top = st.columns([1, 1, 3])
    top[0].button("Random test customer", on_click=load_random_customer, width="stretch")
    top[1].button("Find a disagreement", on_click=load_disagreement_customer, width="stretch")

    left, right = st.columns([3, 2], gap="large")

    with left:
        chart_df = pd.DataFrame({
            "Model": list(probs),
            "Risk": list(probs.values()),
            "Decision": ["Flagged" if f else "Not flagged" for f in flags.values()],
        })
        bars = alt.Chart(chart_df).mark_bar().encode(
            y=alt.Y("Model:N", sort=list(MODEL_FILES), title=None),
            x=alt.X("Risk:Q", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%"), title="Predicted risk"),
            color=alt.Color(
                "Decision:N",
                scale=alt.Scale(domain=["Flagged", "Not flagged"],
                                range=["#C0392B", "#1F4E79"]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
        )
        line = alt.Chart(pd.DataFrame({"x": [THRESHOLD]})).mark_rule(
            color="black", strokeDash=[5, 4]
        ).encode(x="x:Q")
        st.altair_chart(bars + line, width="stretch")
        st.caption(f"Dashed line = cutoff ({THRESHOLD:.1%}). Red bars are flagged.")

        with st.expander("Edit customer inputs"):
            a, b = st.columns(2)
            a.number_input(LABELS["LIMIT_BAL"], *LIMITS["LIMIT_BAL"], step=10_000, key="LIMIT_BAL")
            b.selectbox(LABELS["EDUCATION"], list(EDU), format_func=EDU.get, key="EDUCATION")
            st.caption("Repayment status: -2 = no usage, -1 = paid in full, 0 = paid minimum, "
                       "1 or more = months late")
            cols = st.columns(6)
            for c, name, month in zip(cols, PAY_COLS, MONTH):
                c.number_input(f"Status, {month.lower()}", *LIMITS[name], key=name)
            cols = st.columns(6)
            for c, name, month in zip(cols, BILL_COLS, MONTH):
                c.number_input(f"Bill, {month.lower()}", *LIMITS[name], step=1000, key=name)
            cols = st.columns(6)
            for c, name, month in zip(cols, PAYAMT_COLS, MONTH):
                c.number_input(f"Payment, {month.lower()}", *LIMITS[name], step=500, key=name)

    with right:
        with st.container(border=True):
            n_flag = sum(flags.values())
            if n_flag in (0, len(flags)):
                st.success("All models agree: " + ("FLAGGED" if n_flag else "NOT FLAGGED"))
            else:
                st.warning(f"Models disagree: {n_flag} of {len(flags)} flag this customer")

            table = pd.DataFrame({
                "Model": list(probs),
                "Risk": [f"{p:.1%}" for p in probs.values()],
                "Decision": ["Flagged" if f else "Not flagged" for f in flags.values()],
            })
            st.dataframe(table, hide_index=True, width="stretch")
            st.metric("Gap between highest and lowest",
                      f"{(max(probs.values()) - min(probs.values())) * 100:.1f} points")

            notes = range_warnings(row)
            if notes:
                with st.expander(f"Unusual inputs ({len(notes)}): disagreement may be larger here"):
                    for n in notes:
                        st.write("- " + n)

            loaded = st.session_state.get("loaded")
            if loaded and loaded["row"] == row:
                outcome = "defaulted" if loaded["actual"] else "did not default"
                st.info(f"Real test customer #{loaded['index']}: they actually **{outcome}**.")