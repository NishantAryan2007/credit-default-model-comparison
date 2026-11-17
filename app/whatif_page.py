import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import (EDU, FEATURES, LIMITS, MODEL_FILES, THRESHOLD, current_row,
                    init_state, load_models, load_random_customer)

NAMES = list(MODEL_FILES)
COLOR = alt.Scale(domain=NAMES, range=["#1F4E79", "#E67E22", "#27AE60", "#C0392B"])

# Sweep ranges: a sensible window, not the full input limits (those are too wide to read).
SWEEP = {
    "PAY_0": list(range(-2, 9)),
    "PAY_2": list(range(-2, 9)),
    "PAY_3": list(range(-2, 9)),
    "PAY_4": list(range(-2, 9)),
    "PAY_5": list(range(-2, 9)),
    "PAY_6": list(range(-2, 9)),
    "EDUCATION": [1, 2, 3, 4],
    "LIMIT_BAL": list(range(10_000, 510_000, 10_000)),
}
for i in range(1, 7):
    SWEEP[f"BILL_AMT{i}"] = list(range(0, 405_000, 5_000))
    SWEEP[f"PAY_AMT{i}"] = list(range(0, 101_000, 1_000))

DISCRETE = {"PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "EDUCATION"}


def render():
    init_state()
    models = load_models()

    st.header("What-if sweep")
    st.caption("Change one input across its range, keep everything else fixed, "
               "and see how each model's predicted risk responds.")

    top = st.columns([2, 1, 3])
    feature = top[0].selectbox("Feature to vary", list(SWEEP), key="sweep_feature")
    top[1].markdown("&nbsp;")
    top[1].button("Random test customer", on_click=load_random_customer, width="stretch")

    row = current_row()
    values = SWEEP[feature]
    grid = pd.DataFrame([row] * len(values))
    grid[feature] = values

    parts = []
    for name, model in models.items():
        p = model.predict_proba(grid)[:, 1]
        parts.append(pd.DataFrame({"Value": values, "Risk": p, "Model": name}))
    df = pd.concat(parts)

    current = row[feature]
    left, right = st.columns([3, 2], gap="large")

    with left:
        base = alt.Chart(df)
        if feature in DISCRETE:
            lines = base.mark_line(point=True).encode(
                x=alt.X("Value:O", title=feature),
                y=alt.Y("Risk:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"),
                        title="Predicted risk"),
                color=alt.Color("Model:N", scale=COLOR, legend=alt.Legend(title=None, orient="bottom")),
            )
            cutoff = alt.Chart(pd.DataFrame({"y": [THRESHOLD]})).mark_rule(
                color="black", strokeDash=[5, 4]).encode(y="y:Q")
            chart = lines + cutoff
        else:
            lines = base.mark_line().encode(
                x=alt.X("Value:Q", title=feature),
                y=alt.Y("Risk:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"),
                        title="Predicted risk"),
                color=alt.Color("Model:N", scale=COLOR, legend=alt.Legend(title=None, orient="bottom")),
            )
            cutoff = alt.Chart(pd.DataFrame({"y": [THRESHOLD]})).mark_rule(
                color="black", strokeDash=[5, 4]).encode(y="y:Q")
            here = alt.Chart(pd.DataFrame({"x": [current]})).mark_rule(
                color="gray", strokeDash=[2, 3]).encode(x="x:Q")
            chart = lines + cutoff + here
        st.altair_chart(chart.properties(height=400), width="stretch")
        st.caption(f"Dashed horizontal line = cutoff ({THRESHOLD:.1%}). "
                   f"Current customer's {feature} = {current:,}"
                   + (" (dotted vertical line)." if feature not in DISCRETE else "."))

    with right:
        with st.container(border=True):
            st.markdown("**Where each model flips**")
            st.caption("First value of the swept feature at which the model flags this customer "
                       "(risk at or above the cutoff).")
            rows = []
            for name in NAMES:
                d = df[df.Model == name]
                above = d[d.Risk >= THRESHOLD]
                if len(above) == 0:
                    flip = "never flags"
                elif len(above) == len(d):
                    flip = "always flags"
                else:
                    flip = f"{int(above.Value.iloc[0]):,}"
                rows.append({"Model": name, "Flags from": flip,
                             "Risk range": f"{d.Risk.min():.0%} to {d.Risk.max():.0%}"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("'Flags from' is the first value, scanning upward, where the risk is at or above "
                       "the cutoff. If a model flags at low values instead, read the chart.")
            st.info("This is what each model does with this one customer. "
                    "It is not a statement about the real world, and unusual combinations of inputs "
                    "may be places where the models have little real data.")