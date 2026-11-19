import json
import altair as alt

import pandas as pd
import streamlit as st

from common import MODEL_FILES, ROOT, THRESHOLD

NAMES = list(MODEL_FILES)

TEXT = {
    "Logistic Regression": {
        "how": "Gives every input a weight, adds them up, and squashes the total into a 0 to 100% risk. "
               "Here the PAY columns and EDUCATION are one-hot encoded, and the money columns are "
               "log-squeezed and scaled.",
        "strength": "Simple, fast, tiny file, and every weight can be read. Smooth risk curves.",
        "weakness": "Can only add up effects. It cannot learn rules like 'if A and B together, then risky' "
                    "unless we build them in by hand.",
    },
    "Decision Tree": {
        "how": "Asks a chain of yes/no questions about the inputs. Each customer lands in an end box (leaf), "
               "and the leaf's default rate becomes the predicted risk.",
        "strength": "Easy to draw and explain. No scaling needed. Can capture rules and thresholds.",
        "weakness": "Unstable: small data changes can change the questions. Few distinct leaves give coarse, "
                    "jumpy probabilities. Deep trees memorize.",
    },
    "Random Forest": {
        "how": "Builds many trees, each on a random sample of customers and a random subset of columns at "
               "each question, then averages their risks.",
        "strength": "Averaging cancels much of one tree's noise. Robust to settings. Needs no scaling.",
        "weakness": "Heavier and harder to explain than one tree. Still memorizes training data if leaves are tiny.",
    },
    "XGBoost": {
        "how": "Builds small trees one after another. Each new tree tries to fix the mistakes the earlier "
               "trees still make, and the final risk is the sum of all corrections.",
        "strength": "Often the strongest on tabular data. Fast to predict. Small file here.",
        "weakness": "Many settings to tune. Overfits if you add too many trees or make them too deep.",
    },
}


@st.cache_data
def load():
    a = ROOT / "artifacts"
    out = []
    for f in ["test_results.json", "tuning_results.json", "disagreement_results.json",
              "bootstrap_results.json"]:
        with open(a / f) as fh:
            out.append(json.load(fh))
    test, tune, dis, boot = out
    return test["results"], tune, dis, boot


def model_tab(name, test, tune, dis):
    t = TEXT[name]
    left, right = st.columns([3, 2], gap = "large")
    with left:
        st.markdown("**How it works**")
        st.write(t["how"])
        st.markdown("**Strength**")
        st.write(t["strength"])
        st.markdown("**Weakness**")
        st.write(t["weakness"])
    with right:
        with st.container(border = True):
            st.markdown("**What we measured**")
            r = test[name]
            c1, c2 = st.columns(2)
            c1.metric("Test ROC-AUC", f"{r['roc_auc']:.3f}")
            c2.metric("Cost per customer", f"{r['cost']:.3f}")
            c3, c4 = st.columns(2)
            c3.metric("Training time", f"{r['fit_seconds']:.2f} s")
            c4.metric("File size", f"{r.get('file_mb', 0):.2f} MB")
            d = dis["models"][name]
            st.caption(
                f"On the {dis['n_split']:,} customers where models split, this model was the odd one out "
                f"{d['alone']} times and matched the real outcome in {d['alone_and_right']} of them."
            )
            st.markdown("**Tuned settings**")
            st.json(tune[name]["best_params"], expanded = False)


def overview_tab(test, dis, boot):
    n_test = 6000
    aucs = [test[n]["roc_auc"] for n in NAMES]
    costs = [test[n]["cost"] for n in NAMES]

    st.subheader("The short version")
    c = st.columns(4)
    c[0].metric("Models compared", len(NAMES))
    c[1].metric("Test ROC-AUC range", f"{min(aucs):.3f} to {max(aucs):.3f}")
    c[2].metric("Cost per customer range", f"{min(costs):.3f} to {max(costs):.3f}")
    c[3].metric("Customers where models split", f"{dis['n_split'] / n_test:.0%}")
    st.markdown(
        "- **All four rank customers about equally well.** The scores sit within about 0.02 of each other.\n"
        "- **All four are reasonably well calibrated.** Predicted risk tracked the real default rate closely, "
        "with small deviations (the decision tree is jumpier).\n"
        f"- **Their decisions still differ.** At the {THRESHOLD:.1%} cutoff they split on about 1 in 4 "
        "test customers, mostly borderline ones.\n"
        "- **A simple model gets close, but a small gap remains.** Logistic regression trails the two "
        "ensembles by about 0.015 AUC, which is small but measurable."
    )

    st.subheader("Is a difference real, or luck?")
    st.caption("Each bar is the plausible range of the AUC difference between two models on this test set. "
               "A positive value means the first model is better. Bars that cross the dashed line at 0 "
               "mean the two models can't be told apart.")
    diff = pd.DataFrame([
        {"Comparison": k.replace(" - ", " vs "), "Mean": m, "Low": lo, "High": hi,
         "Result": "Too close to call" if lo <= 0 <= hi else "Clear difference"}
        for k, (m, lo, hi) in boot["diff_ci"].items()
    ]).sort_values("Mean", ascending=False)
    order = list(diff["Comparison"])
    color = alt.Color(
        "Result:N",
        scale=alt.Scale(domain=["Clear difference", "Too close to call"], range=["#1F4E79", "#9AA5B1"]),
        legend=alt.Legend(title = None, orient = "bottom"),
    )
    base = alt.Chart(diff).encode(
        y=alt.Y("Comparison:N", sort=order, title=None, axis=alt.Axis(labelLimit=320)))
    ranges = base.mark_rule(strokeWidth=4).encode(
        x=alt.X("Low:Q", title="Difference in ROC-AUC (first model minus second)"), x2="High:Q", color=color)
    dots = base.mark_point(filled=True, size=90).encode(x="Mean:Q", color=color)
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color="black", strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart((ranges + dots + zero).properties(height=240), width="stretch")
    st.caption("Method: paired bootstrap, 1,000 resamples of the 6,000 test customers. "
               "It covers test-set noise only, not noise from tuning or training data.")

    with st.expander("Why accuracy is not the headline number"):
        st.write("A model that says 'no default' to everyone is right 78% of the time. Moving the cutoff "
                 "from 0.5 to the cost-based 0.167 made accuracy fall and cost improve.")
    with st.expander("Limits to keep in mind"):
        st.markdown(
            "- The 5-to-1 cost of a miss versus a false alarm is my assumption, not a real bank's number.\n"
            "- The data is from Taiwan in 2005.\n"
            "- SEX, AGE and MARRIAGE are not given to the models, but that does not make a model fair, "
            "because other columns can stand in for them. No fairness audit was done.\n"
            "- The test set has 6,000 customers, so small differences are noise."
        )


def render():
    test, tune, dis, boot = load()
    st.header("How the models work")
    st.caption("Short explanations, with numbers taken from this project's own results.")
    tabs = st.tabs(["Overview"] + NAMES)
    with tabs[0]:
        overview_tab(test, dis, boot)
    for tab, name in zip(tabs[1:], NAMES):
        with tab:
            model_tab(name, test, tune, dis)