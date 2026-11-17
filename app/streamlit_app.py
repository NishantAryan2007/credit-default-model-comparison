import streamlit as st

st.set_page_config(page_title="Credit default model comparison", layout="wide")

from compare_page import render as compare
from performance_page import render as performance
from predict_page import render as predict
from whatif_page import render as whatif

page = st.navigation([
    st.Page(predict, title="Make a prediction", url_path="predict", default=True),
    st.Page(compare, title="Compare models", url_path="compare"),
    st.Page(performance, title="Model performance", url_path="performance"),
    st.Page(whatif, title="What-if sweep", url_path="whatif"),
])
page.run()