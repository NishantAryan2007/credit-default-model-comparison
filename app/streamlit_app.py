import streamlit as st

st.set_page_config(page_title="Credit default model comparison", layout="wide")

from compare_page import render as compare
from predict_page import render as predict

page = st.navigation([
    st.Page(predict, title="Make a prediction", url_path="predict", default=True),
    st.Page(compare, title="Compare models", url_path="compare"),
])
page.run()