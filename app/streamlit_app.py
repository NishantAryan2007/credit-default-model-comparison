import streamlit as st

st.set_page_config(page_title="Credit default model comparison", layout="wide")

from predict_page import render as predict

page = st.navigation([st.Page(predict, title="Make a prediction", default=True)])
page.run()