# app.py

import streamlit as st
from sklearn import set_config

from src.loader import load_css
from src.ui_components import render_root_welcome, render_sidebar_summary


set_config(transform_output="pandas")

st.set_page_config(
    page_title="EduPredict | Student Performance Predictor",
    page_icon=":mortar_board:",
    layout="wide",
    initial_sidebar_state="expanded",
)

css = load_css()
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

render_sidebar_summary()
render_root_welcome()
