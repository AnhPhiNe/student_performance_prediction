# app.py

import streamlit as st
from sklearn import set_config

from src.loader import load_css, load_model_assets
from src.ui_components import render_sidebar_summary

set_config(transform_output="pandas")

st.set_page_config(
    page_title="EduPredict | Student Performance App",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

css = load_css()
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

try:
    full_pipeline, core_model, raw_feature_names, raw_survivors, best_params = load_model_assets()
except Exception as e:
    st.error(f"Failed to load model assets: {e}")
    st.stop()

render_sidebar_summary(raw_feature_names)

st.markdown(
    """
    <div class="ep-app-shell">
        <div class="ep-app-header">
            <div class="ep-app-title">🎓 EduPredict</div>
            <div class="ep-app-subtitle">
                Predict student exam performance with a clean end-to-end machine learning workflow.
            </div>
        </div>

        <div class="ep-app-banner">
            👉 Use the sidebar to open <strong>Home</strong>, <strong>Single Prediction</strong>, or <strong>Batch Prediction</strong>.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)