# pages/5_About.py

import streamlit as st

st.set_page_config(
    page_title="About | EduPredict",
    page_icon=":mortar_board:",
    layout="wide",
)

from src.loader import load_css, load_model_assets
from src.ui_components import render_kpi_cards, render_section_title


css = load_css()
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# =========================================================
# 1) LOAD OPTIONAL MODEL INFO
# =========================================================
assets_loaded = False
assets_error = None
raw_feature_names = []
best_params = None
model_name = "Ridge Regression"

try:
    full_pipeline, core_model, raw_feature_names, raw_survivors, best_params = load_model_assets()
    assets_loaded = True
    if best_params and "model_name" in best_params:
        model_name = best_params["model_name"]
except Exception as e:
    assets_error = str(e)


def render_page_link(path: str, label: str):
    try:
        st.page_link(path, label=label)
    except Exception:
        st.write(f"- {label}: use the sidebar navigation.")


# =========================================================
# 2) HEADER
# =========================================================
render_section_title(
    "About EduPredict",
    "A compact AI portfolio case study for student performance prediction."
)

if assets_error:
    st.warning(
        "Model metadata could not be loaded on this page, but the portfolio summary is still available."
    )


# =========================================================
# 3) PROJECT SUMMARY
# =========================================================
render_kpi_cards([
    ("Project Type", "ML App"),
    ("Task", "Regression"),
    ("Model", model_name),
    ("Input Features", str(len(raw_feature_names)) if assets_loaded else "N/A"),
])

st.markdown("### Project Summary")
st.write(
    "EduPredict is an end-to-end machine learning application that estimates student exam scores "
    "from academic, lifestyle, family, and school-related inputs. The goal is to show practical ML "
    "engineering: not only training a model, but packaging it into a usable Streamlit product."
)


# =========================================================
# 4) WHAT I BUILT
# =========================================================
st.markdown("### What I Built")
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("**Single Prediction**")
    st.write("Interactive form for scoring one student profile and returning a clear result.")

with col2:
    st.markdown("**Batch Prediction**")
    st.write("CSV upload workflow with schema validation, preview, summary, and result download.")

with col3:
    st.markdown("**Model Insights**")
    st.write("Coefficient-based interpretation to explain positive, negative, and overall drivers.")


# =========================================================
# 5) TECHNICAL ARCHITECTURE
# =========================================================
st.markdown("### Technical Architecture")
st.write(
    "The app uses a saved sklearn pipeline for inference, so feature engineering, preprocessing, "
    "and prediction stay consistent between training and deployment."
)

st.code(
    "User input -> Validation -> Feature engineering -> Preprocessing -> Ridge prediction -> Result",
    language="text",
)

with st.expander("View module structure", expanded=False):
    st.markdown(
        """
        - `src/config.py`: shared configuration and UI labels
        - `src/loader.py`: loads CSS, model assets, and data
        - `src/validators.py`: validates input schema and values
        - `src/predictor.py`: prediction flow and recommendations
        - `src/explainer.py`: coefficient-based model insight helpers
        - `src/ui_components.py`: reusable Streamlit UI blocks
        - `pages/`: user-facing Streamlit pages
        """
    )

with st.expander("View model artifacts", expanded=False):
    st.markdown(
        """
        - `hcmue_student_full_pipeline_v1_0.joblib`: full inference pipeline
        - `ridge_core_model.joblib`: core Ridge model used for coefficient interpretation
        - `raw_feature_names.joblib`: input schema contract for forms and CSV uploads
        - `raw_survivors.joblib`: retained raw features from the feature selection workflow
        """
    )

with st.expander("View hyperparameters", expanded=False):
    if best_params:
        st.json(best_params)
    else:
        st.info("Hyperparameter metadata is not available on this page.")


# =========================================================
# 6) ENGINEERING HIGHLIGHTS
# =========================================================
st.markdown("### Engineering Highlights")
highlight_col1, highlight_col2 = st.columns(2, gap="large")

with highlight_col1:
    st.markdown(
        """
        - End-to-end sklearn pipeline for consistent inference
        - Multi-page Streamlit interface with single and batch workflows
        - Input validation before prediction
        """
    )

with highlight_col2:
    st.markdown(
        """
        - CSV template and result download flow
        - Coefficient-based model insight page
        - Modular project structure suitable for maintenance
        """
    )


# =========================================================
# 7) LIMITATIONS & NEXT STEPS
# =========================================================
st.markdown("### Limitations & Next Steps")
st.markdown(
    """
    This project is designed for portfolio demonstration, not real educational decision-making.
    The model is useful for showing an ML product workflow, but predictions should be interpreted
    carefully and with awareness of dataset limitations.
    """
)

limit_col1, limit_col2 = st.columns(2, gap="large")

with limit_col1:
    st.markdown("**Current limitations**")
    st.markdown(
        """
        - High-score samples are relatively limited.
        - Coefficient insights are global, not personalized explanations.
        - Recommendations are partly rule-based.
        """
    )

with limit_col2:
    st.markdown("**Next steps**")
    st.markdown(
        """
        - Add model metadata/version tracking.
        - Improve local explanation for individual predictions.
        - Add more targeted tests for validation and inference.
        """
    )

with st.expander("View future improvements", expanded=False):
    st.markdown(
        """
        - Deployment polish for Streamlit Cloud
        - Prediction report export
        - Model performance summary page
        - More robust monitoring/logging for production-style inference
        - Expanded test coverage for edge cases
        """
    )


# =========================================================
# 8) CTA
# =========================================================
st.markdown("### Explore the App")
cta_col1, cta_col2, cta_col3 = st.columns(3, gap="large")

with cta_col1:
    render_page_link("pages/2_Single_Prediction.py", "Try Single Prediction")

with cta_col2:
    render_page_link("pages/3_Batch_Prediction.py", "Run Batch Prediction")

with cta_col3:
    render_page_link("pages/4_Explainability.py", "View Model Insights")
