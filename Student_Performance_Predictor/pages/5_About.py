# pages/5_About.py

import streamlit as st

st.set_page_config(
    page_title="About | EduPredict",
    page_icon=":mortar_board:",
    layout="wide",
)

from src.loader import load_css, load_model_assets
from src.ui_components import (
    render_cta_card,
    render_feature_cards,
    render_page_header,
    render_text_cards,
    render_workflow_strip,
)


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


# =========================================================
# 2) HEADER
# =========================================================
render_page_header(
    "About EduPredict",
    "A compact case study showing how a trained ML model becomes a usable prediction product.",
)

if assets_error:
    st.warning(
        "Model metadata could not be loaded on this page, but the portfolio summary is still available."
    )


# =========================================================
# 3) PROJECT SUMMARY
# =========================================================
feature_count = str(len(raw_feature_names)) if assets_loaded and raw_feature_names else "19"

st.markdown("### Project Summary")
summary_col, snapshot_col = st.columns([1.55, 1], gap="large")

with summary_col:
    st.markdown(
        """
        <div class="ep-about-summary">
            <div>
                <span>Problem</span>
                <p>Estimate student exam performance from academic, lifestyle, family, and school context factors.</p>
            </div>
            <div>
                <span>What I built</span>
                <p>A Streamlit ML product with single prediction, batch CSV scoring, validation, and model insight views.</p>
            </div>
            <div>
                <span>Outcome</span>
                <p>A polished app that shows the full path from trained model artifact to usable inference workflow.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with snapshot_col:
    st.markdown(
        f"""
        <div class="ep-about-snapshot">
            <div class="ep-about-snapshot-title">Project Snapshot</div>
            <div class="ep-about-snapshot-grid">
                <div><span>Product</span><strong>ML App</strong></div>
                <div><span>Model</span><strong>{model_name}</strong></div>
                <div><span>Workflow</span><strong>Single + Batch</strong></div>
                <div><span>Inputs</span><strong>{feature_count} features</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 4) WHAT I BUILT
# =========================================================
st.markdown("### What I Built")
render_feature_cards([
    ("1", "Single Prediction", "Interactive profile scoring with a clear result and concise recommendation."),
    ("2", "Batch Prediction", "CSV validation, preview, prediction summary, and downloadable results."),
    ("3", "Model Insights", "Directional drivers and overall influence views from Ridge coefficients."),
])
st.markdown("<div class='ep-section-gap'></div>", unsafe_allow_html=True)


# =========================================================
# 5) TECHNICAL ARCHITECTURE
# =========================================================
st.markdown("### Technical Architecture")
st.caption("Saved sklearn pipeline for consistent inference.")

render_workflow_strip([
    "Input",
    "Validate",
    "Pipeline",
    "Predict",
    "Result",
])

with st.expander("View technical details", expanded=False):
    st.markdown("#### Module structure")
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

    st.markdown("#### Model artifacts")
    st.markdown(
        """
        - `hcmue_student_full_pipeline_v1_0.joblib`: full inference pipeline
        - `ridge_core_model.joblib`: core Ridge model used for coefficient interpretation
        - `raw_feature_names.joblib`: input schema contract for forms and CSV uploads
        - `raw_survivors.joblib`: retained raw features from the feature selection workflow
        """
    )

    st.markdown("#### Training metadata")
    if best_params:
        st.json(best_params)
    else:
        st.info("Hyperparameter metadata is not available on this page.")


# =========================================================
# 6) ENGINEERING HIGHLIGHTS
# =========================================================
st.markdown("### Engineering Highlights")
render_text_cards([
    ("ML Pipeline", "Artifact-based inference with consistent preprocessing."),
    ("Product UX", "Single and batch workflows with validation."),
    ("Model Insight", "Readable coefficient driver views."),
])


# =========================================================
# 7) LIMITATIONS & NEXT STEPS
# =========================================================
st.markdown("### Limitations & Next Steps")
st.warning("Demo only. Not for real educational decisions.")

limit_col1, limit_col2 = st.columns(2, gap="large")

with limit_col1:
    st.markdown(
        """
        <div class="ep-list-card">
            <div class="ep-list-card-title">Current limitations</div>
            <div class="ep-list-row">Limited high-score samples</div>
            <div class="ep-list-row">Global, non-causal insights</div>
            <div class="ep-list-row">Partly rule-based recommendations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with limit_col2:
    st.markdown(
        """
        <div class="ep-list-card">
            <div class="ep-list-card-title">Next steps</div>
            <div class="ep-list-row">Add model versioning</div>
            <div class="ep-list-row">Improve local explanations</div>
            <div class="ep-list-row">Expand inference tests</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div class='ep-tight-spacer'></div>", unsafe_allow_html=True)

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
    render_cta_card(
        "Predict one student",
        "Use the interactive form to estimate a single student's exam score.",
        "pages/2_Single_Prediction.py",
        "Predict One Student",
    )

with cta_col2:
    render_cta_card(
        "Score a CSV",
        "Validate a batch file, run predictions, and download the results.",
        "pages/3_Batch_Prediction.py",
        "Score CSV File",
    )

with cta_col3:
    render_cta_card(
        "Inspect model drivers",
        "See which model drivers push predictions upward or downward.",
        "pages/4_Explainability.py",
        "Inspect Drivers",
    )
