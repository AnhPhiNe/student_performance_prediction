# pages/1_Home.py

import streamlit as st
st.set_page_config(
    page_title="Home | EduPredict",
    page_icon=":mortar_board:",
    layout="wide",
)

from src.loader import load_css, load_model_assets, load_dataset
from src.ui_components import (
    render_page_header,
    render_section_title,
    render_kpi_cards,
    render_feature_cards,
    render_workflow_strip,
    render_empty_state,
    render_cta_card,
)


css = load_css()
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# =========================================================
# 1) LOAD ASSETS
# =========================================================
try:
    full_pipeline, core_model, raw_feature_names, raw_survivors, best_params = load_model_assets()
    df = load_dataset()
except Exception as e:
    st.error(f"Failed to load project assets: {e}")
    st.stop()


def get_model_label() -> str:
    if best_params and "model_name" in best_params:
        return str(best_params["model_name"])
    return "Ridge Regression"


# =========================================================
# 3) HERO
# =========================================================
render_page_header(
    title="EduPredict",
    subtitle="Predict student exam performance from structured academic, lifestyle, family, and school context data.",
    badge_text="Student Performance Machine Learning App",
)

st.info(
    "Choose a workflow below to score one student, process a CSV batch, or inspect model drivers."
)


# =========================================================
# 4) PROJECT STATUS
# =========================================================
dataset_rows = f"{len(df):,}" if df is not None else "Missing"
render_section_title(
    "Project Snapshot",
    "A quick view of the prediction product currently loaded in the app."
)

render_kpi_cards([
    ("Dataset Rows", dataset_rows),
    ("Input Features", str(len(raw_feature_names))),
    ("Target", "Exam Score"),
    ("Current Model", get_model_label()),
])


# =========================================================
# 5) MAIN TOOLS
# =========================================================
render_section_title(
    "What would you like to do?",
    "Choose one of the main tools below to predict scores, process a CSV file, or inspect model insights."
)

cta_col1, cta_col2, cta_col3 = st.columns(3, gap="large")

with cta_col1:
    render_cta_card(
        "Single Prediction",
        "Estimate one student's exam score from an interactive profile.",
        "pages/2_Single_Prediction.py",
        "Open Single Prediction",
    )

with cta_col2:
    render_cta_card(
        "Batch Prediction",
        "Upload a CSV file, validate records, score the full batch, and export results.",
        "pages/3_Batch_Prediction.py",
        "Open Batch Prediction",
    )

with cta_col3:
    render_cta_card(
        "Model Insights",
        "Review the strongest positive and negative drivers behind model predictions.",
        "pages/4_Explainability.py",
        "Open Model Insights",
    )


# =========================================================
# 6) HOW IT WORKS
# =========================================================
render_section_title(
    "How the App Works",
    "A high-level flow from uploaded or entered data to usable prediction results."
)

render_workflow_strip([
    "Input Data",
    "Validate Schema",
    "Predict Score",
    "Review Results",
])


# =========================================================
# 7) OPTIONAL DATASET SAMPLE
# =========================================================
with st.expander("View sample dataset", expanded=False):
    if df is None:
        render_empty_state("Dataset is not available, so a sample cannot be shown.")
    else:
        st.caption("Showing a small sample for context only.")
        st.dataframe(df.head(5), use_container_width=True, hide_index=True)
