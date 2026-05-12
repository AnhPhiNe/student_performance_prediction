# pages/1_Home.py

import streamlit as st
import plotly.express as px

from src.loader import load_model_assets, load_dataset
from src.ui_components import (
    render_hero_section,
    render_section_title,
    render_kpi_cards,
    render_feature_cards,
    render_workflow_strip,
    render_next_step_cards,
    render_empty_state
)

# =========================================================
# 1) LOAD ASSETS
# =========================================================
try:
    full_pipeline, core_model, raw_feature_names, raw_survivors, best_params = load_model_assets()
    df = load_dataset()
except Exception as e:
    st.error(f"Failed to load project assets: {e}")
    st.stop()

# =========================================================
# 2) HERO
# =========================================================
render_hero_section(
    title="EduPredict: Student Performance Prediction",
    subtitle="A clean end-to-end machine learning application for predicting student exam scores from academic, lifestyle, family, and school-related factors.",
    badge_text="Final Model • Ridge Regression"
)

# =========================================================
# 3) KPI SUMMARY
# =========================================================
dataset_status = "Available" if df is not None else "Missing"
render_kpi_cards([
    ("Final Model", "Ridge Regression"),
    ("Raw Features", str(len(raw_feature_names))),
    ("Prediction Type", "Regression"),
    ("Dataset", dataset_status),
])

# =========================================================
# 4) WHAT THE APP CAN DO
# =========================================================
render_section_title(
    "What this application can do",
    "Explore the main workflows supported by the app."
)

render_feature_cards([
    ("🎯", "Single Prediction", "Predict one student profile interactively and get recommendations."),
    ("📂", "Batch Prediction", "Upload a CSV file and generate predictions for many students at once."),
    ("📈", "Model Insights", "Inspect Ridge coefficient-based interpretation in a simple format."),
])

# =========================================================
# 5) WORKFLOW
# =========================================================
render_section_title(
    "How the system works",
    "The prediction flow is designed to be robust, modular, and deployment-friendly."
)

render_workflow_strip([
    "Input",
    "Validation",
    "Feature Engineering",
    "Preprocessing",
    "Ridge Prediction"
])

# =========================================================
# 6) DATA PREVIEW + SCORE DISTRIBUTION
# =========================================================
render_section_title(
    "Dataset overview",
    "A quick view of the source data and target score distribution."
)

if df is None:
    render_empty_state("Dataset is not available, so preview charts cannot be shown.")
else:
    left_col, right_col = st.columns([1.05, 1.25], gap="large")

    with left_col:
        st.markdown("### Dataset Preview")
        preview_df = df.head(5).copy()
        st.dataframe(preview_df, width="stretch", hide_index=True)

    with right_col:
        st.markdown("### Exam Score Distribution")

        if "Exam_Score" in df.columns:
            fig = px.histogram(
                df,
                x="Exam_Score",
                nbins=30,
                title=None
            )
            fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Exam Score",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, width="stretch")
        else:
            render_empty_state("Column 'Exam_Score' was not found in the dataset.")

# =========================================================
# 7) NEXT ACTIONS
# =========================================================
render_section_title(
    "Where should you go next?",
    "Choose the workflow that matches what you want to do first."
)

render_next_step_cards()