# pages/1_Home.py

import streamlit as st
st.set_page_config(
    page_title="Home | EduPredict",
    page_icon=":mortar_board:",
    layout="wide",
)

from src.loader import load_css, load_model_assets, load_dataset
from src.ui_components import (
    render_hero_section,
    render_section_title,
    render_kpi_cards,
    render_feature_cards,
    render_workflow_strip,
    render_empty_state,
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


# =========================================================
# 2) SMALL HELPERS
# =========================================================
def page_link_or_sidebar_note(page_path: str, label: str):
    if hasattr(st, "page_link"):
        st.page_link(page_path, label=label)
    else:
        st.caption(f"Open {label} from the sidebar.")


def get_model_label() -> str:
    if best_params and "model_name" in best_params:
        return str(best_params["model_name"])
    return "Ridge Regression"


# =========================================================
# 3) HERO
# =========================================================
render_hero_section(
    title="EduPredict",
    subtitle=(
        "A Streamlit portfolio app that predicts student exam performance from academic, "
        "lifestyle, family, and school context factors."
    ),
    badge_text="ML Portfolio Project",
)

st.info(
    "Use the sidebar to open Single Prediction for one student, Batch Prediction for CSV files, "
    "or Explainability to inspect model drivers."
)


# =========================================================
# 4) PROJECT STATUS
# =========================================================
dataset_rows = f"{len(df):,}" if df is not None else "Missing"
render_kpi_cards([
    ("Dataset Rows", dataset_rows),
    ("Input Features", str(len(raw_feature_names))),
    ("Target", "Exam Score"),
    ("Current Model", get_model_label()),
])


# =========================================================
# 5) WHAT THE APP DOES
# =========================================================
render_section_title(
    "What the app does",
    "Three practical workflows for model inference and interpretation."
)

render_feature_cards([
    ("1", "Single Prediction", "Estimate one student's exam score from an interactive profile."),
    ("2", "Batch Prediction", "Upload a CSV file, validate it, and score many students at once."),
    ("3", "Model Insights", "Review the strongest positive and negative model drivers."),
])


# =========================================================
# 6) HOW IT WORKS
# =========================================================
render_section_title(
    "How it works",
    "A short end-to-end flow from user input to interpretable result."
)

render_workflow_strip([
    "Enter Profile",
    "Validate Input",
    "Predict Score",
    "Interpret Result",
])


# =========================================================
# 7) NEXT ACTIONS
# =========================================================
render_section_title(
    "Next actions",
    "Choose the workflow that matches what you want to try first."
)

cta_col1, cta_col2, cta_col3 = st.columns(3, gap="large")

with cta_col1:
    st.markdown("### Start single prediction")
    st.write("Build one student profile and see the predicted score immediately.")
    page_link_or_sidebar_note("pages/2_Single_Prediction.py", "Single Prediction")

with cta_col2:
    st.markdown("### Try batch prediction")
    st.write("Download the template, upload a CSV, validate records, and export results.")
    page_link_or_sidebar_note("pages/3_Batch_Prediction.py", "Batch Prediction")

with cta_col3:
    st.markdown("### View model insights")
    st.write("Understand which features push predictions upward or downward.")
    page_link_or_sidebar_note("pages/4_Explainability.py", "Explainability")


# =========================================================
# 8) OPTIONAL DATASET SAMPLE
# =========================================================
with st.expander("View sample dataset", expanded=False):
    if df is None:
        render_empty_state("Dataset is not available, so a sample cannot be shown.")
    else:
        st.caption("A small sample is shown for context only. Prediction workflows are available from the sidebar.")
        st.dataframe(df.head(5), use_container_width=True, hide_index=True)
