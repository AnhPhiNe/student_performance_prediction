# pages/3_Batch_Prediction.py

import pandas as pd
import streamlit as st

from src.loader import load_model_assets
from src.helpers import build_default_input
from src.predictor import coerce_input_types, predict_batch
from src.validators import validate_input_dataframe
from src.ui_components import render_section_title, render_empty_state

from sklearn import set_config
set_config(transform_output="pandas")

# =========================================================
# 1) LOAD MODEL
# =========================================================
try:
    full_pipeline, core_model, raw_feature_names, raw_survivors, best_params = load_model_assets()
except Exception as e:
    st.error(f"Failed to load model assets: {e}")
    st.stop()

# =========================================================
# 2) TEMPLATE CSV
# =========================================================
def create_template_dataframe(required_columns: list[str]) -> pd.DataFrame:
    default_row = build_default_input(required_columns)
    return pd.DataFrame([default_row])


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

# =========================================================
# 3) HEADER
# =========================================================
render_section_title(
    "Batch Prediction",
    "Upload a CSV file, validate the input schema, predict many student records at once, and download the result."
)

# =========================================================
# 4) TEMPLATE DOWNLOAD
# =========================================================
template_df = create_template_dataframe(raw_feature_names)

st.download_button(
    label="📥 Download CSV Template",
    data=dataframe_to_csv_bytes(template_df),
    file_name="student_prediction_template.csv",
    mime="text/csv",
    width="content"
)

st.caption("Use this template if you are not sure how the input CSV should be structured.")

# =========================================================
# 5) FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# =========================================================
# 6) PROCESS
# =========================================================
if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded CSV file: {e}")
        st.stop()

    st.markdown("### Uploaded File Preview")
    st.dataframe(batch_df.head(10), width="stretch", hide_index=True)

    batch_df = coerce_input_types(batch_df)
    validation = validate_input_dataframe(batch_df, raw_feature_names)

    if validation["warnings"]:
        for warn in validation["warnings"]:
            st.warning(warn)

    if not validation["is_valid"]:
        st.error("The uploaded file is invalid. Please fix the following issues:")
        for err in validation["errors"]:
            st.write(f"- {err}")
        st.stop()

    st.success("The uploaded file passed validation successfully.")

    if st.button("🚀 Run Batch Prediction", type="primary", width="stretch"):
        with st.spinner("Scoring student records..."):
            result_df = predict_batch(full_pipeline, batch_df, raw_feature_names)

        st.markdown("### Prediction Results")
        st.dataframe(result_df.head(20), width="stretch", hide_index=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Records", len(result_df))
        with c2:
            st.metric("Average Predicted Score", f"{result_df['Predicted_Score'].mean():.2f}")
        with c3:
            high_count = (result_df["Predicted_Score"] >= 80).sum()
            st.metric("Students >= 80", int(high_count))

        st.download_button(
            label="📤 Download Prediction Results",
            data=dataframe_to_csv_bytes(result_df),
            file_name="student_prediction_results.csv",
            mime="text/csv",
            width="content"
        )
else:
    render_empty_state("Upload a CSV file to validate the schema and run batch prediction.")