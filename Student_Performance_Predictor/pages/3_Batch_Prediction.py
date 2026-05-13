# pages/3_Batch_Prediction.py

import pandas as pd
import streamlit as st
st.set_page_config(
    page_title="Batch Prediction | EduPredict",
    page_icon=":mortar_board:",
    layout="wide",
)

from sklearn import set_config

from src.loader import load_css, load_model_assets
from src.helpers import build_default_input
from src.predictor import coerce_input_types, predict_batch
from src.validators import validate_input_dataframe
from src.ui_components import render_page_header, render_empty_state


set_config(transform_output="pandas")

css = load_css()
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# =========================================================
# 1) LOAD MODEL
# =========================================================
try:
    full_pipeline, core_model, raw_feature_names, raw_survivors, best_params = load_model_assets()
except Exception as e:
    st.error(f"Failed to load model assets: {e}")
    st.stop()


# =========================================================
# 2) HELPERS
# =========================================================
def create_template_dataframe(required_columns: list[str]) -> pd.DataFrame:
    default_row = build_default_input(required_columns)
    return pd.DataFrame([default_row])


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def extract_quoted_column(message: str) -> str | None:
    parts = message.split("'")
    if len(parts) >= 2:
        return parts[1]
    return None


def parse_extra_columns(warnings: list[str]) -> list[str]:
    extra_columns = []
    for warning in warnings:
        if "Unexpected extra columns" not in warning:
            continue

        raw_columns = warning.split(":", 1)[-1]
        raw_columns = raw_columns.replace("[", "").replace("]", "")
        for item in raw_columns.split(","):
            column = item.strip().strip("'").strip('"')
            if column:
                extra_columns.append(column)

    return extra_columns


def group_validation_errors(errors: list[str]) -> dict[str, list[str]]:
    grouped_errors = {
        "missing_values": [],
        "out_of_range": [],
        "other": [],
    }

    for error in errors:
        column = extract_quoted_column(error)

        if "contains missing values" in error:
            grouped_errors["missing_values"].append(column or error)
        elif "must be between" in error:
            grouped_errors["out_of_range"].append(error)
        else:
            grouped_errors["other"].append(error)

    return grouped_errors


def render_validation_report(validation: dict):
    extra_columns = parse_extra_columns(validation["warnings"])
    grouped_errors = group_validation_errors(validation["errors"])

    if extra_columns:
        st.warning("Extra columns were found. They are not used by the prediction model.")

        for column in extra_columns:
            if column == "Exam_Score":
                st.write(
                    "- `Exam_Score` is the target column used for training. "
                    "Remove it before prediction; the app will ignore it if present."
                )
            else:
                st.write(f"- `{column}` will be ignored.")

    if not validation["is_valid"]:
        st.error("Validation failed. Please clean the file and upload it again.")

        if grouped_errors["missing_values"]:
            st.markdown("#### Missing values")
            st.write("These required input columns contain empty values:")
            for column in grouped_errors["missing_values"]:
                st.write(f"- `{column}`")

        if grouped_errors["out_of_range"]:
            st.markdown("#### Out-of-range values")
            for error in grouped_errors["out_of_range"]:
                st.write(f"- {error}")

        if grouped_errors["other"]:
            st.markdown("#### Other validation issues")
            for error in grouped_errors["other"]:
                st.write(f"- {error}")

        st.markdown("#### Suggested fix")
        st.write("- Download the template from Step 1.")
        st.write("- Fill all required input feature columns.")
        st.write("- Remove `Exam_Score` before prediction.")
        st.write("- Re-upload the cleaned CSV file.")
        return False

    st.success("Validation passed. The file is ready for prediction.")
    return True


def render_result_summary(result_df: pd.DataFrame):
    total_records = len(result_df)
    average_score = result_df["Predicted_Score"].mean()
    highest_score = result_df["Predicted_Score"].max()
    lowest_score = result_df["Predicted_Score"].min()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows Predicted", total_records)
    with c2:
        st.metric("Average Score", f"{average_score:.2f}")
    with c3:
        st.metric("Highest Score", f"{highest_score:.2f}")
    with c4:
        st.metric("Lowest Score", f"{lowest_score:.2f}")

    if "Predicted_Band" in result_df.columns:
        st.markdown("#### Band Counts")
        band_counts = (
            result_df["Predicted_Band"]
            .value_counts()
            .rename_axis("Band")
            .reset_index(name="Count")
        )
        st.dataframe(band_counts, use_container_width=True, hide_index=True)
        st.bar_chart(band_counts.set_index("Band"))


# =========================================================
# 3) HEADER
# =========================================================
render_page_header(
    "Batch Prediction",
    "Upload a CSV file, validate the schema, score many student records, and download the result."
)


# =========================================================
# 4) STEP 1 + STEP 2
# =========================================================
template_df = create_template_dataframe(raw_feature_names)

step_col1, step_col2 = st.columns([1, 1.3], gap="large")

with step_col1:
    st.markdown("### Step 1: Download template")
    st.write("Use the template to match the expected input columns.")
    st.download_button(
        label="Download CSV Template",
        data=dataframe_to_csv_bytes(template_df),
        file_name="student_prediction_template.csv",
        mime="text/csv",
    )

with step_col2:
    st.markdown("### Step 2: Upload CSV")
    st.info(
        "Batch prediction expects input features only. Do not upload the original training dataset directly. "
        "Remove the target column `Exam_Score` before prediction, or use the downloadable template for the safest format. "
        "`Hours_Studied` is treated as weekly study hours and has no upper range limit."
    )
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])


# =========================================================
# 5) STEP 3 + STEP 4
# =========================================================
if uploaded_file is None:
    render_empty_state("Upload a CSV file to validate the schema and run batch prediction.")
else:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded CSV file: {e}")
        st.stop()

    st.markdown("### Step 3: Validate file")
    st.caption("Previewing the first 5 rows only. Full results are available after prediction.")
    st.dataframe(uploaded_df.head(5), use_container_width=True, hide_index=True)

    batch_df = coerce_input_types(uploaded_df)
    validation = validate_input_dataframe(batch_df, raw_feature_names)
    is_valid = render_validation_report(validation)

    if is_valid:
        st.markdown("### Step 4: Predict and download result")

        if st.button("Run Batch Prediction", type="primary", use_container_width=True):
            with st.spinner("Scoring student records..."):
                result_df = predict_batch(full_pipeline, batch_df, raw_feature_names)

            st.success("Batch prediction completed.")
            render_result_summary(result_df)

            st.markdown("#### Result Preview")
            st.dataframe(result_df.head(5), use_container_width=True, hide_index=True)

            st.download_button(
                label="Download Prediction Results",
                data=dataframe_to_csv_bytes(result_df),
                file_name="student_prediction_results.csv",
                mime="text/csv",
            )
