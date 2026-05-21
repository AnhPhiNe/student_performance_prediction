# pages/3_Batch_Prediction.py

import pandas as pd
import plotly.express as px
import streamlit as st
st.set_page_config(
    page_title="Batch Prediction | EduPredict",
    page_icon=":mortar_board:",
    layout="wide",
)

from sklearn import set_config

from src.config import CATEGORICAL_DEFAULTS, CATEGORICAL_OPTIONS, NUMERIC_RANGES
from src.loader import load_css, load_model_assets
from src.helpers import build_default_input
from src.inference_client import (
    get_inference_mode_label,
    predict_student_dataframe_via_client,
)
from src.prediction_service import PredictionInputError
from src.predictor import coerce_input_types
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


def get_missing_mask(series: pd.Series) -> pd.Series:
    text_values = series.astype("string").str.strip()
    return series.isna() | text_values.isna() | text_values.eq("")


def render_extra_column_warning(uploaded_columns: list[str], required_columns: list[str]):
    extra_columns = [column for column in uploaded_columns if column not in required_columns]

    if not extra_columns:
        return

    st.warning("Extra columns were found. They are not used by the prediction model.")

    for column in extra_columns:
        if column == "Exam_Score":
            st.write(
                "- `Exam_Score` is the target column used for training. "
                "Remove it before prediction; the app will ignore it if present."
            )
        else:
            st.write(f"- `{column}` will be ignored.")


def validate_required_columns(uploaded_df: pd.DataFrame, required_columns: list[str]) -> bool:
    missing_columns = [column for column in required_columns if column not in uploaded_df.columns]

    if not missing_columns:
        return True

    st.error("Validation failed. The uploaded file is missing required input columns.")
    st.write("Add these columns to the CSV file before running batch prediction:")
    for column in missing_columns:
        st.write(f"- `{column}`")

    st.markdown("#### Suggested fix")
    st.write("- Download the template from Step 1.")
    st.write("- Copy your batch data into the matching columns.")
    st.write("- Re-upload the completed CSV file.")
    return False


def render_numeric_missing_warning(uploaded_df: pd.DataFrame, required_columns: list[str]):
    missing_counts = []

    for column in NUMERIC_RANGES:
        if column not in required_columns or column not in uploaded_df.columns:
            continue

        missing_count = int(get_missing_mask(uploaded_df[column]).sum())
        if missing_count:
            missing_counts.append((column, missing_count))

    if not missing_counts:
        return

    st.warning(
        "Some numeric values are missing. They will be handled by the model pipeline median imputer."
    )
    for column, missing_count in missing_counts:
        st.write(f"- `{column}`: {missing_count} missing value(s)")


def validate_numeric_values(uploaded_df: pd.DataFrame, required_columns: list[str]) -> bool:
    errors = []

    for column, (min_value, max_value) in NUMERIC_RANGES.items():
        if column not in required_columns or column not in uploaded_df.columns:
            continue

        missing_mask = get_missing_mask(uploaded_df[column])
        numeric_values = pd.to_numeric(uploaded_df[column], errors="coerce")
        invalid_numeric_mask = numeric_values.isna() & ~missing_mask

        if invalid_numeric_mask.any():
            errors.append(f"`{column}` contains non-numeric value(s).")
            continue

        valid_numeric_values = numeric_values[~missing_mask]
        out_of_range_mask = (valid_numeric_values < min_value) | (valid_numeric_values > max_value)
        if out_of_range_mask.any():
            errors.append(f"`{column}` must be between {min_value} and {max_value}.")

    if not errors:
        return True

    st.error("Validation failed. Please fix invalid numeric values before prediction.")
    for error in errors:
        st.write(f"- {error}")
    return False


def fill_allowed_categorical_missing_values(batch_df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    batch_df = batch_df.copy()
    filled_columns = []

    for column, default_value in CATEGORICAL_DEFAULTS.items():
        if column not in required_columns or column not in batch_df.columns:
            continue

        missing_mask = get_missing_mask(batch_df[column])
        missing_count = int(missing_mask.sum())
        if not missing_count:
            continue

        batch_df.loc[missing_mask, column] = default_value
        filled_columns.append((column, missing_count, default_value))

    if filled_columns:
        st.warning("Some categorical values were missing and were filled with neutral defaults.")
        for column, missing_count, default_value in filled_columns:
            st.write(f"- `{column}`: {missing_count} missing value(s) filled with `{default_value}`")

    return batch_df


def validate_unfilled_categorical_missing_values(batch_df: pd.DataFrame, required_columns: list[str]) -> bool:
    blocking_missing = []

    for column in CATEGORICAL_OPTIONS:
        if column not in required_columns or column not in batch_df.columns:
            continue
        if column in CATEGORICAL_DEFAULTS:
            continue

        missing_count = int(get_missing_mask(batch_df[column]).sum())
        if missing_count:
            blocking_missing.append((column, missing_count))

    if not blocking_missing:
        return True

    st.error("Validation failed. Some categorical columns have missing values that should not be auto-filled.")
    st.write("Please complete these fields in the CSV file and upload it again:")
    for column, missing_count in blocking_missing:
        st.write(f"- `{column}`: {missing_count} missing value(s)")
    return False


def validate_categorical_values(batch_df: pd.DataFrame, required_columns: list[str]) -> bool:
    errors = []

    for column, valid_options in CATEGORICAL_OPTIONS.items():
        if column not in required_columns or column not in batch_df.columns:
            continue

        actual_values = set(batch_df[column].dropna().astype(str).str.strip().unique())
        actual_values.discard("")
        invalid_values = sorted(actual_values - set(valid_options))

        if invalid_values:
            errors.append(f"`{column}` has invalid value(s): {invalid_values}. Allowed values: {valid_options}")

    if not errors:
        return True

    st.error("Validation failed. Please fix invalid categorical values before prediction.")
    for error in errors:
        st.write(f"- {error}")
    return False


def prepare_batch_dataframe(uploaded_df: pd.DataFrame, required_columns: list[str]) -> tuple[pd.DataFrame, bool]:
    render_extra_column_warning(uploaded_df.columns.tolist(), required_columns)

    if not validate_required_columns(uploaded_df, required_columns):
        return uploaded_df, False

    render_numeric_missing_warning(uploaded_df, required_columns)
    if not validate_numeric_values(uploaded_df, required_columns):
        return uploaded_df, False

    batch_df = coerce_input_types(uploaded_df)
    batch_df = fill_allowed_categorical_missing_values(batch_df, required_columns)

    if not validate_unfilled_categorical_missing_values(batch_df, required_columns):
        return batch_df, False

    if not validate_categorical_values(batch_df, required_columns):
        return batch_df, False

    st.success("Validation passed. The file is ready for prediction.")
    return batch_df, True


def render_result_preview(result_df: pd.DataFrame):
    st.markdown("#### Result Preview")

    preview_df = result_df.head(100)
    if len(result_df) > 100:
        st.caption("Showing first 100 rows only. Download CSV to view full results.")
    else:
        st.caption(f"Showing all {len(result_df)} prediction result row(s).")

    st.dataframe(preview_df, use_container_width=True, hide_index=True)


def build_result_dataframe_from_predictions(batch_df: pd.DataFrame, predictions: list[dict]) -> pd.DataFrame:
    if len(predictions) != len(batch_df):
        raise ValueError(
            f"Prediction count mismatch: expected {len(batch_df)} row(s), got {len(predictions)}."
        )

    result_df = batch_df.copy().reset_index(drop=True)
    result_df["Predicted_Score"] = [
        round(float(prediction["predicted_score"]), 2)
        for prediction in predictions
    ]
    result_df["Predicted_Band"] = [
        str(prediction["predicted_band"])
        for prediction in predictions
    ]
    return result_df


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

    render_score_distribution(result_df)


def render_score_distribution(result_df: pd.DataFrame):
    st.markdown("#### Predicted Score Range Distribution")

    score_bins = [0, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    score_labels = ["0-50","50-55","55-60","60-65","65-70","70-75","75-80","80-85","85-90","90-95", "95-100"]
    chart_df = result_df[["Predicted_Score"]].copy()
    chart_df["Score Range"] = pd.cut(
        chart_df["Predicted_Score"],
        bins=score_bins,
        labels=score_labels,
        include_lowest=True,
        right=True,
    )
    range_counts = (
        chart_df["Score Range"]
        .value_counts(sort=False)
        .reindex(score_labels, fill_value=0)
        .rename_axis("Score Range")
        .reset_index(name="Number of Students")
    )

    chart = px.bar(
        range_counts,
        x="Score Range",
        y="Number of Students",
        text="Number of Students",
        labels={
            "Score Range": "Score Range",
            "Number of Students": "Number of Students",
        },
    )
    chart.update_traces(
        marker_color="#2563EB",
        marker_line_color="white",
        marker_line_width=1,
        textposition="outside",
        cliponaxis=False,
    )
    chart.update_layout(
        height=380,
        margin=dict(l=20, r=32, t=20, b=20),
        xaxis_title="Score Range",
        yaxis_title="Number of Students",
        yaxis=dict(range=[0, max(1, range_counts["Number of Students"].max()) * 1.2]),
        showlegend=False,
        template="plotly_white",
    )

    st.plotly_chart(chart, use_container_width=True)
    st.caption(
        "This chart groups predicted scores into business-friendly ranges, making it easier to see "
        "whether the uploaded batch is concentrated in lower, middle, or higher performance segments."
    )


# =========================================================
# 3) HEADER
# =========================================================
render_page_header(
    "Batch Prediction",
    "Upload a CSV file, validate the schema, score many student records, and download the result."
)
st.caption(f"Inference backend: {get_inference_mode_label()}")


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
        "`Hours_Studied` is treated as weekly study hours and must be between 0 and 80."
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

    batch_df, is_valid = prepare_batch_dataframe(uploaded_df, raw_feature_names)

    if is_valid:
        st.markdown("### Step 4: Predict and download result")

        if st.button("Run Batch Prediction", type="primary", use_container_width=True):
            try:
                with st.spinner("Scoring student records..."):
                    prediction_result = predict_student_dataframe_via_client(
                        batch_df.loc[:, raw_feature_names],
                        filename=getattr(uploaded_file, "name", "students.csv"),
                    )
                    result_df = build_result_dataframe_from_predictions(
                        batch_df,
                        prediction_result.predictions,
                    )

                for warning in prediction_result.warnings:
                    st.warning(warning)

                st.success("Batch prediction completed.")
                st.caption(f"Inference mode: {prediction_result.inference_mode}")
                render_result_summary(result_df)
                render_result_preview(result_df)

                st.download_button(
                    label="Download Prediction Results",
                    data=dataframe_to_csv_bytes(result_df),
                    file_name="student_prediction_results.csv",
                    mime="text/csv",
                )
            except PredictionInputError as e:
                if e.warnings:
                    for warning in e.warnings:
                        st.warning(warning)
                st.error("Prediction failed because the batch data is not valid.")
                for error in e.errors:
                    st.write(f"- {error}")
            except Exception as e:
                st.error(f"Batch prediction could not be completed: {e}")
