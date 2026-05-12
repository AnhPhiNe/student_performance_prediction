# pages/2_Single_Prediction.py

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn import set_config
set_config(transform_output="pandas")

from src.config import (
    SAMPLE_PROFILES,
    CATEGORICAL_OPTIONS,
    FORM_GROUPS,
    DEFAULT_VALUES,
    NUMERIC_RANGES
)
from src.loader import load_model_assets
from src.helpers import get_friendly_label, format_score
from src.predictor import (
    build_input_dataframe,
    coerce_input_types,
    predict_single,
    score_band,
    generate_recommendations
)
from src.validators import validate_input_dataframe
from src.ui_components import (
    render_section_title,
    render_result_card,
    render_empty_state
)

# =========================================================
# 1) LOAD MODEL
# =========================================================
try:
    full_pipeline, core_model, raw_feature_names, raw_survivors, best_params = load_model_assets()
except Exception as e:
    st.error(f"Failed to load model assets: {e}")
    st.stop()

# =========================================================
# 2) HÀM PHỤ CHO MÀU BAND
# =========================================================
def get_band_color(score: float) -> str:
    if score >= 90:
        return "#16a34a"
    elif score >= 80:
        return "#22c55e"
    elif score >= 70:
        return "#2563eb"
    elif score >= 60:
        return "#f59e0b"
    return "#ef4444"


def render_score_gauge(score: float):
    color = get_band_color(score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 34}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 60], "color": "rgba(239,68,68,0.18)"},
                {"range": [60, 80], "color": "rgba(245,158,11,0.18)"},
                {"range": [80, 100], "color": "rgba(34,197,94,0.18)"}
            ]
        }
    ))

    fig.update_layout(height=280, margin=dict(l=10, r=10, t=25, b=10))
    st.plotly_chart(fig, width="stretch")


# =========================================================
# 3) STATE
# =========================================================
if "selected_profile" not in st.session_state:
    st.session_state.selected_profile = "Balanced Student"

if "form_data" not in st.session_state:
    st.session_state.form_data = SAMPLE_PROFILES["Balanced Student"].copy()

# =========================================================
# 4) HEADER
# =========================================================
render_section_title(
    "Single Student Prediction",
    "Build one student profile, run prediction, and read the result in a simple way."
)

# =========================================================
# 5) QUICK PROFILE
# =========================================================
selected_profile = st.selectbox(
    "Quick preset profile",
    options=list(SAMPLE_PROFILES.keys()),
    index=list(SAMPLE_PROFILES.keys()).index(st.session_state.selected_profile)
)

if selected_profile != st.session_state.selected_profile:
    st.session_state.selected_profile = selected_profile
    st.session_state.form_data = SAMPLE_PROFILES[selected_profile].copy()

# =========================================================
# 6) FORM
# =========================================================
with st.form("single_prediction_form"):
    left_col, right_col = st.columns([1.35, 1], gap="large")

    with left_col:
        for group_name, features in FORM_GROUPS.items():
            st.markdown(f"### {group_name}")

            group_features = [f for f in features if f in raw_feature_names]

            for feature in group_features:
                label = get_friendly_label(feature)
                current_value = st.session_state.form_data.get(feature, DEFAULT_VALUES.get(feature))

                if feature in NUMERIC_RANGES:
                    min_val, max_val = NUMERIC_RANGES[feature]

                    if feature in ["Hours_Studied", "Attendance", "Sleep_Hours", "Physical_Activity"]:
                        st.session_state.form_data[feature] = st.slider(
                            label,
                            min_value=min_val,
                            max_value=max_val,
                            value=int(current_value)
                        )
                    else:
                        st.session_state.form_data[feature] = st.number_input(
                            label,
                            min_value=min_val,
                            max_value=max_val,
                            value=int(current_value),
                            step=1
                        )

                elif feature in CATEGORICAL_OPTIONS:
                    options = CATEGORICAL_OPTIONS[feature]
                    current_value = current_value if current_value in options else options[0]

                    st.session_state.form_data[feature] = st.selectbox(
                        label,
                        options=options,
                        index=options.index(current_value)
                    )

    with right_col:
        st.markdown("### Prediction Panel")
        st.caption("After you click predict, the system will estimate the score and provide basic recommendations.")

        submit_btn = st.form_submit_button(
            "Predict Student Score",
            width="stretch",
            type="primary"
        )

# =========================================================
# 7) PREDICT
# =========================================================
if submit_btn:
    user_input = st.session_state.form_data.copy()

    input_df = build_input_dataframe(user_input, raw_feature_names)
    input_df = coerce_input_types(input_df)

    validation = validate_input_dataframe(input_df, raw_feature_names)

    if not validation["is_valid"]:
        st.error("The input data is not valid.")
        for err in validation["errors"]:
            st.write(f"- {err}")
        for warn in validation["warnings"]:
            st.warning(warn)
        st.stop()

    for warn in validation["warnings"]:
        st.warning(warn)

    with st.spinner("Running prediction..."):
        predicted_score = predict_single(full_pipeline, input_df, raw_feature_names)

    band = score_band(predicted_score)
    band_color = get_band_color(predicted_score)
    recommendations = generate_recommendations(input_df)

    st.markdown("---")
    result_col, insight_col = st.columns([1, 1.1], gap="large")

    with result_col:
        render_result_card(
            score_text=format_score(predicted_score),
            band=band,
            band_color=band_color
        )
        render_score_gauge(predicted_score)

    with insight_col:
        st.markdown("### Interpretation")

        if predicted_score >= 80:
            st.success("This student profile is likely to perform strongly based on the current input factors.")
        elif predicted_score >= 60:
            st.warning("This student profile is in the middle range. Several factors could still be improved.")
        else:
            st.error("This student profile may be academically at risk and may need stronger support.")

        st.markdown("### Recommendations")
        for i, rec in enumerate(recommendations, start=1):
            st.write(f"**{i}.** {rec}")

        st.markdown("### Input Summary")
        summary_df = pd.DataFrame({
            "Feature": [get_friendly_label(col) for col in input_df.columns],
            "Value": [str(input_df.iloc[0][col]) for col in input_df.columns]
        })
        st.dataframe(summary_df, width="stretch", hide_index=True)

else:
    render_empty_state("Choose a preset or adjust the form, then click **Predict Student Score** to see the result.")