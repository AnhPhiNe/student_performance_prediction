# pages/2_Single_Prediction.py

import streamlit as st
st.set_page_config(
    page_title="Single Prediction | EduPredict",
    page_icon=":mortar_board:",
    layout="wide",
)

from sklearn import set_config

from src.config import (
    SAMPLE_PROFILES,
    CATEGORICAL_OPTIONS,
    FORM_GROUPS,
    DEFAULT_VALUES,
    NUMERIC_RANGES,
)
from src.loader import load_css, load_model_assets
from src.helpers import get_friendly_label, format_score
from src.predictor import (
    build_input_dataframe,
    coerce_input_types,
    predict_single,
    score_band,
    generate_recommendations,
)
from src.validators import validate_input_dataframe
from src.ui_components import render_section_title


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
# 2) SMALL UI HELPERS
# =========================================================
def get_band_color(score: float) -> str:
    if score >= 90:
        return "#16a34a"
    if score >= 80:
        return "#22c55e"
    if score >= 70:
        return "#2563eb"
    if score >= 60:
        return "#f59e0b"
    return "#ef4444"


def render_compact_score(score: float, band: str):
    band_color = get_band_color(score)
    progress_value = int(max(0, min(100, round(score))))

    st.markdown(
        f"""
        <div class="ep-score-label">Predicted Score</div>
        <div class="ep-score-value">{format_score(score)}</div>
        <div class="ep-score-band" style="border-color:{band_color}; color:{band_color};">
            {band}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(progress_value)
    st.caption(f"{format_score(score)} / 100")
    st.caption("Bands: Average 60+, Good 70+, Very Good 80+.")


def shorten_recommendation(recommendation: str) -> str:
    recommendation_lower = recommendation.lower()

    if "attendance" in recommendation_lower:
        return "Maintain attendance above 85%."
    if "study" in recommendation_lower or "cramming" in recommendation_lower:
        return "Increase study consistency gradually."
    if "sleep" in recommendation_lower:
        return "Keep sleep duration around 7-8 hours."
    if "motivation" in recommendation_lower:
        return "Set small weekly learning goals."
    if "internet" in recommendation_lower or "resources" in recommendation_lower:
        return "Use available learning resources consistently."
    if "parental" in recommendation_lower:
        return "Strengthen learning support at home."
    if "balanced" in recommendation_lower or "maintain" in recommendation_lower:
        return "Maintain current learning habits."

    first_sentence = recommendation.split(".")[0].strip()
    if len(first_sentence) > 90:
        first_sentence = first_sentence[:87].rstrip() + "..."
    return f"{first_sentence}."


def get_input_value(input_df, column_name: str):
    if column_name not in input_df.columns:
        return None
    return input_df.iloc[0][column_name]


def build_display_recommendations(input_df, recommendations: list[str], score: float) -> list[str]:
    display_recommendations = []

    attendance = get_input_value(input_df, "Attendance")
    if attendance is not None and float(attendance) < 90:
        display_recommendations.append("Improve attendance consistency toward 90%.")

    hours_studied = get_input_value(input_df, "Hours_Studied")
    if hours_studied is not None and float(hours_studied) < 6:
        display_recommendations.append("Increase weekly study consistency gradually.")

    previous_scores = get_input_value(input_df, "Previous_Scores")
    if previous_scores is not None and float(previous_scores) < 80:
        display_recommendations.append("Review past exam gaps before new topics.")

    sleep_hours = get_input_value(input_df, "Sleep_Hours")
    if sleep_hours is not None and float(sleep_hours) < 7:
        display_recommendations.append("Keep sleep duration around 7-8 hours.")

    motivation_level = get_input_value(input_df, "Motivation_Level")
    if motivation_level is not None and str(motivation_level) in ["Low", "Medium"]:
        display_recommendations.append("Set small weekly learning goals.")

    access_to_resources = get_input_value(input_df, "Access_to_Resources")
    if access_to_resources is not None and str(access_to_resources) in ["Low", "Medium"]:
        display_recommendations.append("Use learning resources consistently.")

    for recommendation in recommendations:
        display_recommendations.append(shorten_recommendation(recommendation))

    if score < 70:
        display_recommendations.append("Pick one weak area and improve it for the next prediction.")

    unique_recommendations = []
    for recommendation in display_recommendations:
        if recommendation not in unique_recommendations:
            unique_recommendations.append(recommendation)

    return unique_recommendations or ["Maintain current learning habits."]


def render_input_widget(feature: str):
    label = get_friendly_label(feature)
    current_value = st.session_state.form_data.get(feature, DEFAULT_VALUES.get(feature))

    if feature == "Hours_Studied":
        st.session_state.form_data[feature] = st.number_input(
            label,
            min_value=0,
            value=int(current_value),
            step=1,
            help="Weekly study hours. No upper limit is enforced by validation.",
        )

    elif feature in NUMERIC_RANGES:
        min_val, max_val = NUMERIC_RANGES[feature]
        value = int(current_value)

        if feature in ["Hours_Studied", "Attendance", "Sleep_Hours", "Physical_Activity"]:
            st.session_state.form_data[feature] = st.slider(
                label,
                min_value=min_val,
                max_value=max_val,
                value=value,
            )
        else:
            st.session_state.form_data[feature] = st.number_input(
                label,
                min_value=min_val,
                max_value=max_val,
                value=value,
                step=1,
            )

    elif feature in CATEGORICAL_OPTIONS:
        options = CATEGORICAL_OPTIONS[feature]
        value = current_value if current_value in options else options[0]

        st.session_state.form_data[feature] = st.selectbox(
            label,
            options=options,
            index=options.index(value),
        )


def render_prediction_result():
    try:
        user_input = st.session_state.form_data.copy()
        input_df = build_input_dataframe(user_input, raw_feature_names)
        input_df = coerce_input_types(input_df)

        validation = validate_input_dataframe(input_df, raw_feature_names)

        if validation["warnings"]:
            for warning in validation["warnings"]:
                st.warning(warning)

        if not validation["is_valid"]:
            st.error("The input data is not valid. Please review the items below.")
            for error in validation["errors"]:
                st.write(f"- {error}")
            return

        with st.spinner("Running prediction..."):
            predicted_score = predict_single(full_pipeline, input_df, raw_feature_names)

        band = score_band(predicted_score)
        recommendations = generate_recommendations(input_df)
        display_recommendations = build_display_recommendations(
            input_df,
            recommendations,
            predicted_score,
        )

        st.markdown(
            "<div class='ep-prediction-status'>Prediction completed</div>",
            unsafe_allow_html=True,
        )
        render_compact_score(predicted_score, band)

        if display_recommendations:
            st.markdown(
                f"""
                <div class="ep-main-recommendation">
                    {display_recommendations[0]}
                </div>
                """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"Prediction could not be completed: {e}")


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
    "Adjust a student profile and run an end-to-end model prediction."
)


# =========================================================
# 5) QUICK PROFILE
# =========================================================
profile_names = list(SAMPLE_PROFILES.keys())
selected_profile = st.selectbox(
    "Quick preset profile",
    options=profile_names,
    index=profile_names.index(st.session_state.selected_profile),
)

if selected_profile != st.session_state.selected_profile:
    st.session_state.selected_profile = selected_profile
    st.session_state.form_data = SAMPLE_PROFILES[selected_profile].copy()


# =========================================================
# 6) FORM + RESULT PANEL
# =========================================================
tab_groups = [
    ("Academic", "Academic Habits"),
    ("Lifestyle", "Lifestyle"),
    ("Family & Learning", "Family & Learning Environment"),
    ("School & Personal", "School & Personal Context"),
]

with st.form("single_prediction_form"):
    form_col, result_col = st.columns([1.75, 1], gap="large")

    with form_col:
        st.markdown("### Student Profile")
        st.caption("Switch tabs to adjust lifestyle, family, and school context factors.")
        tabs = st.tabs([tab_label for tab_label, _ in tab_groups])

        for tab, (_, group_name) in zip(tabs, tab_groups):
            with tab:
                group_features = [
                    feature
                    for feature in FORM_GROUPS.get(group_name, [])
                    if feature in raw_feature_names
                ]
                field_cols = st.columns(2, gap="medium")

                for index, feature in enumerate(group_features):
                    with field_cols[index % 2]:
                        render_input_widget(feature)

    with result_col:
        st.markdown("### Prediction Result")

        submit_btn = st.form_submit_button(
            "Predict Student Score",
            type="primary",
            use_container_width=True,
        )

        if submit_btn:
            st.caption("Latest prediction for the selected profile.")
            render_prediction_result()
        else:
            st.caption("Ready when the profile is set.")
            st.info("Fill in the form and click Predict to see the estimated score.")
