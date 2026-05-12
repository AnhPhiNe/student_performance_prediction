# src/predictor.py

import numpy as np
import pandas as pd
from src.config import DEFAULT_VALUES


def build_input_dataframe(user_input: dict, raw_feature_names: list[str]) -> pd.DataFrame:
    """
    Tạo DataFrame đầu vào từ dữ liệu người dùng nhập.
    Đảm bảo:
    - đúng tên cột
    - đúng thứ tự cột
    """
    raw_feature_names = list(raw_feature_names)

    row = {}
    for feature in raw_feature_names:
        row[feature] = user_input.get(feature, DEFAULT_VALUES.get(feature, None))

    input_df = pd.DataFrame([row], columns=raw_feature_names)
    return input_df


def coerce_input_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ép kiểu dữ liệu nhưng vẫn giữ DataFrame và tên cột.
    """
    df = df.copy()

    numeric_columns = [
        "Hours_Studied",
        "Attendance",
        "Previous_Scores",
        "Tutoring_Sessions",
        "Sleep_Hours",
        "Physical_Activity",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.columns:
        if col not in numeric_columns:
            df[col] = df[col].astype("string")

    return df


def predict_single(model_pipeline, input_df: pd.DataFrame, raw_feature_names: list[str]) -> float:
    """
    Dự đoán điểm cho 1 học sinh.
    """
    raw_feature_names = list(raw_feature_names)

    input_df = input_df.copy()
    input_df = input_df.loc[:, raw_feature_names]

    if not isinstance(input_df, pd.DataFrame):
        input_df = pd.DataFrame(input_df, columns=raw_feature_names)

    prediction = model_pipeline.predict(input_df)[0]
    prediction = float(np.clip(prediction, 0, 100))
    return prediction


def predict_batch(model_pipeline, batch_df: pd.DataFrame, raw_feature_names: list[str]) -> pd.DataFrame:
    """
    Dự đoán điểm cho nhiều học sinh cùng lúc.
    """
    raw_feature_names = list(raw_feature_names)

    ordered_df = batch_df.copy().loc[:, raw_feature_names]

    predictions = model_pipeline.predict(ordered_df)
    predictions = np.clip(predictions, 0, 100)

    result_df = batch_df.copy()
    result_df["Predicted_Score"] = np.round(predictions, 2)
    result_df["Predicted_Band"] = result_df["Predicted_Score"].apply(score_band)

    return result_df


def score_band(score: float) -> str:
    if score >= 90:
        return "Excellent"
    elif score >= 80:
        return "Very Good"
    elif score >= 70:
        return "Good"
    elif score >= 60:
        return "Average"
    return "Needs Improvement"


def generate_recommendations(input_df: pd.DataFrame) -> list[str]:
    recommendations = []

    if "Attendance" in input_df.columns and float(input_df["Attendance"].iloc[0]) < 80:
        recommendations.append(
            "Improve attendance consistency because regular class participation often supports better academic performance."
        )

    if "Hours_Studied" in input_df.columns and float(input_df["Hours_Studied"].iloc[0]) < 4:
        recommendations.append(
            "Increase daily study time gradually instead of relying on last-minute cramming."
        )

    if "Sleep_Hours" in input_df.columns and float(input_df["Sleep_Hours"].iloc[0]) < 6:
        recommendations.append(
            "Sleep duration is relatively low. Better sleep may improve concentration and learning retention."
        )

    if "Motivation_Level" in input_df.columns and str(input_df["Motivation_Level"].iloc[0]) == "Low":
        recommendations.append(
            "Try setting small weekly study goals to improve motivation and learning consistency."
        )

    if "Internet_Access" in input_df.columns and str(input_df["Internet_Access"].iloc[0]) == "No":
        recommendations.append(
            "Limited internet access may reduce access to learning materials and online support resources."
        )

    if "Parental_Involvement" in input_df.columns and str(input_df["Parental_Involvement"].iloc[0]) == "Low":
        recommendations.append(
            "More parental encouragement and learning support may help improve academic stability."
        )

    if not recommendations:
        recommendations.append(
            "This student profile looks relatively balanced. The main priority is to maintain consistency in current habits."
        )

    return recommendations