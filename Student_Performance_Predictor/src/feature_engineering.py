# src/feature_engineering.py

import pandas as pd
def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features used in the final ML pipeline.
    """
    df = df.copy()

    attendance_safe = df["Attendance"].replace(0, 1)

    df["Study_Efficiency"] = df["Hours_Studied"] / attendance_safe
    df["Total_Study_Time"] = df["Hours_Studied"] + df["Tutoring_Sessions"]
    df["Engagement_Index"] = (
        df["Attendance"] * 0.1
        + df["Extracurricular_Activities"].map({"Yes": 1, "No": 0})
    )

    return df


def feature_engineering_for_pipeline(X, raw_survivors=None, raw_feature_names=None):
    """
    Feature engineering logic used inside the saved pipeline.
    """
    if not isinstance(X, pd.DataFrame):
        if raw_feature_names is None:
            raise ValueError("raw_feature_names must be provided when X is not a DataFrame.")
        X = pd.DataFrame(X, columns=raw_feature_names)

    X = X.copy()
    X = add_new_features(X)

    if raw_survivors is not None:
        X = X[raw_survivors].copy()

    return X