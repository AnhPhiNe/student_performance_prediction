# src/explainer.py

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_OPTIONS, FRIENDLY_LABELS


def humanize_feature_name(name: str) -> str:
    return FRIENDLY_LABELS.get(name, name.replace("_", " "))


def clean_display_name(name: str) -> str:
    """
    Convert encoded pipeline feature names into user-facing labels.
    """
    clean_name = name.split("__")[-1]

    for feature, options in CATEGORICAL_OPTIONS.items():
        prefix = f"{feature}_"
        if clean_name.startswith(prefix):
            category = clean_name[len(prefix):].replace("_", " ")
            if category in options:
                return f"{humanize_feature_name(feature)} = {category}"

    return humanize_feature_name(clean_name)


def build_ridge_coefficient_table(core_model, encoded_feature_names: list[str]) -> pd.DataFrame:
    """
    Build a coefficient table from the Ridge model.
    """
    coef_series = pd.Series(core_model.coef_, index=encoded_feature_names)

    coef_df = pd.DataFrame({
        "Feature": coef_series.index,
        "Display_Name": [clean_display_name(col) for col in coef_series.index],
        "Coefficient": coef_series.values,
    })

    coef_df = (
        coef_df.groupby("Display_Name", as_index=False)["Coefficient"]
        .sum()
        .sort_values(by="Coefficient", ascending=False)
        .reset_index(drop=True)
    )

    coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
    coef_df["Direction"] = np.where(
        coef_df["Coefficient"] >= 0,
        "Positive",
        "Negative",
    )

    return coef_df


def get_top_positive_negative(coef_df: pd.DataFrame, top_n: int = 10):
    """
    Return top positive and top negative coefficients.
    """
    top_positive = (
        coef_df[coef_df["Coefficient"] > 0]
        .sort_values(by="Coefficient", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    top_negative = (
        coef_df[coef_df["Coefficient"] < 0]
        .sort_values(by="Coefficient", ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )

    return top_positive, top_negative
