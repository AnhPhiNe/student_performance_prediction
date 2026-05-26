# src/helpers.py

import pandas as pd

from src.config import DEFAULT_VALUES, FRIENDLY_LABELS


def get_friendly_label(column_name: str) -> str:
    """
    Convert a technical column name into a user-facing label.
    """
    return FRIENDLY_LABELS.get(column_name, column_name.replace("_", " "))


def build_default_input(raw_feature_names: list[str]) -> dict:
    """
    Build a default input dictionary for forms and CSV templates.
    """
    data = {}

    for feature in raw_feature_names:
        data[feature] = DEFAULT_VALUES.get(feature, None)

    return data


def dict_to_single_row_df(data: dict, raw_feature_names: list[str]) -> pd.DataFrame:
    """
    Convert an input dictionary into a schema-ordered one-row DataFrame.
    """
    row = {
        feature: data.get(feature, DEFAULT_VALUES.get(feature, None))
        for feature in raw_feature_names
    }
    return pd.DataFrame([row])


def format_score(score: float) -> str:
    """
    Format a numeric score for UI display.
    """
    try:
        return f"{float(score):.1f}"
    except Exception:
        return "N/A"


def safe_get_first_value(df: pd.DataFrame, column_name: str, default=None):
    """
    Return the first value from a DataFrame column, or a fallback default.
    """
    try:
        return df[column_name].iloc[0]
    except Exception:
        return default
