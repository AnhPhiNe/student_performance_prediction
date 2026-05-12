# src/explainer.py

import numpy as np
import pandas as pd


def clean_display_name(name: str) -> str:
    """
    Làm sạch tên feature để hiển thị đẹp hơn.
    """
    clean_name = name.split("__")[-1]

    suffixes = [
        "_Yes", "_No",
        "_Male", "_Female",
        "_Public", "_Private",
        "_Urban", "_Rural"
    ]

    for s in suffixes:
        if clean_name.endswith(s):
            clean_name = clean_name.replace(s, "")

    return clean_name


def build_ridge_coefficient_table(core_model, encoded_feature_names: list[str]) -> pd.DataFrame:
    """
    Tạo bảng coefficient từ Ridge model.
    """
    coef_series = pd.Series(core_model.coef_, index=encoded_feature_names)

    coef_df = pd.DataFrame({
        "Feature": coef_series.index,
        "Display_Name": [clean_display_name(col) for col in coef_series.index],
        "Coefficient": coef_series.values,
    })

    # Nếu tên hiển thị trùng nhau thì cộng hệ số lại
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
        "Negative"
    )

    return coef_df


def get_top_positive_negative(coef_df: pd.DataFrame, top_n: int = 10):
    """
    Trả về top positive và top negative coefficients.
    """
    top_positive = (
        coef_df.sort_values(by="Coefficient", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    top_negative = (
        coef_df.sort_values(by="Coefficient", ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )

    return top_positive, top_negative