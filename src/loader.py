import os
import pandas as pd
import streamlit as st

from src.config import (
    DATA_PATH,
    CSS_PATH
)
from src.artifact_loader import load_model_assets as load_backend_model_assets


@st.cache_resource
def load_model_assets():
    """
    Tải các file model/artifact chỉ 1 lần và cache lại.
    Khi người dùng đổi tab hoặc bấm widget, Streamlit sẽ rerun script.
    Có cache thì app không phải load model lại liên tục.
    """
    assets = load_backend_model_assets()
    return (
        assets.full_pipeline,
        assets.core_model,
        assets.raw_feature_names,
        assets.raw_survivors,
        assets.best_params,
    )


@st.cache_data
def load_dataset():
    """
    Tải dataset để dùng cho EDA hoặc preview.
    cache_data phù hợp cho dữ liệu bảng hơn cache_resource.
    """
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


def load_css():
    """
    Đọc file CSS nếu có để style giao diện.
    Nếu chưa tạo styles.css thì hàm này trả về chuỗi rỗng.
    """
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return ""
