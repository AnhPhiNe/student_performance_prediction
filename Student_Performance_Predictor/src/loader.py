# src/loader.py

import json
import os
import joblib
import pandas as pd
import streamlit as st
from sklearn import set_config

set_config(transform_output="pandas")

from src.config import (
    PIPELINE_PATH,
    CORE_MODEL_PATH,
    RAW_FEATURES_PATH,
    RAW_SURVIVORS_PATH,
    BEST_PARAMS_PATH,
    DATA_PATH,
    CSS_PATH
)


@st.cache_resource
def load_model_assets():
    """
    Tải các file model/artifact chỉ 1 lần và cache lại.
    Khi người dùng đổi tab hoặc bấm widget, Streamlit sẽ rerun script.
    Có cache thì app không phải load model lại liên tục.
    """
    full_pipeline = joblib.load(PIPELINE_PATH)
    core_model = joblib.load(CORE_MODEL_PATH)
    raw_feature_names = joblib.load(RAW_FEATURES_PATH)
    raw_survivors = joblib.load(RAW_SURVIVORS_PATH)

    if not isinstance(raw_feature_names, list):
        raw_feature_names = list(raw_feature_names)

    if not isinstance(raw_survivors, list):
        raw_survivors = list(raw_survivors)

    best_params = None
    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
            best_params = json.load(f)

    return full_pipeline, core_model, raw_feature_names, raw_survivors, best_params


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