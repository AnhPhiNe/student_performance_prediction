import os

import pandas as pd
import streamlit as st

from src.artifact_loader import load_model_assets as load_backend_model_assets
from src.config import CSS_PATH, DATA_PATH


@st.cache_resource
def load_model_assets():
    """
    Load model artifacts once and cache them across Streamlit reruns.
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
    Load the dataset for previews and lightweight project context.
    """
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


def load_css():
    """
    Load the app stylesheet if it exists.
    """
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return ""
