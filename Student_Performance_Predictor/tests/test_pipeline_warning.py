from pathlib import Path

import joblib
import pandas as pd

from src.config import PIPELINE_PATH, RAW_FEATURES_PATH, SAMPLE_PROFILES


def test_model_artifacts_exist():
    assert Path(PIPELINE_PATH).is_file(), f"Missing pipeline artifact: {PIPELINE_PATH}"
    assert Path(RAW_FEATURES_PATH).is_file(), f"Missing raw feature artifact: {RAW_FEATURES_PATH}"


def test_load_pipeline_and_raw_features_successfully():
    pipeline = joblib.load(PIPELINE_PATH)
    raw_feature_names = joblib.load(RAW_FEATURES_PATH)

    assert hasattr(pipeline, "predict")
    assert raw_feature_names is not None
    assert len(raw_feature_names) > 0


def test_pipeline_prediction_is_numeric_and_in_valid_range():
    pipeline = joblib.load(PIPELINE_PATH)
    raw_feature_names = joblib.load(RAW_FEATURES_PATH)

    if not isinstance(raw_feature_names, list):
        raw_feature_names = list(raw_feature_names)

    sample = SAMPLE_PROFILES["Balanced Student"]
    input_df = pd.DataFrame([sample], columns=raw_feature_names)

    prediction = pipeline.predict(input_df)[0]
    prediction = float(prediction)

    assert 0 <= prediction <= 100
