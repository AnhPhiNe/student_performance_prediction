from pathlib import Path
import warnings

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
    pipeline.set_output(transform="pandas")
    raw_feature_names = joblib.load(RAW_FEATURES_PATH)

    if not isinstance(raw_feature_names, list):
        raw_feature_names = list(raw_feature_names)

    sample = SAMPLE_PROFILES["Balanced Student"]
    input_df = pd.DataFrame([sample], columns=raw_feature_names)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        prediction = pipeline.predict(input_df)[0]

    feature_name_warnings = [
        warning
        for warning in caught_warnings
        if "does not have valid feature names" in str(warning.message)
    ]

    assert not feature_name_warnings
    prediction = float(prediction)

    assert 0 <= prediction <= 100
