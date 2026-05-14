from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.artifact_loader import ModelAssets
from src.config import CATEGORICAL_OPTIONS, NUMERIC_RANGES
from src.predictor import (
    build_input_dataframe,
    coerce_input_types,
    generate_recommendations,
    predict_batch,
    predict_single,
    score_band,
)
from src.validators import validate_input_dataframe


CATEGORICAL_DEFAULTS = {
    "Teacher_Quality": "Medium",
    "Parental_Education_Level": "High School",
    "Distance_from_Home": "Moderate",
    "Access_to_Resources": "Medium",
    "Motivation_Level": "Medium",
    "Parental_Involvement": "Medium",
    "Family_Income": "Medium",
    "Peer_Influence": "Neutral",
}


@dataclass(frozen=True)
class PredictionResult:
    predicted_score: float
    predicted_band: str
    recommendations: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class BatchPredictionResult:
    count: int
    average_score: float
    predictions: list[dict[str, Any]]
    warnings: list[str]


class PredictionInputError(ValueError):
    def __init__(self, errors: list[str], warnings: list[str] | None = None):
        super().__init__("Prediction input validation failed.")
        self.errors = errors
        self.warnings = warnings or []


def _validate_or_raise(df: pd.DataFrame, raw_feature_names: list[str]) -> list[str]:
    validation = validate_input_dataframe(df, raw_feature_names)
    if not validation["is_valid"]:
        raise PredictionInputError(
            errors=validation["errors"],
            warnings=validation["warnings"],
        )

    return validation["warnings"]


def _missing_mask(series: pd.Series) -> pd.Series:
    text_values = series.astype("string").str.strip()
    return series.isna() | text_values.isna() | text_values.eq("")


def _prepare_batch_dataframe(records: list[dict[str, Any]], raw_feature_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    batch_df = pd.DataFrame(records)
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = [column for column in raw_feature_names if column not in batch_df.columns]
    extra_columns = [column for column in batch_df.columns if column not in raw_feature_names]

    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    if extra_columns:
        warnings.append(f"Unexpected extra columns will be ignored: {extra_columns}")

    if errors:
        raise PredictionInputError(errors=errors, warnings=warnings)

    batch_df = batch_df.copy()

    for column, (min_value, max_value) in NUMERIC_RANGES.items():
        if column not in raw_feature_names or column not in batch_df.columns:
            continue

        missing = _missing_mask(batch_df[column])
        numeric_values = pd.to_numeric(batch_df[column], errors="coerce")
        invalid_numeric = numeric_values.isna() & ~missing

        if invalid_numeric.any():
            errors.append(f"Column '{column}' contains non-numeric value(s).")
            continue

        valid_values = numeric_values[~missing]
        out_of_range = (valid_values < min_value) | (valid_values > max_value)
        if out_of_range.any():
            errors.append(f"Column '{column}' must be between {min_value} and {max_value}.")

        missing_count = int(missing.sum())
        if missing_count:
            warnings.append(
                f"Column '{column}' has {missing_count} missing value(s); the model pipeline will use median imputation."
            )

        batch_df.loc[:, column] = numeric_values

    for column, default_value in CATEGORICAL_DEFAULTS.items():
        if column not in raw_feature_names or column not in batch_df.columns:
            continue

        missing = _missing_mask(batch_df[column])
        missing_count = int(missing.sum())
        if not missing_count:
            continue

        batch_df.loc[missing, column] = default_value
        warnings.append(
            f"Column '{column}' had {missing_count} missing value(s) filled with '{default_value}'."
        )

    for column, valid_options in CATEGORICAL_OPTIONS.items():
        if column not in raw_feature_names or column not in batch_df.columns:
            continue

        missing = _missing_mask(batch_df[column])
        if missing.any():
            errors.append(f"Column '{column}' contains missing values.")
            continue

        actual_values = set(batch_df[column].dropna().astype(str).str.strip().unique())
        invalid_values = sorted(actual_values - set(valid_options))
        if invalid_values:
            errors.append(
                f"Column '{column}' has invalid values: {invalid_values}. Allowed values: {valid_options}"
            )

        batch_df.loc[:, column] = batch_df[column].astype("string")

    if errors:
        raise PredictionInputError(errors=errors, warnings=warnings)

    return batch_df, warnings


def predict_student_profile(record: dict[str, Any], assets: ModelAssets) -> PredictionResult:
    input_df = build_input_dataframe(record, assets.raw_feature_names)
    input_df = coerce_input_types(input_df)
    warnings = _validate_or_raise(input_df, assets.raw_feature_names)

    predicted_score = round(
        predict_single(assets.full_pipeline, input_df, assets.raw_feature_names),
        2,
    )

    return PredictionResult(
        predicted_score=predicted_score,
        predicted_band=score_band(predicted_score),
        recommendations=generate_recommendations(input_df),
        warnings=warnings,
    )


def predict_student_batch(records: list[dict[str, Any]], assets: ModelAssets) -> BatchPredictionResult:
    if not records:
        raise PredictionInputError(errors=["At least one record is required for batch prediction."])

    batch_df, warnings = _prepare_batch_dataframe(records, assets.raw_feature_names)

    result_df = predict_batch(assets.full_pipeline, batch_df, assets.raw_feature_names)

    predictions = []
    for index, row in result_df.reset_index(drop=True).iterrows():
        predicted_score = float(row["Predicted_Score"])
        predictions.append(
            {
                "row_id": int(index + 1),
                "predicted_score": predicted_score,
                "predicted_band": str(row["Predicted_Band"]),
            }
        )

    return BatchPredictionResult(
        count=len(predictions),
        average_score=float(np.round(result_df["Predicted_Score"].mean(), 2)),
        predictions=predictions,
        warnings=warnings,
    )


def build_model_metadata(assets: ModelAssets) -> dict[str, Any]:
    metadata = assets.metadata or {}
    metrics = metadata.get("metrics") if isinstance(metadata.get("metrics"), dict) else {}

    return {
        "model_name": metadata.get("final_model_name", "Ridge Regression"),
        "target": "Exam_Score",
        "features": assets.raw_feature_names,
        "selected_features": assets.raw_survivors,
        "metrics": metrics,
        "training_date": metadata.get("training_date"),
    }
