from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.artifact_loader import ModelAssets
from src.predictor import (
    build_input_dataframe,
    coerce_input_types,
    generate_recommendations,
    predict_batch,
    predict_single,
    score_band,
)
from src.validators import validate_input_dataframe


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

    batch_df = pd.DataFrame(records)
    batch_df = coerce_input_types(batch_df)
    warnings = _validate_or_raise(batch_df, assets.raw_feature_names)

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
