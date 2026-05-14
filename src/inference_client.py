import os
from dataclasses import dataclass
from typing import Any

import httpx
import pandas as pd

from src.artifact_loader import load_model_assets
from src.prediction_service import (
    PredictionInputError,
    predict_student_batch,
    predict_student_profile,
)


DEFAULT_TIMEOUT_SECONDS = 10.0
API_BASE_URL_ENV_VAR = "API_BASE_URL"


@dataclass(frozen=True)
class SinglePredictionResult:
    predicted_score: float
    predicted_band: str
    recommendations: list[str]
    warnings: list[str]
    inference_mode: str


@dataclass(frozen=True)
class BatchPredictionResult:
    count: int
    average_score: float
    predictions: list[dict[str, Any]]
    warnings: list[str]
    inference_mode: str


def get_api_base_url() -> str | None:
    value = os.getenv(API_BASE_URL_ENV_VAR)
    if value and value.strip():
        return value.strip().rstrip("/")
    return None


def get_inference_mode_label() -> str:
    if get_api_base_url():
        return "FastAPI backend enabled"
    return "Local machine learning service"


def _api_detail_to_error(detail: Any) -> PredictionInputError:
    if isinstance(detail, dict):
        errors = detail.get("errors")
        warnings = detail.get("warnings")
        if isinstance(errors, list):
            return PredictionInputError(
                errors=[str(error) for error in errors],
                warnings=[str(warning) for warning in warnings or []],
            )

    return PredictionInputError(errors=[str(detail)])


def _post_json(api_base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(
        f"{api_base_url}{path}",
        json=payload,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )

    if response.status_code == 422:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text
        raise _api_detail_to_error(detail)

    response.raise_for_status()
    return response.json()


def _post_csv(api_base_url: str, path: str, csv_bytes: bytes, filename: str) -> dict[str, Any]:
    response = httpx.post(
        f"{api_base_url}{path}",
        files={"file": (filename, csv_bytes, "text/csv")},
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )

    if response.status_code == 422:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text
        raise _api_detail_to_error(detail)

    response.raise_for_status()
    return response.json()


def _predict_student_profile_local(record: dict[str, Any], warnings: list[str] | None = None) -> SinglePredictionResult:
    assets = load_model_assets()
    result = predict_student_profile(record, assets)
    combined_warnings = [*(warnings or []), *result.warnings]

    return SinglePredictionResult(
        predicted_score=result.predicted_score,
        predicted_band=result.predicted_band,
        recommendations=result.recommendations,
        warnings=combined_warnings,
        inference_mode="local",
    )


def _predict_student_batch_local(records: list[dict[str, Any]], warnings: list[str] | None = None) -> BatchPredictionResult:
    assets = load_model_assets()
    result = predict_student_batch(records, assets)
    combined_warnings = [*(warnings or []), *result.warnings]

    return BatchPredictionResult(
        count=result.count,
        average_score=result.average_score,
        predictions=result.predictions,
        warnings=combined_warnings,
        inference_mode="local",
    )


def predict_student_profile_via_client(record: dict[str, Any]) -> SinglePredictionResult:
    api_base_url = get_api_base_url()
    if not api_base_url:
        return _predict_student_profile_local(record)

    try:
        payload = _post_json(api_base_url, "/predict", record)
    except PredictionInputError:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        return _predict_student_profile_local(
            record,
            warnings=[f"FastAPI backend unavailable; used local fallback. Details: {exc}"],
        )

    return SinglePredictionResult(
        predicted_score=float(payload["predicted_score"]),
        predicted_band=str(payload["predicted_band"]),
        recommendations=[str(item) for item in payload.get("recommendations", [])],
        warnings=[str(item) for item in payload.get("warnings", [])],
        inference_mode="api",
    )


def predict_student_batch_via_client(records: list[dict[str, Any]]) -> BatchPredictionResult:
    api_base_url = get_api_base_url()
    if not api_base_url:
        return _predict_student_batch_local(records)

    try:
        payload = _post_json(api_base_url, "/batch-predict", {"records": records})
    except PredictionInputError:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        return _predict_student_batch_local(
            records,
            warnings=[f"FastAPI backend unavailable; used local fallback. Details: {exc}"],
        )

    return BatchPredictionResult(
        count=int(payload["count"]),
        average_score=float(payload["average_score"]),
        predictions=list(payload.get("predictions", [])),
        warnings=[str(item) for item in payload.get("warnings", [])],
        inference_mode="api",
    )


def predict_student_dataframe_via_client(batch_df: pd.DataFrame, filename: str = "students.csv") -> BatchPredictionResult:
    records = batch_df.to_dict(orient="records")
    api_base_url = get_api_base_url()
    if not api_base_url:
        return _predict_student_batch_local(records)

    try:
        csv_bytes = batch_df.to_csv(index=False).encode("utf-8-sig")
        payload = _post_csv(api_base_url, "/batch-predict-csv", csv_bytes, filename)
    except PredictionInputError:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        return _predict_student_batch_local(
            records,
            warnings=[f"FastAPI backend unavailable; used local fallback. Details: {exc}"],
        )

    return BatchPredictionResult(
        count=int(payload["count"]),
        average_score=float(payload["average_score"]),
        predictions=list(payload.get("predictions", [])),
        warnings=[str(item) for item in payload.get("warnings", [])],
        inference_mode="api",
    )
