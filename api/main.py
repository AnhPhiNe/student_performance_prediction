from io import StringIO

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    MetadataResponse,
    PredictionResponse,
    StudentProfileRequest,
)
from src.artifact_loader import load_model_assets
from src.prediction_service import (
    PredictionInputError,
    build_model_metadata,
    predict_student_batch,
    predict_student_profile,
)


app = FastAPI(
    title="Student Performance Predictor API",
    description="FastAPI inference backend for the Student Performance Predictor model.",
    version="1.0.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Student Performance Predictor API",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata", response_model=MetadataResponse)
def get_metadata() -> dict:
    assets = load_model_assets()
    return build_model_metadata(assets)


@app.post("/predict", response_model=PredictionResponse)
def predict(profile: StudentProfileRequest) -> PredictionResponse:
    assets = load_model_assets()

    try:
        result = predict_student_profile(profile.model_dump(), assets)
    except PredictionInputError as exc:
        raise HTTPException(
            status_code=422,
            detail={"errors": exc.errors, "warnings": exc.warnings},
        ) from exc

    return PredictionResponse(
        predicted_score=result.predicted_score,
        predicted_band=result.predicted_band,
        recommendations=result.recommendations,
        warnings=result.warnings,
    )


@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    assets = load_model_assets()
    records = [record.model_dump() for record in payload.records]

    try:
        result = predict_student_batch(records, assets)
    except PredictionInputError as exc:
        raise HTTPException(
            status_code=422,
            detail={"errors": exc.errors, "warnings": exc.warnings},
        ) from exc

    return BatchPredictionResponse(
        count=result.count,
        average_score=result.average_score,
        predictions=result.predictions,
        warnings=result.warnings,
    )


@app.post("/batch-predict-csv", response_model=BatchPredictionResponse)
async def batch_predict_csv(file: UploadFile = File(...)) -> BatchPredictionResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail={"errors": ["Uploaded file must use the .csv extension."]},
        )

    try:
        contents = await file.read()
        if not contents or not contents.strip():
            raise HTTPException(
                status_code=400,
                detail={"errors": ["Uploaded CSV file is empty."]},
            )

        csv_text = contents.decode("utf-8-sig")
        uploaded_df = pd.read_csv(StringIO(csv_text))
        if uploaded_df.empty:
            raise HTTPException(
                status_code=400,
                detail={"errors": ["Uploaded CSV must contain at least one data row."]},
            )

        assets = load_model_assets()
        matching_columns = [
            column for column in uploaded_df.columns if column in assets.raw_feature_names
        ]
        if not matching_columns:
            raise HTTPException(
                status_code=400,
                detail={
                    "errors": [
                        "Uploaded CSV does not contain any expected input columns.",
                        f"Expected at least one of: {assets.raw_feature_names}",
                    ]
                },
            )
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail={"errors": ["CSV file must be encoded as UTF-8 or UTF-8 with BOM."]},
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise HTTPException(
            status_code=400,
            detail={"errors": ["Uploaded CSV file is empty or has no columns."]},
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={"errors": [f"Could not read CSV file: {exc}"]},
        ) from exc

    records = uploaded_df.to_dict(orient="records")

    try:
        result = predict_student_batch(records, assets)
    except PredictionInputError as exc:
        raise HTTPException(
            status_code=422,
            detail={"errors": exc.errors, "warnings": exc.warnings},
        ) from exc

    return BatchPredictionResponse(
        count=result.count,
        average_score=result.average_score,
        predictions=result.predictions,
        warnings=result.warnings,
    )
