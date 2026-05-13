from fastapi import FastAPI, HTTPException

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
