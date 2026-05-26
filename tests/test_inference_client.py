import httpx
import pandas as pd

from src import inference_client
from src.inference_client import BatchPredictionResult, SinglePredictionResult


def test_get_api_base_url_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:8000/")

    assert inference_client.get_api_base_url() == "http://127.0.0.1:8000"


def test_predict_student_profile_via_client_uses_api(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:8000")

    def fake_post_json(api_base_url, path, payload):
        assert api_base_url == "http://127.0.0.1:8000"
        assert path == "/predict"
        assert payload["Hours_Studied"] == 5
        return {
            "predicted_score": 78.5,
            "predicted_band": "Good",
            "recommendations": ["Maintain current learning habits."],
            "warnings": [],
        }

    monkeypatch.setattr(inference_client, "_post_json", fake_post_json)

    result = inference_client.predict_student_profile_via_client({"Hours_Studied": 5})

    assert result.inference_mode == "api"
    assert result.predicted_score == 78.5
    assert result.predicted_band == "Good"


def test_predict_student_profile_via_client_falls_back_when_api_is_unavailable(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:8000")

    def fake_post_json(api_base_url, path, payload):
        raise httpx.ConnectError("connection refused")

    def fake_local(record, warnings=None):
        return SinglePredictionResult(
            predicted_score=75.0,
            predicted_band="Good",
            recommendations=["Fallback recommendation."],
            warnings=warnings or [],
            inference_mode="local",
        )

    monkeypatch.setattr(inference_client, "_post_json", fake_post_json)
    monkeypatch.setattr(inference_client, "_predict_student_profile_local", fake_local)

    result = inference_client.predict_student_profile_via_client({"Hours_Studied": 5})

    assert result.inference_mode == "local"
    assert result.predicted_score == 75.0
    assert any("FastAPI backend unavailable" in warning for warning in result.warnings)


def test_predict_student_dataframe_via_client_uses_csv_endpoint(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:8000")

    def fake_post_csv(api_base_url, path, csv_bytes, filename):
        assert api_base_url == "http://127.0.0.1:8000"
        assert path == "/batch-predict-csv"
        assert filename == "students.csv"
        assert b"Hours_Studied" in csv_bytes
        return {
            "count": 1,
            "average_score": 81.0,
            "predictions": [
                {
                    "row_id": 1,
                    "predicted_score": 81.0,
                    "predicted_band": "Very Good",
                }
            ],
            "warnings": [],
        }

    monkeypatch.setattr(inference_client, "_post_csv", fake_post_csv)

    result = inference_client.predict_student_dataframe_via_client(
        pd.DataFrame([{"Hours_Studied": 5}])
    )

    assert isinstance(result, BatchPredictionResult)
    assert result.inference_mode == "api"
    assert result.count == 1
    assert result.predictions[0]["predicted_band"] == "Very Good"
