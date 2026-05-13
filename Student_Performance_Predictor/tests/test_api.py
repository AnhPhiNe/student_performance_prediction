from fastapi.testclient import TestClient

from api.main import app
from src.config import SAMPLE_PROFILES


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metadata_endpoint_contains_model_contract():
    response = client.get("/metadata")
    payload = response.json()

    assert response.status_code == 200
    assert payload["model_name"] == "Ridge Regression"
    assert payload["target"] == "Exam_Score"
    assert "Hours_Studied" in payload["features"]
    assert len(payload["features"]) == 19


def test_predict_endpoint_returns_valid_prediction():
    response = client.post("/predict", json=SAMPLE_PROFILES["Balanced Student"])
    payload = response.json()

    assert response.status_code == 200
    assert 0 <= payload["predicted_score"] <= 100
    assert payload["predicted_band"] in [
        "Excellent",
        "Very Good",
        "Good",
        "Average",
        "Needs Improvement",
    ]
    assert payload["recommendations"]


def test_predict_endpoint_rejects_invalid_input():
    invalid_profile = SAMPLE_PROFILES["Balanced Student"].copy()
    invalid_profile["Attendance"] = 150

    response = client.post("/predict", json=invalid_profile)

    assert response.status_code == 422


def test_batch_predict_endpoint_returns_all_records():
    records = [
        SAMPLE_PROFILES["Balanced Student"],
        SAMPLE_PROFILES["High Performer"],
        SAMPLE_PROFILES["At-Risk Student"],
    ]

    response = client.post("/batch-predict", json={"records": records})
    payload = response.json()

    assert response.status_code == 200
    assert payload["count"] == 3
    assert len(payload["predictions"]) == 3
    assert 0 <= payload["average_score"] <= 100
