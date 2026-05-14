from fastapi.testclient import TestClient
import pandas as pd

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


def test_batch_predict_csv_endpoint_returns_all_records():
    records = [
        SAMPLE_PROFILES["Balanced Student"],
        SAMPLE_PROFILES["High Performer"],
    ]
    csv_bytes = pd.DataFrame(records).to_csv(index=False).encode("utf-8")

    response = client.post(
        "/batch-predict-csv",
        files={"file": ("students.csv", csv_bytes, "text/csv")},
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["count"] == 2
    assert len(payload["predictions"]) == 2
    assert 0 <= payload["average_score"] <= 100


def test_batch_predict_csv_endpoint_fills_allowed_categorical_missing_values():
    records = [
        SAMPLE_PROFILES["Balanced Student"].copy(),
        SAMPLE_PROFILES["High Performer"].copy(),
    ]
    records[0]["Teacher_Quality"] = ""
    records[0]["Parental_Education_Level"] = ""
    records[1]["Distance_from_Home"] = None
    csv_bytes = pd.DataFrame(records).to_csv(index=False).encode("utf-8")

    response = client.post(
        "/batch-predict-csv",
        files={"file": ("students.csv", csv_bytes, "text/csv")},
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["count"] == 2
    assert any("Teacher_Quality" in warning for warning in payload["warnings"])
    assert any("Parental_Education_Level" in warning for warning in payload["warnings"])
    assert any("Distance_from_Home" in warning for warning in payload["warnings"])


def test_batch_predict_csv_endpoint_allows_numeric_missing_values():
    records = [
        SAMPLE_PROFILES["Balanced Student"].copy(),
        SAMPLE_PROFILES["High Performer"].copy(),
    ]
    records[0]["Hours_Studied"] = ""
    csv_bytes = pd.DataFrame(records).to_csv(index=False).encode("utf-8")

    response = client.post(
        "/batch-predict-csv",
        files={"file": ("students.csv", csv_bytes, "text/csv")},
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["count"] == 2
    assert any("Hours_Studied" in warning for warning in payload["warnings"])


def test_batch_predict_csv_endpoint_rejects_non_csv_file():
    response = client.post(
        "/batch-predict-csv",
        files={"file": ("students.txt", b"not,a,csv", "text/plain")},
    )

    assert response.status_code == 400
    assert "extension" in response.json()["detail"]["errors"][0]


def test_batch_predict_csv_endpoint_rejects_empty_file():
    response = client.post(
        "/batch-predict-csv",
        files={"file": ("students.csv", b"", "text/csv")},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]["errors"][0].lower()


def test_batch_predict_csv_endpoint_rejects_header_only_csv():
    csv_bytes = pd.DataFrame(columns=list(SAMPLE_PROFILES["Balanced Student"].keys())).to_csv(index=False).encode("utf-8")

    response = client.post(
        "/batch-predict-csv",
        files={"file": ("students.csv", csv_bytes, "text/csv")},
    )

    assert response.status_code == 400
    assert "at least one data row" in response.json()["detail"]["errors"][0]


def test_batch_predict_csv_endpoint_rejects_unmatched_schema():
    csv_bytes = pd.DataFrame([{"Wrong_Column": "value"}]).to_csv(index=False).encode("utf-8")

    response = client.post(
        "/batch-predict-csv",
        files={"file": ("students.csv", csv_bytes, "text/csv")},
    )

    assert response.status_code == 400
    assert "does not contain any expected input columns" in response.json()["detail"]["errors"][0]
