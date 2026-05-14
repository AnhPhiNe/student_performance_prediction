# Student Performance Predictor

Student Performance Predictor is a portfolio-oriented machine learning application that estimates a student's exam score from academic, lifestyle, family, and school-context features.

The project is packaged as a multi-page Streamlit app with single-profile prediction, batch CSV prediction, input validation, saved model artifacts, Ridge coefficient-based model interpretation, and a lightweight FastAPI inference backend.

The Streamlit app supports two inference modes:

```text
Default mode:
Streamlit -> local prediction service -> sklearn pipeline

API-backed mode:
Streamlit -> FastAPI -> prediction service -> sklearn pipeline
```

This keeps the deployed Streamlit app simple while allowing the same project to demonstrate a real HTTP inference boundary when `API_BASE_URL` is configured.

Current deployment status: **deployed**
Live app: [Streamlit App](https://student-performance-predictor-ap.streamlit.app)
Live API: [FastAPI Backend](https://student-performance-prediction-8xv5.onrender.com)

## Project Scope

This is a supervised regression project. The target variable is:

```text
Exam_Score
```

The app is intended for ML engineering demonstration and portfolio review, not for real educational decision-making.

## Tech Stack

- **Language:** Python
- **App:** Streamlit
- **API:** FastAPI, Uvicorn
- **Data:** pandas, NumPy
- **Machine Learning:** scikit-learn
- **Visualization:** Plotly, Matplotlib
- **Serialization:** joblib
- **Testing:** pytest
- **Notebook:** Jupyter Notebook

XGBoost is used only as an optional training comparison in `scripts/train_model.py`; the deployed app uses a Ridge Regression pipeline.

## Folder Structure

```text
+-- app.py
+-- api/
|   +-- __init__.py
|   +-- main.py
|   +-- schemas.py
+-- pages/
|   +-- 1_Home.py
|   +-- 2_Single_Prediction.py
|   +-- 3_Batch_Prediction.py
|   +-- 4_Explainability.py
|   +-- 5_Project_Details.py
+-- src/
|   +-- __init__.py
|   +-- artifact_loader.py
|   +-- config.py
|   +-- inference_client.py
|   +-- prediction_service.py
|   +-- predictor.py
|   +-- validators.py
|   +-- loader.py
|   +-- explainer.py
|   +-- helpers.py
|   +-- ui_components.py
+-- data/
|   +-- Student_Performance.csv
+-- models/
|   +-- hcmue_student_full_pipeline_v1_0.joblib
|   +-- ridge_core_model.joblib
|   +-- raw_feature_names.joblib
|   +-- raw_survivors.joblib
|   +-- best_hyperparameters.json
|   +-- model_metadata.json
+-- notebooks/
|   +-- model_exploration.ipynb
+-- scripts/
|   +-- train_model.py
+-- assets/
|   +-- css/
|       +-- styles.css
+-- tests/
|   +-- test_api.py
|   +-- test_inference_client.py
|   +-- test_model_pipeline.py
+-- requirements.txt
+-- README.md
+-- LICENSE
+-- .gitignore
```

## App Features

- Multi-page Streamlit user interface.
- Single student prediction from an interactive profile form.
- Preset profiles for quick testing.
- Batch prediction from uploaded CSV files.
- Downloadable CSV input template.
- Input schema and value validation.
- Prediction bands: `Excellent`, `Very Good`, `Good`, `Average`, and `Needs Improvement`.
- Rule-based recommendations for the current prediction.
- Ridge coefficient-based model insight page.
- Project Details page for dataset, pipeline, artifacts, limitations, and roadmap.
- FastAPI endpoints for service info, health checks, metadata, single prediction, and batch prediction.
- Optional API-backed Streamlit inference through `API_BASE_URL`, with local inference fallback.

## Model Pipeline

The inference flow is:

```text
Raw input
    -> Validation
    -> Preprocessing
    -> Train-only feature selection
    -> Ridge Regression
    -> Clipped score
    -> Score band and recommendations
```

Main model artifacts:

- `hcmue_student_full_pipeline_v1_0.joblib`: deployable raw-input prediction pipeline.
- `ridge_core_model.joblib`: Ridge model used by the explainability page.
- `raw_feature_names.joblib`: expected raw input schema.
- `raw_survivors.joblib`: selected raw features retained after feature screening.
- `best_hyperparameters.json`: final model hyperparameter snapshot.
- `model_metadata.json`: training environment, metrics, residual summary, and model comparison.

## Training Workflow

The production-oriented training workflow lives in:

```bash
scripts/train_model.py
```

The script performs data loading, cleaning, train/test split, train-only feature screening, model comparison, final Ridge pipeline fitting, artifact export, metadata export, and an exported-pipeline smoke check.

Existing artifacts are not overwritten silently. Move or remove the current files in `models/` before intentionally exporting a fresh model run.

The notebook in `notebooks/model_exploration.ipynb` is kept for EDA and modeling reference.

## Run Locally

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Run the FastAPI backend:

```bash
uvicorn api.main:app --reload
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Use Streamlit with the FastAPI backend:

```powershell
$env:API_BASE_URL="http://127.0.0.1:8000"
streamlit run app.py
```

If `API_BASE_URL` is not set, Streamlit uses the local prediction service directly. If the API is configured but unavailable, the app falls back to local inference and shows a warning.

Run tests:

```bash
python -m pytest tests
```

## FastAPI Backend

The API is a stateless inference service that reuses the same saved sklearn pipeline and prediction service as the Streamlit app.

Available endpoints:

```text
GET  /
GET  /health
GET  /metadata
POST /predict
POST /batch-predict
POST /batch-predict-csv
```

`GET /` returns a compact service status and docs pointer for browser checks.

`POST /predict` accepts one student profile and returns a predicted score, score band, recommendations, and validation warnings.

`POST /batch-predict` accepts a JSON payload with multiple records and returns row-level predictions plus a batch average.

`POST /batch-predict-csv` accepts a UTF-8 CSV upload and returns the same batch prediction response format.

For CSV batch prediction, selected categorical fields with missing values are filled with neutral defaults, while missing numeric values are left for the model pipeline's median imputer.

CSV uploads must use the `.csv` extension, contain at least one data row, and include at least one expected input column before schema validation runs.

The Streamlit UI can use this backend in API-backed mode, but the API is intentionally optional so the portfolio demo remains easy to deploy as a single Streamlit app.

## Current Model Snapshot

The current saved model is Ridge Regression. The active hyperparameters and evaluation metadata are stored under `models/`.

This project reports model metrics as portfolio evidence, not as a locked production benchmark.

## Limitations

- Predictions are correlational and should not be treated as causal explanations.
- The dataset may not generalize to every school system or student population.
- The app is designed for demonstration, not high-stakes academic decisions.
- Local explanations such as SHAP are not part of the current deployed app.

## Roadmap

- Add final screenshots after deployment.
- Expand tests for validators, schema alignment, and batch prediction preparation.
- Add model/data source notes before public release.
