# Student Performance Predictor

Student Performance Predictor is a portfolio-oriented machine learning application that estimates a student's exam score from academic, lifestyle, family, and school-context features.

The project is packaged as a multi-page Streamlit app with single-profile prediction, batch CSV prediction, input validation, saved model artifacts, and Ridge coefficient-based model interpretation.

Current deployment status: **deployed**
Live app: [Streamlit App](https://your-app-name.streamlit.app)

## Project Scope

This is a supervised regression project. The target variable is:

```text
Exam_Score
```

The app is intended for ML engineering demonstration and portfolio review, not for real educational decision-making.

## Tech Stack

- **Language:** Python
- **App:** Streamlit
- **Data:** pandas, NumPy
- **Machine Learning:** scikit-learn
- **Visualization:** Plotly, Matplotlib
- **Serialization:** joblib
- **Testing:** pytest
- **Notebook:** Jupyter Notebook

XGBoost is used only as an optional training comparison in `scripts/train_model.py`; the deployed app uses a Ridge Regression pipeline.

## Folder Structure

```text
Student_Performance_Predictor/
+-- app.py
+-- pages/
|   +-- 1_Home.py
|   +-- 2_Single_Prediction.py
|   +-- 3_Batch_Prediction.py
|   +-- 4_Explainability.py
|   +-- 5_Project_Details.py
+-- src/
|   +-- __init__.py
|   +-- config.py
|   +-- feature_engineering.py
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

## Model Pipeline

The inference flow is:

```text
Raw input
    -> Validation
    -> Feature engineering
    -> Selected feature filtering
    -> Preprocessing
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

The script performs data loading, cleaning, train/test split, feature engineering, train-only feature screening, model comparison, final Ridge pipeline fitting, artifact export, metadata export, and an exported-pipeline smoke check.

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

Run tests:

```bash
python -m pytest tests
```

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
- Add deployment instructions and public app URL.
- Add a small FastAPI layer only if API demonstration becomes a project goal.
- Expand tests for validators, schema alignment, and batch prediction preparation.
- Add model/data source notes before public release.
