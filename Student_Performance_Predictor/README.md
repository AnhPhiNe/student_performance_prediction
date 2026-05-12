# Student Performance Predictor

> Draft README for a portfolio machine learning project. This document describes the current state of the repository and will be refined as the project evolves.

## 1. Project Overview

Student Performance Predictor is an end-to-end machine learning application that predicts a student's exam score from academic, lifestyle, family, and school-related factors.

The project is built as a multi-page Streamlit app with a saved machine learning pipeline, input validation, single-student prediction, batch prediction, and coefficient-based model interpretation.

Current deployment status: **not deployed yet**.

## 2. Problem Statement

The goal is to estimate a student's final exam score using structured input features such as study hours, attendance, previous scores, sleep hours, tutoring sessions, motivation level, parental involvement, school type, and other contextual factors.

This is a **supervised regression problem** where the target variable is:

```text
Exam_Score
```

## 3. Tech Stack

- **Language:** Python
- **App Framework:** Streamlit
- **Data Processing:** pandas, NumPy
- **Machine Learning:** scikit-learn, XGBoost
- **Visualization:** Plotly, Matplotlib
- **Model Serialization:** joblib
- **Notebook Workflow:** Jupyter Notebook
- **Testing:** pytest-compatible test structure

Note: SHAP is listed as a dependency, but the current explainability page uses **Ridge coefficient-based interpretation**, not a complete SHAP workflow.

## 4. Current Folder Structure

```text
Student_Performance_Predictor/
+-- app.py
+-- pages/
|   +-- 1_Home.py
|   +-- 2_Single_Prediction.py
|   +-- 3_Batch_Prediction.py
|   +-- 4_Explainability.py
|   +-- 5_About.py
+-- src/
|   +-- __init__.py
|   +-- config.py
|   +-- loader.py
|   +-- predictor.py
|   +-- validators.py
|   +-- feature_engineering.py
|   +-- explainer.py
|   +-- helpers.py
|   +-- ui_components.py
|   +-- utils.py
+-- data/
|   +-- Student_Performance.csv
+-- models/
|   +-- hcmue_student_full_pipeline_v1_0.joblib
|   +-- ridge_core_model.joblib
|   +-- raw_feature_names.joblib
|   +-- raw_survivors.joblib
|   +-- best_hyperparameters.json
+-- notebooks/
|   +-- 01_eda_model_training.ipynb
+-- artifacts/
|   +-- figures/
|   +-- model_comparison.csv
+-- assets/
|   +-- css/
|   +-- icons/
|   +-- images/
+-- tests/
|   +-- test_pipeline_warning.py
+-- requirements.txt
+-- README.md
+-- .gitignore
```

## 5. Current Features

- Multi-page Streamlit application.
- Home page with dataset preview and score distribution.
- Single student prediction using an interactive form.
- Preset student profiles for quick testing.
- Batch prediction from uploaded CSV files.
- Downloadable CSV template for batch inference.
- Input schema and value validation before prediction.
- Prediction result bands such as `Excellent`, `Very Good`, `Good`, `Average`, and `Needs Improvement`.
- Basic rule-based recommendations based on input values.
- Ridge Regression coefficient-based model interpretation.
- Modular code structure under `src/`.

## 6. Model Pipeline

The current inference flow is:

```text
Raw User Input
    -> Input Validation
    -> Feature Engineering
    -> Selected Feature Filtering
    -> Preprocessing
    -> Ridge Regression Prediction
    -> Score Clipping to 0-100
    -> Result Band + Recommendations
```

Main model artifacts:

- `hcmue_student_full_pipeline_v1_0.joblib`: full prediction pipeline.
- `ridge_core_model.joblib`: Ridge model used for coefficient-based interpretation.
- `raw_feature_names.joblib`: required raw input schema.
- `raw_survivors.joblib`: selected raw features retained by the pipeline.
- `best_hyperparameters.json`: saved best model metadata/hyperparameters snapshot.

## 7. How to Run Locally

Clone the repository and move into the project folder:

```bash
cd Student_Performance_Predictor
```

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

## 8. Screenshots Placeholder

Screenshots will be added after UI review.

Planned screenshot slots:

- Home page
- Single prediction page
- Batch prediction page
- Explainability page
- About page

## 9. Planned Improvements

- Add a more complete test suite for validators and prediction utilities.
- Improve README with actual screenshots.
- Add deployment instructions after the app is deployed.
- Add model training and evaluation scripts outside the notebook.
- Add clearer model/version metadata for saved artifacts.
- Add Docker support.
- Add Streamlit secrets/config documentation if needed.
- Explore SHAP-based explainability as a future enhancement.

## 10. CV Bullet Points

- Built an end-to-end student exam score prediction app using Python, scikit-learn, and Streamlit.
- Packaged a trained Ridge Regression pipeline into a multi-page app supporting single and batch inference.
- Implemented input validation, feature engineering, model loading, prediction logic, and coefficient-based interpretation in modular `src/` components.
- Designed a portfolio-ready ML project structure with notebooks, model artifacts, app pages, tests, and reusable utility modules.
