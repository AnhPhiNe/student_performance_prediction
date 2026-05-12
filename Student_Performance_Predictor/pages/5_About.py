# pages/5_About.py

import streamlit as st

from src.loader import load_model_assets
from src.ui_components import render_section_title, render_kpi_cards

# =========================================================
# 1) LOAD ASSETS
# =========================================================
try:
    full_pipeline, core_model, raw_feature_names, raw_survivors, best_params = load_model_assets()
except Exception as e:
    st.error(f"Failed to load project assets: {e}")
    st.stop()

# =========================================================
# 2) HEADER
# =========================================================
render_section_title(
    "About This Project",
    "A summary of the technical architecture, design decisions, and portfolio value of this application."
)

# =========================================================
# 3) QUICK SUMMARY
# =========================================================
render_kpi_cards([
    ("App Type", "Multi-page Streamlit App"),
    ("Task", "Regression"),
    ("Raw Features", str(len(raw_feature_names))),
    ("Final Model", "Ridge Regression"),
])

# =========================================================
# 4) PROJECT OVERVIEW
# =========================================================
st.markdown("### Project Overview")
st.markdown("""
This project is an **end-to-end machine learning application** for predicting student exam scores.

It was designed not only to train a model, but also to package the model into a user-facing application
that supports:

- **single prediction**
- **batch prediction**
- **input validation**
- **model interpretation**
- **modular project structure**

The goal is to demonstrate practical machine learning engineering skills in a format suitable for
a personal portfolio and internship applications.
""")

# =========================================================
# 5) TECHNICAL ARCHITECTURE
# =========================================================
st.markdown("### Technical Architecture")
st.code(
    "Raw User Input → Validation → Feature Engineering → Selected Feature Filtering → Preprocessing → Ridge Prediction",
    language="text"
)

st.markdown("""
The project separates responsibilities into different modules:

- **`src/config.py`** → central configuration
- **`src/loader.py`** → loads models, data, and CSS
- **`src/validators.py`** → validates schema and input values
- **`src/predictor.py`** → prediction logic and recommendations
- **`src/explainer.py`** → coefficient-based model interpretation
- **`src/ui_components.py`** → reusable UI building blocks
- **`pages/`** → multi-page Streamlit user interface
""")

# =========================================================
# 6) MODEL ARTIFACTS
# =========================================================
st.markdown("### Model Artifacts Used")
st.markdown("""
This application uses four main artifacts:

#### 1. `hcmue_student_full_pipeline_v1_0.joblib`
Used for **end-to-end prediction**.
It contains:
- feature engineering
- selected feature filtering
- preprocessing
- final Ridge Regression model

#### 2. `ridge_core_model.joblib`
Used for **model interpretation**.
It supports:
- coefficient-based feature importance
- positive / negative driver analysis

#### 3. `raw_feature_names.joblib`
Used as the **input schema contract**.
It tells the app:
- what columns are required
- how the form should be built
- what the uploaded CSV should contain

#### 4. `raw_survivors.joblib`
Stores the selected raw features retained after the feature selection workflow.
""")

if best_params is not None:
    st.markdown("### Best Hyperparameters Snapshot")
    st.json(best_params)
else:
    st.info("No best hyperparameters file was loaded. This is optional for the application.")

# =========================================================
# 7) KEY DESIGN DECISIONS
# =========================================================
st.markdown("### Key Design Decisions")
st.markdown("""
#### Why use a multi-page app?
A multi-page structure makes the app easier to use and easier to maintain.
Instead of putting everything into one file, each page has one main purpose.

#### Why choose Ridge Regression as the final model?
Ridge Regression was selected because it achieved the best balance between:
- predictive performance
- stability
- simplicity
- interpretability

#### Why use a full pipeline artifact?
The full pipeline is safer for real app inference because it includes:
- feature engineering
- selected feature filtering
- preprocessing
- final model prediction

#### Why add validation?
A machine learning app should not blindly trust user input.
Validation improves:
- reliability
- user trust
- error handling
""")

# =========================================================
# 8) LIMITATIONS
# =========================================================
st.markdown("### Current Limitations")
st.markdown("""
No project is perfect. This app still has some limitations:

- High-score samples (especially 80+ and 90+) are relatively scarce in the dataset.
- Coefficient-based interpretation is simpler than local explanation methods such as SHAP.
- The recommendation logic is partly rule-based, not fully model-generated.
- The app is designed for portfolio/demo purposes, not for real educational decision-making.
""")

# =========================================================
# 9) FUTURE IMPROVEMENTS
# =========================================================
st.markdown("### Future Improvements")
st.markdown("""
Potential future upgrades include:

- model versioning
- artifact metadata tracking
- richer coefficient interpretation views
- deployment to Streamlit Cloud
- Docker support
- unit tests for validators and predictors
- monitoring and logging
""")

# =========================================================
# 10) PORTFOLIO VALUE
# =========================================================
st.markdown("### Why This Project Is Valuable for a CV")
st.markdown("""
This project demonstrates more than model training.
It shows the ability to:

- organize a machine learning project with a clean structure
- deploy a model into a user-facing application
- validate user input before inference
- support both single and batch workflows
- communicate model behavior through interpretable outputs

That makes it stronger than a notebook-only project for internship applications.
""")