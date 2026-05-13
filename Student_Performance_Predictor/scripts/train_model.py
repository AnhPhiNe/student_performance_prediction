"""
Train a reproducible Student Performance Predictor model.

This script moves the production-oriented training/export flow out of the
notebook while keeping the same raw-input contract used by the Streamlit app.

Default safety behavior:
- training and evaluation run when the script is executed;
- artifacts are exported unless --no-export is passed;
- existing artifacts are never overwritten silently.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import add_new_features, feature_engineering_for_pipeline


RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TARGET_COLUMN = "Exam_Score"

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "Student_Performance.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"

ARTIFACT_FILENAMES = {
    "full_pipeline": "hcmue_student_full_pipeline_v1_0.joblib",
    "core_model": "ridge_core_model.joblib",
    "raw_feature_names": "raw_feature_names.joblib",
    "raw_survivors": "raw_survivors.joblib",
    "best_hyperparameters": "best_hyperparameters.json",
    "model_metadata": "model_metadata.json",
}

NEW_ENGINEERED_FEATURES = [
    "Study_Efficiency",
    "Total_Study_Time",
    "Engagement_Index",
]

BINARY_OR_NOMINAL_FEATURES = [
    "Extracurricular_Activities",
    "Internet_Access",
    "Learning_Disabilities",
    "Gender",
    "School_Type",
]

ORDINAL_FEATURES = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Motivation_Level",
    "Family_Income",
    "Teacher_Quality",
    "Peer_Influence",
    "Parental_Education_Level",
    "Distance_from_Home",
]

ORDINAL_CATEGORIES = [["missing", "Low", "Medium", "High"]] * 5 + [
    ["missing", "Negative", "Neutral", "Positive"],
    ["missing", "High School", "College", "Postgraduate"],
    ["missing", "Near", "Moderate", "Far"],
]

KNOWN_ONE_HOT_SUFFIXES = [
    "_Yes",
    "_No",
    "_Male",
    "_Female",
    "_Public",
    "_Private",
    "_Urban",
    "_Rural",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and optionally export the Student Performance Predictor artifacts."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the training CSV. Default: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for exported artifacts. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Run training/evaluation only and skip writing model artifacts.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_and_clean_data(data_path: Path) -> pd.DataFrame:
    data_path = data_path.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    logging.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain target column: {TARGET_COLUMN}")

    before_rows = len(df)
    df = df.drop_duplicates()
    after_dedup_rows = len(df)
    df = df[df[TARGET_COLUMN] <= 100].copy()

    logging.info("Rows loaded: %s", before_rows)
    logging.info("Dropped duplicate rows: %s", before_rows - after_dedup_rows)
    logging.info("Dropped rows with Exam_Score > 100: %s", after_dedup_rows - len(df))
    logging.info("Clean dataset shape: %s", df.shape)

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    logging.info("Train shape: %s | Test shape: %s", X_train_raw.shape, X_test_raw.shape)
    return X_train_raw, X_test_raw, y_train, y_test


def get_feature_groups(engineered_df: pd.DataFrame) -> dict[str, list[Any]]:
    original_numeric = engineered_df.select_dtypes(include=["number"]).columns.tolist()
    numeric_raw = [col for col in original_numeric if col not in NEW_ENGINEERED_FEATURES]
    numeric_features = numeric_raw + [col for col in NEW_ENGINEERED_FEATURES if col in engineered_df.columns]

    binary_or_nominal = [col for col in BINARY_OR_NOMINAL_FEATURES if col in engineered_df.columns]

    ordinal_features = []
    ordinal_categories = []
    for index, col in enumerate(ORDINAL_FEATURES):
        if col in engineered_df.columns:
            ordinal_features.append(col)
            ordinal_categories.append(ORDINAL_CATEGORIES[index])

    if len(numeric_features) + len(ordinal_features) + len(binary_or_nominal) == 0:
        raise ValueError("No usable feature groups were found in the training data.")

    logging.info(
        "Feature groups: %s numeric, %s ordinal, %s binary/nominal",
        len(numeric_features),
        len(ordinal_features),
        len(binary_or_nominal),
    )

    return {
        "numeric": numeric_features,
        "ordinal": ordinal_features,
        "ordinal_categories": ordinal_categories,
        "binary_or_nominal": binary_or_nominal,
    }


def build_preprocessor(feature_groups: dict[str, list[Any]]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_groups["numeric"],
            ),
            (
                "ord",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                categories=feature_groups["ordinal_categories"],
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                feature_groups["ordinal"],
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "encoder",
                            OneHotEncoder(
                                drop="first",
                                sparse_output=False,
                                handle_unknown="ignore",
                            ),
                        ),
                    ]
                ),
                feature_groups["binary_or_nominal"],
            ),
        ],
        remainder="drop",
    )


def clean_encoded_feature_name(feature_name: str) -> str:
    raw_name = feature_name.split("__")[-1]
    for suffix in KNOWN_ONE_HOT_SUFFIXES:
        if raw_name.endswith(suffix):
            return raw_name[: -len(suffix)]
    return raw_name


def select_raw_survivor_features(
    X_train_engineered: pd.DataFrame,
    y_train: pd.Series,
    feature_groups: dict[str, list[Any]],
) -> list[str]:
    logging.info("Running train-only feature screening")
    screening_preprocessor = build_preprocessor(feature_groups)
    X_train_encoded = screening_preprocessor.fit_transform(X_train_engineered)

    _, p_values = f_regression(X_train_encoded, y_train)
    mi_scores = mutual_info_regression(X_train_encoded, y_train, random_state=RANDOM_STATE)

    selection_df = pd.DataFrame(
        {
            "feature": X_train_encoded.columns,
            "p_value": p_values,
            "mi_score": mi_scores,
        }
    ).sort_values(by="mi_score", ascending=False)

    selected_mask = (selection_df["p_value"] < 0.05) | (selection_df["mi_score"] > 0.01)
    encoded_selected = selection_df.loc[selected_mask, "feature"].tolist()

    if not encoded_selected:
        raise ValueError("Feature screening selected no features. Check data quality and thresholds.")

    X_selected = X_train_encoded[encoded_selected].copy()
    corr_matrix = X_selected.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    features_to_drop: list[str] = []
    corr_threshold = 0.85

    for col in upper_triangle.columns:
        high_corr_features = upper_triangle.index[upper_triangle[col] > corr_threshold].tolist()
        for row_feature in high_corr_features:
            row_info = selection_df[selection_df["feature"] == row_feature]
            col_info = selection_df[selection_df["feature"] == col]
            if row_info.empty or col_info.empty:
                continue

            row_mi = row_info["mi_score"].iloc[0]
            col_mi = col_info["mi_score"].iloc[0]
            row_p = row_info["p_value"].iloc[0]
            col_p = col_info["p_value"].iloc[0]

            if row_mi > col_mi:
                drop_feature = col
            elif row_mi < col_mi:
                drop_feature = row_feature
            else:
                drop_feature = col if row_p > col_p else row_feature

            if drop_feature not in features_to_drop:
                features_to_drop.append(drop_feature)

    encoded_survivors = [col for col in encoded_selected if col not in features_to_drop]
    raw_survivors = [clean_encoded_feature_name(col) for col in encoded_survivors]
    raw_survivors = list(dict.fromkeys(raw_survivors))

    logging.info("Selected encoded features before correlation filter: %s", len(encoded_selected))
    logging.info("Dropped highly correlated encoded features: %s", len(features_to_drop))
    logging.info("Final raw survivor features: %s", raw_survivors)

    return raw_survivors


def filter_feature_groups_for_survivors(
    feature_groups: dict[str, list[Any]],
    raw_survivors: list[str],
) -> dict[str, list[Any]]:
    ordinal_features = []
    ordinal_categories = []

    for feature, categories in zip(feature_groups["ordinal"], feature_groups["ordinal_categories"]):
        if feature in raw_survivors:
            ordinal_features.append(feature)
            ordinal_categories.append(categories)

    selected_groups = {
        "numeric": [col for col in feature_groups["numeric"] if col in raw_survivors],
        "ordinal": ordinal_features,
        "ordinal_categories": ordinal_categories,
        "binary_or_nominal": [col for col in feature_groups["binary_or_nominal"] if col in raw_survivors],
    }

    if (
        len(selected_groups["numeric"])
        + len(selected_groups["ordinal"])
        + len(selected_groups["binary_or_nominal"])
        == 0
    ):
        raise ValueError("No feature groups remain after feature screening.")

    return selected_groups


def prepare_modeling_matrices(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], ColumnTransformer]:
    X_train_engineered = add_new_features(X_train_raw)
    X_test_engineered = add_new_features(X_test_raw)

    feature_groups = get_feature_groups(X_train_engineered)
    raw_survivors = select_raw_survivor_features(X_train_engineered, y_train, feature_groups)
    selected_feature_groups = filter_feature_groups_for_survivors(feature_groups, raw_survivors)

    selected_preprocessor = build_preprocessor(selected_feature_groups)

    X_train_selected = X_train_engineered[raw_survivors].copy()
    X_test_selected = X_test_engineered[raw_survivors].copy()

    X_train_encoded = selected_preprocessor.fit_transform(X_train_selected)
    X_test_encoded = selected_preprocessor.transform(X_test_selected)

    logging.info("Encoded train shape: %s | Encoded test shape: %s", X_train_encoded.shape, X_test_encoded.shape)
    return X_train_encoded, X_test_encoded, raw_survivors, selected_preprocessor


def tune_random_forest(
    X_train_encoded: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[RandomForestRegressor, float, dict[str, Any]]:
    """
    Tune the same Random Forest search space without putting None in param_grid.

    Passing None inside GridSearchCV param_grid can trigger a numpy.ma cast
    warning while sklearn builds masked parameter arrays. We keep the original
    candidate set by running the max_depth=None candidates through the estimator
    default, then searching numeric max_depth values separately.
    """
    search_spaces = [
        {
            "fixed_params": {"max_depth": None},
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE, max_depth=None),
            "param_grid": {
                "n_estimators": [200, 400],
                "min_samples_leaf": [2, 5],
                "max_features": ["sqrt", 0.8],
            },
        },
        {
            "fixed_params": {},
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE),
            "param_grid": {
                "n_estimators": [200, 400],
                "max_depth": [6, 10],
                "min_samples_leaf": [2, 5],
                "max_features": ["sqrt", 0.8],
            },
        },
    ]

    best_estimator: RandomForestRegressor | None = None
    best_score = -np.inf
    best_params: dict[str, Any] = {}

    for search_space in search_spaces:
        grid = GridSearchCV(
            estimator=search_space["estimator"],
            param_grid=search_space["param_grid"],
            cv=CV_FOLDS,
            scoring="r2",
            n_jobs=-1,
            error_score="raise",
        )
        grid.fit(X_train_encoded, y_train)

        if grid.best_score_ > best_score:
            best_estimator = grid.best_estimator_
            best_score = float(grid.best_score_)
            best_params = {
                **search_space["fixed_params"],
                **grid.best_params_,
            }

    if best_estimator is None:
        raise RuntimeError("Random Forest tuning did not produce a fitted estimator.")

    return best_estimator, best_score, best_params


def tune_and_compare_models(
    X_train_encoded: pd.DataFrame,
    X_test_encoded: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    models: dict[str, Any] = {}
    cv_scores: dict[str, float] = {}
    best_hyperparameters: dict[str, Any] = {}

    logging.info("Training DummyRegressor baseline")
    dummy_model = DummyRegressor(strategy="mean")
    dummy_model.fit(X_train_encoded, y_train)
    models["Dummy Baseline"] = dummy_model
    cv_scores["Dummy Baseline"] = cross_val_score(
        dummy_model,
        X_train_encoded,
        y_train,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
    ).mean()

    logging.info("Training Linear Regression")
    linear_model = LinearRegression()
    linear_model.fit(X_train_encoded, y_train)
    models["Linear Regression"] = linear_model
    cv_scores["Linear Regression"] = cross_val_score(
        linear_model,
        X_train_encoded,
        y_train,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
    ).mean()

    logging.info("Tuning Ridge Regression")
    ridge_grid = GridSearchCV(
        estimator=Ridge(random_state=RANDOM_STATE),
        param_grid={"alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]},
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
    )
    ridge_grid.fit(X_train_encoded, y_train)
    models["Ridge Regression"] = ridge_grid.best_estimator_
    cv_scores["Ridge Regression"] = ridge_grid.best_score_
    best_hyperparameters["Ridge Regression"] = ridge_grid.best_params_

    logging.info("Tuning Lasso Regression")
    lasso_grid = GridSearchCV(
        estimator=Lasso(random_state=RANDOM_STATE, max_iter=10000),
        param_grid={"alpha": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]},
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
    )
    lasso_grid.fit(X_train_encoded, y_train)
    models["Lasso Regression"] = lasso_grid.best_estimator_
    cv_scores["Lasso Regression"] = lasso_grid.best_score_
    best_hyperparameters["Lasso Regression"] = lasso_grid.best_params_

    logging.info("Tuning Random Forest")
    rf_model, rf_score, rf_params = tune_random_forest(
        X_train_encoded=X_train_encoded,
        y_train=y_train,
    )
    models["Random Forest"] = rf_model
    cv_scores["Random Forest"] = rf_score
    best_hyperparameters["Random Forest"] = rf_params

    try:
        from xgboost import XGBRegressor

        logging.info("Tuning XGBoost")
        xgb_grid = GridSearchCV(
            estimator=XGBRegressor(
                random_state=RANDOM_STATE,
                tree_method="hist",
                objective="reg:squarederror",
            ),
            param_grid={
                "n_estimators": [300],
                "learning_rate": [0.03],
                "max_depth": [2, 3],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
                "reg_lambda": [10, 30],
                "min_child_weight": [5, 10],
            },
            cv=CV_FOLDS,
            scoring="r2",
            n_jobs=-1,
        )
        xgb_grid.fit(X_train_encoded, y_train)
        models["XGBoost"] = xgb_grid.best_estimator_
        cv_scores["XGBoost"] = xgb_grid.best_score_
        best_hyperparameters["XGBoost"] = xgb_grid.best_params_
    except ImportError:
        logging.info("XGBoost is not installed. Skipping XGBoost.")

    results = []
    for model_name, model in models.items():
        y_pred_train = model.predict(X_train_encoded)
        y_pred_test = model.predict(X_test_encoded)
        metrics = calculate_metrics(y_test, y_pred_test)

        results.append(
            {
                "model": model_name,
                "cv_r2": float(cv_scores[model_name]),
                "r2_train": float(r2_score(y_train, y_pred_train)),
                "r2_test": metrics["r2"],
                "train_test_gap": float(r2_score(y_train, y_pred_train) - metrics["r2"]),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
            }
        )

    results_df = pd.DataFrame(results).sort_values(by="cv_r2", ascending=False).reset_index(drop=True)
    logging.info("Model comparison complete")
    logging.info("\n%s", results_df.to_string(index=False))

    return results_df, models, best_hyperparameters


def calculate_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def residual_summary(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "min": float(np.min(residuals)),
        "median": float(np.median(residuals)),
        "max": float(np.max(residuals)),
    }


def build_final_pipeline(
    selected_preprocessor: ColumnTransformer,
    raw_survivors: list[str],
    raw_feature_names: list[str],
    ridge_params: dict[str, Any],
) -> Pipeline:
    feature_engineer = FunctionTransformer(
        partial(
            feature_engineering_for_pipeline,
            raw_survivors=raw_survivors,
            raw_feature_names=raw_feature_names,
        ),
        validate=False,
    )

    final_ridge = Ridge(
        alpha=ridge_params.get("alpha", 1.0),
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("feature_eng", feature_engineer),
            ("preprocess", selected_preprocessor),
            ("model", final_ridge),
        ]
    )


def build_metadata(
    data_path: Path,
    metrics: dict[str, float],
    residuals: dict[str, float],
    raw_feature_names: list[str],
    raw_survivors: list[str],
    best_hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        "metrics": metrics,
        "residual_summary": residuals,
        "raw_feature_names": raw_feature_names,
        "selected_raw_survivor_features": raw_survivors,
        "final_model_name": "Ridge Regression",
        "best_hyperparameters": best_hyperparameters,
        "dataset_path": str(data_path.resolve()),
        "limitation_note": (
            "This is a portfolio-oriented tabular regression model. Feature selection uses "
            "train-only statistical screening, and final holdout metrics should be treated as "
            "project evidence rather than a locked production benchmark."
        ),
    }
    return metadata


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {key: output_dir / filename for key, filename in ARTIFACT_FILENAMES.items()}


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def log_export_plan(paths: dict[str, Path]) -> None:
    logging.info("Artifacts prepared for export:")
    for path in paths.values():
        status = "would overwrite existing file" if path.exists() else "new file"
        logging.info("  - %s (%s)", path, status)


def ensure_no_existing_artifacts(paths: dict[str, Path]) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if not existing:
        return

    formatted = "\n".join(f"  - {path}" for path in existing)
    raise FileExistsError(
        "Export would overwrite existing artifacts. Move or back up these files first:\n"
        f"{formatted}"
    )


def export_artifacts(
    output_dir: Path,
    full_pipeline: Pipeline,
    raw_feature_names: list[str],
    raw_survivors: list[str],
    best_hyperparameters: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(output_dir)

    ensure_no_existing_artifacts(paths)

    ridge_model = full_pipeline.named_steps["model"]
    ridge_params = best_hyperparameters.get("Ridge Regression", {})

    joblib.dump(full_pipeline, paths["full_pipeline"])
    joblib.dump(ridge_model, paths["core_model"])
    joblib.dump(raw_feature_names, paths["raw_feature_names"])
    joblib.dump(raw_survivors, paths["raw_survivors"])

    with paths["best_hyperparameters"].open("w", encoding="utf-8") as f:
        json.dump({"model_name": "Ridge", **ridge_params}, f, indent=4, ensure_ascii=False)

    with paths["model_metadata"].open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    logging.info("Artifacts exported to %s", output_dir)


def smoke_test_exported_pipeline(output_dir: Path, sample_input: dict[str, Any]) -> float:
    paths = artifact_paths(output_dir)
    loaded_pipeline = joblib.load(paths["full_pipeline"])
    raw_feature_names = joblib.load(paths["raw_feature_names"])
    sample_df = pd.DataFrame([sample_input], columns=raw_feature_names)
    prediction = float(loaded_pipeline.predict(sample_df)[0])
    logging.info("Smoke test prediction: %.2f", prediction)
    return prediction


def sample_input(raw_feature_names: list[str]) -> dict[str, Any]:
    defaults = {
        "Hours_Studied": 12,
        "Attendance": 100,
        "Parental_Involvement": "High",
        "Access_to_Resources": "High",
        "Extracurricular_Activities": "Yes",
        "Sleep_Hours": 8,
        "Previous_Scores": 100,
        "Motivation_Level": "High",
        "Internet_Access": "Yes",
        "Tutoring_Sessions": 12,
        "Family_Income": "High",
        "Teacher_Quality": "High",
        "School_Type": "Public",
        "Peer_Influence": "Positive",
        "Physical_Activity": 8,
        "Learning_Disabilities": "No",
        "Parental_Education_Level": "Postgraduate",
        "Distance_from_Home": "Near",
        "Gender": "Male",
    }
    return {feature: defaults.get(feature) for feature in raw_feature_names}


def main() -> None:
    configure_logging()
    set_config(transform_output="pandas")

    args = parse_args()
    data_path = resolve_project_path(args.data_path)
    output_dir = resolve_project_path(args.output_dir)

    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Data path: %s", data_path)
    logging.info("Output directory: %s", output_dir)

    df = load_and_clean_data(data_path)
    X_train_raw, X_test_raw, y_train, y_test = split_features_target(df)
    raw_feature_names = list(X_train_raw.columns)

    X_train_encoded, X_test_encoded, raw_survivors, selected_preprocessor = prepare_modeling_matrices(
        X_train_raw,
        X_test_raw,
        y_train,
    )

    results_df, _, best_hyperparameters = tune_and_compare_models(
        X_train_encoded,
        X_test_encoded,
        y_train,
        y_test,
    )

    ridge_params = best_hyperparameters["Ridge Regression"]
    final_pipeline = build_final_pipeline(
        selected_preprocessor=selected_preprocessor,
        raw_survivors=raw_survivors,
        raw_feature_names=raw_feature_names,
        ridge_params=ridge_params,
    )

    logging.info("Training final raw-input Ridge pipeline")
    final_pipeline.fit(X_train_raw, y_train)
    final_predictions = final_pipeline.predict(X_test_raw)
    final_metrics = calculate_metrics(y_test, final_predictions)
    final_residuals = residual_summary(y_test, final_predictions)

    logging.info("Final model is fixed to Ridge Regression for deployability and explainability")
    logging.info("Final raw-input pipeline metrics: %s", final_metrics)
    logging.info("Residual summary: %s", final_residuals)

    metadata = build_metadata(
        data_path=data_path,
        metrics=final_metrics,
        residuals=final_residuals,
        raw_feature_names=raw_feature_names,
        raw_survivors=raw_survivors,
        best_hyperparameters=best_hyperparameters,
    )
    metadata["model_comparison"] = results_df.to_dict(orient="records")

    paths = artifact_paths(output_dir)
    log_export_plan(paths)

    if args.no_export:
        logging.info("Artifact export skipped because --no-export was provided.")
        logging.info("Smoke test skipped because no artifacts were exported.")
        return

    export_artifacts(
        output_dir=output_dir,
        full_pipeline=final_pipeline,
        raw_feature_names=raw_feature_names,
        raw_survivors=raw_survivors,
        best_hyperparameters=best_hyperparameters,
        metadata=metadata,
    )

    # Smoke test is intentionally tied to a successful export. It is not run in Phase 2
    # because this script is only being created, not executed.
    smoke_test_exported_pipeline(output_dir, sample_input(raw_feature_names))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("Training failed: %s", exc)
        raise
