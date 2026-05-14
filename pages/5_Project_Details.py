# pages/5_Project_Details.py

import json
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

st.set_page_config(
    page_title="Project Details | EduPredict",
    page_icon=":mortar_board:",
    layout="wide",
)

from src.config import BASE_DIR, FORM_GROUPS
from src.loader import load_css, load_dataset, load_model_assets


css = load_css()
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# =========================================================
# 1) LOAD OPTIONAL PROJECT METADATA
# =========================================================
models_dir = Path(BASE_DIR) / "models"
metadata_path = models_dir / "model_metadata.json"

df = None
raw_feature_names: list[str] = []
raw_survivors: list[str] = []
best_params: dict[str, Any] | None = None
model_metadata: dict[str, Any] | None = None
metadata_source_path: Path | None = None
model_name = "Ridge Regression"
required_metrics = ("r2", "mae", "rmse")

try:
    df = load_dataset()
except Exception:
    df = None

try:
    _, _, raw_feature_names, raw_survivors, best_params = load_model_assets()
    if best_params:
        configured_model_name = best_params.get("model_name")
        if configured_model_name and configured_model_name != "Ridge":
            model_name = str(configured_model_name)
except Exception:
    raw_feature_names = []
    raw_survivors = []
    best_params = None

if metadata_path.exists():
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            candidate_metadata = json.load(f)

        metrics = candidate_metadata.get("metrics") if isinstance(candidate_metadata, dict) else None
        if isinstance(metrics, dict):
            for metric_name in required_metrics:
                float(metrics[metric_name])

            model_metadata = candidate_metadata
            metadata_source_path = metadata_path
            model_name = str(candidate_metadata.get("final_model_name", model_name))
    except Exception:
        model_metadata = None
        metadata_source_path = None


# =========================================================
# 2) SMALL UI HELPERS
# =========================================================
def metric_from_metadata(metric_name: str, decimals: int = 3) -> str:
    metrics = model_metadata.get("metrics") if isinstance(model_metadata, dict) else None
    if not isinstance(metrics, dict):
        return "Not available"

    value = metrics.get(metric_name)
    if value is None:
        return "Not available"

    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "Not available"


def dataset_rows() -> str:
    if df is not None:
        return f"{len(df):,}"
    return "6,607"


def feature_count() -> str:
    if raw_feature_names:
        return str(len(raw_feature_names))
    return "19"


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    header_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    row_html = []
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        row_html.append(f"<tr>{cells}</tr>")

    return f"""
    <table class="ep-doc-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{''.join(row_html)}</tbody>
    </table>
    """


def render_doc_table(headers: list[str], rows: list[list[str]]):
    st.markdown(html_table(headers, rows), unsafe_allow_html=True)


def label_value_rows_html(rows: list[tuple[str, str]]) -> str:
    row_html = "".join(
        (
            "<div class='ep-doc-row'>"
            f"<span>{escape(label)}</span>"
            f"<strong>{value}</strong>"
            "</div>"
        )
        for label, value in rows
    )
    return f"<div class='ep-doc-rows'>{row_html}</div>"


def render_badges(labels: list[str]):
    badge_html = "".join(f"<span>{escape(label)}</span>" for label in labels)
    st.markdown(f"<div class='ep-doc-badges'>{badge_html}</div>", unsafe_allow_html=True)


def render_pipeline_flow(steps: list[str]):
    step_html = []
    for index, step in enumerate(steps):
        step_html.append(f"<span class='ep-flow-step'>{escape(step)}</span>")
        if index < len(steps) - 1:
            step_html.append("<span class='ep-flow-arrow'>&rarr;</span>")

    st.markdown(f"<div class='ep-flow-row'>{''.join(step_html)}</div>", unsafe_allow_html=True)


def render_section_title(title: str, caption: str | None = None):
    st.markdown(f"<h2 class='ep-doc-section-title'>{escape(title)}</h2>", unsafe_allow_html=True)
    if caption:
        st.markdown(f"<p class='ep-doc-section-caption'>{escape(caption)}</p>", unsafe_allow_html=True)


def render_checklist(items: list[str]):
    item_html = "".join(f"<li>{escape(item)}</li>" for item in items)
    st.markdown(f"<ul class='ep-doc-checklist'>{item_html}</ul>", unsafe_allow_html=True)


def code_list(items: list[str]) -> str:
    return ", ".join(f"<code>{escape(item)}</code>" for item in items)


st.markdown(
    """
    <style>
        .ep-doc-hero {
            margin: 0 0 1.2rem 0;
            padding: 0.2rem 0 0.35rem 0;
        }

        .ep-doc-title {
            margin: 0 0 0.45rem 0;
            color: var(--ep-text);
            font-size: 2.45rem;
            font-weight: 850;
            line-height: 1.12;
        }

        .ep-doc-subtitle {
            max-width: 860px;
            margin: 0;
            color: var(--ep-muted);
            font-size: 1rem;
            line-height: 1.6;
        }

        .ep-doc-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.8rem;
        }

        .ep-doc-badges span {
            display: inline-flex;
            align-items: center;
            min-height: 1.8rem;
            padding: 0.28rem 0.68rem;
            border: 1px solid #dbe3ef;
            border-radius: 999px;
            background: #ffffff;
            color: #334155;
            font-size: 0.82rem;
            font-weight: 750;
        }

        .ep-doc-section-title {
            margin: 1.55rem 0 0.35rem 0;
            color: var(--ep-text);
            font-size: 1.34rem;
            font-weight: 850;
            line-height: 1.25;
        }

        .ep-doc-section-caption {
            margin: 0 0 0.7rem 0;
            color: var(--ep-muted);
            font-size: 0.94rem;
            line-height: 1.55;
        }

        .ep-doc-panel {
            height: 100%;
            min-height: 232px;
            padding: 0.95rem 1rem;
            border: 1px solid var(--ep-border);
            border-radius: var(--ep-radius);
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.035);
            box-sizing: border-box;
        }

        .ep-doc-panel h3 {
            margin: 0 0 0.62rem 0;
            color: var(--ep-text);
            font-size: 1rem;
            font-weight: 850;
        }

        .ep-doc-rows {
            display: grid;
            gap: 0;
            border: 1px solid var(--ep-border);
            border-radius: var(--ep-radius);
            overflow: hidden;
            background: #ffffff;
        }

        .ep-doc-row {
            display: grid;
            grid-template-columns: minmax(140px, 0.75fr) minmax(0, 1.25fr);
            gap: 0.75rem;
            padding: 0.62rem 0.72rem;
            border-top: 1px solid #eef2f7;
            align-items: center;
        }

        .ep-doc-row:first-child {
            border-top: 0;
        }

        .ep-doc-row span {
            color: var(--ep-muted);
            font-size: 0.84rem;
            font-weight: 750;
        }

        .ep-doc-row strong {
            color: var(--ep-text);
            font-size: 0.95rem;
            font-weight: 800;
            line-height: 1.35;
        }

        .ep-doc-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            overflow: hidden;
            border: 1px solid var(--ep-border);
            border-radius: var(--ep-radius);
            background: #ffffff;
            font-size: 0.93rem;
        }

        .ep-doc-table th {
            text-align: left;
            padding: 0.62rem 0.72rem;
            background: #f8fafc;
            border-bottom: 1px solid var(--ep-border);
            color: #475569;
            font-size: 0.82rem;
            font-weight: 800;
        }

        .ep-doc-table td {
            vertical-align: top;
            padding: 0.62rem 0.72rem;
            border-bottom: 1px solid #eef2f7;
            color: #334155;
            line-height: 1.45;
        }

        .ep-doc-table tbody tr:last-child td {
            border-bottom: 0;
        }

        .ep-doc-table code {
            color: #1e3a8a;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 6px;
            padding: 0.1rem 0.3rem;
            font-size: 0.84rem;
        }

        .ep-doc-checklist {
            display: grid;
            gap: 0.42rem;
            margin: 0.2rem 0 0 0;
            padding: 0;
            list-style: none;
        }

        .ep-doc-checklist li {
            position: relative;
            padding: 0.48rem 0.65rem 0.48rem 1.95rem;
            border: 1px solid #e6edf6;
            border-radius: var(--ep-radius);
            background: #ffffff;
            color: #334155;
            font-size: 0.93rem;
            line-height: 1.4;
        }

        .ep-doc-checklist li:before {
            content: "";
            position: absolute;
            left: 0.72rem;
            top: 0.78rem;
            width: 0.42rem;
            height: 0.42rem;
            border-radius: 999px;
            background: var(--ep-primary);
        }

        .ep-doc-note {
            margin: 0.2rem 0 0.55rem 0;
            padding: 0.72rem 0.85rem;
            border: 1px solid #fed7aa;
            border-radius: var(--ep-radius);
            background: #fff7ed;
            color: #7c2d12;
            font-size: 0.9rem;
            line-height: 1.45;
        }

        .ep-doc-footer {
            margin-top: 1rem;
            padding-top: 0.8rem;
            border-top: 1px solid var(--ep-border);
            color: #475569;
            font-size: 0.94rem;
            line-height: 1.5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3) HEADER / HERO
# =========================================================
st.markdown(
    """
    <section class="ep-doc-hero">
        <h1 class="ep-doc-title">Project Details</h1>
        <p class="ep-doc-subtitle">
            Technical overview of the dataset, model pipeline, evaluation, and deployment workflow behind EduPredict.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)
render_badges([
    "Ridge Regression",
    "6.6K Records",
    "19 Input Features",
    "Preprocessing Pipeline",
    "FastAPI Backend",
])


# =========================================================
# 4) EXECUTIVE SUMMARY
# =========================================================
render_section_title(
    "Executive Summary",
    "A recruiter-facing snapshot of the product goal, model choice, and evaluation status.",
)

goal_col, model_col = st.columns([1.05, 0.95], gap="large")
with goal_col:
    goal_rows = label_value_rows_html(
        [
            ("Objective", "Predict student exam performance"),
            ("Data", "Structured tabular inputs"),
            ("Inference", "Single profile and batch CSV scoring"),
            ("Product Lens", "Inspectable machine learning workflow"),
        ]
    )
    st.markdown(
        f"""
        <div class="ep-doc-panel">
            <h3>Project Goal</h3>
            {goal_rows}
        </div>
        """,
        unsafe_allow_html=True,
    )

with model_col:
    final_model_rows = label_value_rows_html(
        [
            ("Model", escape(model_name)),
            ("R2", metric_from_metadata("r2")),
            ("MAE", metric_from_metadata("mae", decimals=2)),
            ("RMSE", metric_from_metadata("rmse", decimals=2)),
        ]
    )
    st.markdown(
        f"""
        <div class="ep-doc-panel">
            <h3>Final Model</h3>
            {final_model_rows}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 5) DATASET OVERVIEW
# =========================================================
render_section_title("Data & Model", "Dataset shape, target contract, and feature taxonomy.")

render_doc_table(
    ["Item", "Value"],
    [
        ["Rows", dataset_rows()],
        ["Target", "<code>Exam_Score</code>"],
        ["Inputs", feature_count()],
        ["Input Type", "Structured Tabular Data"],
    ],
)

st.markdown("#### Feature Categories")
render_doc_table(
    ["Category", "Description", "Count"],
    [
        ["Academic Habits", "Study behavior and academic history", "4"],
        ["Lifestyle", "Sleep, activity, and motivation indicators", "4"],
        ["Family & Resources", "Household support and access context", "5"],
        ["School & Personal", "School environment and student-specific context", "6"],
    ],
)

with st.expander("View raw feature names", expanded=False):
    render_doc_table(
        ["Category", "Raw features"],
        [
            ["Academic Habits", code_list(FORM_GROUPS["Academic Habits"])],
            ["Lifestyle", code_list(FORM_GROUPS["Lifestyle"])],
            ["Family & Resources", code_list(FORM_GROUPS["Family & Learning Environment"])],
            ["School & Personal", code_list(FORM_GROUPS["School & Personal Context"])],
        ],
    )
    if raw_survivors:
        st.caption(f"Selected raw features available: {len(raw_survivors)}.")
    else:
        st.caption("Selected raw features are shown when raw_survivors.joblib is available.")


# =========================================================
# 6) MODEL PIPELINE
# =========================================================
render_section_title("Pipeline & Artifacts", "Compact view of inference flow and saved production artifacts.")

render_pipeline_flow(
    [
        "Streamlit or API Input",
        "Inference Adapter",
        "Validation",
        "Preprocessing",
        "Feature Selection",
        "Ridge Regression",
        "Predicted Score",
    ]
)

st.markdown("#### Pipeline Notes")
render_checklist(
    [
        "Numeric median imputation",
        "Ordinal/categorical encoding",
        "Train-only feature selection",
        "Raw-input inference pipeline",
        "Optional FastAPI-backed inference mode",
        "Schema-aligned batch validation",
    ]
)

st.markdown("#### Artifacts & Reproducibility")
render_doc_table(
    ["Artifact", "Purpose"],
    [
        ["<code>hcmue_student_full_pipeline_v1_0.joblib</code>", "deployable inference pipeline"],
        ["<code>ridge_core_model.joblib</code>", "coefficient-based insight support"],
        ["<code>raw_feature_names.joblib</code>", "input schema contract"],
        ["<code>raw_survivors.joblib</code>", "selected raw features"],
        ["<code>best_hyperparameters.json</code>", "selected configuration"],
        ["<code>model_metadata.json</code>", "metrics and metadata"],
    ],
)

render_checklist(
    [
        "Dedicated training script",
        "random_state=42",
        "Export overwrite protection",
        "Notebook separated from production training",
    ]
)


# =========================================================
# 7) RELIABILITY, LIMITATIONS, ROADMAP
# =========================================================
render_section_title("Limitations & Responsible Use")

render_checklist(
    [
        "Predictions are correlational, not causal explanations",
        "Dataset limitations may affect generalization",
        "Not for real educational decision-making",
    ]
)

render_section_title(
    "Next Improvements",
    "Focused engineering follow-ups for model reliability, deployment, and maintainability.",
)

render_doc_table(
    ["Area", "Planned improvement"],
    [
        ["Validation", "Schema tests"],
        ["Model lifecycle", "Model versioning"],
        ["Operations", "Lightweight monitoring"],
        ["Explainability", "Local explanation support"],
        ["Deployment", "Optional public API hosting"],
    ],
)

st.markdown(
    """
    <div class="ep-doc-footer">
        <strong>GitHub Repository:</strong> <a href="https://github.com/AnhPhiNe/student_performance_prediction" target="_blank" rel="noopener noreferrer">student_performance_prediction</a><br>
        <strong>Author:</strong> Nguyen Anh Phi
    </div>
    """,
    unsafe_allow_html=True,
)
