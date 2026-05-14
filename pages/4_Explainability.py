# pages/4_Explainability.py

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.patches import Patch

st.set_page_config(
    page_title="Model Insights | EduPredict",
    page_icon=":mortar_board:",
    layout="wide",
)

from src.loader import load_css, load_model_assets
from src.explainer import build_ridge_coefficient_table, get_top_positive_negative
from src.ui_components import render_page_header


css = load_css()
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

TOP_DRIVER_COUNT = 5
DEFAULT_IMPORTANCE_COUNT = 10


# =========================================================
# 1) LOAD ASSETS
# =========================================================
try:
    full_pipeline, core_model, raw_feature_names, raw_survivors, best_params = load_model_assets()
except Exception as e:
    st.error(f"Failed to load model insight assets: {e}")
    st.stop()


# =========================================================
# 2) SMALL UI HELPERS
# =========================================================
def format_coefficient(value: float) -> str:
    return f"{value:+.3f}"


def build_feature_importance_view(coef_df: pd.DataFrame, top_n: int | None = DEFAULT_IMPORTANCE_COUNT) -> pd.DataFrame:
    view_df = coef_df.sort_values(by="Abs_Coefficient", ascending=False)
    if top_n is not None:
        view_df = view_df.head(top_n)

    return view_df.copy().rename(
        columns={
            "Display_Name": "Driver",
            "Abs_Coefficient": "Importance",
        }
    )[["Driver", "Importance", "Coefficient", "Direction"]]


def render_driver_chart(driver_df: pd.DataFrame, color: str):
    plot_df = driver_df.copy()
    plot_df["Importance"] = plot_df["Coefficient"].abs()
    plot_df = plot_df.sort_values(by="Importance", ascending=True)

    fig_height = max(2.8, len(plot_df) * 0.5)
    fig, ax = plt.subplots(figsize=(6.2, fig_height), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    bars = ax.barh(plot_df["Display_Name"], plot_df["Importance"], color=color)
    max_importance = max(plot_df["Importance"].max(), 0.001)

    for bar, coefficient in zip(bars, plot_df["Coefficient"]):
        ax.text(
            bar.get_width() + max_importance * 0.02,
            bar.get_y() + bar.get_height() / 2,
            format_coefficient(coefficient),
            va="center",
            color="#334155",
            fontsize=8,
        )

    ax.set_xlabel("Relative influence", color="#334155", fontsize=8)
    ax.set_ylabel("")
    ax.set_xlim(0, max_importance * 1.22)
    ax.tick_params(axis="x", colors="#64748b", labelsize=8)
    ax.tick_params(axis="y", colors="#1f2937", labelsize=8)
    ax.grid(axis="x", linestyle="--", color="#cbd5e1", alpha=0.55)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_feature_importance_chart(importance_df: pd.DataFrame):
    plot_df = importance_df.sort_values(by="Importance", ascending=True)
    colors = plot_df["Direction"].map({
        "Positive": "#2563eb",
        "Negative": "#ef4444",
    })

    fig_height = max(3.6, len(plot_df) * 0.32)
    fig, ax = plt.subplots(figsize=(8.5, fig_height), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    bars = ax.barh(plot_df["Driver"], plot_df["Importance"], color=colors)
    max_importance = max(plot_df["Importance"].max(), 0.001)

    for bar, coefficient in zip(bars, plot_df["Coefficient"]):
        ax.text(
            bar.get_width() + max_importance * 0.015,
            bar.get_y() + bar.get_height() / 2,
            format_coefficient(coefficient),
            va="center",
            color="#334155",
            fontsize=8,
        )

    ax.set_xlabel("Overall model influence", color="#334155", fontsize=9)
    ax.set_ylabel("")
    ax.set_xlim(0, max_importance * 1.18)
    ax.tick_params(axis="x", colors="#64748b", labelsize=8)
    ax.tick_params(axis="y", colors="#1f2937", labelsize=8)
    ax.grid(axis="x", linestyle="--", color="#cbd5e1", alpha=0.65)
    legend_items = [
        Patch(facecolor="#2563eb", label="Positive driver"),
        Patch(facecolor="#ef4444", label="Negative driver"),
    ]
    legend = ax.legend(
        handles=legend_items,
        loc="lower right",
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        fontsize=8,
    )
    for text in legend.get_texts():
        text.set_color("#334155")

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# =========================================================
# 3) HEADER
# =========================================================
render_page_header(
    "Model Insights",
    "Explain what factors influence predicted exam scores."
)

st.info(
    "Ridge coefficients summarize how the model uses each feature during prediction."
)

st.warning(
    "Coefficients show association, not causation."
)


# =========================================================
# 4) EXTRACT COEFFICIENTS
# =========================================================
try:
    preprocessor = full_pipeline.named_steps["preprocess"]
    encoded_feature_names = preprocessor.get_feature_names_out().tolist()
except Exception as e:
    st.error(f"Could not extract encoded feature names from the pipeline: {e}")
    st.stop()

coef_df = build_ridge_coefficient_table(core_model, encoded_feature_names)
top_positive, top_negative = get_top_positive_negative(coef_df, top_n=TOP_DRIVER_COUNT)


# =========================================================
# 5) COMPACT MODEL CONTEXT
# =========================================================
model_name = best_params["model_name"] if best_params and "model_name" in best_params else "Ridge Regression"

st.caption(
    f"Method: Ridge coefficients | Model: {model_name} | "
    f"Showing: top {len(top_positive)} positive and top {len(top_negative)} negative drivers"
)


# =========================================================
# 6) TOP DRIVERS
# =========================================================
st.markdown("### Prediction Drivers")
st.caption(
    f"Directional view: showing only the strongest top {TOP_DRIVER_COUNT} positive and negative drivers."
)
positive_col, negative_col = st.columns(2, gap="large")

with positive_col:
    st.markdown("#### Top Positive Drivers")
    st.caption("Positive coefficients are associated with higher predicted scores.")
    if top_positive.empty:
        st.info("No positive drivers were found in the coefficient table.")
    else:
        render_driver_chart(top_positive, color="#2563eb")

with negative_col:
    st.markdown("#### Top Negative Drivers")
    st.caption("Negative coefficients are associated with lower predicted scores.")
    if top_negative.empty:
        st.info("No negative drivers were found in the coefficient table.")
    else:
        render_driver_chart(top_negative, color="#ef4444")


# =========================================================
# 7) HOW TO INTERPRET
# =========================================================
st.markdown("### How to interpret this")

interpret_col1, interpret_col2, interpret_col3 = st.columns(3, gap="large")
with interpret_col1:
    st.markdown(
        """
        **Direction**

        Positive drivers are associated with higher predicted scores. Negative drivers are associated with lower predicted scores.
        """
    )

with interpret_col2:
    st.markdown(
        """
        **Influence**

        Longer bars mean stronger model influence, based on coefficient size after preprocessing.
        """
    )

with interpret_col3:
    st.markdown(
        """
        **Limit**

        These are model associations, not proof of real-world cause and effect.
        """
    )


# =========================================================
# 8) OVERALL FEATURE IMPORTANCE
# =========================================================
with st.expander("View overall feature importance", expanded=False):
    show_all_drivers = st.checkbox(f"Show all {len(coef_df)} drivers", value=False)
    importance_count = None if show_all_drivers else DEFAULT_IMPORTANCE_COUNT
    importance_df = build_feature_importance_view(coef_df, top_n=importance_count)
    view_label = f"all {len(coef_df)} drivers" if show_all_drivers else f"top {DEFAULT_IMPORTANCE_COUNT} of {len(coef_df)} drivers"

    st.markdown(
        (
            "<p style='text-align:center; color:#64748b; font-size:0.92rem; margin:0.2rem 0 0.9rem 0;'>"
            f"Showing {view_label}. Overall influence is based on the absolute Ridge coefficient."
            "</p>"
        ),
        unsafe_allow_html=True,
    )
    render_feature_importance_chart(importance_df)
