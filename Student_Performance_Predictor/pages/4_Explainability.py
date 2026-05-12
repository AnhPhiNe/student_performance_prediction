# pages/4_Explainability.py

import matplotlib.pyplot as plt
import streamlit as st

from src.loader import load_model_assets
from src.explainer import build_ridge_coefficient_table, get_top_positive_negative
from src.ui_components import render_section_title, render_empty_state

# =========================================================
# 1) LOAD ASSETS
# =========================================================
try:
    full_pipeline, core_model, raw_feature_names, raw_survivors, best_params = load_model_assets()
except Exception as e:
    st.error(f"Failed to load explainability assets: {e}")
    st.stop()

# =========================================================
# 2) HEADER
# =========================================================
render_section_title(
    "Model Explainability",
    "Interpret the final Ridge Regression model through coefficient-based insights."
)

# =========================================================
# 3) EXTRACT ENCODED FEATURE NAMES
# =========================================================
try:
    preprocessor = full_pipeline.named_steps["preprocess"]
    encoded_feature_names = preprocessor.get_feature_names_out().tolist()
except Exception as e:
    st.error(f"Could not extract encoded feature names from the pipeline: {e}")
    st.stop()

# =========================================================
# 4) BUILD COEFFICIENT TABLE
# =========================================================
coef_df = build_ridge_coefficient_table(core_model, encoded_feature_names)
top_positive, top_negative = get_top_positive_negative(coef_df, top_n=10)

# =========================================================
# 5) QUICK SUMMARY
# =========================================================
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Raw Features", len(raw_feature_names))
with c2:
    st.metric("Selected Raw Survivors", len(raw_survivors))
with c3:
    model_name = best_params["model_name"] if best_params and "model_name" in best_params else "Ridge"
    st.metric("Final Model", model_name)

# =========================================================
# 6) TABS
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "Top Positive Drivers",
    "Top Negative Drivers",
    "Full Coefficient Table"
])

with tab1:
    st.subheader("Top Positive Drivers")
    st.caption("These features push the predicted score upward when their values increase or when that category is active.")
    st.dataframe(top_positive[["Display_Name", "Coefficient"]], width="stretch", hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = top_positive.sort_values(by="Coefficient", ascending=True)
    ax.barh(plot_df["Display_Name"], plot_df["Coefficient"])
    ax.set_title("Top Positive Coefficients")
    ax.set_xlabel("Coefficient")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    st.pyplot(fig)

with tab2:
    st.subheader("Top Negative Drivers")
    st.caption("These features are associated with lower predicted scores when they increase or when that category is active.")
    st.dataframe(top_negative[["Display_Name", "Coefficient"]], width="stretch", hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = top_negative.sort_values(by="Coefficient", ascending=True)
    ax.barh(plot_df["Display_Name"], plot_df["Coefficient"])
    ax.set_title("Top Negative Coefficients")
    ax.set_xlabel("Coefficient")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    st.pyplot(fig)

with tab3:
    st.subheader("Full Coefficient Table")
    st.dataframe(coef_df, width="stretch", hide_index=True)

st.info(
    "Because the final model is Ridge Regression, explainability is based on learned coefficients rather than SHAP tree explanations. "
    "Positive coefficients support higher predicted scores, while negative coefficients push predictions downward."
)