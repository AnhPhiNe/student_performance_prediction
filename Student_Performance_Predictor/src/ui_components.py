# src/ui_components.py

import streamlit as st


def render_hero_section(title: str, subtitle: str, badge_text: str | None = None):
    badge_html = f"<div class='ep-badge'>{badge_text}</div>" if badge_text else ""

    st.markdown(
        f"""
        <section class="ep-hero">
            {badge_html}
            <h1 class="ep-hero-title">{title}</h1>
            <p class="ep-hero-subtitle">{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title: str, caption: str | None = None):
    st.markdown(f"<h2 class='ep-section-title'>{title}</h2>", unsafe_allow_html=True)
    if caption:
        st.markdown(f"<p class='ep-section-caption'>{caption}</p>", unsafe_allow_html=True)


def render_kpi_cards(kpis: list[tuple[str, str]]):
    cols = st.columns(len(kpis), gap="medium")
    for col, (label, value) in zip(cols, kpis):
        with col:
            st.markdown(
                f"""
                <div class="ep-kpi-card">
                    <div class="ep-kpi-label">{label}</div>
                    <div class="ep-kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_feature_cards(cards: list[tuple[str, str, str]]):
    cols = st.columns(len(cards), gap="medium")
    for col, (icon, title, desc) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="ep-feature-card">
                    <div class="ep-feature-icon">{icon}</div>
                    <div class="ep-feature-title">{title}</div>
                    <div class="ep-feature-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_workflow_strip(steps: list[str]):
    step_html = []
    for index, step in enumerate(steps):
        step_html.append(f"<span class='ep-flow-step'>{step}</span>")
        if index < len(steps) - 1:
            step_html.append("<span class='ep-flow-arrow'>/</span>")

    st.markdown(
        f"""
        <div class="ep-flow-row">
            {''.join(step_html)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(score_text: str, band: str, band_color: str):
    st.markdown(
        f"""
        <div class="ep-result-card">
            <div class="ep-result-label">Predicted Score</div>
            <div class="ep-result-score">{score_text}</div>
            <div class="ep-result-band" style="border-color:{band_color}; color:{band_color};">
                {band}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(message: str):
    st.markdown(
        f"""
        <div class="ep-empty-state">{message}</div>
        """,
        unsafe_allow_html=True,
    )


def render_next_step_cards():
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div class="ep-next-card">
                <div class="ep-next-title">Single Prediction</div>
                <div class="ep-next-desc">Build one profile and estimate the expected exam score instantly.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="ep-next-card">
                <div class="ep-next-title">Batch Prediction</div>
                <div class="ep-next-desc">Upload a CSV file, validate the schema, and score multiple students at once.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_root_welcome():
    st.caption("ML Portfolio Project")
    st.title("EduPredict")
    st.subheader("Student Performance Predictor")
    st.write(
        "A compact Streamlit app for estimating student exam performance from academic, "
        "lifestyle, family, and school-related factors."
    )

    st.info(
        "Use the sidebar navigation to open Home, Single Prediction, Batch Prediction, "
        "Explainability, or About. Start with Home for context, or Single Prediction "
        "to try the model immediately."
    )


def render_sidebar_summary(raw_feature_names: list[str] | None = None):
    feature_count = len(raw_feature_names) if raw_feature_names is not None else None
    feature_line = (
        f"<div><strong>Input Features</strong><br>{feature_count}</div>"
        if feature_count is not None
        else "<div><strong>Input Schema</strong><br>Loaded on prediction pages</div>"
    )

    with st.sidebar:
        st.markdown(
            """
            <div class="ep-sidebar-brand">
                <div class="ep-sidebar-logo">EP</div>
                <div>
                    <div class="ep-sidebar-title">EduPredict</div>
                    <div class="ep-sidebar-subtitle">Student Performance Predictor</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            f"""
            <div class="ep-sidebar-mini">
                <div><strong>Model</strong><br>Ridge Regression</div>
                {feature_line}
                <div><strong>Output</strong><br>Exam Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            """
            <div class="ep-sidebar-nav-note">
                Navigation: Home, Single Prediction, Batch Prediction, Explainability, About.
            </div>
            """,
            unsafe_allow_html=True,
        )
