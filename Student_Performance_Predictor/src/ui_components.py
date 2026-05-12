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
        unsafe_allow_html=True
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
                unsafe_allow_html=True
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
                unsafe_allow_html=True
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
        unsafe_allow_html=True
    )


def render_empty_state(message: str):
    st.markdown(
        f"""
        <div class="ep-empty-state">{message}</div>
        """,
        unsafe_allow_html=True
    )


def render_next_step_cards():
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div class="ep-next-card">
                <div class="ep-next-title">🎯 Single Prediction</div>
                <div class="ep-next-desc">Build one profile and estimate the expected exam score instantly.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="ep-next-card">
                <div class="ep-next-title">📂 Batch Prediction</div>
                <div class="ep-next-desc">Upload a CSV file, validate the schema, and score multiple students at once.</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_sidebar_summary(raw_feature_names: list[str]):
    with st.sidebar:
        st.markdown("## 🎓 EduPredict")
        st.caption("Portfolio ML App")

        st.markdown("---")
        st.markdown(
            f"""
            <div class="ep-sidebar-mini">
                <div><strong>Model</strong><br>Ridge Regression</div>
                <div><strong>Features</strong><br>{len(raw_feature_names)}</div>
                <div><strong>Output</strong><br>Exam Score</div>
            </div>
            """,
            unsafe_allow_html=True
        )