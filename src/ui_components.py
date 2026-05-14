# src/ui_components.py

import streamlit as st


def render_page_header(title: str, subtitle: str, badge_text: str | None = None):
    st.markdown(
        (
            "<div class='ep-page-header'>"
            f"<h1 class='ep-page-title'>{title}</h1>"
            f"<p class='ep-page-subtitle'>{subtitle}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_hero_section(title: str, subtitle: str):
    st.markdown(
        f"""
        <section class="ep-hero">
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


def render_text_cards(cards: list[tuple[str, str]]):
    cols = st.columns(len(cards), gap="medium")
    for col, (title, desc) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="ep-text-card">
                    <div class="ep-text-card-title">{title}</div>
                    <div class="ep-text-card-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_cta_card(title: str, desc: str, page_path: str, link_label: str):
    button_key = f"cta_{page_path}_{link_label}".replace("/", "_").replace(" ", "_")

    st.markdown(
        f"""
        <div class="ep-cta-card">
            <div class="ep-cta-title">{title}</div>
            <div class="ep-cta-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(link_label, key=button_key, use_container_width=True):
        try:
            st.switch_page(page_path)
        except Exception:
            st.caption(f"Open {link_label} from the sidebar.")


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
    render_page_header(
        "EduPredict",
        (
            "A compact Streamlit app for estimating student exam performance from "
            "academic, lifestyle, family, and school-related factors."
        ),
    )

    st.info(
        "Start with Home for context, or jump straight into a prediction workflow."
    )

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        render_cta_card(
            "Open Home",
            "Review the project overview, workflows, and model status.",
            "pages/1_Home.py",
            "Go to Home",
        )

    with col2:
        render_cta_card(
            "Try one profile",
            "Estimate one student's score with the interactive form.",
            "pages/2_Single_Prediction.py",
            "Open Single Prediction",
        )

    with col3:
        render_cta_card(
            "Inspect drivers",
            "See which factors influence predicted scores.",
            "pages/4_Explainability.py",
            "Open Model Insights",
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

