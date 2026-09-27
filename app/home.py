"""Product landing page, built with Streamlit-native navigation links."""

from __future__ import annotations

from html import escape

import streamlit as st

import ui


FEATURES = (
    ("Compare Phones", "▣", "See how two phones differ across the features people mention.", "Compare phones"),
    ("Review Analyzer", "✦", "Paste a review and explore its likes, dislikes and estimated rating.", "Analyze a review"),
    ("Phone Features", "◫", "Explore opinions about battery, camera, speed, design and display.", "Explore features"),
    ("Common Problems", "◈", "Find recurring negative comments and inspect the reviews behind them.", "Find common problems"),
    ("Discover Patterns", "◎", "Group reviews with similar wording and explore their shared themes.", "Discover patterns"),
    ("Find Similar Phones", "⇄", "Compare phones through the language used in their reviews.", "Find similar phones"),
)

ASPECTS = (
    ("Battery", "Charge and endurance"),
    ("Camera", "Photos and video"),
    ("Performance", "Speed and fluidity"),
    ("Design", "Look and feel"),
    ("Display", "Screen experience"),
)


def render_home(pages: dict, review_count: int, model_count: int) -> None:
    with st.container(key="home-hero-shell"):
        left, right = st.columns([1.28, 0.9], gap="large", vertical_alignment="center")
        with left:
            st.markdown(
                """
                <div class="home-hero-copy">
                  <div class="hero-kicker"><span class="hero-kicker__dot"></span> A clearer view of phone reviews</div>
                  <h1>Understand what people think about their phones<span class="hero-stop">.</span></h1>
                  <p>Explore reviews, compare smartphone experiences, discover common problems,
                  and see how opinions change over time.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                """
                <div class="hero-visual" aria-label="Illustration of phone features considered separately">
                  <div class="hero-visual__orbit hero-visual__orbit--one"></div>
                  <div class="hero-visual__orbit hero-visual__orbit--two"></div>
                  <div class="hero-device">
                    <div class="hero-device__top"><span>REVIEW LENS</span><span class="hero-device__signal">● ● ●</span></div>
                    <div class="hero-device__screen">
                      <div class="hero-device__eyebrow">ONE REVIEW, MANY DETAILS</div>
                      <div class="hero-device__line hero-device__line--long"></div>
                      <div class="hero-device__line"></div>
                      <div class="hero-device__line hero-device__line--short"></div>
                      <div class="hero-device__divider"></div>
                      <div class="hero-device__feature"><span class="feature-dot feature-dot--indigo"></span>Camera <i></i></div>
                      <div class="hero-device__feature"><span class="feature-dot feature-dot--teal"></span>Battery <i></i></div>
                      <div class="hero-device__feature"><span class="feature-dot feature-dot--amber"></span>Display <i></i></div>
                    </div>
                    <div class="hero-device__foot">FIVE FEATURE LENSES</div>
                  </div>
                  <div class="hero-visual__badge">Insights beyond an overall rating <span>↗</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with st.container(key="hero-actions"):
            links = (
                ("Compare Phones", "Compare phones"),
                ("Review Analyzer", "Analyze a review"),
                ("Review Trends", "Explore review trends"),
            )
            for column, (key, label) in zip(st.columns(3, gap="small"), links):
                with column:
                    st.page_link(pages[key], label=label, use_container_width=True)

    st.markdown(
        '<div class="dataset-ribbon"><span class="dataset-ribbon__icon">i</span>'
        '<p><strong>Research demonstration.</strong> These reviews, ratings and dates are synthetically '
        'generated. The named phones are real, but the opinions are not verified customer feedback.</p></div>',
        unsafe_allow_html=True,
    )

    ui.section_header("01", "Explore by task", "Start with a question. Each tool lets you dig into the generated reviews.")
    for row in (FEATURES[:3], FEATURES[3:]):
        for column, (page, icon, description, action) in zip(st.columns(3, gap="medium"), row):
            with column, st.container(border=True, key=f"feature-{page.lower().replace(' ', '-')}"):
                st.markdown(
                    f'<div class="feature-card__icon" aria-hidden="true">{escape(icon)}</div>'
                    f'<h3 class="feature-card__title">{escape(page)}</h3>'
                    f'<p class="feature-card__description">{escape(description)}</p>',
                    unsafe_allow_html=True,
                )
                st.page_link(pages[page], label=f"{action}  →")

    ui.section_header("02", "A phone is more than one score",
                      "Explore five parts of the experience, not just a single positive or negative label.")
    st.markdown(
        '<div class="aspect-strip">'
        + "".join(
            f'<div class="aspect-strip__item"><span>{escape(ui.FEATURE_ICONS[name])}</span>'
            f'<strong>{escape(name)}</strong><small>{escape(detail)}</small></div>'
            for name, detail in ASPECTS
        )
        + '</div>',
        unsafe_allow_html=True,
    )

    ui.section_header("03", "How the experience works",
                      "A simple path from written reviews to useful, feature-level views.")
    st.markdown(
        '<div class="journey">'
        '<div><span>01</span><strong>Choose a phone or review</strong><p>Start with a product or your own text.</p></div>'
        '<div><span>02</span><strong>Explore its features</strong><p>See which parts of the phone are discussed.</p></div>'
        '<div><span>03</span><strong>Read the evidence</strong><p>Inspect examples behind each result.</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.container(key="home-data-panel"):
        st.markdown('<div class="data-panel__eyebrow">ABOUT THIS RESEARCH DATASET</div>', unsafe_allow_html=True)
        st.markdown(
            f'<h2>{review_count:,} generated reviews. {model_count} real phone model names.</h2>'
            '<p>The text, dates, ratings and opinions are simulated. Charts and predictions describe '
            'this research dataset—not measured performance or actual customer experiences.</p>',
            unsafe_allow_html=True,
        )
        with st.expander("How the analysis works — technical details"):
            st.write(
                "The app detects mentions of battery, camera, performance, design and display, "
                "then uses a saved DistilBERT model to classify each opinion. Separate text-only "
                "XGBoost models estimate negative-review likelihood and a 1–5 rating. "
                "Sentence-BERT embeddings support review grouping and similar-phone search. "
                "SHAP shows which words affected a prediction. All results are based on the "
                "synthetic dataset and its trained models."
            )
