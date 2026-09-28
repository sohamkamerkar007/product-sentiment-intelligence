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


def render_home(pages: dict, dataset: dict) -> None:
    with st.container(key="home-hero-shell"):
        left, right = st.columns([1.28, 0.9], gap="large", vertical_alignment="center")
        with left:
            st.markdown(
                """
                <div class="home-hero-copy">
                  <div class="hero-kicker"><span class="hero-kicker__dot"></span> A clearer view of phone reviews</div>
                  <h1>Understand What People Think About Their Phones<span class="hero-stop">.</span></h1>
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

    ui.section_header("01", "Explore by task", "Start with a question. Each tool lets you dig into the reviews.")
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
      with st.expander("About the Dataset", expanded=True):
        st.caption("A quick look at the review table powering these views.")
        facts = [
            ("Reviews", f'{dataset["reviews"]:,}'),
            ("Columns", f'{dataset["columns"]:,}'),
        ]
        if dataset["models"] is not None:
            facts.append(("Smartphone models", f'{dataset["models"]:,}'))
        if dataset["brands"] is not None:
            facts.append(("Brands", f'{dataset["brands"]:,}'))
        st.markdown(
            '<div class="dataset-facts">' + ''.join(
                f'<div class="dataset-fact"><strong>{escape(value)}</strong><span>{escape(label)}</span></div>'
                for label, value in facts
            ) + '</div>',
            unsafe_allow_html=True,
        )
        details = []
        if dataset["features"]:
            details.append('<div><strong>Review features</strong><span>' +
                           escape(', '.join(dataset["features"])) + '</span></div>')
        if dataset["rating_range"]:
            low, high = dataset["rating_range"]
            details.append(f'<div><strong>Rating range</strong><span>{low:g}–{high:g}</span></div>')
        if dataset["date_range"]:
            first, last = dataset["date_range"]
            details.append('<div><strong>Review dates</strong><span>' +
                           f'{first:%d %b %Y} – {last:%d %b %Y}</span></div>')
        if details:
            st.markdown('<div class="dataset-details">' + ''.join(details) + '</div>',
                        unsafe_allow_html=True)
        st.markdown(
            '<p class="dataset-note">The review text, ratings, dates and opinions are simulated for '
            'research and demonstration. They are not verified customer feedback.</p>',
            unsafe_allow_html=True,
        )
        with st.popover("How the analysis works"):
            st.write(
                "The app detects mentions of battery, camera, performance, design and display, "
                "then uses a saved DistilBERT model to classify each opinion. Separate text-only "
                "XGBoost models estimate negative-review likelihood and a 1–5 rating. "
                "Sentence-BERT embeddings support review grouping and similar-phone search. "
                "SHAP shows which words affected a prediction."
            )
