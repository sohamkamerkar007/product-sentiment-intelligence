"""Plain-language landing page for the Streamlit application."""

from __future__ import annotations

import streamlit as st


def render_home(pages: dict, review_count: int, model_count: int) -> None:
    st.markdown(
        """
        <section class="home-hero">
          <div class="home-eyebrow">Explore phone reviews</div>
          <h1>See what a review says about each part of a phone.</h1>
          <p>Compare phones, explore comments about battery or camera, and spot
          patterns that are easy to miss in a long list of reviews.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="home-note">These reviews are synthetically generated for research and '
        'demonstration. They are not actual customer reviews. Real phone names are used only '
        'as identifiers in the example dataset.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="home-section"><h2>Where would you like to start?</h2>'
                '<p>Choose a task below, or use the menu at the top.</p></div>', unsafe_allow_html=True)
    actions = [
        ("Compare Phones", "Put two or more phones side by side."),
        ("Review Analyzer", "Paste a review and see what it says."),
        ("Common Problems", "Explore frequently mentioned complaints."),
        ("Discover Patterns", "Find groups of reviews that sound alike."),
    ]
    for first, second in ((actions[0], actions[1]), (actions[2], actions[3])):
        cols = st.columns(2)
        for col, (label, description) in zip(cols, (first, second)):
            with col:
                st.markdown(f'<div class="home-card"><h3>{label}</h3><p>{description}</p></div>',
                            unsafe_allow_html=True)
                st.page_link(pages[label], label=f"Open {label}")

    st.markdown('<div class="home-section"><h2>Look beyond one overall rating</h2>'
                '<p>One review can praise the camera and complain about the battery. '
                'This app looks at five parts of a phone separately.</p></div>', unsafe_allow_html=True)
    aspects = [
        ("Battery", "How long it lasts and how charging feels."),
        ("Camera", "Photos, video, focus and low-light results."),
        ("Performance", "Speed, apps, multitasking and games."),
        ("Design", "How the phone looks and feels in the hand."),
        ("Display", "Brightness, colour, sharpness and touch."),
    ]
    for first, second, third in (aspects[:3], (*aspects[3:], None)):
        items = [item for item in (first, second, third) if item is not None]
        cols = st.columns(len(items))
        for col, (name, description) in zip(cols, items):
            with col:
                st.markdown(f'<div class="home-card"><h3>{name}</h3><p>{description}</p></div>',
                            unsafe_allow_html=True)

    st.markdown('<div class="home-section"><h2>About the example data</h2></div>', unsafe_allow_html=True)
    st.write(
        f"This research dataset contains {review_count:,} generated reviews across "
        f"{model_count} commercially released phone models. The review dates, ratings "
        "and opinions are simulated. A chart or prediction here does not describe "
        "verified feedback about a real product."
    )
    with st.expander("How this works — technical details"):
        st.write(
            "The app detects mentions of battery, camera, performance, design and display, "
            "then uses a saved DistilBERT model to classify each opinion. Separate text-only "
            "XGBoost models estimate negative-review likelihood and a 1–5 rating. "
            "Sentence-BERT embeddings support review grouping and similar-phone search. "
            "SHAP shows which words affected a prediction. All results are based on the "
            "synthetic dataset and its trained models."
        )
