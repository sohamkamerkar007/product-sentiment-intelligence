"""Small presentation components shared by the Streamlit pages.

These helpers render only HTML/CSS. They never alter analytical data or models.
"""

from __future__ import annotations

from html import escape

import streamlit as st


FEATURE_ICONS = {
    "Battery": "◴",
    "Camera": "◎",
    "Performance": "✦",
    "Design": "◇",
    "Display": "▣",
}


def brand_header() -> None:
    st.markdown(
        """
        <div class="brandbar">
          <div class="brandbar__identity">
            <span class="brandbar__mark" aria-hidden="true"><i></i><i></i><i></i></span>
            <span class="brandbar__name">PRODUCT <strong>SENTIMENT</strong> INTELLIGENCE</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <section class="page-intro">
          <div class="page-intro__text">
            <div class="page-eyebrow"><span class="eyebrow-rule"></span>{escape(eyebrow)}</div>
            <h1>{escape(title)}</h1>
            <p>{escape(description)}</p>
          </div>
          <div class="page-intro__ornament" aria-hidden="true"><span></span><span></span><span></span></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section_header(number: str, title: str, description: str = "") -> None:
    detail = f"<p>{escape(description)}</p>" if description else ""
    st.markdown(
        f'<div class="section-heading"><span class="section-heading__number">{escape(number)}</span>'
        f'<div><h2>{escape(title)}</h2>{detail}</div></div>',
        unsafe_allow_html=True,
    )


def phone_card(name: str, brand: str, label: str) -> None:
    st.markdown(
        f'<div class="phone-card"><div class="phone-card__top"><span class="phone-card__label">'
        f'{escape(label)}</span><span class="phone-card__glyph" aria-hidden="true">▣</span></div>'
        f'<h3>{escape(name)}</h3><p>{escape(brand)}</p></div>',
        unsafe_allow_html=True,
    )


def review_quote(text: str, source: str = "Review example") -> None:
    st.markdown(
        f'<blockquote class="review-quote"><p>{escape(text)}</p>'
        f'<footer>{escape(source)}</footer></blockquote>',
        unsafe_allow_html=True,
    )


def sentiment_card(label: str, count: int, tone: str) -> None:
    """Show the existing calculated count with text and a redundant tone cue."""
    if tone not in {"positive", "neutral", "negative"}:
        raise ValueError("Unknown sentiment-card tone")
    st.markdown(
        f'<div class="sentiment-card sentiment-card--{tone}">'
        f'<div class="sentiment-card__label"><span aria-hidden="true"></span>{escape(label)}</div>'
        f'<strong>{int(count)}</strong><small>phone features</small></div>',
        unsafe_allow_html=True,
    )
