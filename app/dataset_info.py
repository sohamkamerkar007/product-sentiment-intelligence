"""Overview-only, schema-aware facts about the active review table."""

from __future__ import annotations

import pandas as pd


FEATURE_NAMES = {
    "battery_life_rating": "Battery life",
    "camera_rating": "Camera",
    "performance_rating": "Performance",
    "design_rating": "Design",
    "display_rating": "Display",
}


def summarize_reviews(reviews: pd.DataFrame) -> dict:
    """Calculate display facts from the loaded data without fixed corpus counts."""
    columns = set(reviews.columns)
    facts = {
        "reviews": len(reviews),
        "columns": len(reviews.columns),
        "models": reviews["model"].nunique(dropna=True) if "model" in columns else None,
        "brands": reviews["brand"].nunique(dropna=True) if "brand" in columns else None,
        "features": [name for column, name in FEATURE_NAMES.items() if column in columns],
        "rating_range": None,
        "date_range": None,
    }
    if "rating" in columns:
        ratings = pd.to_numeric(reviews["rating"], errors="coerce").dropna()
        if not ratings.empty:
            facts["rating_range"] = (ratings.min(), ratings.max())
    if "review_date" in columns:
        dates = pd.to_datetime(reviews["review_date"], errors="coerce").dropna()
        if not dates.empty:
            facts["date_range"] = (dates.min(), dates.max())
    return facts
