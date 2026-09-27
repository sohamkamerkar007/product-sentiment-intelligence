import pandas as pd
import numpy as np


def calculate_sentiment_profile(df):
    """
    Create sentiment statistics for every
    brand-model-aspect combination.
    """

    profile = (
        df.groupby(
            ["brand", "model", "aspect"]
        )
        .agg(
            mention_count=("predicted_sentiment", "size"),

            positive_pct=(
                "predicted_sentiment",
                lambda x: (x == "Positive").mean() * 100
            ),

            neutral_pct=(
                "predicted_sentiment",
                lambda x: (x == "Neutral").mean() * 100
            ),

            negative_pct=(
                "predicted_sentiment",
                lambda x: (x == "Negative").mean() * 100
            ),

            avg_confidence=(
                "prediction_confidence",
                "mean"
            )
        )
        .reset_index()
    )

    profile["sentiment_score"] = (
        profile["positive_pct"]
        - profile["negative_pct"]
    )

    return profile


def add_model_order(profile, model_order):
    """
    Add chronological/product-line ordering.
    """

    profile = profile.copy()

    profile["model_order"] = (
        profile["model"]
        .map(model_order)
    )

    return profile


def calculate_evolution(profile):
    """
    Calculate sentiment change between
    consecutive models.
    """

    profile = profile.copy()

    profile = profile.sort_values(
        [
            "brand",
            "aspect",
            "model_order"
        ]
    ).reset_index(drop=True)

    profile["sentiment_change"] = (
        profile
        .groupby(
            ["brand", "aspect"]
        )["sentiment_score"]
        .diff()
    )

    return profile


def classify_trend(change):
    """
    Convert sentiment change into
    an interpretable trend.
    """

    if pd.isna(change):
        return "Baseline"

    if change >= 5:
        return "Improving"

    if change <= -5:
        return "Declining"

    return "Stable"


def add_trend_labels(profile):
    """
    Add trend labels.
    """

    profile = profile.copy()

    profile["trend"] = (
        profile["sentiment_change"]
        .apply(classify_trend)
    )

    return profile


def get_evolution_summary(profile):
    """
    Count improving, declining,
    stable and baseline entries.
    """

    summary = (
        profile["trend"]
        .value_counts()
        .rename_axis("trend")
        .reset_index(
            name="count"
        )
    )

    return summary


def get_declining_aspects(profile):
    """
    Return strongest declining aspects.
    """

    declining = profile[
        profile["trend"] == "Declining"
    ].copy()

    return declining.sort_values(
        "sentiment_change",
        ascending=True
    )


def get_improving_aspects(profile):
    """
    Return strongest improving aspects.
    """

    improving = profile[
        profile["trend"] == "Improving"
    ].copy()

    return improving.sort_values(
        "sentiment_change",
        ascending=False
    )