import pandas as pd
import numpy as np


def create_phone_aspect_profile(df):
    """
    Create sentiment statistics for every
    phone model and aspect.
    """

    profile = (
        df.groupby(
            ["brand", "model", "aspect"]
        )
        .agg(
            mention_count=(
                "predicted_sentiment",
                "size"
            ),

            positive_pct=(
                "predicted_sentiment",
                lambda x: (
                    x == "Positive"
                ).mean() * 100
            ),

            neutral_pct=(
                "predicted_sentiment",
                lambda x: (
                    x == "Neutral"
                ).mean() * 100
            ),

            negative_pct=(
                "predicted_sentiment",
                lambda x: (
                    x == "Negative"
                ).mean() * 100
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


def detect_aspect_conflicts(
    profile,
    positive_threshold=50,
    negative_threshold=25
):
    """
    Detect phones where one aspect is strongly positive
    while another aspect is strongly negative.

    This represents a potential product trade-off.
    """

    conflicts = []

    for (brand, model), phone_data in profile.groupby(
        ["brand", "model"]
    ):

        aspects = phone_data.to_dict("records")

        for i in range(len(aspects)):

            for j in range(i + 1, len(aspects)):

                aspect_a = aspects[i]
                aspect_b = aspects[j]

                # A positive / B negative
                if (
                    aspect_a["positive_pct"]
                    >= positive_threshold
                    and
                    aspect_b["negative_pct"]
                    >= negative_threshold
                ):

                    conflicts.append({
                        "brand": brand,
                        "model": model,
                        "aspect_positive":
                            aspect_a["aspect"],
                        "positive_pct":
                            aspect_a["positive_pct"],
                        "aspect_negative":
                            aspect_b["aspect"],
                        "negative_pct":
                            aspect_b["negative_pct"],
                        "conflict_strength":
                            (
                                aspect_a["positive_pct"]
                                + aspect_b["negative_pct"]
                            ) / 2
                    })

                # B positive / A negative
                elif (
                    aspect_b["positive_pct"]
                    >= positive_threshold
                    and
                    aspect_a["negative_pct"]
                    >= negative_threshold
                ):

                    conflicts.append({
                        "brand": brand,
                        "model": model,
                        "aspect_positive":
                            aspect_b["aspect"],
                        "positive_pct":
                            aspect_b["positive_pct"],
                        "aspect_negative":
                            aspect_a["aspect"],
                        "negative_pct":
                            aspect_a["negative_pct"],
                        "conflict_strength":
                            (
                                aspect_b["positive_pct"]
                                + aspect_a["negative_pct"]
                            ) / 2
                    })

    conflicts_df = pd.DataFrame(conflicts)

    if not conflicts_df.empty:

        conflicts_df = (
            conflicts_df
            .sort_values(
                "conflict_strength",
                ascending=False
            )
            .reset_index(drop=True)
        )

    return conflicts_df


def detect_review_level_conflicts(df):
    """
    Detect individual reviews where different aspects
    have opposite predicted sentiments.

    Example:
    Camera = Positive
    Battery = Negative
    """

    review_conflicts = []

    for review_id, review_data in df.groupby(
        "review_id"
    ):

        sentiments = (
            review_data
            .set_index("aspect")[
                "predicted_sentiment"
            ]
            .to_dict()
        )

        aspects = list(sentiments.keys())

        for i in range(len(aspects)):

            for j in range(i + 1, len(aspects)):

                aspect_a = aspects[i]
                aspect_b = aspects[j]

                sentiment_a = sentiments[aspect_a]
                sentiment_b = sentiments[aspect_b]

                if (
                    sentiment_a == "Positive"
                    and
                    sentiment_b == "Negative"
                ):

                    review_conflicts.append({
                        "review_id": review_id,
                        "aspect_positive": aspect_a,
                        "aspect_negative": aspect_b
                    })

                elif (
                    sentiment_b == "Positive"
                    and
                    sentiment_a == "Negative"
                ):

                    review_conflicts.append({
                        "review_id": review_id,
                        "aspect_positive": aspect_b,
                        "aspect_negative": aspect_a
                    })

    return pd.DataFrame(
        review_conflicts
    )


def summarize_conflicts(conflicts_df):
    """
    Summarize conflict frequency by
    aspect combination.
    """

    if conflicts_df.empty:
        return pd.DataFrame()

    summary = (
        conflicts_df
        .groupby(
            [
                "aspect_positive",
                "aspect_negative"
            ]
        )
        .size()
        .reset_index(
            name="conflict_count"
        )
        .sort_values(
            "conflict_count",
            ascending=False
        )
    )

    return summary