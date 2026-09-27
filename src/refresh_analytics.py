"""Refresh conflict summaries after model-keyed ABSA predictions change."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"


def run() -> dict:
    predictions = pd.read_csv(DATA / "absa_expanded_predictions.csv", dtype={"review_id": str})
    reviews = pd.read_csv(DATA / "reviews_expanded.csv")
    rows = []
    for review_id, group in predictions.groupby("review_id", sort=False):
        records = group.drop_duplicates("aspect").to_dict("records")
        for left, right in combinations(records, 2):
            if {left["predicted_sentiment"], right["predicted_sentiment"]} != {"Positive", "Negative"}:
                continue
            first, second = sorted((left, right), key=lambda row: row["aspect"])
            strength = (float(first["prediction_confidence"]) + float(second["prediction_confidence"])) / 2
            rows.append({"review_id": review_id, "brand": first["brand"], "model": first["model"],
                         "review_text": first["review_text"], "aspect_1": first["aspect"], "aspect_2": second["aspect"],
                         "sentiment_1": first["predicted_sentiment"], "sentiment_2": second["predicted_sentiment"],
                         "confidence_1": first["prediction_confidence"], "confidence_2": second["prediction_confidence"],
                         "conflict_strength": strength})
    conflicts = pd.DataFrame(rows)
    conflicts.to_csv(DATA / "review_aspect_conflicts.csv", index=False)
    pair_columns = ["aspect_1", "aspect_2"]
    global_pairs = conflicts.groupby(pair_columns).agg(
        conflict_count=("review_id", "size"), unique_reviews=("review_id", "nunique"),
        avg_conflict_strength=("conflict_strength", "mean")
    ).reset_index().sort_values("conflict_count", ascending=False)
    global_pairs.to_csv(DATA / "aspect_conflict_summary.csv", index=False)
    phone_pairs = conflicts.groupby(["brand", "model", *pair_columns]).agg(
        conflict_count=("review_id", "size"), unique_reviews=("review_id", "nunique"),
        avg_conflict_strength=("conflict_strength", "mean")
    ).reset_index().sort_values("conflict_count", ascending=False)
    phone_pairs.to_csv(DATA / "phone_aspect_conflicts.csv", index=False)
    counts = reviews.groupby(["brand", "model"]).size().rename("total_reviews")
    summaries = conflicts.groupby(["brand", "model"]).agg(
        total_conflict_pairs=("review_id", "size"), conflict_reviews=("review_id", "nunique"),
        avg_conflict_strength=("conflict_strength", "mean")
    )
    by_phone = counts.to_frame().join(summaries).fillna(0).reset_index()
    by_phone["conflict_rate_pct"] = 100 * by_phone["conflict_reviews"] / by_phone["total_reviews"]
    by_phone.to_csv(DATA / "conflicts_by_phone.csv", index=False)
    pd.DataFrame([{"total_reviews": len(reviews), "reviews_with_conflict": conflicts["review_id"].nunique(),
                   "total_conflict_pairs": len(conflicts),
                   "conflict_rate_pct": 100 * conflicts["review_id"].nunique() / len(reviews),
                   "unique_aspect_pairs": len(global_pairs)}]).to_csv(DATA / "conflict_summary.csv", index=False)
    return {"reviews": len(reviews), "phone_models": reviews["model"].nunique(), "conflict_pairs": len(conflicts)}


if __name__ == "__main__":
    print(run())
