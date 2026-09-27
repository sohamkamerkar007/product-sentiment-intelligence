"""Volume-adjusted detection of growing complaint themes over time."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from semantic import ROOT, create_embeddings


TOPICS_CSV = ROOT / "data" / "processed" / "complaint_topics.csv"
TOPIC_MODEL = ROOT / "models" / "trained_models" / "product_intelligence" / "complaint_kmeans.joblib"
ASPECT_TERMS = {
    "battery": r"battery|charg|power|drain|top-up",
    "camera": r"camera|photo|video|focus|portrait|image",
    "performance": r"perform|app|gam|stutter|lag|slow|responsive",
    "design": r"design|build|frame|button|weight|hold|finish|bulky",
    "display": r"display|screen|panel|bright|colour|color|touch|scroll",
}


def build_topics(n_topics: int = 10) -> pd.DataFrame:
    reviews = pd.read_csv(ROOT / "data" / "processed" / "reviews_expanded.csv")
    embeddings = create_embeddings()
    dates = pd.to_datetime(reviews["review_date"])
    historical = (reviews["sentiment"] == "Negative") & (dates < "2026-01-01")
    vectors = np.asarray(embeddings[historical.to_numpy()], dtype=np.float32)
    model = KMeans(n_clusters=n_topics, random_state=42, n_init=10).fit(vectors)
    negative = reviews["sentiment"] == "Negative"
    labels = pd.Series(-1, index=reviews.index, dtype=int)
    labels.loc[negative] = model.predict(np.asarray(embeddings[negative.to_numpy()], dtype=np.float32))
    topics = pd.DataFrame({"review_id": reviews["review_id"], "topic": labels})
    vocabulary = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_features=1500)
    matrix = vocabulary.fit_transform(reviews.loc[historical, "review_text"].astype(str))
    terms = vocabulary.get_feature_names_out()
    historical_labels = model.labels_
    descriptions = {}
    for label in range(n_topics):
        mean = np.asarray(matrix[historical_labels == label].mean(axis=0)).ravel()
        descriptions[label] = ", ".join(terms[np.argsort(mean)[-5:][::-1]])
    topics["top_terms"] = topics["topic"].map(descriptions).fillna("")
    TOPICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    TOPIC_MODEL.parent.mkdir(parents=True, exist_ok=True)
    topics.to_csv(TOPICS_CSV, index=False)
    joblib.dump(model, TOPIC_MODEL, compress=3)
    return topics


def detect_issues(reviews: pd.DataFrame, topics: pd.DataFrame, *, brand: str | None = None,
                  model: str | None = None, aspect: str | None = None,
                  start: str | None = None, end: str | None = None,
                  min_reviews: int = 40) -> pd.DataFrame:
    """One-sided Fisher test on topic prevalence in consecutive calendar years.

    Benjamini-Hochberg adjusted p <= 0.10, >= 5 current topic reviews, and
    >= 40 reviews in both periods are required before reporting an issue.
    """
    data = reviews.merge(topics, on="review_id", how="left", validate="one_to_one")
    if brand:
        data = data[data["brand"] == brand]
    if model:
        data = data[data["model"] == model]
    if aspect:
        if aspect not in ASPECT_TERMS:
            raise ValueError("Unknown aspect filter")
        data = data[data["review_text"].str.contains(ASPECT_TERMS[aspect], case=False, regex=True, na=False)]
    data = data.copy()
    data["date"] = pd.to_datetime(data["review_date"], errors="coerce")
    if start:
        data = data[data["date"] >= pd.Timestamp(start)]
    if end:
        data = data[data["date"] <= pd.Timestamp(end)]
    if data.empty:
        return pd.DataFrame()
    # Annual windows provide enough observations for stable complaint shares in
    # model-level slices; unequal year lengths are normalized by review volume.
    data["period"] = data["date"].dt.year.astype(str)
    periods = sorted(data["period"].unique())
    rows = []
    for previous, current in zip(periods[:-1], periods[1:]):
        old = data[data["period"] == previous]
        new = data[data["period"] == current]
        if min(len(old), len(new)) < min_reviews:
            continue
        for topic in sorted(data.loc[data["topic"] >= 0, "topic"].dropna().unique()):
            earlier = int((old["topic"] == topic).sum())
            later = int((new["topic"] == topic).sum())
            old_rate = earlier / len(old)
            new_rate = later / len(new)
            _, p_value = fisher_exact([[later, len(new) - later], [earlier, len(old) - earlier]], alternative="greater")
            evidence = new[new["topic"] == topic].head(4)["review_text"].tolist()
            topic_rows = data[data["topic"] == topic]
            terms = topic_rows["top_terms"].iloc[0] if not topic_rows.empty else ""
            rows.append({"topic": int(topic), "top_terms": terms, "previous_period": previous,
                         "current_period": current, "previous_count": earlier, "current_count": later,
                         "previous_total": len(old), "current_total": len(new),
                         "previous_share": old_rate, "current_share": new_rate,
                         "increase_pp": 100 * (new_rate - old_rate), "p_value": p_value,
                         "evidence": evidence})
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    # Include every topic-period comparison in the correction, even declining
    # themes, so filtered findings cannot evade the multiple-testing penalty.
    ranks = np.arange(1, len(result) + 1)
    result["adjusted_p"] = np.minimum.accumulate((result["p_value"].to_numpy() * len(result) / ranks)[::-1])[::-1].clip(max=1)
    return result[(result["adjusted_p"] <= 0.10) & (result["current_count"] >= 5)
                  & (result["increase_pp"] > 0)].sort_values("increase_pp", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    topics = build_topics()
    reviews = pd.read_csv(ROOT / "data" / "processed" / "reviews_expanded.csv")
    print("Topics:", topics["topic"].value_counts().to_dict())
    print("Detected issue comparisons:", len(detect_issues(reviews, topics)))
