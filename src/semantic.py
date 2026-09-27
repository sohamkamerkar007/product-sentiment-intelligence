"""Sentence embedding, unsupervised discovery and model relationships."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS = ROOT / "data" / "processed" / "review_embeddings.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings(force: bool = False) -> np.ndarray:
    """Cache normalized embeddings in the same row order as reviews_expanded.csv."""
    reviews = pd.read_csv(ROOT / "data" / "processed" / "reviews_expanded.csv")
    if EMBEDDINGS.exists() and not force:
        result = np.load(EMBEDDINGS, mmap_mode="r")
        if len(result) == len(reviews):
            return result
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    result = model.encode(reviews["review_text"].astype(str).tolist(),
                          batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.save(EMBEDDINGS, np.asarray(result, dtype=np.float32))
    return np.load(EMBEDDINGS, mmap_mode="r")


def discover(reviews: pd.DataFrame, embeddings: np.ndarray, indices: np.ndarray,
             method: str = "K-Means", n_clusters: int = 8,
             min_cluster_size: int = 20) -> tuple[pd.DataFrame, dict]:
    """Cluster only selected reviews; labels are data-derived top terms."""
    if len(indices) < max(30, n_clusters * 4):
        raise ValueError("Choose at least 30 reviews before clustering.")
    selected = reviews.iloc[indices].copy().reset_index(drop=True)
    vectors = np.asarray(embeddings[indices], dtype=np.float32)
    if method == "K-Means":
        fitted = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(vectors)
        labels = fitted.labels_
    elif method == "HDBSCAN":
        reduced = PCA(n_components=min(25, len(vectors) - 1), random_state=42).fit_transform(vectors)
        # A small core-neighbour requirement avoids discarding this sparse,
        # multi-topic review corpus wholesale as noise. Cluster size remains
        # an explicit user-controlled density threshold.
        labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=3,
                         copy=True).fit_predict(reduced)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    selected["cluster"] = labels.astype(int)
    valid = labels >= 0
    unique = np.unique(labels[valid])
    score = None
    if len(unique) >= 2 and valid.sum() >= 50:
        subset = np.flatnonzero(valid)
        if len(subset) > 1200:
            subset = np.random.default_rng(42).choice(subset, size=1200, replace=False)
        if len(np.unique(labels[subset])) >= 2:
            score = float(silhouette_score(vectors[subset], labels[subset], metric="cosine"))

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1200, ngram_range=(1, 2), min_df=2)
    terms = vectorizer.fit_transform(selected["review_text"].astype(str))
    names = vectorizer.get_feature_names_out()
    summaries = []
    for label in unique:
        group = np.flatnonzero(labels == label)
        weights = np.asarray(terms[group].mean(axis=0)).ravel()
        top_terms = names[np.argsort(weights)[-5:][::-1]].tolist()
        centroid = vectors[group].mean(axis=0, keepdims=True)
        representatives = group[np.argsort(cosine_similarity(vectors[group], centroid).ravel())[-3:][::-1]]
        summaries.append({"cluster": int(label), "size": len(group), "top_terms": ", ".join(top_terms),
                          "representative_reviews": selected.iloc[representatives]["review_text"].tolist()})
    if (labels == -1).any():
        summaries.append({"cluster": -1, "size": int((labels == -1).sum()),
                          "top_terms": "Unassigned / noise", "representative_reviews": []})
    return selected, {"silhouette_cosine": score, "clusters": summaries, "method": method}


def related_models(reviews: pd.DataFrame, embeddings: np.ndarray, model: str,
                   brand: str | None = None, top_k: int = 8) -> pd.DataFrame:
    """Cosine similarity of normalized mean review embeddings per phone."""
    if model not in set(reviews["model"]):
        raise ValueError("Choose a smartphone model present in the dataset.")
    vectors = np.asarray(embeddings, dtype=np.float32)
    profiles = {}
    for name, group in reviews.groupby("model", sort=False):
        if len(group) < 30:
            continue
        mean = vectors[group.index.to_numpy()].mean(axis=0)
        profiles[name] = mean / max(np.linalg.norm(mean), 1e-9)
    if model not in profiles:
        raise ValueError("This model has too few reviews for a reliable relationship profile.")
    source = profiles[model]
    candidates = []
    for name, vector in profiles.items():
        if name == model:
            continue
        group = reviews[reviews["model"] == name]
        candidate_brand = str(group["brand"].iloc[0])
        if brand and candidate_brand != brand:
            continue
        candidates.append({"model": name, "brand": candidate_brand, "similarity": float(np.dot(source, vector)),
                           "review_count": len(group)})
    return pd.DataFrame(candidates).sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)


if __name__ == "__main__":
    print("Embedding shape:", create_embeddings().shape)
