"""Record reproducible, descriptive checks for the unsupervised modules."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from emerging import TOPICS_CSV, detect_issues
from semantic import ROOT, create_embeddings, discover, related_models


def main() -> None:
    reviews = pd.read_csv(ROOT / "data" / "processed" / "reviews_expanded.csv")
    vectors = create_embeddings()
    rng = np.random.default_rng(42)
    sample = np.sort(rng.choice(len(reviews), size=1200, replace=False))
    clustering = {}
    for method in ("K-Means", "HDBSCAN"):
        _, result = discover(reviews, vectors, sample, method=method,
                             n_clusters=8, min_cluster_size=25)
        clustering[method] = {
            "sample_reviews": len(sample),
            "clusters_excluding_noise": sum(c["cluster"] >= 0 for c in result["clusters"]),
            "noise_reviews": sum(c["size"] for c in result["clusters"] if c["cluster"] == -1),
            "cosine_silhouette": result["silhouette_cosine"],
        }
    topics = pd.read_csv(TOPICS_CSV)
    issues = detect_issues(reviews, topics)
    example_model = str(reviews["model"].value_counts().index[0])
    relatives = related_models(reviews, vectors, example_model, top_k=5)
    report = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_shape": list(vectors.shape),
        "clustering": clustering,
        "emerging_issue_count": len(issues),
        "emerging_issue_method": "One-sided Fisher exact test on adjacent calendar years; Benjamini-Hochberg FDR <= 0.10 across every topic-period comparison; minimum 40 reviews per period and 5 current complaints. The partial 2026 year is normalized by review volume.",
        "example_related_model": example_model,
        "example_related_models": relatives[["model", "similarity"]].to_dict("records"),
        "limitations": "All reviews are synthetic. Clusters and similarities are descriptive, and the later-period Google battery change was deliberately simulated in generated rows.",
    }
    path = ROOT / "evaluation" / "discovery_metrics.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
