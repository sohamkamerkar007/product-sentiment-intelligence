"""Focused checks of the upgraded data and model workflows."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emerging import detect_issues  # noqa: E402
from generate_dataset import build, validate  # noqa: E402
from semantic import discover, related_models  # noqa: E402
from supervised import load_predictors  # noqa: E402


class DatasetTests(unittest.TestCase):
    def test_generated_schema_and_group_split(self):
        reviews, partitions, annotations = build()
        validate(reviews)
        self.assertEqual(reviews.model.nunique(), 50)
        self.assertGreater(len(annotations), 10_000)
        merged = reviews[["review_id", "model"]].merge(partitions, on="review_id")
        self.assertEqual(merged.groupby("model")["split"].nunique().max(), 1)

    def test_validation_rejects_duplicate(self):
        reviews = pd.read_csv(ROOT / "data" / "processed" / "reviews_expanded.csv")
        reviews.loc[1, "review_id"] = reviews.loc[0, "review_id"]
        with self.assertRaises(AssertionError):
            validate(reviews)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dissatisfaction, cls.rating = load_predictors()

    def test_prediction_and_empty_review(self):
        text = "The camera takes sharp photos, but the battery drains quickly."
        self.assertTrue(0 <= self.dissatisfaction.predict(text)["probability"] <= 1)
        self.assertTrue(1 <= self.rating.predict(text)["predicted_rating"] <= 5)
        with self.assertRaises(ValueError):
            self.rating.predict("   ")

    def test_shap_uses_model_features(self):
        explanations = self.dissatisfaction.explain("The battery drains quickly.", top_n=5)
        self.assertTrue(explanations)
        self.assertTrue(all(np.isfinite(item["contribution"]) for item in explanations))

    def test_absa_model_loads_and_predicts(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = ROOT / "models" / "trained_models" / "distilbert_absa"
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        with torch.inference_mode():
            output = model(**tokenizer("aspect: camera review: photos look sharp", return_tensors="pt"))
        self.assertEqual(output.logits.shape[-1], 3)


class DiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reviews = pd.read_csv(ROOT / "data" / "processed" / "reviews_expanded.csv")
        cls.embeddings = np.load(ROOT / "data" / "processed" / "review_embeddings.npy", mmap_mode="r")

    def test_kmeans_and_hdbscan(self):
        indices = np.arange(120)
        for method in ("K-Means", "HDBSCAN"):
            selected, report = discover(self.reviews, self.embeddings, indices, method=method,
                                        n_clusters=4, min_cluster_size=12)
            self.assertEqual(len(selected), 120)
            self.assertTrue(report["clusters"])
        with self.assertRaises(ValueError):
            discover(self.reviews, self.embeddings, np.arange(8))

    def test_related_models_and_unknown_model(self):
        model = str(self.reviews["model"].iloc[0])
        related = related_models(self.reviews, self.embeddings, model)
        self.assertFalse(related.empty)
        self.assertTrue(related["similarity"].between(-1, 1).all())
        with self.assertRaises(ValueError):
            related_models(self.reviews, self.embeddings, "Unknown model")

    def test_emerging_empty_and_valid_results(self):
        topics = pd.read_csv(ROOT / "data" / "processed" / "complaint_topics.csv")
        self.assertTrue(detect_issues(self.reviews, topics, model="Unknown model").empty)
        result = detect_issues(self.reviews, topics)
        self.assertTrue(result.empty or (result["current_share"] > result["previous_share"]).all())


if __name__ == "__main__":
    unittest.main()
