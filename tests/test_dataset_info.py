"""Check that Overview facts follow the loaded review schema and values."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from dataset_info import summarize_reviews  # noqa: E402


class DatasetInfoTests(unittest.TestCase):
    def test_active_dataset(self):
        reviews = pd.read_csv(ROOT / "data" / "processed" / "reviews_expanded.csv")
        summary = summarize_reviews(reviews)
        self.assertEqual(summary["reviews"], len(reviews))
        self.assertEqual(summary["columns"], len(reviews.columns))
        self.assertEqual(summary["models"], reviews["model"].nunique())
        self.assertEqual(summary["brands"], reviews["brand"].nunique())
        self.assertEqual(summary["rating_range"], (reviews["rating"].min(), reviews["rating"].max()))
        self.assertEqual(summary["date_range"][0], pd.to_datetime(reviews["review_date"]).min())

    def test_optional_fields_and_changed_rows(self):
        example = pd.DataFrame({"model": ["A", "B"], "camera_rating": [4, 5]})
        summary = summarize_reviews(example)
        self.assertEqual((summary["reviews"], summary["columns"], summary["models"]), (2, 2, 2))
        self.assertEqual(summary["features"], ["Camera"])
        self.assertIsNone(summary["brands"])
        self.assertIsNone(summary["date_range"])
        self.assertIsNone(summary["rating_range"])


if __name__ == "__main__":
    unittest.main()
