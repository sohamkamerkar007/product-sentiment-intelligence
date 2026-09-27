"""Apply the existing DistilBERT ABSA model to added synthetic aspect mentions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "trained_models" / "distilbert_absa"
OUTPUT = ROOT / "data" / "processed" / "absa_expanded_predictions.csv"
REPORT = ROOT / "evaluation" / "expanded_absa_metrics.json"


def run(batch_size: int = 48) -> dict:
    reviews = pd.read_csv(ROOT / "data" / "processed" / "reviews_expanded.csv")
    annotations = pd.read_csv(ROOT / "data" / "processed" / "synthetic_aspect_annotations.csv")
    splits = pd.read_csv(ROOT / "data" / "processed" / "review_splits.csv")
    rating_columns = ["battery_life_rating", "camera_rating", "performance_rating", "design_rating", "display_rating"]
    added = annotations.merge(reviews[["review_id", "brand", "model", "review_text", *rating_columns]], on="review_id", validate="many_to_one")
    added = added.merge(splits, on="review_id", validate="many_to_one")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    labels = {int(k): v for k, v in model.config.id2label.items()}
    text_inputs = ("aspect: " + added["aspect"] + " review: " + added["evidence"]).astype(str).tolist()
    predicted, confidence = [], []
    with torch.inference_mode():
        for start in range(0, len(text_inputs), batch_size):
            encoded = tokenizer(text_inputs[start:start + batch_size], truncation=True, max_length=128,
                                padding=True, return_tensors="pt")
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)
            ids = probabilities.argmax(dim=-1)
            predicted.extend(labels[int(i)] for i in ids)
            confidence.extend(probabilities.gather(1, ids[:, None]).squeeze(1).tolist())
    added["aspect_rating"] = [row[{"battery": "battery_life_rating", "camera": "camera_rating",
                                             "performance": "performance_rating", "design": "design_rating",
                                             "display": "display_rating"}[row["aspect"]]] for _, row in added.iterrows()]
    added["predicted_sentiment"] = predicted
    added["prediction_confidence"] = confidence
    old = pd.read_csv(ROOT / "data" / "processed" / "absa_full_predictions.csv")
    old["prediction_source"] = "Original saved DistilBERT predictions"
    added["prediction_source"] = "DistilBERT on expanded synthetic reviews"
    combined = pd.concat([old, added[old.columns]], ignore_index=True)
    combined.to_csv(OUTPUT, index=False)
    test = added["split"] == "test"
    y_true = added.loc[test, "aspect_sentiment"]
    y_pred = added.loc[test, "predicted_sentiment"]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    report = {"test_aspect_mentions": int(test.sum()), "test_review_models": int(added.loc[test, "model"].nunique()),
              "accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(precision),
              "macro_recall": float(recall), "macro_f1": float(f1),
              "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["Negative", "Neutral", "Positive"]).tolist(),
              "labels": ["Negative", "Neutral", "Positive"],
              "caveat": "Synthetic generated labels; the saved ABSA model was not retrained."}
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
