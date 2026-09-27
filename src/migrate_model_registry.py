"""Refresh active model identifiers without retraining text-only models.

Run after updating the authoritative registry. The raw source and model weights
are intentionally untouched. Cached embeddings stay valid because review order
and text are checked before any active CSV is written.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from generate_dataset import ANNOTATIONS, OUTPUT, SPLITS, build
from model_registry import ROOT, validate_reviews
from refresh_analytics import run as refresh_conflicts


MODEL_KEYED_FILES = (
    ROOT / "data" / "processed" / "absa_expanded_predictions.csv",
    ROOT / "data" / "processed" / "supervised_predictions.csv",
)


def migrate() -> dict:
    previous = pd.read_csv(OUTPUT)
    previous_splits = pd.read_csv(SPLITS)
    previous_annotations = pd.read_csv(ANNOTATIONS)
    updated, splits, annotations = build()
    if not previous.drop(columns="model").equals(updated.drop(columns="model")):
        raise ValueError("A non-model review field changed; migration stopped.")
    if not previous_splits.equals(splits) or not previous_annotations.equals(annotations):
        raise ValueError("A split or aspect annotation changed; migration stopped.")
    if len(np.load(ROOT / "data" / "processed" / "review_embeddings.npy", mmap_mode="r")) != len(updated):
        raise ValueError("Cached embeddings do not align with the review rows.")
    validate_reviews(updated)

    by_id = updated.set_index("review_id")[["brand", "model"]]
    keyed_outputs = []
    for path in MODEL_KEYED_FILES:
        frame = pd.read_csv(path)
        replacement = frame["review_id"].map(by_id["model"])
        expected_brand = frame["review_id"].map(by_id["brand"])
        # The predecessor ABSA export uses numeric aspect-row identifiers for
        # its original rows, while added reviews retain review IDs. Keep those
        # original rows unchanged and require every placeholder row to match.
        placeholders = frame["model"].str.contains(r"\bStudy\s+[A-Z]\b", regex=True)
        if (placeholders & replacement.isna()).any():
            raise ValueError(f"Placeholder IDs do not align in {path.name}; migration stopped.")
        matched = replacement.notna()
        if not frame.loc[matched, "brand"].equals(expected_brand.loc[matched]):
            raise ValueError(f"Brands do not align in {path.name}; migration stopped.")
        frame["model"] = replacement.fillna(frame["model"])
        keyed_outputs.append((path, frame))

    updated.to_csv(OUTPUT, index=False)
    for path, frame in keyed_outputs:
        frame.to_csv(path, index=False)
    result = refresh_conflicts()
    result["mapped_placeholder_models"] = 18
    return result


if __name__ == "__main__":
    print(migrate())
