"""Leakage-aware text models for dissatisfaction and rating prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score)
from xgboost import XGBClassifier, XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "trained_models" / "product_intelligence"
EVAL_DIR = ROOT / "evaluation"
NUMERIC_NAMES = ["characters", "words", "exclamations", "questions", "sentence_count"]


def numeric_features(texts: list[str]) -> csr_matrix:
    values = []
    for text in texts:
        value = str(text)
        values.append([min(len(value), 1000) / 1000, min(len(value.split()), 200) / 200,
                       min(value.count("!"), 5) / 5, min(value.count("?"), 5) / 5,
                       min(value.count("."), 10) / 10])
    return csr_matrix(np.asarray(values, dtype=np.float32))


@dataclass
class ReviewPredictor:
    vectorizer: TfidfVectorizer
    model: object
    task: str
    threshold: float = 0.5

    @property
    def feature_names(self) -> list[str]:
        return self.vectorizer.get_feature_names_out().tolist() + NUMERIC_NAMES

    def transform(self, texts: list[str]):
        cleaned = [str(t or "") for t in texts]
        return hstack([self.vectorizer.transform(cleaned), numeric_features(cleaned)], format="csr")

    def predict(self, text: str) -> dict:
        if not str(text).strip():
            raise ValueError("Enter review text before requesting a prediction.")
        matrix = self.transform([text])
        if self.task == "dissatisfaction":
            probability = float(self.model.predict_proba(matrix)[0, 1])
            return {"dissatisfied": probability >= self.threshold, "probability": probability,
                    "threshold": self.threshold}
        rating = float(np.clip(self.model.predict(matrix)[0], 1, 5))
        return {"predicted_rating": rating}

    def explain(self, text: str, top_n: int = 8) -> list[dict]:
        """SHAP contributions for exactly the feature vector fed to XGBoost."""
        import shap

        row = self.transform([text])
        values = np.asarray(shap.TreeExplainer(self.model).shap_values(row))[0]
        dense = row.toarray()[0]
        strongest = np.argsort(np.abs(values))[-top_n:][::-1]
        return [{"feature": self.feature_names[i] + (" (absent)" if dense[i] == 0 else ""),
                 "value": float(dense[i]), "contribution": float(values[i])} for i in strongest]


def classification_metrics(y_true, probability, threshold: float = 0.5) -> dict:
    predicted = (probability >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "decision_threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }


def regression_metrics(y_true, predicted) -> dict:
    return {"mae": float(mean_absolute_error(y_true, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, predicted))),
            "r2": float(r2_score(y_true, predicted))}


def train() -> dict:
    data = pd.read_csv(ROOT / "data" / "processed" / "reviews_expanded.csv")
    partitions = pd.read_csv(ROOT / "data" / "processed" / "review_splits.csv")
    data = data.merge(partitions, on="review_id", validate="one_to_one")
    groups = {key: data[data["split"] == key].reset_index(drop=True) for key in ("train", "validation", "test")}
    vectorizer = TfidfVectorizer(max_features=600, ngram_range=(1, 2), min_df=4,
                                 sublinear_tf=True, strip_accents="unicode")
    train_text = groups["train"]["review_text"].astype(str).tolist()
    matrices = {"train": hstack([vectorizer.fit_transform(train_text), numeric_features(train_text)], format="csr")}
    for key in ("validation", "test"):
        texts = groups[key]["review_text"].astype(str).tolist()
        matrices[key] = hstack([vectorizer.transform(texts), numeric_features(texts)], format="csr")

    # Target uses the provided whole-review label. Rating and all aspect rating
    # columns are excluded from inputs. The source's labels are synthetic.
    y_class = {key: group["sentiment"].eq("Negative").astype(int).to_numpy() for key, group in groups.items()}
    y_rating = {key: group["rating"].astype(float).to_numpy() for key, group in groups.items()}
    baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
    baseline.fit(matrices["train"], y_class["train"])
    baseline_score = classification_metrics(y_class["test"], baseline.predict_proba(matrices["test"])[:, 1])

    trials = []
    for depth in (3, 5):
        candidate = XGBClassifier(n_estimators=180, max_depth=depth, learning_rate=0.07,
                                  subsample=0.9, colsample_bytree=0.85, tree_method="hist",
                                  eval_metric="logloss", random_state=42, n_jobs=4)
        candidate.fit(matrices["train"], y_class["train"])
        probability = candidate.predict_proba(matrices["validation"])[:, 1]
        threshold, score = max(
            ((float(t), classification_metrics(y_class["validation"], probability, float(t)))
             for t in np.arange(0.2, 0.61, 0.05)),
            key=lambda result: result[1]["f1"]
        )
        trials.append((score["f1"], depth, threshold, candidate))
    _, chosen_depth, chosen_threshold, classifier = max(trials, key=lambda result: result[0])
    class_score = classification_metrics(y_class["test"], classifier.predict_proba(matrices["test"])[:, 1], chosen_threshold)

    rating_baseline = DummyRegressor(strategy="mean")
    rating_baseline.fit(matrices["train"], y_rating["train"])
    rating_base_score = regression_metrics(y_rating["test"], rating_baseline.predict(matrices["test"]))
    rating_trials = []
    for depth in (3, 5):
        candidate = XGBRegressor(n_estimators=180, max_depth=depth, learning_rate=0.07,
                                 subsample=0.9, colsample_bytree=0.85, tree_method="hist",
                                 objective="reg:squarederror", random_state=42, n_jobs=4)
        candidate.fit(matrices["train"], y_rating["train"])
        score = regression_metrics(y_rating["validation"], candidate.predict(matrices["validation"]))
        rating_trials.append((score["mae"], depth, candidate))
    _, rating_depth, regressor = min(rating_trials, key=lambda result: result[0])
    rating_score = regression_metrics(y_rating["test"], regressor.predict(matrices["test"]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ReviewPredictor(vectorizer, classifier, "dissatisfaction", chosen_threshold), MODEL_DIR / "dissatisfaction.joblib", compress=3)
    joblib.dump(ReviewPredictor(vectorizer, regressor, "rating"), MODEL_DIR / "rating.joblib", compress=3)
    all_text = data["review_text"].astype(str).tolist()
    all_matrix = hstack([vectorizer.transform(all_text), numeric_features(all_text)], format="csr")
    pd.DataFrame({"review_id": data["review_id"], "model": data["model"], "brand": data["brand"],
                  "dissatisfaction_probability": classifier.predict_proba(all_matrix)[:, 1],
                  "predicted_rating": np.clip(regressor.predict(all_matrix), 1, 5)})\
        .to_csv(ROOT / "data" / "processed" / "supervised_predictions.csv", index=False)
    report = {
        "data": {"rows": int(len(data)), "models": int(data.model.nunique()),
                 "train": len(groups["train"]), "validation": len(groups["validation"]), "test": len(groups["test"]),
                 "split": "grouped by smartphone model; final test held out from fitting and selection",
                 "target_dissatisfaction": "sentiment == Negative (synthetic source label)",
                 "target_rating": "integer rating on the observed 1–5 scale"},
        "dissatisfaction": {"baseline_logistic": baseline_score, "xgboost": class_score,
                            "selected_max_depth": chosen_depth},
        "rating": {"baseline_mean": rating_base_score, "xgboost": rating_score,
                   "selected_max_depth": rating_depth},
    }
    (EVAL_DIR / "supervised_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_predictors() -> tuple[ReviewPredictor, ReviewPredictor]:
    return (joblib.load(MODEL_DIR / "dissatisfaction.joblib"), joblib.load(MODEL_DIR / "rating.joblib"))


if __name__ == "__main__":
    # Import through the stable module name so persisted ReviewPredictor
    # instances can be loaded from Streamlit and notebooks.
    from supervised import train as canonical_train

    print(json.dumps(canonical_train(), indent=2))
