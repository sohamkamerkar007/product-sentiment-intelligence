# Product Sentiment Intelligence

An interactive NLP research application for studying smartphone reviews at both review and product level. It extends the completed **Fine-Grained Consumer Opinion Intelligence System** with dissatisfaction and rating prediction, semantic review discovery, related-product exploration, and screened complaint trends.

> **Research-data notice:** The 50 phone names refer to real released models, but **every review, rating, and review date in this demonstration is synthetic**. The data is not authentic customer feedback. Predictions and emerging issues demonstrate a method on generated data—not measured market performance or consumer advice.

## Application modules

| Page | Purpose |
| --- | --- |
| Home | Research question, five phone features and analysis pipeline. |
| Compare Phones | Feature-level opinion comparison and predicted consumer response. |
| Review Analyzer | Aspect detection, saved DistilBERT ABSA inference, dissatisfaction/rating prediction and SHAP explanations. |
| Phone Features | A phone's battery, camera, performance, design and display opinions with review evidence. |
| Common Problems | Recurrent negative feedback across brands and models. |
| Review Trends | Earlier-versus-later opinion patterns using the original six-month analysis. |
| Likes & Dislikes | Within-review trade-offs: one positive feature alongside a negative one. |
| Discover Patterns | K-Means or HDBSCAN groups of semantically similar reviews with terms and examples. |
| Find Similar Phones | Phones ranked by cosine similarity of aggregated review embeddings. |
| Spot New Problems | Complaint themes whose share of all reviews increased from 2025 to 2026. |

The original seven analytical pages and saved ABSA weights remain in place. The new modules use the same Python, Streamlit and Plotly architecture.

The interface uses Streamlit's native top navigation, a shared light design system in `app/assets/style.css`, and reusable presentation helpers in `app/ui.py`. The home page is a guided entry point; technical explanations remain available in optional sections. UI styling does not alter model or dataset outputs.

## Dataset and provenance

- `data/raw/Mobile_Reviews_GenData.csv`: original 4,200 `Synthetic-V2` reviews, unchanged.
- `data/processed/reviews_expanded.csv`: 7,969 reviews for 50 models across 9 brands; 140–180 reviews per model. The 3,769 added rows are marked `Synthetic-Expansion`, with IDs beginning `SYNEXP_`.
- The 18 former `Brand Study A/B` identifiers were replaced with real released phone names. Their [canonical registry and manufacturer references](src/model_registry.py) make each substitution traceable. All 50 names are real product names, **not** evidence that the generated reviews were written about those products.
- Review dates are simulated for research comparisons and do not establish actual release chronology; some generated review dates may predate a phone's launch. Do not interpret time trends as historical market observations.
- `src/generate_dataset.py` uses a fixed seed and phrase templates, preserves original records at field level, validates schema/duplicates/labels/dates/model counts, and writes a train/validation/test split grouped by phone model. `src/migrate_model_registry.py` updates saved model-keyed analytics without retraining models or changing review text, IDs, splits, or row order.
- The generator **deliberately** increases negative battery wording in some later-period Google synthetic rows to exercise emerging-issue detection. This is simulated drift, not a discovery about Google products.

Because the corpus is templated and synthetic, lexical overlap may inflate generalization estimates even with held-out models. Independent real-review validation is required before substantive use.

## Methods

**Aspect analysis:** Sentence Transformer `all-MiniLM-L6-v2` helps match review text to battery, camera, performance, design and display. The existing locally saved DistilBERT ABSA model labels detected aspect evidence Positive, Neutral or Negative. Its weights were not retrained. Original TF-IDF/Logistic Regression and DistilBERT research notebooks remain available.

**Consumer-response prediction:** TF-IDF unigrams/bigrams (600 features) and five text-length/punctuation features feed an XGBoost dissatisfaction classifier and XGBoost rating regressor. Review text is the only input: rating, labels, aspect ratings and phone metadata are excluded. Dissatisfaction means the source whole-review label is `Negative`; rating uses the source 1–5 integer scale. Logistic Regression and mean-rating predictors provide baselines. Train/validation/test partitions are grouped by model. Validation selects XGBoost depth and the dissatisfaction threshold; test is held out. SHAP TreeExplainer attributes model feature contributions, not causes.

**Unsupervised discovery:** Cached, normalized 384-dimensional Sentence-BERT review embeddings support K-Means and HDBSCAN. TF-IDF terms and representative reviews aid interpretation. Related phones are ranked by cosine similarity of mean review embeddings. This is language similarity, not a product-quality measure.

**Emerging issues:** K-Means complaint topics are fitted to pre-2026 negative reviews and then assigned to all negative reviews. A one-sided Fisher exact test compares each theme's share of **all reviews** in adjacent calendar years. Benjamini–Hochberg correction covers every topic/period comparison. Reporting requires adjusted *p* ≤ 0.10, at least 40 reviews in each period and five current-period complaints. The 2026 window is incomplete; shares normalize for review volume but not all seasonal effects. Filters change the analysis population and its statistical test.

## Evaluation snapshot

These are saved synthetic-data results from `evaluation/`, not live-market performance:

| Task / held-out set | Result |
| --- | --- |
| XGBoost dissatisfaction; 1,273 reviews from held-out phone models | Accuracy 0.802, positive F1 0.641, macro F1 0.752, ROC-AUC 0.884 |
| Logistic Regression dissatisfaction baseline; same set | Positive F1 0.648, macro F1 0.755, ROC-AUC 0.892 |
| XGBoost rating; same set | MAE 0.650, RMSE 0.781, R² 0.392 |
| Mean-rating baseline; same set | MAE 0.916 |
| Unchanged ABSA model; 1,488 synthetic aspect-evidence mentions from 8 held-out models | Accuracy 0.804, macro F1 0.793 |
| K-Means / HDBSCAN; fixed 1,200-review sample | Cosine silhouette 0.106 / 0.094; HDBSCAN noise 583 reviews |

The Logistic Regression baseline slightly **outperforms XGBoost** on reported dissatisfaction F1 and ROC-AUC. XGBoost is retained as the requested explainable tree-based comparison, not claimed as the better classifier. ABSA was evaluated on generated evidence spans, not independently annotated real reviews. Clustering scores are descriptive, not ground-truth accuracy. See [evaluation notes](evaluation/README.md) for caveats and an original-notebook metric discrepancy.

## Run locally

Python 3.11+ is recommended. The saved ABSA weights use Git LFS.

```powershell
git lfs install
git lfs pull
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/app.py
```

On macOS/Linux, activate with `source .venv/bin/activate`. First use may download the public `all-MiniLM-L6-v2` Sentence Transformer unless already cached. Saved XGBoost models, CSV analytics and embeddings are included; app startup does not retrain or regenerate them. Streamlit caching retains loaded models and data across reruns.

## Reproduce research outputs

The app does not require these scripts at startup. Run them only to intentionally rebuild generated artifacts, in this order:

```powershell
python src/generate_dataset.py
python src/supervised.py
python src/semantic.py
python src/emerging.py
python src/expand_absa.py
python src/refresh_analytics.py
python src/evaluate_discovery.py
python -m unittest discover -s tests -v
```

`src/expand_absa.py` performs CPU-heavy transformer inference and may take several minutes. Results can vary slightly across dependency/platform versions. Original notebooks document predecessor experiments; added methods live in `src/`.

## Repository layout

```text
product-sentiment-intelligence/
├── app/             Streamlit app, CSS, shared UI helpers
├── data/raw/        Unchanged original synthetic source
├── data/processed/  Expanded data, splits, predictions, topics, embeddings
├── evaluation/      Metrics and evaluation notes
├── models/          ABSA, baseline and new XGBoost/topic models
├── notebooks/       Original research notebooks
├── src/             Data generation, modelling and analysis scripts
├── tests/           Data, model and discovery smoke tests
├── requirements.txt
└── README.md
```

## Limits and responsible use

This is a portfolio/research demonstration, not a production system. It has no real customer data, independent human-rated evaluation, product specifications, causal inference or calibration study. Templates repeat linguistic patterns and synthetic drift was deliberately introduced. Topic labels are automatically derived terms; inspect example reviews before interpreting them. Do not use these outputs to rank commercial products or make purchasing decisions without independent real-world evidence.
