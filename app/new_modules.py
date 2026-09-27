"""Interactive research modules shared with the existing Streamlit application."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emerging import ASPECT_TERMS, TOPICS_CSV, detect_issues  # noqa: E402
from semantic import EMBEDDINGS, discover, related_models  # noqa: E402
from supervised import load_predictors  # noqa: E402


@st.cache_data
def expanded_reviews() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "reviews_expanded.csv"
    return pd.read_csv(path)


@st.cache_data
def expanded_aspects() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "absa_expanded_predictions.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def saved_predictions() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "supervised_predictions.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def complaint_topics() -> pd.DataFrame:
    return pd.read_csv(TOPICS_CSV) if TOPICS_CSV.exists() else pd.DataFrame()


@st.cache_resource
def embedding_matrix() -> np.ndarray:
    if not EMBEDDINGS.exists():
        raise FileNotFoundError("Review embeddings are unavailable. Run python src/semantic.py first.")
    return np.load(EMBEDDINGS, mmap_mode="r")


@st.cache_resource
def predictors():
    return load_predictors()


def prediction_panel(review_text: str) -> None:
    st.markdown("### Consumer response predictions")
    st.caption("These are model estimates from synthetic review text, not an actual customer's rating or a measured outcome.")
    try:
        dissatisfied, rating = predictors()
        outcome = dissatisfied.predict(review_text)
        rating_outcome = rating.predict(review_text)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        st.info(f"Predictions are unavailable: {exc}")
        return
    first, second = st.columns(2)
    first.metric("Dissatisfaction", "Likely" if outcome["dissatisfied"] else "Unlikely",
                 help=f"Estimated probability: {outcome['probability']:.1%}; decision threshold: {outcome['threshold']:.0%}.")
    second.metric("Predicted rating", f"{rating_outcome['predicted_rating']:.1f} / 5",
                  help="A text-based regression estimate on the dataset's 1–5 rating scale.")
    st.caption(f"Dissatisfaction probability: {outcome['probability']:.1%}. The label uses a validation-selected threshold of {outcome['threshold']:.0%}.")
    if st.checkbox("Explain these predictions", key="show_shap"):
        for title, model in (("Dissatisfaction", dissatisfied), ("Rating", rating)):
            try:
                contributions = pd.DataFrame(model.explain(review_text, top_n=8))
            except (ImportError, ValueError) as exc:
                st.info(f"{title} explanation is unavailable: {exc}")
                continue
            if contributions.empty:
                st.info(f"No informative features were found for the {title.lower()} explanation.")
                continue
            contributions["Direction"] = np.where(contributions["contribution"] >= 0, "Raises estimate", "Lowers estimate")
            figure = px.bar(contributions.sort_values("contribution"), x="contribution", y="feature",
                            color="Direction", orientation="h", title=f"What influenced the {title.lower()} estimate",
                            color_discrete_map={"Raises estimate": "#f49b80", "Lowers estimate": "#6ee7d1"})
            figure.update_layout(yaxis_title="Review word or text feature", xaxis_title="SHAP contribution")
            st.plotly_chart(figure, width="stretch")
        st.caption("SHAP describes this model's calculation for this review. It does not establish cause and effect.")


def comparison_prediction_panel(models: list[str]) -> None:
    predictions = saved_predictions()
    if predictions.empty:
        return
    chosen = predictions[predictions["model"].isin(models)]
    summary = chosen.groupby("model").agg(
        reviews=("review_id", "size"),
        mean_dissatisfaction_probability=("dissatisfaction_probability", "mean"),
        mean_predicted_rating=("predicted_rating", "mean")
    ).reset_index()
    st.subheader("Predicted opinion patterns")
    st.caption("Averages of model predictions over synthetic review text. These are exploratory summaries, not observed ratings.")
    st.dataframe(summary.round(3), width="stretch", hide_index=True)


def aspect_evidence_panel(model: str) -> None:
    aspects = expanded_aspects()
    if aspects.empty:
        return
    selected = aspects[aspects["model"] == model]
    if selected.empty:
        return
    st.subheader("Supporting review evidence")
    left, right = st.columns(2)
    aspect = left.selectbox("Aspect to inspect", sorted(selected["aspect"].unique()), key="evidence_aspect")
    sentiment = right.selectbox("Opinion", ["Negative", "Positive", "Neutral"], key="evidence_sentiment")
    evidence = selected[(selected["aspect"] == aspect) & (selected["predicted_sentiment"] == sentiment)]
    if evidence.empty:
        st.info("No matching aspect predictions were found for this selection.")
    else:
        for _, row in evidence.head(4).iterrows():
            st.markdown(f"> {row['review_text']}")
        st.caption(f"Showing {min(4, len(evidence))} of {len(evidence)} matching aspect predictions.")


def weakness_evidence_panel(aspect: str, model: str | None = None, brand: str | None = None) -> None:
    aspects = expanded_aspects()
    if aspects.empty:
        return
    filtered = aspects[(aspects["aspect"] == aspect) & (aspects["predicted_sentiment"] == "Negative")]
    if model:
        filtered = filtered[filtered["model"] == model]
    if brand:
        filtered = filtered[filtered["brand"] == brand]
    st.subheader("Negative review evidence")
    if filtered.empty:
        st.info("No negative reviews match these filters.")
    else:
        for _, row in filtered.head(5).iterrows():
            st.markdown(f"**{row['model']}** — {row['review_text']}")


def discovery_page(page_intro) -> None:
    page_intro("Unsupervised analysis", "Consumer Opinion Discovery",
               "Explore groups of semantically similar reviews without assigning predefined categories to them.")
    reviews = expanded_reviews()
    if not EMBEDDINGS.exists():
        st.info("Sentence embeddings are missing. Run `python src/semantic.py` to prepare this module.")
        return
    brands = ["All brands", *sorted(reviews["brand"].unique())]
    selected_brand = st.selectbox("Brand", brands, key="discovery_brand")
    subset = reviews if selected_brand == "All brands" else reviews[reviews["brand"] == selected_brand]
    selected_model = st.selectbox("Model", ["All models", *sorted(subset["model"].unique())], key="discovery_model")
    if selected_model != "All models":
        subset = subset[subset["model"] == selected_model]
    method = st.radio("Clustering method", ["K-Means", "HDBSCAN"], horizontal=True)
    if method == "K-Means":
        parameter = st.slider("Number of clusters", 3, 16, 8)
    else:
        parameter = st.slider("Minimum reviews per cluster", 10, 80, 25, step=5)
    st.caption(f"{len(subset):,} matching reviews. For responsive exploration, clustering uses up to 2,500 sampled reviews.")
    if st.button("Discover opinion patterns", type="primary"):
        if subset.empty:
            st.info("Choose a brand or model with reviews first.")
            return
        indices = subset.index.to_numpy()
        if len(indices) > 2500:
            indices = np.sort(np.random.default_rng(42).choice(indices, 2500, replace=False))
        with st.spinner("Grouping semantically similar reviews..."):
            try:
                labels, report = discover(reviews, embedding_matrix(), indices, method=method,
                                          n_clusters=parameter if method == "K-Means" else 8,
                                          min_cluster_size=parameter if method == "HDBSCAN" else 25)
            except (ValueError, MemoryError) as exc:
                st.info(f"Clustering could not produce a useful result: {exc}")
                return
        st.session_state["discovery_result"] = (labels, report)
        st.session_state["discovery_signature"] = (selected_brand, selected_model, method, parameter)
    if st.session_state.get("discovery_signature") != (selected_brand, selected_model, method, parameter):
        st.info("Choose settings and run discovery to inspect clusters.")
        return
    labels, report = st.session_state["discovery_result"]
    summaries = pd.DataFrame([{k: v for k, v in cluster.items() if k != "representative_reviews"}
                              for cluster in report["clusters"]])
    st.subheader("Discovered patterns")
    st.dataframe(summaries, width="stretch", hide_index=True)
    if report["silhouette_cosine"] is not None:
        st.caption(f"Cosine silhouette score on a sample: {report['silhouette_cosine']:.3f}. This measures separation, not category truth.")
    selected_cluster = st.selectbox("Inspect cluster", summaries["cluster"].tolist(),
                                    format_func=lambda value: "Unassigned reviews" if value == -1 else f"Cluster {value}")
    cluster_info = next(item for item in report["clusters"] if item["cluster"] == selected_cluster)
    st.markdown(f"**Frequent terms:** {cluster_info['top_terms']}")
    for review in cluster_info["representative_reviews"]:
        st.markdown(f"> {review}")
    st.dataframe(labels[labels["cluster"] == selected_cluster][["brand", "model", "review_text", "sentiment"]].head(100),
                 width="stretch", hide_index=True)
    st.caption("Cluster names are derived from frequent words. A cluster can contain multiple aspects or sentiment labels.")


def relationships_page(page_intro) -> None:
    page_intro("Semantic exploration", "Model Relationships",
               "Find phones whose review language is similar, then compare the opinion patterns behind the match.")
    reviews = expanded_reviews()
    if not EMBEDDINGS.exists():
        st.info("Sentence embeddings are missing. Run `python src/semantic.py` first.")
        return
    first, second = st.columns(2)
    model = first.selectbox("Reference smartphone", sorted(reviews["model"].unique()), key="relation_model")
    brand = second.selectbox("Related brand", ["All brands", *sorted(reviews["brand"].unique())], key="relation_brand")
    aspect = st.selectbox("Opinion focus", ["All aspects", *ASPECT_TERMS.keys()], key="relation_aspect")
    subset = reviews
    if aspect != "All aspects":
        subset = reviews[reviews["review_text"].str.contains(ASPECT_TERMS[aspect], case=False, regex=True, na=False)]
    try:
        neighbors = related_models(subset, embedding_matrix(), model,
                                   brand=None if brand == "All brands" else brand)
    except ValueError as exc:
        st.info(str(exc))
        return
    if neighbors.empty:
        st.info("No related models match this brand and aspect filter.")
        return
    neighbors["similarity"] = neighbors["similarity"].round(3)
    st.dataframe(neighbors, width="stretch", hide_index=True)
    selected = st.selectbox("Compare with", neighbors["model"].tolist(), key="related_comparison")
    st.caption("Similarity is cosine similarity between mean normalized Sentence-BERT review embeddings. It measures language overlap, not product quality.")
    aspects = expanded_aspects()
    if not aspects.empty:
        comparison = aspects[aspects["model"].isin([model, selected])]
        comparison = comparison.groupby(["model", "aspect"])["predicted_sentiment"].apply(
            lambda values: 100 * (values.eq("Positive").mean() - values.eq("Negative").mean())).reset_index(name="sentiment_score")
        st.subheader("Aspect sentiment patterns")
        st.dataframe(comparison.pivot(index="aspect", columns="model", values="sentiment_score").round(1), width="stretch")
    topics = complaint_topics()
    if not topics.empty:
        joined = reviews[["review_id", "model"]].merge(topics, on="review_id")
        each = {name: set(joined[(joined["model"] == name) & (joined["topic"] >= 0)]["topic"]) for name in (model, selected)}
        shared = each[model] & each[selected]
        if shared:
            descriptions = topics[topics["topic"].isin(shared)]["top_terms"].drop_duplicates().head(3).tolist()
            st.markdown("**Shared complaint vocabulary:** " + "; ".join(descriptions))
    comparison_prediction_panel([model, selected])
    st.caption("These relationships describe patterns within a synthetic dataset; they do not establish equivalent product performance.")


def emerging_page(page_intro, selected_model: str | None = None) -> None:
    page_intro("Temporal signal detection", "Emerging Issues",
               "Compare complaint-theme prevalence across calendar years while accounting for review volume.")
    reviews = expanded_reviews()
    topics = complaint_topics()
    if topics.empty:
        st.info("Complaint topics are missing. Run `python src/emerging.py` first.")
        return
    a, b, c = st.columns(3)
    brand = a.selectbox("Brand", ["All brands", *sorted(reviews["brand"].unique())], key="issues_brand")
    candidates = reviews if brand == "All brands" else reviews[reviews["brand"] == brand]
    model_options = ["All models", *sorted(candidates["model"].unique())]
    model_index = model_options.index(selected_model) if selected_model in model_options else 0
    model = b.selectbox("Smartphone", model_options, index=model_index, key="issues_model")
    aspect = c.selectbox("Aspect", ["All aspects", *ASPECT_TERMS.keys()], key="issues_aspect")
    dates = pd.to_datetime(reviews["review_date"])
    date_range = st.date_input("Review window", value=(dates.min().date(), dates.max().date()),
                               min_value=dates.min().date(), max_value=dates.max().date(), key="issues_dates")
    if st.button("Find increasing complaint themes", type="primary"):
        if len(date_range) != 2:
            st.info("Choose both a start and an end date.")
            return
        with st.spinner("Comparing complaint prevalence by period..."):
            result = detect_issues(reviews, topics, brand=None if brand == "All brands" else brand,
                                   model=None if model == "All models" else model,
                                   aspect=None if aspect == "All aspects" else aspect,
                                   start=str(date_range[0]), end=str(date_range[1]))
        st.session_state["issue_result"] = result
        st.session_state["issue_signature"] = (brand, model, aspect, tuple(date_range))
    if st.session_state.get("issue_signature") != (brand, model, aspect, tuple(date_range)):
        st.info("Select filters and run detection to investigate period-over-period changes.")
        return
    result = st.session_state["issue_result"]
    if result.empty:
        st.info("No complaint theme met the sample-size and statistical evidence requirements for these filters.")
        return
    display = result[["top_terms", "previous_period", "current_period", "previous_share", "current_share", "increase_pp", "adjusted_p"]].copy()
    display[["previous_share", "current_share"]] = (display[["previous_share", "current_share"]] * 100).round(1)
    st.dataframe(display.round(3), width="stretch", hide_index=True)
    selected = st.selectbox("Inspect an increasing theme", result.index.tolist(),
                            format_func=lambda i: f"{result.loc[i, 'top_terms']} · +{result.loc[i, 'increase_pp']:.1f} pp")
    row = result.loc[selected]
    st.markdown(f"**{row['previous_period']}:** {row['previous_count']} of {row['previous_total']} reviews · "
                f"**{row['current_period']}:** {row['current_count']} of {row['current_total']} reviews")
    for review in row["evidence"]:
        st.markdown(f"> {review}")
    st.caption("Topics come from historical negative-review clustering. A result requires at least 40 reviews in each period, five current topic reviews and a one-sided Fisher test with false-discovery adjustment ≤ 0.10. These are synthetic patterns, not market trends.")
