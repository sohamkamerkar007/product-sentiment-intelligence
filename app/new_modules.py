"""Interactive research modules shared with the existing Streamlit application."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import ui

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
    st.markdown("### What this review suggests")
    st.caption("These estimates come from generated research data, not a real buyer's experience.")
    try:
        dissatisfied, rating = predictors()
        outcome = dissatisfied.predict(review_text)
        rating_outcome = rating.predict(review_text)
    except (FileNotFoundError, ValueError, ImportError):
        st.info("These estimates are temporarily unavailable. The review analysis above is still available.")
        return
    first, second = st.columns(2)
    first.metric("Chance of a negative review", f"{outcome['probability']:.0%}",
                 help="An estimate based on the words in this review and synthetic training examples.")
    second.metric("Estimated rating", f"{rating_outcome['predicted_rating']:.1f} / 5",
                  help="An estimate on the dataset's 1–5 scale, not an observed rating.")
    st.caption("The negative-review estimate is " + ("above" if outcome["dissatisfied"] else "below") +
               " the system's decision cutoff.")
    if st.checkbox("Why did the system give these results?", key="show_shap"):
        for title, model in (("negative-review chance", dissatisfied), ("estimated rating", rating)):
            try:
                contributions = pd.DataFrame(model.explain(review_text, top_n=8))
            except (ImportError, ValueError):
                st.info(f"The explanation for {title} is unavailable right now.")
                continue
            if contributions.empty:
                st.info(f"No informative features were found for the {title.lower()} explanation.")
                continue
            contributions["Direction"] = np.where(contributions["contribution"] >= 0, "Raises estimate", "Lowers estimate")
            figure = px.bar(contributions.sort_values("contribution"), x="contribution", y="feature",
                            color="Direction", orientation="h", title=f"Words affecting the {title}",
                            color_discrete_map={"Raises estimate": "#c24158", "Lowers estimate": "#0f766e"})
            figure.update_layout(yaxis_title="Word or text pattern", xaxis_title="Effect on this estimate")
            st.plotly_chart(figure, width="stretch")
        with st.expander("Technical details about these explanations"):
            st.write("SHAP shows how the saved XGBoost models used text features for this review. "
                     "It explains a calculation, not why a person would feel a certain way.")


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
    st.subheader("Estimated review patterns")
    st.caption("Averages across generated reviews. These are not observed customer ratings.")
    summary["mean_dissatisfaction_probability"] = (100 * summary["mean_dissatisfaction_probability"]).round(1)
    st.dataframe(summary.rename(columns={"model": "Phone", "reviews": "Reviews",
                                         "mean_dissatisfaction_probability": "Estimated negative reviews (%)",
                                         "mean_predicted_rating": "Estimated rating / 5"}).round(1),
                 width="stretch", hide_index=True)


def aspect_evidence_panel(model: str, *, show_heading: bool = True,
                          key_prefix: str = "evidence") -> None:
    aspects = expanded_aspects()
    if aspects.empty:
        return
    selected = aspects[aspects["model"] == model]
    if selected.empty:
        return
    if show_heading:
        st.subheader("Example reviews behind these results")
    left, right = st.columns(2)
    aspect = left.selectbox("Phone feature", sorted(selected["aspect"].unique()), key=f"{key_prefix}_aspect")
    sentiment = right.selectbox("Opinion", ["Negative", "Positive", "Neutral"], key=f"{key_prefix}_sentiment")
    evidence = selected[(selected["aspect"] == aspect) & (selected["predicted_sentiment"] == sentiment)]
    if evidence.empty:
        st.info("No reviews match this feature and opinion choice.")
    else:
        for _, row in evidence.head(4).iterrows():
            ui.review_quote(str(row["review_text"]))
        st.caption(f"Showing {min(4, len(evidence))} of {len(evidence)} matching review examples.")


def weakness_evidence_panel(aspect: str, model: str | None = None, brand: str | None = None) -> None:
    aspects = expanded_aspects()
    if aspects.empty:
        return
    filtered = aspects[(aspects["aspect"] == aspect) & (aspects["predicted_sentiment"] == "Negative")]
    if model:
        filtered = filtered[filtered["model"] == model]
    if brand:
        filtered = filtered[filtered["brand"] == brand]
    st.subheader("Examples of negative comments")
    if filtered.empty:
        st.info("No negative reviews match these filters.")
    else:
        for _, row in filtered.head(5).iterrows():
            ui.review_quote(str(row["review_text"]), str(row["model"]))


def discovery_page(page_intro) -> None:
    page_intro("Explore reviews", "Discover Review Patterns",
               "Group reviews that use similar language, then read examples from each group.")
    ui.section_header("01", "Choose how to explore", "Filter the generated reviews and pick a grouping style.")
    reviews = expanded_reviews()
    if not EMBEDDINGS.exists():
        st.info("Review patterns are temporarily unavailable because prepared review data is missing.")
        return
    brands = ["All brands", *sorted(reviews["brand"].unique())]
    selected_brand = st.selectbox("Brand", brands, key="discovery_brand")
    subset = reviews if selected_brand == "All brands" else reviews[reviews["brand"] == selected_brand]
    selected_model = st.selectbox("Phone", ["All models", *sorted(subset["model"].unique())], key="discovery_model")
    if selected_model != "All models":
        subset = subset[subset["model"] == selected_model]
    method_label = st.radio("How should reviews be grouped?", ["Choose the number of groups", "Find natural groups"], horizontal=True)
    method = "K-Means" if method_label == "Choose the number of groups" else "HDBSCAN"
    if method == "K-Means":
        parameter = st.slider("Number of review groups", 3, 16, 8)
    else:
        parameter = st.slider("Smallest group size", 10, 80, 25, step=5)
    st.caption(f"{len(subset):,} matching reviews. Up to 2,500 are sampled to keep this page responsive.")
    if st.button("Find review groups", type="primary"):
        if subset.empty:
            st.info("Choose a brand or model with reviews first.")
            return
        indices = subset.index.to_numpy()
        if len(indices) > 2500:
            indices = np.sort(np.random.default_rng(42).choice(indices, 2500, replace=False))
        with st.spinner("Grouping reviews with similar wording..."):
            try:
                labels, report = discover(reviews, embedding_matrix(), indices, method=method,
                                          n_clusters=parameter if method == "K-Means" else 8,
                                          min_cluster_size=parameter if method == "HDBSCAN" else 25)
            except (ValueError, MemoryError):
                st.info("These reviews could not be grouped with the current choices. Try another phone, brand, or group setting.")
                return
        st.session_state["discovery_result"] = (labels, report)
        st.session_state["discovery_signature"] = (selected_brand, selected_model, method, parameter)
    if st.session_state.get("discovery_signature") != (selected_brand, selected_model, method, parameter):
        st.info("Choose your settings and select ‘Find review groups’ to begin.")
        return
    labels, report = st.session_state["discovery_result"]
    summaries = pd.DataFrame([{k: v for k, v in cluster.items() if k != "representative_reviews"}
                              for cluster in report["clusters"]])
    st.subheader("Review groups")
    st.dataframe(summaries.rename(columns={"cluster": "Group", "size": "Reviews",
                                            "top_terms": "Common words"}), width="stretch", hide_index=True)
    selected_cluster = st.selectbox("Explore a group", summaries["cluster"].tolist(),
                                    format_func=lambda value: "Reviews without a clear group" if value == -1 else f"Group {value + 1}")
    cluster_info = next(item for item in report["clusters"] if item["cluster"] == selected_cluster)
    st.markdown(f"**Common words:** {cluster_info['top_terms']}")
    for review in cluster_info["representative_reviews"]:
        ui.review_quote(str(review))
    st.dataframe(labels[labels["cluster"] == selected_cluster][["brand", "model", "review_text", "sentiment"]].head(100),
                 width="stretch", hide_index=True)
    st.caption("A group can contain different phone features and both positive and negative comments.")
    with st.expander("How the grouping works — technical details"):
        st.write("The first option uses K-Means; the second uses HDBSCAN. Both work on saved "
                 "Sentence-BERT review embeddings. Groups are described by common TF-IDF terms. "
                 "No human-verified topic labels are available.")
        if report["silhouette_cosine"] is not None:
            st.write(f"Sample cosine silhouette score: {report['silhouette_cosine']:.3f}.")


def relationships_page(page_intro) -> None:
    page_intro("Compare review language", "Find Similar Phones",
               "Find phones with similar-sounding reviews and inspect what those reviews discuss.")
    ui.section_header("01", "Start with a phone", "Choose a model and explore others with similar review wording.")
    reviews = expanded_reviews()
    if not EMBEDDINGS.exists():
        st.info("Similar-phone results are temporarily unavailable because prepared review data is missing.")
        return
    first, second = st.columns(2)
    model = first.selectbox("Start with this phone", sorted(reviews["model"].unique()), key="relation_model")
    brand = second.selectbox("Show phones from", ["All brands", *sorted(reviews["brand"].unique())], key="relation_brand")
    aspect = st.selectbox("Focus on a feature", ["All features", *ASPECT_TERMS.keys()], key="relation_aspect")
    subset = reviews
    if aspect != "All features":
        subset = reviews[reviews["review_text"].str.contains(ASPECT_TERMS[aspect], case=False, regex=True, na=False)]
    try:
        neighbors = related_models(subset, embedding_matrix(), model,
                                   brand=None if brand == "All brands" else brand)
    except ValueError:
        st.info("There are not enough matching reviews to compare this selection. Try another feature or brand.")
        return
    if neighbors.empty:
        st.info("No other phones match these choices. Try another brand or feature.")
        return
    neighbors["similarity"] = (100 * neighbors["similarity"]).round(1)
    st.dataframe(neighbors.rename(columns={"model": "Phone", "brand": "Brand",
                                               "similarity": "How similar the reviews are (%)",
                                               "review_count": "Reviews"}), width="stretch", hide_index=True)
    selected = st.selectbox("Compare with", neighbors["model"].tolist(), key="related_comparison")
    st.caption("This measures similarity in review wording, not how well the phones perform.")
    aspects = expanded_aspects()
    if not aspects.empty:
        comparison = aspects[aspects["model"].isin([model, selected])]
        comparison = comparison.groupby(["model", "aspect"])["predicted_sentiment"].apply(
            lambda values: 100 * (values.eq("Positive").mean() - values.eq("Negative").mean())).reset_index(name="sentiment_score")
        st.subheader("Opinions about each feature")
        st.dataframe(comparison.pivot(index="aspect", columns="model", values="sentiment_score").round(1), width="stretch")
    topics = complaint_topics()
    if not topics.empty:
        joined = reviews[["review_id", "model"]].merge(topics, on="review_id")
        each = {name: set(joined[(joined["model"] == name) & (joined["topic"] >= 0)]["topic"]) for name in (model, selected)}
        shared = each[model] & each[selected]
        if shared:
            descriptions = topics[topics["topic"].isin(shared)]["top_terms"].drop_duplicates().head(3).tolist()
            st.markdown("**Complaint themes seen in both phones:** " + "; ".join(descriptions))
    comparison_prediction_panel([model, selected])
    with st.expander("How similarity is calculated — technical details"):
        st.write("Each phone is represented by the mean of its saved Sentence-BERT review "
                 "embeddings. Cosine similarity compares those representations. Review wording "
                 "in this synthetic dataset does not establish equivalent product performance.")


def emerging_page(page_intro, selected_model: str | None = None) -> None:
    page_intro("Changes in complaints", "Spot New Problems",
               "See which complaint themes appear more often in the later review period.")
    ui.section_header("01", "Check for growing complaints", "Choose the phones, feature and time window to compare.")
    reviews = expanded_reviews()
    topics = complaint_topics()
    if topics.empty:
        st.info("Complaint patterns are temporarily unavailable because prepared review data is missing.")
        return
    a, b, c = st.columns(3)
    brand = a.selectbox("Brand", ["All brands", *sorted(reviews["brand"].unique())], key="issues_brand")
    candidates = reviews if brand == "All brands" else reviews[reviews["brand"] == brand]
    model_options = ["All models", *sorted(candidates["model"].unique())]
    model_index = model_options.index(selected_model) if selected_model in model_options else 0
    model = b.selectbox("Phone", model_options, index=model_index, key="issues_model")
    aspect = c.selectbox("Phone feature", ["All features", *ASPECT_TERMS.keys()], key="issues_aspect")
    dates = pd.to_datetime(reviews["review_date"])
    date_range = st.date_input("Dates to include", value=(dates.min().date(), dates.max().date()),
                               min_value=dates.min().date(), max_value=dates.max().date(), key="issues_dates")
    if st.button("Find growing complaints", type="primary"):
        if len(date_range) != 2:
            st.info("Choose both a start and an end date.")
            return
        with st.spinner("Comparing complaints across the selected periods..."):
            result = detect_issues(reviews, topics, brand=None if brand == "All brands" else brand,
                                   model=None if model == "All models" else model,
                                   aspect=None if aspect == "All features" else aspect,
                                   start=str(date_range[0]), end=str(date_range[1]))
        st.session_state["issue_result"] = result
        st.session_state["issue_signature"] = (brand, model, aspect, tuple(date_range))
    if st.session_state.get("issue_signature") != (brand, model, aspect, tuple(date_range)):
        st.info("Choose a brand, phone or feature, then select ‘Find growing complaints’.")
        return
    result = st.session_state["issue_result"]
    if result.empty:
        st.info("No growing complaint theme passed the reliability checks for these choices.")
        return
    display = result[["top_terms", "previous_period", "current_period", "previous_share", "current_share", "increase_pp"]].copy()
    display[["previous_share", "current_share"]] = (display[["previous_share", "current_share"]] * 100).round(1)
    display = display.rename(columns={"top_terms": "Complaint theme", "previous_period": "Earlier year",
                                      "current_period": "Later year", "previous_share": "Earlier reviews (%)",
                                      "current_share": "Later reviews (%)", "increase_pp": "Increase (percentage points)"})
    st.dataframe(display.round(1), width="stretch", hide_index=True)
    selected = st.selectbox("Read examples", result.index.tolist(),
                            format_func=lambda i: f"{result.loc[i, 'top_terms']} · +{result.loc[i, 'increase_pp']:.1f} percentage points")
    row = result.loc[selected]
    st.markdown(f"**{row['previous_period']}:** {row['previous_count']} of {row['previous_total']} reviews · "
                f"**{row['current_period']}:** {row['current_count']} of {row['current_total']} reviews")
    for review in row["evidence"]:
        ui.review_quote(str(review))
    st.caption("These changes describe generated reviews, not actual problems with the named phones.")
    with st.expander("How growing complaints are checked — technical details"):
        st.write("Complaint topics are learned from earlier negative reviews. A result requires "
                 "at least 40 reviews in each calendar year, five later complaints, and a "
                 "one-sided Fisher exact test with Benjamini–Hochberg adjusted p ≤ 0.10. "
                 "The later year is incomplete, so shares are divided by review volume.")
        st.write(f"Adjusted p-value for this theme: {row['adjusted_p']:.3f}.")
