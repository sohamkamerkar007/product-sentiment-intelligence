import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import torch
import re

from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sentence_transformers import SentenceTransformer
import new_modules as intelligence
import home
import ui


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Product Sentiment Intelligence",
    page_icon="📱",
    layout="wide"
)


# ============================================================
# APPLICATION PRESENTATION
# ============================================================

STYLE_PATH = Path(__file__).resolve().parent / "assets" / "style.css"


def load_styles():
    """Load the shared visual system without affecting application logic."""
    if STYLE_PATH.exists():
        st.markdown(
            f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True
        )


def page_intro(eyebrow, title, description):
    """Render a consistent title treatment for every analysis module."""
    ui.page_header(eyebrow, title, description)


load_styles()

research_template = pio.templates["plotly_white"]
research_template.layout.update(
    font={"family": "Aptos, Segoe UI, sans-serif", "color": "#344256", "size": 13},
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    colorway=["#4f46e5", "#0f766e", "#d97706", "#dc2626", "#64748b"],
    margin={"l": 36, "r": 30, "t": 72, "b": 45},
    xaxis={"gridcolor": "#edf0f5", "linecolor": "#d3dae6", "zeroline": False},
    yaxis={"gridcolor": "#edf0f5", "linecolor": "#d3dae6", "zeroline": False},
    legend={"orientation": "h", "y": -0.25, "title": {"text": ""}},
    hoverlabel={"bgcolor": "#172033", "font_color": "#ffffff"},
)
pio.templates["consumer_intelligence"] = research_template
px.defaults.template = "consumer_intelligence"
SENTIMENT_COLORS = {"Positive": "#15803d", "Negative": "#dc2626", "Neutral": "#b45309"}


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    files = {
        "reviews": "reviews_expanded.csv",
        "absa": "absa_expanded_predictions.csv",
        "weakness": "weakness_analysis.csv",
        "evolution": "evolution_analysis.csv",
        "phone_conflicts": "phone_aspect_conflicts.csv",
        "review_conflicts": "review_aspect_conflicts.csv",
        "conflict_summary": "conflict_summary.csv",
        "conflicts_by_phone": "conflicts_by_phone.csv",
        "aspect_conflicts": "aspect_conflict_summary.csv"
    }

    data = {}

    for key, filename in files.items():

        file_path = DATA_DIR / filename

        if file_path.exists():

            data[key] = pd.read_csv(file_path)

        else:

            data[key] = pd.DataFrame()

    return data


data = load_data()


# ============================================================
# LOAD ABSA TRANSFORMER MODEL
# ============================================================

ABSA_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "trained_models"
    / "distilbert_absa"
)


@st.cache_resource
def load_absa_model():

    tokenizer = AutoTokenizer.from_pretrained(
        ABSA_MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        ABSA_MODEL_PATH
    )

    model.eval()

    return tokenizer, model


absa_tokenizer, absa_model = load_absa_model()


# ============================================================
# LOAD SEMANTIC ASPECT MODEL
# ============================================================

@st.cache_resource
def load_semantic_model():

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


semantic_model = load_semantic_model()


# ============================================================
# SEMANTIC ASPECT PROFILES
# ============================================================

aspect_profiles = {

    "battery": [
        "battery life",
        "battery lasts a long time",
        "phone lasts all day",
        "phone lasts throughout the day",
        "long battery life",
        "poor battery life",
        "battery drains quickly",
        "fast battery drain",
        "battery capacity",
        "charging",
        "fast charging",
        "slow charging",
        "overnight battery drain",
        "power consumption",
        "how long the phone lasts",
        "daily battery endurance"
    ],

    "camera": [
        "camera quality",
        "photos look good",
        "photos look sharp",
        "photo quality",
        "taking pictures",
        "portrait photography",
        "night photography",
        "video recording",
        "camera focus",
        "autofocus",
        "image quality",
        "low light photography",
        "camera produces detailed photos",
        "sharp detailed images",
        "photography performance"
    ],

    "performance": [
        "phone feels smooth",
        "smooth performance",
        "phone is responsive",
        "fast and responsive",
        "apps open quickly",
        "apps run smoothly",
        "multitasking performance",
        "gaming performance",
        "games run smoothly",
        "no lag",
        "no stuttering",
        "phone feels fast",
        "slow performance",
        "phone feels sluggish",
        "apps are slow",
        "lag while using the phone",
        "processor performance",
        "system feels responsive",
        "fast everyday performance",
        "smooth user experience"
    ],

    "design": [
        "phone design",
        "build quality",
        "premium build",
        "feels premium",
        "comfortable to hold",
        "comfortable in the hand",
        "phone feels lightweight",
        "phone feels heavy",
        "phone is bulky",
        "phone looks beautiful",
        "physical buttons",
        "frame quality",
        "materials used",
        "overall appearance",
        "looks beautiful",
        "premium materials",
        "ergonomics of the phone"
    ],

    "display": [
        "display quality",
        "screen quality",
        "screen is bright",
        "display is bright",
        "bright screen",
        "screen looks beautiful",
        "colors look accurate",
        "vibrant colors",
        "screen is sharp",
        "high refresh rate",
        "smooth scrolling",
        "touch response",
        "screen visibility outdoors",
        "display brightness",
        "panel quality",
        "color reproduction",
        "screen visibility",
        "touch sensitivity"
    ]
}


# ============================================================
# PRECOMPUTE SEMANTIC ASPECT EMBEDDINGS
# ============================================================

@st.cache_resource
def create_aspect_embeddings():

    aspect_embeddings = {}

    for aspect, phrases in aspect_profiles.items():

        embeddings = semantic_model.encode(
            phrases,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        aspect_embeddings[aspect] = embeddings

    return aspect_embeddings


aspect_embeddings = create_aspect_embeddings()


# ============================================================
# EXPLICIT KEYWORDS
# ============================================================

explicit_keywords = {

    "battery": [
        "battery",
        "charging",
        "charge",
        "drain",
        "endurance"
    ],

    "camera": [
        "camera",
        "photo",
        "photos",
        "picture",
        "pictures",
        "portrait",
        "video",
        "focus",
        "photography"
    ],

    "performance": [
        "performance",
        "processor",
        "gaming",
        "game",
        "multitasking",
        "lag",
        "slow",
        "sluggish",
        "smooth",
        "responsive",
        "speed"
    ],

    "design": [
        "design",
        "build",
        "weight",
        "heavy",
        "lightweight",
        "premium",
        "frame",
        "buttons",
        "materials",
        "ergonomic"
    ],

    "display": [
        "display",
        "screen",
        "brightness",
        "bright",
        "colors",
        "refresh rate",
        "panel",
        "touch",
        "resolution"
    ]
}


# ============================================================
# SEMANTIC ASPECT DETECTOR
# ============================================================

def detect_aspects_semantically(review_text):

    text = review_text.lower().strip()

    # --------------------------------------------------------
    # Split review into sentences
    # --------------------------------------------------------

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    if not sentences:

        sentences = [text]


    aspect_scores = {
        aspect: 0.0
        for aspect in aspect_profiles
    }


    # --------------------------------------------------------
    # Analyze every sentence separately
    # --------------------------------------------------------

    for sentence in sentences:

        sentence_embedding = semantic_model.encode(
            sentence,
            convert_to_tensor=True,
            normalize_embeddings=True
        )


        for aspect, embeddings in aspect_embeddings.items():

            similarities = (
                torch.matmul(
                    embeddings,
                    sentence_embedding
                )
            )

            best_similarity = (
                torch.max(
                    similarities
                )
                .item()
            )


            # Keep the strongest semantic evidence
            aspect_scores[aspect] = max(
                aspect_scores[aspect],
                best_similarity
            )


    # --------------------------------------------------------
    # Explicit keyword boost
    # --------------------------------------------------------

    for aspect, keywords in explicit_keywords.items():

        for keyword in keywords:

            pattern = (
                r"\b"
                + re.escape(keyword)
                + r"\b"
            )

            if re.search(
                pattern,
                text
            ):

                aspect_scores[aspect] = min(
                    aspect_scores[aspect] + 0.20,
                    1.0
                )

                break


    # --------------------------------------------------------
    # Select relevant aspects
    # --------------------------------------------------------

    ASPECT_THRESHOLD = 0.42

    detected_aspects = [
        aspect
        for aspect, score in aspect_scores.items()
        if score >= ASPECT_THRESHOLD
    ]


    # --------------------------------------------------------
    # Sort by relevance
    # --------------------------------------------------------

    detected_aspects = sorted(
        detected_aspects,
        key=lambda x: aspect_scores[x],
        reverse=True
    )


    # --------------------------------------------------------
    # Fallback for moderately relevant reviews
    # --------------------------------------------------------

    if not detected_aspects:

        best_aspect = max(
            aspect_scores,
            key=aspect_scores.get
        )

        best_score = aspect_scores[
            best_aspect
        ]

        if best_score >= 0.35:

            detected_aspects = [
                best_aspect
            ]


    return detected_aspects, aspect_scores


# ============================================================
# ABSA SENTIMENT PREDICTION
# ============================================================

def predict_aspect_sentiment(
    aspect,
    review_text
):

    model_input = (
        "aspect: "
        + aspect
        + " review: "
        + review_text
    )


    encoded = absa_tokenizer(
        model_input,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )


    with torch.no_grad():

        outputs = absa_model(
            **encoded
        )


    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]


    predicted_id = (
        torch.argmax(
            probabilities
        )
        .item()
    )


    predicted_sentiment = (
        ABSA_ID2LABEL[
            predicted_id
        ]
    )


    confidence = (
        probabilities[
            predicted_id
        ]
        .item()
        * 100
    )


    return (
        predicted_sentiment,
        confidence
    )


# ============================================================
# ABSA LABEL MAPPING
# ============================================================

ABSA_ID2LABEL = {
    int(k): v
    for k, v in absa_model.config.id2label.items()
}


print(
    "ABSA label mapping:",
    ABSA_ID2LABEL
)


# ============================================================
# DATA REFERENCES
# ============================================================

reviews = data["reviews"]
absa = data["absa"]
weakness = data["weakness"]
evolution = data["evolution"]
phone_conflicts = data["phone_conflicts"]
review_conflicts = data["review_conflicts"]
conflict_summary = data["conflict_summary"]
conflicts_by_phone = data["conflicts_by_phone"]
aspect_conflict_summary = data["aspect_conflicts"]


# ============================================================
# TOP NAVIGATION
# ============================================================


def _show_selected_page():
    """The existing analytics render below this router's common frame."""


NAV_PAGES = {
    "Home": st.Page(_show_selected_page, title="Home", url_path="home", default=True),
    "Compare Phones": st.Page(_show_selected_page, title="Compare Phones", url_path="compare-phones"),
    "Review Analyzer": st.Page(_show_selected_page, title="Review Analyzer", url_path="review-analyzer"),
    "Phone Features": st.Page(_show_selected_page, title="Phone Features", url_path="phone-features"),
    "Common Problems": st.Page(_show_selected_page, title="Common Problems", url_path="common-problems"),
    "Review Trends": st.Page(_show_selected_page, title="Review Trends", url_path="review-trends"),
    "Likes & Dislikes": st.Page(_show_selected_page, title="Likes & Dislikes", url_path="likes-dislikes"),
    "Discover Patterns": st.Page(_show_selected_page, title="Discover Patterns", url_path="discover-patterns"),
    "Find Similar Phones": st.Page(_show_selected_page, title="Find Similar Phones", url_path="similar-phones"),
    "Spot New Problems": st.Page(_show_selected_page, title="Spot New Problems", url_path="new-problems"),
}
selected_page = st.navigation(
    {
        "": [NAV_PAGES["Home"]],
        "Explore": [NAV_PAGES[name] for name in (
            "Compare Phones", "Review Analyzer", "Phone Features", "Common Problems",
            "Review Trends", "Likes & Dislikes")],
        "Discover": [NAV_PAGES[name] for name in (
            "Discover Patterns", "Find Similar Phones", "Spot New Problems")],
    },
    position="top",
)
selected_page.run()
page = {
    "Home": "🏠 Overview",
    "Compare Phones": "📱 Phone Comparison",
    "Review Analyzer": "🔍 Review Analyzer",
    "Phone Features": "🧩 Aspect Sentiment",
    "Common Problems": "📉 Weakness Analyzer",
    "Review Trends": "📈 Sentiment Evolution",
    "Likes & Dislikes": "⚔️ Conflict Detection",
    "Discover Patterns": "🧭 Opinion Discovery",
    "Find Similar Phones": "🔗 Model Relationships",
    "Spot New Problems": "🚨 Emerging Issues",
}[selected_page.title]

ui.brand_header()
with st.expander("About this dataset"):
    st.write(
        "The phone model names identify real released products, but every review, "
        "rating, date and opinion in this project is simulated. "
        "Comparisons and predictions describe this research dataset only."
    )


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "🏠 Overview":
    home.render_home(NAV_PAGES, len(reviews), reviews["model"].nunique() if not reviews.empty else 0)



# ============================================================
# PHONE COMPARISON PAGE
# ============================================================

elif page == "📱 Phone Comparison":

    page_intro(
        "Compare the example reviews",
        "Compare Phones",
        "Choose phones to see what the generated reviews like and dislike about each feature."
    )
    ui.section_header("01", "Build your comparison", "Choose two phones to start. Add more below the results.")


    if reviews.empty or absa.empty:

        st.error(
            "Required review or ABSA data is not available."
        )

    else:

        available_models = sorted(
            reviews["model"]
            .dropna()
            .unique()
        )


        col1, col2 = st.columns(2)


        with col1:

            phone_1 = st.selectbox(
                "Phone one",
                available_models,
                index=0
            )


        with col2:

            phone_2 = st.selectbox(
                "Phone two",
                available_models,
                index=min(
                    1,
                    len(available_models) - 1
                )
            )


        if phone_1 == phone_2:

            st.warning(
                "Please select two different phones."
            )

        else:

            brand_1 = (
                reviews[
                    reviews["model"] == phone_1
                ]["brand"]
                .iloc[0]
            )

            brand_2 = (
                reviews[
                    reviews["model"] == phone_2
                ]["brand"]
                .iloc[0]
            )


            col1, col2 = st.columns(2)

            with col1:
                ui.phone_card(phone_1, brand_1, "Phone one")

            with col2:
                ui.phone_card(phone_2, brand_2, "Phone two")

            ui.section_header("02", "See the differences",
                              "Compare positive and negative comments about each feature in the generated reviews.")


            phone_1_data = absa[
                absa["model"] == phone_1
            ].copy()

            phone_2_data = absa[
                absa["model"] == phone_2
            ].copy()


            def calculate_aspect_sentiment(
                phone_data
            ):

                result = (
                    phone_data
                    .groupby("aspect")[
                        "predicted_sentiment"
                    ]
                    .value_counts(
                        normalize=True
                    )
                    .unstack(
                        fill_value=0
                    )
                    * 100
                )


                for sentiment in [
                    "Positive",
                    "Neutral",
                    "Negative"
                ]:

                    if sentiment not in result.columns:

                        result[sentiment] = 0


                result = result[
                    [
                        "Positive",
                        "Neutral",
                        "Negative"
                    ]
                ]

                return result


            sentiment_1 = (
                calculate_aspect_sentiment(
                    phone_1_data
                )
            )

            sentiment_2 = (
                calculate_aspect_sentiment(
                    phone_2_data
                )
            )


            aspects = [
                "battery",
                "camera",
                "performance",
                "design",
                "display"
            ]


            comparison_rows = []


            for aspect in aspects:

                positive_1 = (
                    sentiment_1
                    .loc[aspect, "Positive"]
                    if aspect in sentiment_1.index
                    else 0
                )

                positive_2 = (
                    sentiment_2
                    .loc[aspect, "Positive"]
                    if aspect in sentiment_2.index
                    else 0
                )

                negative_1 = (
                    sentiment_1
                    .loc[aspect, "Negative"]
                    if aspect in sentiment_1.index
                    else 0
                )

                negative_2 = (
                    sentiment_2
                    .loc[aspect, "Negative"]
                    if aspect in sentiment_2.index
                    else 0
                )


                comparison_rows.append(
                    {
                        "Aspect":
                            aspect.title(),

                        f"{phone_1} Positive (%)":
                            round(
                                positive_1,
                                1
                            ),

                        f"{phone_2} Positive (%)":
                            round(
                                positive_2,
                                1
                            ),

                        f"{phone_1} Negative (%)":
                            round(
                                negative_1,
                                1
                            ),

                        f"{phone_2} Negative (%)":
                            round(
                                negative_2,
                                1
                            )
                    }
                )


            comparison_df = pd.DataFrame(
                comparison_rows
            )


            st.subheader(
                "Opinions side by side"
            )

            st.dataframe(
                comparison_df,
                width="stretch",
                hide_index=True
            )


            st.subheader(
                "Positive comments by feature"
            )


            positive_chart = pd.DataFrame(
                {
                    "Aspect": aspects,

                    phone_1: [
                        sentiment_1
                        .loc[a, "Positive"]
                        if a in sentiment_1.index
                        else 0
                        for a in aspects
                    ],

                    phone_2: [
                        sentiment_2
                        .loc[a, "Positive"]
                        if a in sentiment_2.index
                        else 0
                        for a in aspects
                    ]
                }
            )


            positive_chart = positive_chart.melt(
                id_vars="Aspect",
                var_name="Phone",
                value_name="Positive Sentiment (%)"
            )


            fig_positive = px.bar(
                positive_chart,
                x="Aspect",
                y="Positive Sentiment (%)",
                color="Phone",
                barmode="group",
                title="Positive comments by phone feature"
            )


            fig_positive.update_yaxes(
                range=[0, 100]
            )


            st.plotly_chart(
                fig_positive,
                width="stretch"
            )


            st.subheader(
                "Negative comments by feature"
            )


            negative_chart = pd.DataFrame(
                {
                    "Aspect": aspects,

                    phone_1: [
                        sentiment_1
                        .loc[a, "Negative"]
                        if a in sentiment_1.index
                        else 0
                        for a in aspects
                    ],

                    phone_2: [
                        sentiment_2
                        .loc[a, "Negative"]
                        if a in sentiment_2.index
                        else 0
                        for a in aspects
                    ]
                }
            )


            negative_chart = negative_chart.melt(
                id_vars="Aspect",
                var_name="Phone",
                value_name="Negative Sentiment (%)"
            )


            fig_negative = px.bar(
                negative_chart,
                x="Aspect",
                y="Negative Sentiment (%)",
                color="Phone",
                barmode="group",
                title="Negative comments by phone feature"
            )


            fig_negative.update_yaxes(
                range=[0, 100]
            )


            st.plotly_chart(
                fig_negative,
                width="stretch"
            )


            st.subheader(
                "Which phone has more positive comments?"
            )


            winner_rows = []


            for aspect in aspects:

                score_1 = (
                    sentiment_1
                    .loc[aspect, "Positive"]
                    if aspect in sentiment_1.index
                    else 0
                )

                score_2 = (
                    sentiment_2
                    .loc[aspect, "Positive"]
                    if aspect in sentiment_2.index
                    else 0
                )


                if score_1 > score_2:

                    winner = phone_1

                elif score_2 > score_1:

                    winner = phone_2

                else:

                    winner = "Tie"


                winner_rows.append(
                    {
                        "Aspect":
                            aspect.title(),

                        f"{phone_1}":
                            round(
                                score_1,
                                1
                            ),

                        f"{phone_2}":
                            round(
                                score_2,
                                1
                            ),

                        "Winner":
                            winner
                    }
                )


            winner_df = pd.DataFrame(
                winner_rows
            )


            st.dataframe(
                winner_df,
                width="stretch",
                hide_index=True
            )

            additional_phones = st.multiselect(
                "Add more smartphones to the comparison",
                [model for model in available_models if model not in {phone_1, phone_2}],
                key="additional_comparison_phones"
            )

            if additional_phones:
                chosen_models = [phone_1, phone_2, *additional_phones]
                extra = absa[absa["model"].isin(chosen_models)]
                multi = extra.groupby(["model", "aspect"])["predicted_sentiment"].apply(
                    lambda values: 100 * (values.eq("Positive").mean() - values.eq("Negative").mean())
                ).reset_index(name="Sentiment score")
                st.subheader("Compare more phones by feature")
                st.dataframe(multi.pivot(index="aspect", columns="model", values="Sentiment score").round(1),
                             width="stretch")

            intelligence.comparison_prediction_panel([phone_1, phone_2, *additional_phones])
            ui.section_header("03", "Read the reviews behind the comparison",
                              "Choose a phone, feature and opinion to inspect generated review examples.")
            evidence_phone = st.selectbox(
                "Read examples for",
                [phone_1, phone_2, *additional_phones],
                key="comparison_evidence_phone",
            )
            intelligence.aspect_evidence_panel(evidence_phone, show_heading=False,
                                               key_prefix="comparison_evidence")


# ============================================================
# REVIEW ANALYZER PAGE
# ============================================================

elif page == "🔍 Review Analyzer":

    page_intro(
        "Try it yourself",
        "Review Analyzer",
        "Paste a review to see which phone features it mentions and whether each comment sounds positive or negative."
    )
    ui.section_header("01", "Start with a review", "Write your own text or choose a generated example from the dataset.")

    # ============================================================
    # ASPECT SEMANTIC PROFILES
    # ============================================================

    aspect_profiles = {

        "battery": [
            "battery lasts all day",
            "long battery life",
            "battery backup",
            "battery drains quickly",
            "battery drains fast",
            "phone needs frequent charging",
            "charging is fast",
            "charging is slow",
            "charging takes too long",
            "battery performance",
            "screen on time",
            "phone lasts throughout the day",
            "phone needs to be charged often"
        ],

        "camera": [
            "camera quality",
            "photos look sharp",
            "photos look good",
            "takes good photos",
            "takes great photos",
            "picture quality",
            "photo quality",
            "night photography",
            "night photos",
            "low light photography",
            "portrait photos",
            "video quality",
            "camera focus",
            "camera struggles in low light"
        ],

        "performance": [
            "phone feels smooth",
            "performance is smooth",
            "runs smoothly",
            "apps open quickly",
            "apps run smoothly",
            "gaming performance",
            "gaming is smooth",
            "games run smoothly",
            "multitasking is smooth",
            "phone is fast",
            "phone feels fast",
            "phone lags",
            "phone is slow",
            "apps freeze",
            "games stutter",
            "performance is poor"
        ],

        "design": [
            "design looks premium",
            "premium design",
            "build quality",
            "build feels solid",
            "phone feels lightweight",
            "phone feels heavy",
            "comfortable to hold",
            "looks attractive",
            "looks stylish",
            "frame feels premium",
            "frame feels cheap",
            "overall design",
            "phone design"
        ],

        "display": [
            "screen is bright",
            "display is bright",
            "screen looks vibrant",
            "display looks vibrant",
            "screen colors",
            "display colors",
            "color accuracy",
            "refresh rate",
            "screen quality",
            "display quality",
            "touch response",
            "screen is dim",
            "display is dim",
            "viewing angles",
            "screen looks sharp"
        ]
    }

    # ============================================================
    # EXPLICIT ASPECT KEYWORDS
    # ============================================================

    aspect_keywords = {

        "battery": [
            "battery",
            "charging",
            "charge",
            "drain",
            "screen-on time",
            "screen on time"
        ],

        "camera": [
            "camera",
            "photo",
            "photos",
            "picture",
            "pictures",
            "portrait",
            "photography",
            "video"
        ],

        "performance": [
            "performance",
            "processor",
            "gaming",
            "game",
            "games",
            "lag",
            "lags",
            "slow",
            "slowly",
            "smooth",
            "multitasking",
            "apps",
            "stutter",
            "freezes",
            "freeze"
        ],

        "design": [
            "design",
            "build",
            "frame",
            "weight",
            "lightweight",
            "heavy",
            "premium",
            "stylish",
            "comfortable"
        ],

        "display": [
            "display",
            "screen",
            "brightness",
            "bright",
            "dim",
            "colors",
            "colour",
            "refresh rate",
            "panel",
            "touch"
        ]
    }

    # ============================================================
    # SEMANTIC SETTINGS
    # ============================================================

    SEMANTIC_THRESHOLD = 0.48
    SEMANTIC_MARGIN = 0.06

    @st.cache_resource
    def review_profile_embeddings():
        return {
            aspect: semantic_model.encode(phrases, convert_to_tensor=True)
            for aspect, phrases in aspect_profiles.items()
        }

    cached_review_profiles = review_profile_embeddings()

    # ============================================================
    # REVIEW INPUT
    # ============================================================

    input_mode = st.radio("Review source", ["Write a review", "Choose an existing review"], horizontal=True)
    example_text = ""
    if input_mode == "Choose an existing review":
        available_reviews = intelligence.expanded_reviews()
        source_model = st.selectbox("Smartphone model", sorted(available_reviews["model"].unique()), key="review_source_model")
        examples = available_reviews[available_reviews["model"] == source_model].head(40)
        source_id = st.selectbox("Review ID", examples["review_id"].astype(str).tolist(), key="review_source_id")
        example_text = str(examples.loc[examples["review_id"].astype(str) == source_id, "review_text"].iloc[0])

    review_text = st.text_area(
        "Your smartphone review",
        value=example_text,
        height=150,
        placeholder=(
            "Example: The phone is very lightweight, "
            "and photos are beautiful and sharp, "
            "however it lags and feels very slow."
        )
    )

    analyze_button = st.button(
        "Analyze review",
        width="stretch",
        type="primary",
    )
    if analyze_button:
        st.session_state["last_analyzed_text"] = review_text
        st.session_state["review_analysis_active"] = bool(review_text.strip())
    analysis_active = (
        st.session_state.get("review_analysis_active", False)
        and st.session_state.get("last_analyzed_text") == review_text
    )

    # ============================================================
    # ANALYSIS
    # ============================================================

    if analyze_button or analysis_active:

        if not review_text.strip():

            st.warning(
                "Please enter a smartphone review first."
            )

        else:

            text = review_text.strip()

            # ====================================================
            # CLAUSE / OPINION CHUNK SPLITTING
            # ====================================================
            #
            # We split not only at ".", "!" and "?"
            # but also around:
            #
            #   comma + conjunction
            #   "but"
            #   "however"
            #   "although"
            #   "while"
            #
            # This allows multiple opinions in one sentence
            # to be analyzed independently.
            #
            # ====================================================

            chunks = re.split(
                r"""
                (?<=[.!?;])
                \s+

                |

                ,\s*
                (?=
                    (?:and|but|also|however|although|though|while)
                    \b
                )

                |

                \s+
                (?=
                    (?:but|however|although|though|while)
                    \b
                )
                """,
                text,
                flags=re.IGNORECASE | re.VERBOSE
            )

            chunks = [
                chunk.strip(" ,.")
                for chunk in chunks
                if chunk.strip(" ,.")
            ]

            # ====================================================
            # SECONDARY COMMA SPLIT
            # ====================================================
            #
            # Handle structures such as:
            #
            # "photos are beautiful and sharp, it lags badly"
            #
            # when no conjunction is present.
            #
            # ====================================================

            refined_chunks = []

            for chunk in chunks:

                comma_parts = re.split(
                    r",\s*",
                    chunk
                )

                for part in comma_parts:

                    part = part.strip()

                    if part:
                        refined_chunks.append(
                            part
                        )

            chunks = refined_chunks

            # ====================================================
            # DETECT ASPECTS
            # ====================================================

            detected_results = []

            for chunk in chunks:

                chunk_lower = chunk.lower()

                scores = {}

                # ------------------------------------------------
                # EXPLICIT KEYWORD MATCHING
                # ------------------------------------------------

                for aspect, keywords in (
                    aspect_keywords.items()
                ):

                    matches = []

                    for keyword in keywords:

                        if keyword in chunk_lower:

                            matches.append(
                                keyword
                            )

                    if matches:

                        scores[aspect] = (
                            0.72
                            +
                            min(
                                len(matches) * 0.04,
                                0.18
                            )
                        )

                # ------------------------------------------------
                # SEMANTIC MATCHING
                # ------------------------------------------------

                try:

                    chunk_embedding = (
                        semantic_model.encode(
                            chunk,
                            convert_to_tensor=True
                        )
                    )

                    for aspect, phrases in (
                        aspect_profiles.items()
                    ):

                        profile_embeddings = cached_review_profiles[aspect]

                        similarities = (
                            torch.nn.functional
                            .cosine_similarity(
                                chunk_embedding.unsqueeze(0),
                                profile_embeddings
                            )
                        )

                        semantic_score = (
                            similarities.max().item()
                        )

                        # Only use semantic score when
                        # there isn't already strong explicit
                        # evidence.
                        if aspect not in scores:

                            scores[aspect] = (
                                semantic_score
                            )

                except Exception:

                    pass

                # ------------------------------------------------
                # NO ASPECT
                # ------------------------------------------------

                if not scores:
                    continue

                # ------------------------------------------------
                # SORT ASPECT SCORES
                # ------------------------------------------------

                ranked = sorted(
                    scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                best_aspect = ranked[0][0]
                best_score = ranked[0][1]

                if len(ranked) > 1:

                    second_score = (
                        ranked[1][1]
                    )

                else:

                    second_score = 0

                # ------------------------------------------------
                # ACCEPT ASPECT
                # ------------------------------------------------

                explicit_match = any(
                    keyword in chunk_lower
                    for keyword in
                    aspect_keywords[best_aspect]
                )

                if explicit_match:

                    accepted = (
                        best_score
                        >= 0.70
                    )

                else:

                    accepted = (
                        best_score
                        >= SEMANTIC_THRESHOLD
                        and
                        (
                            best_score
                            - second_score
                            >= SEMANTIC_MARGIN
                        )
                    )

                if accepted:

                    detected_results.append(
                        {
                            "aspect":
                                best_aspect,

                            "evidence":
                                chunk,

                            "score":
                                best_score
                        }
                    )

            # ====================================================
            # REMOVE DUPLICATE ASPECTS
            # ====================================================
            #
            # If multiple chunks discuss the same aspect,
            # keep the strongest evidence.
            #
            # ====================================================

            best_by_aspect = {}

            for result in detected_results:

                aspect = result["aspect"]

                if (
                    aspect not in best_by_aspect
                    or
                    result["score"]
                    >
                    best_by_aspect[
                        aspect
                    ]["score"]
                ):

                    best_by_aspect[
                        aspect
                    ] = result

            # ====================================================
            # ORDER ASPECTS
            # ====================================================

            aspect_order = [
                "performance",
                "camera",
                "battery",
                "design",
                "display"
            ]

            final_results = []

            for aspect in aspect_order:

                if aspect in best_by_aspect:

                    final_results.append(
                        best_by_aspect[
                            aspect
                        ]
                    )

            # ====================================================
            # DISPLAY DETECTED ASPECTS
            # ====================================================

            if not final_results:

                st.warning(
                    """
                    No supported smartphone aspect was detected.

                    Try mentioning things such as battery,
                    camera, performance, design or display.
                    """
                )

            else:

                st.success(
                    f"Detected "
                    f"{len(final_results)} aspect(s): "
                    +
                    ", ".join(
                        result["aspect"].title()
                        for result in final_results
                    )
                )

                # =================================================
                # ASPECT DETECTION EVIDENCE
                # =================================================

                with st.expander(
                    "See the words that mention each feature"
                ):

                    evidence_rows = []

                    for result in final_results:

                        evidence_rows.append(
                            {
                                "Aspect":
                                    result[
                                        "aspect"
                                    ].title(),

                                "Evidence":
                                    result[
                                        "evidence"
                                    ],

                                "Detection Score":
                                    round(
                                        result[
                                            "score"
                                        ],
                                        3
                                    )
                            }
                        )

                    evidence_df = pd.DataFrame(
                        evidence_rows
                    )

                    st.dataframe(
                        evidence_df,
                        width="stretch",
                        hide_index=True
                    )

                # =================================================
                # ASPECT-LEVEL SENTIMENT
                # =================================================

                results = []

                for result in final_results:

                    aspect = result[
                        "aspect"
                    ]

                    evidence = result[
                        "evidence"
                    ]

                    # ---------------------------------------------
                    # IMPORTANT:
                    #
                    # Send the aspect-specific evidence to ABSA,
                    # NOT the entire review.
                    #
                    # ---------------------------------------------

                    model_input = (
                        "aspect: "
                        + aspect
                        + " review: "
                        + evidence
                    )

                    # ---------------------------------------------
                    # TOKENIZE
                    # ---------------------------------------------

                    encoded = absa_tokenizer(
                        model_input,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt"
                    )

                    # ---------------------------------------------
                    # MODEL PREDICTION
                    # ---------------------------------------------

                    with torch.no_grad():

                        outputs = absa_model(
                            **encoded
                        )

                    probabilities = (
                        torch.softmax(
                            outputs.logits,
                            dim=-1
                        )[0]
                    )

                    predicted_id = (
                        torch.argmax(
                            probabilities
                        ).item()
                    )

                    predicted_sentiment = (
                        ABSA_ID2LABEL[
                            predicted_id
                        ]
                    )

                    confidence = (
                        probabilities[
                            predicted_id
                        ].item()
                        * 100
                    )

                    relevance = (
                        result[
                            "score"
                        ]
                        * 100
                    )

                    results.append(
                        {
                            "Aspect":
                                aspect.title(),

                            "Sentiment":
                                predicted_sentiment,

                            "Sentiment Confidence":
                                round(
                                    confidence,
                                    2
                                ),

                            "Aspect Relevance":
                                round(
                                    relevance,
                                    2
                                )
                        }
                    )

                # =================================================
                # RESULTS TABLE
                # =================================================

                results_df = pd.DataFrame(
                    results
                )

                st.markdown("---")

                st.subheader(
                    "What this review says about each feature"
                )

                st.dataframe(
                    results_df.rename(columns={
                        "Aspect": "Phone feature",
                        "Sentiment Confidence": "Confidence (%)",
                        "Aspect Relevance": "Feature relevance",
                    }),
                    width="stretch",
                    hide_index=True
                )

                # =================================================
                # SENTIMENT SUMMARY
                # =================================================

                st.markdown("---")

                st.subheader(
                    "At a glance"
                )

                positive_count = sum(
                    results_df["Sentiment"]
                    == "Positive"
                )

                neutral_count = sum(
                    results_df["Sentiment"]
                    == "Neutral"
                )

                negative_count = sum(
                    results_df["Sentiment"]
                    == "Negative"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    ui.sentiment_card("Positive", positive_count, "positive")

                with col2:
                    ui.sentiment_card("Neutral", neutral_count, "neutral")

                with col3:
                    ui.sentiment_card("Negative", negative_count, "negative")

                # =================================================
                # CONFIDENCE CHART
                # =================================================

                st.markdown("---")

                st.subheader(
                    "How sure the system is"
                )

                fig = px.bar(
                    results_df,
                    x="Aspect",
                    y="Sentiment Confidence",
                    color="Sentiment",
                    text="Sentiment Confidence",
                    title="How sure the system is about each feature",
                    color_discrete_map=SENTIMENT_COLORS,
                )

                fig.update_yaxes(
                    range=[0, 100],
                    title="Confidence (%)"
                )

                fig.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside"
                )

                st.plotly_chart(
                    fig,
                    width="stretch"
                )

    if analysis_active and final_results:
        intelligence.prediction_panel(review_text)

# ============================================================
# ASPECT SENTIMENT PAGE
# ============================================================

# ============================================================
# ASPECT SENTIMENT BY SMARTPHONE
# ============================================================

elif page == "🧩 Aspect Sentiment":

    page_intro(
        "Explore one phone",
        "Phone Features",
        "Choose a phone and see what its generated reviews say about battery, camera, speed, design and display."
    )
    ui.section_header("01", "Choose a phone", "Explore its five features and read example comments behind the results.")


    # ========================================================
    # CHECK DATA
    # ========================================================

    if reviews.empty or absa.empty:

        st.error(
            "Required review or ABSA data is not available."
        )

    else:

        # ====================================================
        # PHONE SELECTION
        # ====================================================

        available_models = sorted(
            reviews["model"]
            .dropna()
            .unique()
        )


        selected_phone = st.selectbox(
            "Choose a smartphone",
            available_models,
            key="aspect_sentiment_phone"
        )


        # ====================================================
        # GET BRAND
        # ====================================================

        phone_rows = reviews[
            reviews["model"] == selected_phone
        ]


        if not phone_rows.empty:

            selected_brand = (
                phone_rows["brand"]
                .iloc[0]
            )

        else:

            selected_brand = "Unknown"


        st.markdown(
            f"### {selected_phone}"
        )

        st.caption(
            f"Brand: {selected_brand}"
        )


        st.markdown("---")


        # ====================================================
        # FILTER ABSA DATA
        # ====================================================

        phone_data = absa[
            absa["model"] == selected_phone
        ].copy()


        if phone_data.empty:

            st.warning(
                "No ABSA predictions are available "
                "for this smartphone."
            )

        else:

            # =================================================
            # ASPECT ORDER
            # =================================================

            aspect_order = [
                "battery",
                "camera",
                "performance",
                "design",
                "display"
            ]


            sentiment_order = [
                "Negative",
                "Neutral",
                "Positive"
            ]


            # =================================================
            # CALCULATE SENTIMENT DISTRIBUTION
            # =================================================

            phone_distribution = (
                phone_data
                .groupby("aspect")[
                    "predicted_sentiment"
                ]
                .value_counts(
                    normalize=True
                )
                .unstack(
                    fill_value=0
                )
                * 100
            )


            # =================================================
            # ENSURE ALL SENTIMENT COLUMNS EXIST
            # =================================================

            for sentiment in sentiment_order:

                if sentiment not in phone_distribution.columns:

                    phone_distribution[sentiment] = 0


            # =================================================
            # ENSURE ALL ASPECTS EXIST
            # =================================================

            phone_distribution = (
                phone_distribution
                .reindex(
                    aspect_order
                )
                .fillna(0)
            )


            phone_distribution = (
                phone_distribution[
                    sentiment_order
                ]
            )


            # =================================================
            # SENTIMENT SCORE
            # =================================================

            phone_distribution[
                "Sentiment Score"
            ] = (
                phone_distribution["Positive"]
                - phone_distribution["Negative"]
            )


            # =================================================
            # CREATE DISPLAY TABLE
            # =================================================

            phone_table = (
                phone_distribution
                .reset_index()
            )


            phone_table.columns = [
                "Aspect",
                "Negative (%)",
                "Neutral (%)",
                "Positive (%)",
                "Sentiment Score"
            ]


            phone_table["Aspect"] = (
                phone_table["Aspect"]
                .str.title()
            )


            phone_table = (
                phone_table
                .round(2)
            )


            # =================================================
            # PHONE-SPECIFIC TABLE
            # =================================================

            st.subheader(
                "Positive and negative comments by feature"
            )


            st.dataframe(
                phone_table.rename(columns={
                    "Aspect": "Phone feature",
                    "Sentiment Score": "Positive minus negative (%)",
                }),
                width="stretch",
                hide_index=True
            )


            # =================================================
            # SENTIMENT CHART
            # =================================================

            st.subheader(
                "Opinions about each feature"
            )


            chart_data = (
                phone_distribution[
                    sentiment_order
                ]
                .reset_index()
                .melt(
                    id_vars="aspect",
                    var_name="Sentiment",
                    value_name="Percentage"
                )
            )


            chart_data["aspect"] = (
                chart_data["aspect"]
                .str.title()
            )


            fig = px.bar(
                chart_data,
                x="aspect",
                y="Percentage",
                color="Sentiment",
                barmode="group",
                text="Percentage",
                title=(
                    f"Opinions about each feature of {selected_phone}"
                ),
                color_discrete_map=SENTIMENT_COLORS,
            )


            fig.update_yaxes(
                range=[0, 100],
                title="Percentage (%)"
            )


            fig.update_xaxes(
                title="Phone feature"
            )


            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )


            st.plotly_chart(
                fig,
                width="stretch"
            )


            # =================================================
            # BEST AND WEAKEST ASPECT
            # =================================================

            st.subheader(
                "Most liked and most criticized features"
            )


            phone_scores = (
                phone_distribution[
                    ["Sentiment Score"]
                ]
                .reset_index()
            )


            phone_scores.columns = [
                "Aspect",
                "Sentiment Score"
            ]


            phone_scores = (
                phone_scores
                .sort_values(
                    "Sentiment Score",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )


            best_aspect = (
                phone_scores
                .iloc[0]
            )


            weakest_aspect = (
                phone_scores
                .iloc[-1]
            )


            col1, col2 = st.columns(2)


            with col1:

                st.success(
                    f"""
                    ### 🥇 Most liked feature

                    **{best_aspect["Aspect"].title()}**

                    Positive minus negative comments:

                    **{best_aspect["Sentiment Score"]:.2f}**
                    """
                )


            with col2:

                st.warning(
                    f"""
                    ### ⚠️ Most criticized feature

                    **{weakest_aspect["Aspect"].title()}**

                    Positive minus negative comments:

                    **{weakest_aspect["Sentiment Score"]:.2f}**
                    """
                )


            # =================================================
            # SENTIMENT SCORE TABLE
            # =================================================

            st.markdown("---")

            st.subheader(
                "Feature comparison"
            )


            score_display = (
                phone_scores
                .copy()
            )


            score_display["Aspect"] = (
                score_display["Aspect"]
                .str.title()
            )


            score_display[
                "Sentiment Score"
            ] = (
                score_display[
                    "Sentiment Score"
                ]
                .round(2)
            )


            st.dataframe(
                score_display.rename(columns={
                    "Aspect": "Phone feature",
                    "Sentiment Score": "Positive minus negative (%)",
                }),
                width="stretch",
                hide_index=True
            )


            # =================================================
            # INTERPRETATION
            # =================================================

            st.markdown("---")

            st.info(
                f"""
                ### 🧠 Consumer Opinion Summary

                For **{selected_phone}**, the strongest
                consumer sentiment is associated with
                **{best_aspect["Aspect"].title()}**.

                The comparatively weakest aspect is
                **{weakest_aspect["Aspect"].title()}**.

                The sentiment score is calculated as:

                **Positive sentiment (%) − Negative sentiment (%)**

                A higher score indicates stronger positive
                consumer perception of that aspect.
                """
            )

            intelligence.aspect_evidence_panel(selected_phone)

# ============================================================
# WEAKNESS ANALYZER PAGE
# ============================================================

elif page == "📉 Weakness Analyzer":

    page_intro(
        "Explore complaints",
        "Common Problems",
        "Find features that receive more negative comments, then read example reviews."
    )
    ui.section_header("01", "Find recurring complaints", "Filter by brand and phone feature, then compare where negative comments appear.")

    # ============================================================
    # CHECK ABSA DATA
    # ============================================================

    if absa.empty:

        st.warning(
            "ABSA prediction data is not available."
        )

    else:

        # --------------------------------------------------------
        # COPY REQUIRED DATA
        # --------------------------------------------------------

        weakness_df = absa[
            [
                "brand",
                "model",
                "aspect",
                "predicted_sentiment"
            ]
        ].copy()

        # --------------------------------------------------------
        # CLEAN VALUES
        # --------------------------------------------------------

        weakness_df["model"] = (
            weakness_df["model"]
            .astype(str)
            .str.strip()
        )

        weakness_df["aspect"] = (
            weakness_df["aspect"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        weakness_df["predicted_sentiment"] = (
            weakness_df["predicted_sentiment"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        # --------------------------------------------------------
        # SUPPORTED ASPECTS
        # --------------------------------------------------------

        supported_aspects = [
            "battery",
            "camera",
            "performance",
            "design",
            "display"
        ]

        weakness_df = weakness_df[
            weakness_df["aspect"].isin(
                supported_aspects
            )
        ]

        # ========================================================
        # ASPECT SELECTION
        # ========================================================

        selected_brand = st.selectbox(
            "Brand",
            ["All brands", *sorted(weakness_df["brand"].dropna().unique())],
            key="weakness_brand"
        )
        if selected_brand != "All brands":
            weakness_df = weakness_df[weakness_df["brand"] == selected_brand]

        selected_aspect = st.selectbox(
            "Select an aspect",
            supported_aspects,
            format_func=lambda x: x.title()
        )

        st.markdown("---")

        # ========================================================
        # CALCULATE PHONE × ASPECT SENTIMENT
        # ========================================================

        aspect_data = weakness_df[
            weakness_df["aspect"]
            == selected_aspect
        ].copy()

        sentiment_distribution = (
            aspect_data
            .groupby("model")[
                "predicted_sentiment"
            ]
            .value_counts(
                normalize=True
            )
            .unstack(
                fill_value=0
            )
        )

        # --------------------------------------------------------
        # ENSURE ALL SENTIMENT COLUMNS EXIST
        # --------------------------------------------------------

        for sentiment in [
            "Negative",
            "Neutral",
            "Positive"
        ]:

            if sentiment not in sentiment_distribution.columns:

                sentiment_distribution[
                    sentiment
                ] = 0.0

        # --------------------------------------------------------
        # CONVERT TO PERCENTAGES
        # --------------------------------------------------------

        sentiment_distribution[
            "Negative"
        ] *= 100

        sentiment_distribution[
            "Neutral"
        ] *= 100

        sentiment_distribution[
            "Positive"
        ] *= 100

        # ========================================================
        # WEAKNESS SCORE
        # ========================================================

        sentiment_distribution[
            "Weakness Score"
        ] = (
            sentiment_distribution["Negative"]
            -
            sentiment_distribution["Positive"]
        )

        # --------------------------------------------------------
        # RESET INDEX
        # --------------------------------------------------------

        comparison = (
            sentiment_distribution
            .reset_index()
        )

        comparison = comparison.rename(
            columns={
                "model": "Phone"
            }
        )

        # ========================================================
        # CROSS-PHONE WEAKNESS COMPARISON
        # ========================================================

        st.subheader(
            f"{selected_aspect.title()} complaints across phones"
        )

        chart_data = (
            comparison
            .sort_values(
                "Weakness Score",
                ascending=False
            )
        )

        # --------------------------------------------------------
        # BAR CHART
        # --------------------------------------------------------

        fig = px.bar(
            chart_data,
            x="Phone",
            y="Weakness Score",
            text="Weakness Score",
            title=(
                f"{selected_aspect.title()} "
                "Negative feedback across phones"
            )
        )

        fig.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside"
        )

        fig.update_xaxes(
            tickangle=-45
        )

        fig.update_yaxes(
            title="Negative feedback score"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.markdown("---")

        # ========================================================
        # TOP 5 WEAKEST PHONES
        # ========================================================

        st.subheader(
            f"Five phones with the most complaints about "
            f"{selected_aspect.title()}"
        )

        top_5 = (
            comparison
            .sort_values(
                "Weakness Score",
                ascending=False
            )
            .head(5)
            .copy()
        )

        top_5_display = top_5[
            [
                "Phone",
                "Negative",
                "Positive",
                "Weakness Score"
            ]
        ].copy()

        top_5_display = (
            top_5_display
            .round(1)
        )

        # --------------------------------------------------------
        # DISPLAY TABLE
        # --------------------------------------------------------

        st.dataframe(
            top_5_display,
            width="stretch",
            hide_index=True
        )

        evidence_model = st.selectbox("Inspect a smartphone's complaints", comparison["Phone"].tolist(), key="weakness_evidence_model")
        intelligence.weakness_evidence_panel(selected_aspect, model=evidence_model)

# ============================================================
# SENTIMENT EVOLUTION PAGE
# ============================================================

elif page == "📈 Sentiment Evolution":

    page_intro(
        "Compare time periods",
        "Review Trends",
        "See whether the tone of generated reviews changes across six-month periods."
    )
    ui.section_header("01", "Set the time window", "Choose a brand, phone and dates to explore the simulated review timeline.")

    # ============================================================
    # CHECK REVIEW DATA
    # ============================================================

    if reviews.empty:

        st.warning(
            "Review data is not available."
        )

    else:

        evolution_df = reviews.copy()

        # ========================================================
        # CHECK REQUIRED COLUMNS
        # ========================================================

        required_columns = [
            "model",
            "review_date",
            "sentiment"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in evolution_df.columns
        ]

        if missing_columns:

            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        else:

            # ====================================================
            # CLEAN DATA
            # ====================================================

            evolution_df["model"] = (
                evolution_df["model"]
                .astype(str)
                .str.strip()
            )

            evolution_df["sentiment"] = (
                evolution_df["sentiment"]
                .astype(str)
                .str.strip()
                .str.title()
            )

            evolution_df["review_date"] = pd.to_datetime(
                evolution_df["review_date"],
                errors="coerce"
            )

            evolution_df = evolution_df.dropna(
                subset=[
                    "model",
                    "review_date",
                    "sentiment"
                ]
            )

            brand_filter = st.selectbox(
                "Brand to explore",
                ["All brands", *sorted(evolution_df["brand"].dropna().unique())],
                key="evolution_brand"
            )
            if brand_filter != "All brands":
                evolution_df = evolution_df[evolution_df["brand"] == brand_filter]

            date_limits = st.date_input(
                "Review date range",
                value=(evolution_df["review_date"].min().date(), evolution_df["review_date"].max().date()),
                key="evolution_dates"
            )
            if len(date_limits) == 2:
                evolution_df = evolution_df[
                    evolution_df["review_date"].between(pd.Timestamp(date_limits[0]), pd.Timestamp(date_limits[1]))
                ]
            if evolution_df.empty:
                st.info("No reviews match that brand and date range. Widen the dates to continue.")
                st.stop()

            # ====================================================
            # PHONE SELECTION
            # ====================================================

            models = sorted(
                evolution_df["model"]
                .unique()
                .tolist()
            )

            selected_model = st.selectbox(
                "Choose a smartphone",
                models
            )

            # ====================================================
            # FILTER SELECTED PHONE
            # ====================================================

            phone_reviews = evolution_df[
                evolution_df["model"]
                == selected_model
            ].copy()

            phone_reviews = phone_reviews.sort_values(
                "review_date"
            )

            # ====================================================
            # REVIEW HISTORY
            # ====================================================

            first_review_date = (
                phone_reviews["review_date"]
                .min()
            )

            latest_review_date = (
                phone_reviews["review_date"]
                .max()
            )

            total_reviews = len(
                phone_reviews
            )

            st.markdown("---")

            st.subheader(
                "📅 Review History"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "🟢 First Review",
                    first_review_date.strftime(
                        "%d %b %Y"
                    )
                )

            with col2:

                st.metric(
                    "🔵 Latest Review",
                    latest_review_date.strftime(
                        "%d %b %Y"
                    )
                )

            with col3:

                st.metric(
                    "Reviews in this period",
                    total_reviews
                )

            # ====================================================
            # CREATE SEMI-ANNUAL PERIOD
            # ====================================================

            phone_reviews["Year"] = (
                phone_reviews[
                    "review_date"
                ].dt.year
            )

            phone_reviews["Half"] = (
                phone_reviews[
                    "review_date"
                ].dt.month
                    .apply(
                        lambda month:
                        1 if month <= 6
                        else 2
                    )
            )

            phone_reviews["Period"] = (
                "H"
                +
                phone_reviews["Half"]
                .astype(str)
                +
                " "
                +
                phone_reviews["Year"]
                .astype(str)
            )

            # ====================================================
            # SEMI-ANNUAL SENTIMENT DISTRIBUTION
            # ====================================================

            sentiment_counts = (
                phone_reviews
                .groupby(
                    [
                        "Year",
                        "Half",
                        "Period"
                    ]
                )["sentiment"]
                .value_counts()
                .unstack(
                    fill_value=0
                )
                .reset_index()
            )

            # Ensure all sentiment columns exist
            for sentiment in [
                "Positive",
                "Neutral",
                "Negative"
            ]:

                if sentiment not in (
                    sentiment_counts.columns
                ):

                    sentiment_counts[
                        sentiment
                    ] = 0

            # ====================================================
            # TOTAL REVIEWS PER PERIOD
            # ====================================================

            sentiment_counts[
                "Total Reviews"
            ] = (
                sentiment_counts[
                    [
                        "Positive",
                        "Neutral",
                        "Negative"
                    ]
                ]
                .sum(axis=1)
            )

            # ====================================================
            # CONVERT TO PERCENTAGES
            # ====================================================

            sentiment_counts[
                "Positive %"
            ] = (
                sentiment_counts["Positive"]
                /
                sentiment_counts["Total Reviews"]
                * 100
            )

            sentiment_counts[
                "Neutral %"
            ] = (
                sentiment_counts["Neutral"]
                /
                sentiment_counts["Total Reviews"]
                * 100
            )

            sentiment_counts[
                "Negative %"
            ] = (
                sentiment_counts["Negative"]
                /
                sentiment_counts["Total Reviews"]
                * 100
            )

            # ====================================================
            # SENTIMENT SCORE
            # ====================================================
            #
            # Positive % - Negative %
            #
            # Higher score = more positive sentiment
            #
            # ====================================================

            sentiment_counts[
                "Sentiment Score"
            ] = (
                sentiment_counts[
                    "Positive %"
                ]
                -
                sentiment_counts[
                    "Negative %"
                ]
            )

            # ====================================================
            # SORT CHRONOLOGICALLY
            # ====================================================

            sentiment_counts = (
                sentiment_counts
                .sort_values(
                    [
                        "Year",
                        "Half"
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            # Create sequential period index
            sentiment_counts[
                "Period Index"
            ] = range(
                len(sentiment_counts)
            )

            # ====================================================
            # SENTIMENT SCORE TREND
            # ====================================================

            st.markdown("---")

            st.subheader(
                "How opinions change over time"
            )

            chart_data = (
                sentiment_counts[
                    [
                        "Period",
                        "Sentiment Score"
                    ]
                ]
                .copy()
            )

            fig_score = px.line(
                chart_data,
                x="Period",
                y="Sentiment Score",
                markers=True,
                text="Sentiment Score",
                title=(
                    f"{selected_model} "
                    "Review tone over time"
                )
            )

            fig_score.update_traces(
                texttemplate="%{text:.1f}",
                textposition="top center"
            )

            fig_score.update_yaxes(
                title="Positive minus negative comments"
            )

            fig_score.update_xaxes(
                title="Six-Month Period"
            )

            st.plotly_chart(
                fig_score,
                width="stretch"
            )

            # ====================================================
            # SENTIMENT COMPOSITION
            # ====================================================

            st.subheader(
                "Positive, neutral and negative reviews"
            )

            composition_data = (
                sentiment_counts[
                    [
                        "Period",
                        "Positive %",
                        "Neutral %",
                        "Negative %"
                    ]
                ]
                .melt(
                    id_vars="Period",
                    var_name="Sentiment",
                    value_name="Percentage"
                )
            )

            composition_data[
                "Sentiment"
            ] = (
                composition_data[
                    "Sentiment"
                ]
                .str.replace(
                    " %",
                    "",
                    regex=False
                )
            )

            fig_composition = px.line(
                composition_data,
                x="Period",
                y="Percentage",
                color="Sentiment",
                markers=True,
                title=(
                    f"{selected_model} "
                    "Positive, neutral and negative reviews over time"
                ),
                color_discrete_map=SENTIMENT_COLORS,
            )

            fig_composition.update_yaxes(
                range=[0, 100],
                title="Percentage of Reviews"
            )

            fig_composition.update_xaxes(
                title="Six-Month Period"
            )

            st.plotly_chart(
                fig_composition,
                width="stretch"
            )

            # ====================================================
            # SEMI-ANNUAL TABLE
            # ====================================================

            st.subheader(
                "Compare six-month periods"
            )

            table_data = (
                sentiment_counts[
                    [
                        "Period",
                        "Total Reviews",
                        "Positive %",
                        "Neutral %",
                        "Negative %",
                        "Sentiment Score"
                    ]
                ]
                .copy()
            )

            table_data = table_data.rename(
                columns={
                    "Positive %":
                        "Positive (%)",

                    "Neutral %":
                        "Neutral (%)",

                    "Negative %":
                        "Negative (%)"
                }
            )

            table_data[
                [
                    "Positive (%)",
                    "Neutral (%)",
                    "Negative (%)",
                    "Sentiment Score"
                ]
            ] = (
                table_data[
                    [
                        "Positive (%)",
                        "Neutral (%)",
                        "Negative (%)",
                        "Sentiment Score"
                    ]
                ]
                .round(1)
            )

            st.dataframe(
                table_data,
                width="stretch",
                hide_index=True
            )

            # ====================================================
            # EVOLUTION ANALYSIS
            # ====================================================

            st.markdown("---")

            st.subheader(
                "What changed?"
            )

            # ----------------------------------------------------
            # FIRST AND LAST PERIOD
            # ----------------------------------------------------

            first_period = (
                sentiment_counts.iloc[0]
            )

            last_period = (
                sentiment_counts.iloc[-1]
            )

            score_change = (
                last_period[
                    "Sentiment Score"
                ]
                -
                first_period[
                    "Sentiment Score"
                ]
            )

            positive_change = (
                last_period[
                    "Positive %"
                ]
                -
                first_period[
                    "Positive %"
                ]
            )

            negative_change = (
                last_period[
                    "Negative %"
                ]
                -
                first_period[
                    "Negative %"
                ]
            )

            # ----------------------------------------------------
            # TREND SLOPE
            # ----------------------------------------------------

            if len(sentiment_counts) >= 2:

                trend_slope = (
                    sentiment_counts[
                        "Sentiment Score"
                    ]
                    .corr(
                        sentiment_counts[
                            "Period Index"
                        ]
                    )
                )

            else:

                trend_slope = 0

            # ----------------------------------------------------
            # CLASSIFY TREND
            # ----------------------------------------------------

            if score_change >= 10:

                trend_label = (
                    "More positive over time"
                )

                trend_message = (
                    "Customer sentiment became "
                    "noticeably more positive over "
                    "the review history."
                )

            elif score_change <= -10:

                trend_label = (
                    "Less positive over time"
                )

                trend_message = (
                    "Customer sentiment became "
                    "noticeably more negative over "
                    "the review history."
                )

            else:

                trend_label = (
                    "Little change over time"
                )

                trend_message = (
                    "Customer sentiment remained "
                    "relatively stable over the "
                    "review history."
                )

            # ====================================================
            # TREND METRIC
            # ====================================================

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Evolution",
                    trend_label
                )

            with col2:

                st.metric(
                    "Score Change",
                    f"{score_change:+.1f}"
                )

            with col3:

                st.metric(
                    "Positive Change",
                    f"{positive_change:+.1f} pp"
                )

            # ====================================================
            # INTERPRETATION
            # ====================================================

            st.info(
                f"""
                **{selected_model}**

                {trend_message}

                The sentiment score changed from
                **{first_period["Sentiment Score"]:.1f}**
                in **{first_period["Period"]}** to
                **{last_period["Sentiment Score"]:.1f}**
                in **{last_period["Period"]}**.

                Positive sentiment changed by
                **{positive_change:+.1f} percentage points**,
                while negative sentiment changed by
                **{negative_change:+.1f} percentage points**.

                The sentiment score is calculated as:

                **Positive Sentiment % − Negative Sentiment %**

                A higher score therefore represents a more
                positive overall customer opinion.
                """
            )



# ============================================================
# CONFLICT DETECTION PAGE
# ============================================================

elif page == "⚔️ Conflict Detection":

    page_intro(
        "Mixed opinions",
        "Likes & Dislikes",
        "Find reviews that praise one phone feature while criticizing another."
    )
    ui.section_header("01", "Explore mixed opinions", "See how often one generated review praises a feature and criticizes another.")

    if review_conflicts.empty:

        st.info(
            "No aspect-level conflicts were detected in the current data."
        )

    else:

        # --------------------------------------------------------
        # OVERALL METRICS
        # --------------------------------------------------------

        total_absa_reviews = absa["review_id"].nunique()
        conflict_reviews = review_conflicts["review_id"].nunique()
        total_conflict_pairs = len(review_conflicts)

        conflict_rate = (
            conflict_reviews / total_absa_reviews * 100
            if total_absa_reviews > 0 else 0
        )

        top_pair = (
            aspect_conflict_summary.iloc[0]
            if not aspect_conflict_summary.empty
            else None
        )

        top_pair_name = (
            f"{top_pair['aspect_1'].title()} ↔ "
            f"{top_pair['aspect_2'].title()}"
            if top_pair is not None
            else "N/A"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Reviews with mixed opinions",
                f"{conflict_reviews:,}"
            )

        with col2:
            st.metric(
                "Positive / negative feature pairs",
                f"{total_conflict_pairs:,}"
            )

        with col3:
            st.metric(
                "Share with mixed opinions",
                f"{conflict_rate:.1f}%"
            )

        with col4:
            st.metric(
                "Most common feature pair",
                top_pair_name
            )

        st.caption(
            "A mixed-opinion review praises at least one feature and "
            "criticizes another. The share is calculated from reviews "
            "covered by the feature analysis."
        )

        # --------------------------------------------------------
        # GLOBAL TRADE-OFF ANALYSIS
        # --------------------------------------------------------

        st.markdown("---")

        st.subheader("Most common likes-and-dislikes combinations")

        global_pairs = aspect_conflict_summary.copy()

        global_pairs["Trade-off"] = (
            global_pairs["aspect_1"].str.title()
            + " ↔ "
            + global_pairs["aspect_2"].str.title()
        )

        display_pairs = global_pairs[
            [
                "Trade-off",
                "conflict_count",
                "unique_reviews",
                "avg_conflict_strength"
            ]
        ].copy()

        display_pairs.columns = [
            "Trade-off",
            "Conflict Pairs",
            "Reviews",
            "Avg. Confidence (%)"
        ]

        display_pairs["Avg. Confidence (%)"] = (
            (display_pairs["Avg. Confidence (%)"] * 100).round(1)
        )

        st.dataframe(
            display_pairs,
            width="stretch",
            hide_index=True
        )

        # --------------------------------------------------------
        # GLOBAL TRADE-OFF CHART
        # --------------------------------------------------------

        chart_data = global_pairs.head(10).copy()

        fig_global = px.bar(
            chart_data,
            x="Trade-off",
            y="conflict_count",
            text="conflict_count",
            title="Feature pairs with mixed opinions"
        )

        fig_global.update_layout(
            xaxis_title="Feature pair",
            yaxis_title="Number of mixed-opinion pairs"
        )

        fig_global.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig_global,
            width="stretch"
        )

        # --------------------------------------------------------
        # PHONE SELECTION
        # --------------------------------------------------------

        st.markdown("---")

        st.subheader("Mixed opinions for one phone")

        available_phones = sorted(
            conflicts_by_phone["model"].dropna().unique()
        )

        selected_phone = st.selectbox(
            "Select a smartphone",
            available_phones
        )

        selected_phone_data = conflicts_by_phone[
            conflicts_by_phone["model"] == selected_phone
        ]

        if selected_phone_data.empty:

            st.info(
                "No conflict information is available for this phone."
            )

        else:

            phone_row = selected_phone_data.iloc[0]

            brand_name = phone_row["brand"]

            st.markdown(
                f"### {brand_name} {selected_phone}"
            )

            # ----------------------------------------------------
            # PHONE METRICS
            # ----------------------------------------------------

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Reviews Analyzed",
                    int(phone_row["total_reviews"])
                )

            with col2:
                st.metric(
                    "Reviews with mixed opinions",
                    int(phone_row["conflict_reviews"])
                )

            with col3:
                st.metric(
                    "Share with mixed opinions",
                    f"{phone_row['conflict_rate_pct']:.1f}%"
                )

            with col4:
                st.metric(
                    "Positive / negative pairs",
                    int(phone_row["total_conflict_pairs"])
                )

            # ----------------------------------------------------
            # PHONE CONFLICT PAIRS
            # ----------------------------------------------------

            phone_pairs = phone_conflicts[
                phone_conflicts["model"] == selected_phone
            ].copy()

            if phone_pairs.empty:

                st.info(
                    "No positive-vs-negative aspect trade-offs "
                    "were detected for this smartphone."
                )

            else:

                phone_pairs["Trade-off"] = (
                    phone_pairs["aspect_1"].str.title()
                    + " ↔ "
                    + phone_pairs["aspect_2"].str.title()
                )

                phone_pairs = phone_pairs.sort_values(
                    "conflict_count",
                    ascending=False
                )

                st.subheader("Feature trade-offs for this phone")

                phone_table = phone_pairs[
                    [
                        "Trade-off",
                        "conflict_count",
                        "unique_reviews",
                        "avg_conflict_strength"
                    ]
                ].copy()

                phone_table.columns = [
                    "Trade-off",
                    "Conflict Pairs",
                    "Reviews",
                    "Avg. Confidence (%)"
                ]

                phone_table["Avg. Confidence (%)"] = (
                    (phone_table["Avg. Confidence (%)"] * 100).round(1)
                )

                st.dataframe(
                    phone_table,
                    width="stretch",
                    hide_index=True
                )

                # ------------------------------------------------
                # PHONE TRADE-OFF CHART
                # ------------------------------------------------

                fig_phone = px.bar(
                    phone_pairs,
                    x="Trade-off",
                    y="conflict_count",
                    text="conflict_count",
                    title=f"Mixed opinions about {selected_phone}"
                )

                fig_phone.update_layout(
                    xaxis_title="Feature pair",
                    yaxis_title="Number of mixed opinions"
                )

                fig_phone.update_traces(
                    textposition="outside"
                )

                st.plotly_chart(
                    fig_phone,
                    width="stretch"
                )

                # ------------------------------------------------
                # EXAMPLE REVIEWS
                # ------------------------------------------------

                st.subheader("Example reviews with mixed opinions")

                phone_reviews = review_conflicts[
                    review_conflicts["model"] == selected_phone
                ].copy()

                phone_reviews = phone_reviews.sort_values(
                    "conflict_strength",
                    ascending=False
                )

                for _, row in phone_reviews.head(8).iterrows():

                    with st.expander(
                        f"{row['aspect_1'].title()} ↔ "
                        f"{row['aspect_2'].title()}"
                    ):

                        st.markdown(
                            f"**{row['aspect_1'].title()}:** "
                            f"{row['sentiment_1']}  \n"
                            f"**{row['aspect_2'].title()}:** "
                            f"{row['sentiment_2']}"
                        )

                        st.write(row["review_text"])

                        st.caption(
                            f"Model confidence: "
                            f"{row['conflict_strength'] * 100:.1f}%"
                        )

            # ----------------------------------------------------
            # INTERPRETATION
            # ----------------------------------------------------

            if not phone_pairs.empty:

                strongest = phone_pairs.iloc[0]

                strongest_pair = (
                    f"{strongest['aspect_1'].title()} ↔ "
                    f"{strongest['aspect_2'].title()}"
                )

                st.markdown("---")
                st.subheader("What this pattern means")

                st.info(
                    f"For **{selected_phone}**, the most frequently "
                    f"observed mixed-opinion feature pair is "
                    f"**{strongest_pair}**, appearing in "
                    f"**{int(strongest['unique_reviews'])} review(s)**. "
                    f"In this generated dataset, reviews often praise "
                    f"one feature while criticizing another. This is not "
                    f"verified feedback about the named phone."
                )

        # --------------------------------------------------------
        # METHODOLOGY NOTE
        # --------------------------------------------------------

        st.markdown("---")

        with st.expander("How mixed opinions are found — technical details"):

            st.markdown(
                """
                **Step 1:** The ABSA Transformer predicts sentiment
                independently for each detected aspect.

                **Step 2:** Predictions belonging to the same review
                are grouped together.

                **Step 3:** If a review contains at least one
                **Positive** aspect and at least one **Negative**
                aspect, a cross-aspect conflict is recorded.

                **Step 4:** Every positive-negative aspect combination
                becomes a trade-off pair.

                **Step 5:** The pairs are aggregated globally and
                by smartphone to reveal recurring consumer trade-offs.

                **Important:** This is a *cross-aspect trade-off*.
                It does not mean the customer contradicted themselves
                about the same aspect.
                """
            )


elif page == "🧭 Opinion Discovery":
    intelligence.discovery_page(page_intro)


elif page == "🔗 Model Relationships":
    intelligence.relationships_page(page_intro)


elif page == "🚨 Emerging Issues":
    intelligence.emerging_page(page_intro)
