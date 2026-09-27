"""Reproducibly extend the original synthetic review corpus for research demos.

The original 4,200 synthetic records retain their source fields. Added rows
are explicitly marked as synthetic; names come from the phone registry and
must never be interpreted as genuine customer feedback about those products.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from model_registry import ADDED_MODELS, legacy_mapping, validate_reviews


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "Mobile_Reviews_GenData.csv"
OUTPUT = ROOT / "data" / "processed" / "reviews_expanded.csv"
SPLITS = ROOT / "data" / "processed" / "review_splits.csv"
ANNOTATIONS = ROOT / "data" / "processed" / "synthetic_aspect_annotations.csv"
ASPECTS = ("battery", "camera", "performance", "design", "display")
PHRASES = {
    "battery": {
        "Positive": ["the charge comfortably lasts into the evening", "I finish most days with power to spare", "battery endurance has been dependable", "it rarely needs a midday top-up", "charging has fitted well into my routine", "the battery holds up during long commutes"],
        "Neutral": ["battery life is adequate for my usual schedule", "charging speed feels fairly ordinary", "the battery gets me through an average day", "power use has been neither remarkable nor troublesome"],
        "Negative": ["the battery drains faster than I expected", "I often need to recharge before dinner", "power drops quickly during navigation", "charging feels slow when I am in a hurry", "the battery struggles on busy days", "standby drain is noticeable overnight"],
    },
    "camera": {
        "Positive": ["photos retain pleasing detail in daylight", "the camera captures natural colours", "portraits look sharp without much effort", "video remains steady when I walk", "night photos have been better than expected", "focus locks on reliably for quick shots"],
        "Neutral": ["the camera is fine for casual photos", "image quality is acceptable in good light", "video quality meets my basic needs", "the camera feels typical for this range"],
        "Negative": ["night photos come out soft", "autofocus misses moving subjects", "the camera loses detail indoors", "video looks shaky when I move", "colours in photos sometimes look unnatural", "portraits often blur the edges"],
    },
    "performance": {
        "Positive": ["apps open quickly during everyday use", "switching between tasks feels smooth", "games run consistently in my sessions", "the phone stays responsive under a busy workload", "animations feel fluid", "performance has remained steady after weeks of use"],
        "Neutral": ["performance is adequate for basic tasks", "apps run at a reasonable pace", "the speed feels normal for routine use", "multitasking is workable without standing out"],
        "Negative": ["apps pause when I switch between them", "gaming sessions bring noticeable stutter", "the phone feels sluggish after a few minutes", "performance drops when several apps are open", "menus sometimes lag behind my taps", "the device warms up and slows down during games"],
    },
    "design": {
        "Positive": ["the build feels solid in my hand", "its shape is comfortable to hold", "the finish has held up well", "the buttons feel thoughtfully placed", "the design is understated and attractive", "the weight feels well balanced"],
        "Neutral": ["the design is practical rather than memorable", "the build feels ordinary for its price", "its weight is manageable", "the physical layout takes little time to learn"],
        "Negative": ["the frame feels less sturdy than expected", "the phone is awkward to hold for long", "the finish picks up marks easily", "the buttons are difficult to reach", "the design feels bulky in a pocket", "the weight becomes tiring after a while"],
    },
    "display": {
        "Positive": ["the display stays readable outdoors", "screen colours look balanced", "text appears crisp on the panel", "scrolling looks smooth", "brightness adjusts well in changing light", "the display makes videos enjoyable"],
        "Neutral": ["the display is usable in most places", "screen brightness is adequate indoors", "colours look fairly standard", "the panel does its job without surprises"],
        "Negative": ["the screen is hard to read in sunlight", "display colours look washed out", "touch response occasionally misses a swipe", "the panel appears dim outdoors", "scrolling does not feel as smooth as expected", "fine text looks less sharp than I hoped"],
    },
}
OPENERS = [
    "After using it for a few weeks,", "On my daily commute,", "For work and messaging,",
    "During a typical day,", "Having switched from another phone,", "In regular use,",
    "When I travel,", "For calls, maps and media,", "From my first month with it,",
    "While testing my usual apps,", "Over several weekends,", "For the way I use a phone,",
]
LINKS = ["Also,", "Meanwhile,", "However,", "At the same time,", "In contrast,", "One thing I noticed is that"]
CONTEXT = [
    "I mostly use the phone for work and messaging.", "My routine includes music, maps and a little photography.",
    "I often use it away from a charger.", "Most of my use happens indoors.",
    "I compare it with phones I have used before.", "My experience may differ from someone who games more heavily.",
    "These impressions come from ordinary daily use.", "I use the camera more often than the other features.",
]


def choose_sentiment(rng: random.Random) -> str:
    return rng.choices(["Positive", "Neutral", "Negative"], weights=[0.42, 0.26, 0.32])[0]


def make_review(rng: random.Random, used: set[str], force_battery_negative: bool = False) -> tuple[str, dict[str, int], str, int, dict[str, str], dict[str, str]]:
    for _ in range(200):
        aspects = rng.sample(ASPECTS, rng.randint(2, 4))
        if force_battery_negative and "battery" not in aspects:
            aspects[0] = "battery"
        opinions = {a: choose_sentiment(rng) for a in aspects}
        if force_battery_negative:
            opinions["battery"] = "Negative"
        parts = [rng.choice(OPENERS)]
        evidence = {}
        for index, aspect in enumerate(aspects):
            phrase = rng.choice(PHRASES[aspect][opinions[aspect]])
            evidence[aspect] = phrase
            parts.append((rng.choice(LINKS) + " " if index else "") + phrase + ".")
        if rng.random() < 0.42:
            parts.append(rng.choice(CONTEXT))
        text = " ".join(parts)
        normalized = " ".join(text.lower().split())
        if normalized in used:
            continue
        used.add(normalized)
        scores = {a: {"Positive": 5, "Neutral": 3, "Negative": 2}[opinions[a]] if a in opinions else 3 for a in ASPECTS}
        overall = sum({"Positive": 1, "Neutral": 0, "Negative": -1}[s] for s in opinions.values())
        label = "Positive" if overall > 0 else "Negative" if overall < 0 else "Neutral"
        rating = max(1, min(5, round(3.3 + 0.65 * overall + rng.choice([-0.5, 0, 0, 0.5]))))
        return text, scores, label, rating, opinions, evidence
    raise RuntimeError("Could not generate a unique review")


def build(seed: int = 20260927) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    original = pd.read_csv(SOURCE)
    assert original["source"].eq("Synthetic-V2").all(), "Unexpected source provenance"
    used = {" ".join(t.lower().split()) for t in original["review_text"].astype(str)}
    brands = sorted(original["brand"].unique())
    new_models = [(brand, entry.model) for brand in brands for entry in ADDED_MODELS[brand]]
    target_counts = {model: rng.randint(154, 168) for model in original["model"].unique()}
    target_counts.update({model: rng.randint(150, 170) for _, model in new_models})
    model_brand = dict(zip(original["model"], original["brand"]))
    model_brand.update({model: brand for brand, model in new_models})
    rows = []
    annotations = []
    for model, target in target_counts.items():
        existing = original[original["model"] == model]
        needed = target - len(existing)
        reference = existing.iloc[0] if len(existing) else original[original["brand"] == model_brand[model]].iloc[0]
        for _ in range(needed):
            day = pd.Timestamp("2025-01-01") + pd.Timedelta(days=rng.randint(0, 635))
            # Controlled synthetic drift tests the emerging-issue workflow.
            # This is a simulation, never evidence about real Google products.
            drift = model_brand[model] == "Google" and day >= pd.Timestamp("2026-01-01") and rng.random() < 0.75
            text, ratings, sentiment, rating, opinions, evidence = make_review(rng, used, force_battery_negative=drift)
            review_id = f"SYNEXP_{len(rows) + 1:05d}"
            annotations.extend({"review_id": review_id, "aspect": aspect, "aspect_sentiment": opinion,
                                "evidence": evidence[aspect]}
                               for aspect, opinion in opinions.items())
            row = reference.to_dict()
            row.update(review_id=review_id, customer_name="Synthetic participant", brand=model_brand[model], model=model,
                       rating=rating, review_text=text, sentiment=sentiment, review_date=day.strftime("%Y-%m-%d"),
                       battery_life_rating=ratings["battery"], camera_rating=ratings["camera"],
                       performance_rating=ratings["performance"], design_rating=ratings["design"],
                       display_rating=ratings["display"], review_length=len(text), word_count=len(text.split()),
                       helpful_votes=rng.randint(0, 35), verified_purchase=False,
                       source="Synthetic-Expansion")
            rows.append(row)
    expanded = pd.concat([original, pd.DataFrame(rows, columns=original.columns)], ignore_index=True)
    expanded = expanded.sort_values(["review_date", "review_id"]).reset_index(drop=True)
    validate(expanded)
    # Preserve the pre-migration grouped split exactly. The stable historical
    # grouping keys are used only for split assignment, never as ML features.
    old_group_key = {entry.model: legacy for legacy, entry in legacy_mapping().items()}
    groups = expanded["model"].map(lambda name: old_group_key.get(name, name)).astype(str)
    train_val, test = next(GroupShuffleSplit(n_splits=1, test_size=0.16, random_state=seed).split(expanded, groups=groups))
    train, val_local = next(GroupShuffleSplit(n_splits=1, test_size=0.19, random_state=seed + 1).split(expanded.iloc[train_val], groups=groups.iloc[train_val]))
    split = pd.Series("test", index=expanded.index)
    split.iloc[train_val[train]] = "train"
    split.iloc[train_val[val_local]] = "validation"
    partitions = pd.DataFrame({"review_id": expanded["review_id"], "split": split})
    return expanded, partitions, pd.DataFrame(annotations)


def validate(frame: pd.DataFrame) -> None:
    assert len(frame) >= 7000 and frame["model"].nunique() == 50
    assert frame.groupby("model").size().between(140, 180).all()
    assert frame["review_id"].is_unique and not frame["review_text"].str.lower().duplicated().any()
    assert frame["rating"].between(1, 5).all()
    assert set(frame["sentiment"]) == {"Positive", "Neutral", "Negative"}
    assert pd.to_datetime(frame["review_date"], errors="coerce").notna().all()
    assert not frame.isna().any().any()
    validate_reviews(frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    expanded, partitions, annotations = build(args.seed)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_csv(OUTPUT, index=False)
    partitions.to_csv(SPLITS, index=False)
    annotations.to_csv(ANNOTATIONS, index=False)
    print(f"Generated {len(expanded):,} synthetic reviews across {expanded.model.nunique()} models")
    print(partitions["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
