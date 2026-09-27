"""Canonical phone identifiers for the synthetic research dataset.

Manufacturer links verify that a *product name* exists. They do not validate
the generated reviews, prices, ratings, dates, or opinions attributed to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "Mobile_Reviews_GenData.csv"


@dataclass(frozen=True)
class RegisteredModel:
    brand: str
    model: str
    manufacturer_url: str


# Ordered pairs replace the two historical placeholder slots per brand.
# All 18 products were released before the synthetic review period begins.
ADDED_MODELS: dict[str, tuple[RegisteredModel, RegisteredModel]] = {
    "Apple": (
        RegisteredModel("Apple", "iPhone 15", "https://www.apple.com/newsroom/2023/09/iphone-15-lineup-and-new-apple-watch-lineup-arrive-worldwide/"),
        RegisteredModel("Apple", "iPhone 15 Pro", "https://www.apple.com/newsroom/2023/09/iphone-15-lineup-and-new-apple-watch-lineup-arrive-worldwide/"),
    ),
    "Google": (
        RegisteredModel("Google", "Pixel 8", "https://blog.google/intl/en-ca/products/devices-services/meet-pixel-8-and-pixel-8-pro-our-newest-phones/"),
        RegisteredModel("Google", "Pixel 8 Pro", "https://blog.google/intl/en-ca/products/devices-services/meet-pixel-8-and-pixel-8-pro-our-newest-phones/"),
    ),
    "Motorola": (
        RegisteredModel("Motorola", "Motorola Edge 50 Pro", "https://motorolanews.com/motorola-announces-a-new-generation-of-edge-family-with-an-impressive-design-and-its-most-impressive-camera-powered-by-moto-ai/"),
        RegisteredModel("Motorola", "Motorola Razr 50", "https://motorolanews.com/pressbox/motorola-razr/motorola-razr-50-motorola-razr/"),
    ),
    "Nothing": (
        RegisteredModel("Nothing", "Nothing Phone (2)", "https://nothing.tech/products/phone-2"),
        RegisteredModel("Nothing", "Nothing Phone (2a)", "https://ie.nothing.tech/pages/phone-2a"),
    ),
    "OPPO": (
        RegisteredModel("OPPO", "OPPO Reno11 Pro 5G", "https://www.oppo.com/en/smartphones/series-reno/reno11-pro/"),
        RegisteredModel("OPPO", "OPPO Reno12 Pro 5G", "https://www.oppo.com/en/newsroom/press/oppo-reno12-series-launch/"),
    ),
    "OnePlus": (
        RegisteredModel("OnePlus", "OnePlus 11 5G", "https://www.oneplus.com/us/11"),
        RegisteredModel("OnePlus", "OnePlus 12", "https://www.oneplus.com/us/12"),
    ),
    "Realme": (
        RegisteredModel("Realme", "Realme GT 6", "https://www.realme.com/ph/realme-gt-6"),
        RegisteredModel("Realme", "Realme 13 Pro+ 5G", "https://www.realme.com/global/realme-13-pro-plus-5g"),
    ),
    "Samsung": (
        RegisteredModel("Samsung", "Galaxy S24", "https://news.samsung.com/global/enter-the-new-era-of-mobile-ai-with-samsung-galaxy-s24-series"),
        RegisteredModel("Samsung", "Galaxy A55 5G", "https://news.samsung.com/in/samsung-launches-galaxy-a55-5g-and-galaxy-a35-5g-with-flagship-like-camera-innovations-and-samsung-knox-vault-protection"),
    ),
    "Xiaomi": (
        RegisteredModel("Xiaomi", "Xiaomi 14", "https://www.mi.com/global/product/xiaomi-14/"),
        RegisteredModel("Xiaomi", "Xiaomi 14 Ultra", "https://www.mi.com/global/product/xiaomi-14-ultra/"),
    ),
}


def legacy_mapping() -> dict[str, RegisteredModel]:
    """Map old placeholder slots to verified names in deterministic order."""
    return {f"{brand} Study {slot}": entry
            for brand, entries in ADDED_MODELS.items()
            for slot, entry in zip(("A", "B"), entries)}


def registry() -> pd.DataFrame:
    """Return one brand/model registry covering inherited and added products."""
    inherited = pd.read_csv(SOURCE, usecols=["brand", "model"]).drop_duplicates()
    inherited["legacy_model"] = inherited["model"]
    inherited["manufacturer_url"] = ""
    inherited["origin"] = "inherited synthetic source"
    added = pd.DataFrame(
        [{"brand": entry.brand, "model": entry.model, "legacy_model": legacy,
          "manufacturer_url": entry.manufacturer_url, "origin": "verified replacement"}
         for legacy, entry in legacy_mapping().items()]
    )
    result = pd.concat([inherited, added], ignore_index=True)
    if len(result) != 50 or result["model"].nunique() != 50:
        raise ValueError("The phone registry must contain 50 distinct model names.")
    if result.groupby("model")["brand"].nunique().max() != 1:
        raise ValueError("A model cannot belong to multiple brands.")
    return result


def canonical_name(name: str) -> str:
    replacement = legacy_mapping().get(name)
    return replacement.model if replacement else name


def validate_reviews(reviews: pd.DataFrame) -> None:
    """Reject stale placeholders, mismatched brands, or unexpected phone IDs."""
    expected = registry()[["brand", "model"]].drop_duplicates()
    observed = reviews[["brand", "model"]].drop_duplicates()
    if len(observed) != 50 or len(observed.merge(expected, on=["brand", "model"])) != 50:
        raise ValueError("Active reviews do not match the registered brand/model pairs.")
    if reviews["model"].str.contains(r"\bStudy\s+[A-Z]\b", regex=True).any():
        raise ValueError("Active reviews still contain placeholder model names.")


if __name__ == "__main__":
    print(registry()[["brand", "model", "origin", "manufacturer_url"]].to_string(index=False))
