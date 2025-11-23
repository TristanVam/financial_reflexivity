from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from ..data_loading.market_data import get_ohlcv
from ..options.options_features import dummy_options_bias
from ..price.price_features import price_bias_score
from ..sentiment.features import sentiment_feature_series
from .fragility import bias_fragility

logger = logging.getLogger(__name__)


@dataclass
class PrevailingBiasResult:
    price_bias: pd.Series
    news_sentiment: pd.Series | None
    social_sentiment: pd.Series | None
    options_bias: pd.Series | None
    polymarket_bias: pd.Series | None
    prevailing_bias: pd.Series
    features: pd.DataFrame
    fragility: pd.Series


def _zscore(series: pd.Series, name: str) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float, name=name)
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index, name=name)
    return ((series - mean) / std).rename(name)


def align_and_standardize(
    price_bias: pd.Series | None,
    news_bias: pd.Series | None = None,
    social_bias: pd.Series | None = None,
    options_bias: pd.Series | None = None,
    polymarket_bias: pd.Series | None = None,
) -> pd.DataFrame:
    """Align bias components on a common index and return z-scored features."""

    components: dict[str, pd.Series] = {}
    if price_bias is not None:
        components["price_bias_z"] = _zscore(price_bias, "price_bias_z")
    if news_bias is not None:
        components["news_bias_z"] = _zscore(news_bias, "news_bias_z")
    if social_bias is not None:
        components["social_bias_z"] = _zscore(social_bias, "social_bias_z")
    if options_bias is not None:
        components["options_bias_z"] = _zscore(options_bias, "options_bias_z")
    if polymarket_bias is not None:
        components["polymarket_bias_z"] = _zscore(polymarket_bias, "polymarket_bias_z")

    if not components:
        return pd.DataFrame()

    aligned = pd.concat(components.values(), axis=1, join="inner")
    return aligned.dropna(how="all")


def compute_prevailing_bias(
    price_bias: pd.Series | None,
    news_bias: pd.Series | None = None,
    social_bias: pd.Series | None = None,
    options_bias: pd.Series | None = None,
    polymarket_bias: pd.Series | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute the prevailing bias as weighted sum of standardized components."""

    default_weights: dict[str, float] = {
        "price": 0.4,
        "news": 0.2,
        "social": 0.15,
        "options": 0.10,
        "polymarket": 0.15,
    }
    weight_map = weights or default_weights

    features = align_and_standardize(
        price_bias,
        news_bias=news_bias,
        social_bias=social_bias,
        options_bias=options_bias,
        polymarket_bias=polymarket_bias,
    )
    if features.empty:
        return features

    available_weights: dict[str, float] = {}
    for key, col in [
        ("price", "price_bias_z"),
        ("news", "news_bias_z"),
        ("social", "social_bias_z"),
        ("options", "options_bias_z"),
        ("polymarket", "polymarket_bias_z"),
    ]:
        if col in features.columns:
            available_weights[col] = weight_map.get(key, 0.0)

    weight_sum = sum(available_weights.values())
    if weight_sum == 0:
        return features

    weighted_sum = sum(features[col] * w for col, w in available_weights.items())
    prevailing = (weighted_sum / weight_sum).rename("prevailing_bias")
    return features.assign(prevailing_bias=prevailing)


class PrevailingBiasModel:
    def __init__(self, ticker: str, start: str, end: str, polymarket_bias: pd.Series | None = None) -> None:
        self.ticker = ticker
        self.start = start
        self.end = end
        self.polymarket_bias = polymarket_bias

    def run(self) -> PrevailingBiasResult:
        logger.info("Running PrevailingBiasModel for %s", self.ticker)
        price_df = get_ohlcv(self.ticker, self.start, self.end)
        price_score = price_bias_score(price_df)

        start_dt = pd.to_datetime(self.start)
        end_dt = pd.to_datetime(self.end)
        sentiment_df = sentiment_feature_series(self.ticker, start_dt, end_dt)
        news_sentiment = sentiment_df.get("news_sentiment")
        social_sentiment = sentiment_df.get("social_sentiment")

        options_bias = dummy_options_bias(price_df.index)

        feature_frame = compute_prevailing_bias(
            price_score,
            news_bias=news_sentiment,
            social_bias=social_sentiment,
            options_bias=options_bias,
            polymarket_bias=self.polymarket_bias,
        )

        if "prevailing_bias" in feature_frame:
            prevailing_bias_series = feature_frame["prevailing_bias"]
        else:
            prevailing_bias_series = pd.Series(dtype=float)

        fragility_score = bias_fragility(price_score, sentiment_df.get("combined_sentiment", pd.Series(dtype=float)))

        return PrevailingBiasResult(
            price_bias=price_score,
            news_sentiment=news_sentiment,
            social_sentiment=social_sentiment,
            options_bias=options_bias,
            polymarket_bias=self.polymarket_bias,
            prevailing_bias=prevailing_bias_series,
            features=feature_frame,
            fragility=fragility_score,
        )


__all__ = [
    "PrevailingBiasModel",
    "PrevailingBiasResult",
    "align_and_standardize",
    "compute_prevailing_bias",
]
