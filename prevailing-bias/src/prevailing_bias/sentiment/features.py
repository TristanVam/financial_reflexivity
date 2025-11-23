from __future__ import annotations

from datetime import datetime
import logging

import pandas as pd

from .engines.finbert_engine import FinBERTSentimentEngine
from .providers.eodhd_news import EODHDNewsProvider
from .providers.social_api import SocialSentimentProvider
from .aggregation import aggregate_daily_sentiment, combine_sentiment_scores

logger = logging.getLogger(__name__)


def fetch_and_score_news(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    provider = EODHDNewsProvider()
    engine = FinBERTSentimentEngine()
    articles = provider.fetch_articles(ticker, start, end)
    if articles.empty:
        return pd.DataFrame(columns=["timestamp", "date", "sentiment"])
    articles["sentiment"] = engine.score(articles["text"])
    return articles


def fetch_social_sentiment(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    provider = SocialSentimentProvider()
    return provider.fetch(ticker, start, end)


def sentiment_feature_series(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    news_df = fetch_and_score_news(ticker, start, end)
    social_df = fetch_social_sentiment(ticker, start, end)

    news_daily = aggregate_daily_sentiment(news_df.assign(channel="news"))
    social_daily = aggregate_daily_sentiment(social_df.assign(channel="social"))

    news_series = news_daily.get("news", pd.Series(dtype=float))
    social_series = social_daily.get("social", pd.Series(dtype=float))

    combined = combine_sentiment_scores(news_series, social_series)
    combined.index.name = "date"

    result = pd.DataFrame(
        {
            "news_sentiment": news_series,
            "social_sentiment": social_series,
            "combined_sentiment": combined,
        }
    )
    return result


__all__ = [
    "fetch_and_score_news",
    "fetch_social_sentiment",
    "sentiment_feature_series",
]
