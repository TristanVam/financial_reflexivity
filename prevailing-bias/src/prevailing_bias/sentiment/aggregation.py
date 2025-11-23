from __future__ import annotations

import pandas as pd


def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "channel", "sentiment"])
    required_cols = {"date", "channel", "sentiment"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    grouped = (
        df.groupby(["date", "channel"])["sentiment"]
        .mean()
        .reset_index()
        .pivot(index="date", columns="channel", values="sentiment")
    )
    return grouped


def combine_sentiment_scores(news: pd.Series, social: pd.Series, weight_news: float = 0.6) -> pd.Series:
    news_aligned, social_aligned = news.align(social, join="outer")
    news_filled = news_aligned.fillna(news_aligned.mean())
    social_filled = social_aligned.fillna(social_aligned.mean())
    combined = weight_news * news_filled + (1 - weight_news) * social_filled
    return combined.rename("combined_sentiment")


__all__ = ["aggregate_daily_sentiment", "combine_sentiment_scores"]
