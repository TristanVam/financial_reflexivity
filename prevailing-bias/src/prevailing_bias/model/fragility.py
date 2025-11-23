from __future__ import annotations

import pandas as pd


def bias_fragility(price_score: pd.Series, sentiment_score: pd.Series, window: int = 30) -> pd.Series:
    """Compute fragility as rolling std of combined bias drivers."""

    combined = pd.concat([price_score, sentiment_score], axis=1)
    z = (combined - combined.mean()) / combined.std(ddof=0)
    fragility = z.rolling(window=window, min_periods=int(window * 0.5)).std(ddof=0)
    frag_score = fragility.mean(axis=1)
    return frag_score.rename("bias_fragility")


def compute_polymarket_divergence(
    prevailing_bias: pd.Series, polymarket_bias: pd.Series, window: int = 20
) -> pd.Series:
    """Measure divergence between prevailing bias and Polymarket bias."""

    if prevailing_bias.empty or polymarket_bias.empty:
        return pd.Series(dtype=float)

    aligned = pd.concat([prevailing_bias.rename("prevailing"), polymarket_bias.rename("polymarket")], axis=1, join="inner")
    if aligned.empty:
        return pd.Series(dtype=float)

    z = (aligned - aligned.mean()) / aligned.std(ddof=0)
    divergence = z["prevailing"] - z["polymarket"]
    if window > 1:
        divergence = divergence.rolling(window=window, min_periods=max(1, window // 2)).mean()
    return divergence.rename("polymarket_divergence")


__all__ = ["bias_fragility", "compute_polymarket_divergence"]
