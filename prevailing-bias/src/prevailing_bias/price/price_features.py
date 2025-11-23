from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _validate_close(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    return df["Close"].astype(float)


def rolling_trend_slope(df: pd.DataFrame, window: int = 60) -> pd.Series:
    close = _validate_close(df)
    log_price = np.log(close)

    slopes = []
    for i in range(len(log_price)):
        if i + 1 < window:
            slopes.append(np.nan)
            continue
        y = log_price.iloc[i + 1 - window : i + 1]
        x = np.arange(window)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        slope = np.sum((x - x_mean) * (y.values - y_mean)) / denom if denom != 0 else np.nan
        slopes.append(slope * window)
    return pd.Series(slopes, index=df.index, name="rolling_trend_slope")


def distance_from_long_ma(df: pd.DataFrame, window: int = 200) -> pd.Series:
    close = _validate_close(df)
    ma = close.rolling(window=window, min_periods=int(window * 0.5)).mean()
    out = (close - ma) / ma
    return out.rename("distance_from_long_ma")


def position_in_52w_range(df: pd.DataFrame) -> pd.Series:
    close = _validate_close(df)
    rolling_min = close.rolling(window=252, min_periods=126).min()
    rolling_max = close.rolling(window=252, min_periods=126).max()
    range_span = rolling_max - rolling_min
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = (close - rolling_min) / range_span
    return pos.replace([np.inf, -np.inf], np.nan).rename("position_in_52w_range")


def _zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros_like(series), index=series.index)
    return (series - mean) / std


def price_bias_score(df: pd.DataFrame) -> pd.Series:
    slope = rolling_trend_slope(df)
    distance = distance_from_long_ma(df)
    position = position_in_52w_range(df)

    z_slope = _zscore(slope)
    z_distance = _zscore(distance)
    z_position = _zscore(position)

    combined = pd.concat([z_slope, z_distance, z_position], axis=1)
    score = combined.mean(axis=1)
    return score.rename("price_bias_score")


__all__ = [
    "rolling_trend_slope",
    "distance_from_long_ma",
    "position_in_52w_range",
    "price_bias_score",
]
