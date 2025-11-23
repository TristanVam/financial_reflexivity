from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

POLYMARKET_BASE_URL = "https://api.polymarket.com"


@dataclass
class PolymarketMarket:
    id: str
    question: str
    ticker_hint: str | None
    tags: list[str]
    raw: dict[str, Any]


def _get_session(session: requests.Session | None = None) -> requests.Session:
    return session or requests.Session()


def search_markets(keyword: str, session: requests.Session | None = None) -> list[PolymarketMarket]:
    """Search Polymarket markets by keyword.

    This function abstracts the REST call so that the underlying API can be swapped easily
    if Polymarket changes its schema.
    """

    sess = _get_session(session)
    url = f"{POLYMARKET_BASE_URL}/markets"
    try:
        resp = sess.get(url, params={"search": keyword}, timeout=10)
        resp.raise_for_status()
        data = resp.json() or []
    except requests.RequestException as exc:  # pragma: no cover - network
        logger.error("Failed to fetch Polymarket markets: %s", exc)
        return []

    markets: list[PolymarketMarket] = []
    for item in data:
        markets.append(
            PolymarketMarket(
                id=str(item.get("id", "")),
                question=str(item.get("question", "")),
                ticker_hint=item.get("ticker") or item.get("ticker_hint"),
                tags=item.get("tags", []) or [],
                raw=item,
            )
        )
    return markets


def get_market_timeseries(
    market_id: str,
    start: datetime,
    end: datetime,
    session: requests.Session | None = None,
    aggregation: Literal["hourly", "daily"] = "daily",
) -> pd.Series:
    """Fetch historical implied probabilities for a given Polymarket market."""

    sess = _get_session(session)
    url = f"{POLYMARKET_BASE_URL}/markets/{market_id}/prices"
    params = {"start": start.isoformat(), "end": end.isoformat(), "aggregation": aggregation}
    try:
        resp = sess.get(url, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json() or []
    except requests.RequestException as exc:  # pragma: no cover - network
        logger.error("Failed to fetch Polymarket prices for %s: %s", market_id, exc)
        return pd.Series(dtype=float)

    records = []
    for entry in payload:
        ts_raw = entry.get("timestamp")
        prob = entry.get("price") or entry.get("probability")
        if ts_raw is None or prob is None:
            continue
        ts = pd.to_datetime(ts_raw)
        if ts < pd.to_datetime(start) or ts > pd.to_datetime(end):
            continue
        records.append((ts, float(prob)))

    if not records:
        return pd.Series(dtype=float)

    series = pd.Series({ts: val for ts, val in records}, dtype=float).sort_index()
    series.index = series.index.tz_localize(None)
    series.name = market_id

    rule = "1D" if aggregation == "daily" else "1H"
    if len(series) > 1:
        series = series.resample(rule).mean().ffill()

    return series


def get_polymarket_timeseries_for_markets(
    market_ids: list[str],
    start: datetime,
    end: datetime,
    aggregation: Literal["hourly", "daily"] = "daily",
) -> dict[str, pd.Series]:
    """Fetch timeseries for multiple Polymarket markets."""

    result: dict[str, pd.Series] = {}
    for market_id in market_ids:
        series = get_market_timeseries(market_id, start, end, aggregation=aggregation)
        if not series.empty:
            result[market_id] = series
    return result


def compute_polymarket_bias(
    series_dict: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Compute an aggregated Polymarket bias score from multiple market series."""

    if not series_dict:
        return pd.Series(dtype=float)

    df = pd.concat(series_dict.values(), axis=1, join="inner")
    if df.empty:
        return pd.Series(dtype=float)

    df.columns = list(series_dict.keys())

    if weights is None:
        weights = {mid: 1.0 for mid in df.columns}

    aligned_weights = {col: weights.get(col, 0.0) for col in df.columns}
    weight_sum = sum(aligned_weights.values())
    if weight_sum == 0:
        return pd.Series(dtype=float)

    centered = df - 0.5
    weighted = sum(centered[col] * w for col, w in aligned_weights.items())
    bias_series = weighted.rename("polymarket_bias")
    return bias_series


__all__ = [
    "POLYMARKET_BASE_URL",
    "PolymarketMarket",
    "search_markets",
    "get_market_timeseries",
    "get_polymarket_timeseries_for_markets",
    "compute_polymarket_bias",
]
