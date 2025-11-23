from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import requests

from ...config import get_settings

logger = logging.getLogger(__name__)


class SocialSentimentProvider:
    name = "social_sentiment"

    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.SOCIAL_SENTIMENT_API_KEY:
            logger.warning("SOCIAL_SENTIMENT_API_KEY not provided; social sentiment will fail.")

    def fetch(self, ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
        url = f"{self.settings.SOCIAL_SENTIMENT_BASE_URL}/sentiment"
        params = {
            "ticker": ticker,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "api_key": self.settings.SOCIAL_SENTIMENT_API_KEY,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            payload: Dict[str, Any] = resp.json()
            items: List[Dict[str, Any]] = payload.get("data", []) if isinstance(payload, dict) else []
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to fetch social sentiment: %s", exc)
            items = []

        records = []
        for item in items:
            ts = pd.to_datetime(item.get("timestamp"))
            records.append(
                {
                    "timestamp": ts,
                    "date": ts.normalize(),
                    "ticker": ticker,
                    "source": item.get("source", "social_api"),
                    "channel": item.get("channel", "social"),
                    "title": item.get("title", ""),
                    "text": item.get("text", ""),
                    "sentiment": item.get("sentiment"),
                    "meta": item,
                }
            )
        return pd.DataFrame(records)


__all__ = ["SocialSentimentProvider"]
