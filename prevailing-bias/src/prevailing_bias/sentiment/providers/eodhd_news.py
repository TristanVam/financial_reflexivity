from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import requests

from ...config import get_settings
from .base import NewsProvider

logger = logging.getLogger(__name__)


class EODHDNewsProvider:
    name = "eodhd_news"

    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.EODHD_API_KEY:
            logger.warning("EODHD_API_KEY not provided; news fetching will fail.")

    def fetch_articles(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        max_items: int | None = None,
    ) -> pd.DataFrame:
        params = {
            "api_token": self.settings.EODHD_API_KEY,
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
            "limit": max_items or 100,
        }
        url = f"https://eodhd.com/api/news/{ticker}"
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data: List[Dict[str, Any]] = resp.json() if resp.text else []
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to fetch EODHD news: %s", exc)
            data = []

        records = []
        for item in data:
            ts = pd.to_datetime(item.get("date"))
            records.append(
                {
                    "timestamp": ts,
                    "date": ts.normalize(),
                    "ticker": ticker,
                    "source": item.get("source") or "eodhd",
                    "channel": "news",
                    "title": item.get("title", ""),
                    "text": item.get("content", ""),
                    "meta": item,
                }
            )
        return pd.DataFrame(records)


__all__ = ["EODHDNewsProvider"]
