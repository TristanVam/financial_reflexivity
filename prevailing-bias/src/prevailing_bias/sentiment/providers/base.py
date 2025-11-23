from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd


class NewsProvider(Protocol):
    name: str

    def fetch_articles(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        max_items: int | None = None,
    ) -> pd.DataFrame:
        """Return articles DataFrame."""
        ...


__all__ = ["NewsProvider"]
