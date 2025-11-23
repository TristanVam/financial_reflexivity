from __future__ import annotations

from typing import Protocol

import pandas as pd


class SentimentEngine(Protocol):
    name: str

    def score(self, texts: pd.Series) -> pd.Series:
        ...


__all__ = ["SentimentEngine"]
