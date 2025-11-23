from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass
class Settings:
    """Configuration settings loaded from environment variables."""

    EODHD_API_KEY: Optional[str] = None
    SOCIAL_SENTIMENT_API_KEY: Optional[str] = None
    SOCIAL_SENTIMENT_BASE_URL: str = "https://api.socialsentiment.example.com"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return singleton settings loaded from environment variables."""

    return Settings(
        EODHD_API_KEY=os.environ.get("EODHD_API_KEY"),
        SOCIAL_SENTIMENT_API_KEY=os.environ.get("SOCIAL_SENTIMENT_API_KEY"),
        SOCIAL_SENTIMENT_BASE_URL=os.environ.get(
            "SOCIAL_SENTIMENT_BASE_URL", "https://api.socialsentiment.example.com"
        ),
    )


__all__ = ["Settings", "get_settings"]
