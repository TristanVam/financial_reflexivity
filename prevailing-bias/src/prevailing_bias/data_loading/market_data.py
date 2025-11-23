from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_cached(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df.index.name = "date"
        return df.sort_index()
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to load cache %s: %s", path, exc)
        return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_csv(path, index_label="date")
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to save cache %s: %s", path, exc)


def get_ohlcv(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV with caching.

    Parameters
    ----------
    ticker: str
        Ticker symbol.
    start: str
        Start date (YYYY-MM-DD).
    end: str
        End date (YYYY-MM-DD).
    interval: str
        Data interval supported by yfinance.
    """

    cache_path = DATA_DIR / f"{ticker}_{interval}.csv"
    cached = _load_cached(cache_path)

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    if cached is not None:
        if cached.index.min() <= start_dt and cached.index.max() >= end_dt:
            logger.info("Using cached data for %s", ticker)
            return cached.loc[start_dt:end_dt].copy()

    logger.info("Downloading data for %s", ticker)
    data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker}")

    data = data.rename(columns=str.title)
    data.index = pd.to_datetime(data.index)
    data.index.name = "date"
    data = data.sort_index()

    _save_cache(data, cache_path)
    return data.loc[start_dt:end_dt].copy()


__all__ = ["get_ohlcv"]
