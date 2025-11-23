from __future__ import annotations

import pandas as pd


def dummy_options_bias(index: pd.DatetimeIndex) -> pd.Series:
    """Placeholder options bias series of zeros."""

    return pd.Series(0.0, index=index, name="options_bias")


__all__ = ["dummy_options_bias"]
