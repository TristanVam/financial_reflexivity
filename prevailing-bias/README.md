# prevailing-bias

Implementation of a prevailing bias model inspired by George Soros. It combines price-based signals, news sentiment, social sentiment, and a bias fragility indicator. A Streamlit dashboard provides quick visualization.

## Features
- Market data loading with caching via `yfinance`
- Price trend, range, and momentum features
- News sentiment from EODHD processed by FinBERT
- Social sentiment aggregation from a REST API
- Bias fragility metric as a proxy for reversal risk
- Streamlit dashboard using Plotly

## Quickstart
```
python -m pip install -e .
streamlit run src/prevailing_bias/visualization/bias_dashboard.py
```

Set environment variables `EODHD_API_KEY` and `SOCIAL_SENTIMENT_API_KEY` for external sentiment providers.
