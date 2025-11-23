from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..model.prevailing_bias_model import PrevailingBiasModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_series(series: pd.Series, name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=name))
    fig.update_layout(title=name, xaxis_title="Date", yaxis_title=name)
    return fig


def main() -> None:
    st.title("Prevailing Bias Dashboard")

    ticker = st.text_input("Ticker", value="SPY")
    start = st.date_input("Start", value=datetime(2020, 1, 1))
    end = st.date_input("End", value=datetime.today())

    if st.button("Run Model"):
        with st.spinner("Running model..."):
            model = PrevailingBiasModel(ticker, start.isoformat(), end.isoformat())
            result = model.run()

        st.subheader("Price Bias")
        st.plotly_chart(plot_series(result.price_bias, "Price Bias"), use_container_width=True)

        st.subheader("Sentiment Bias")
        st.plotly_chart(plot_series(result.sentiment_bias, "Sentiment Bias"), use_container_width=True)

        st.subheader("Combined Bias")
        st.plotly_chart(plot_series(result.combined_bias, "Combined Bias"), use_container_width=True)

        st.subheader("Bias Fragility")
        st.plotly_chart(plot_series(result.fragility, "Bias Fragility"), use_container_width=True)


if __name__ == "__main__":
    main()
