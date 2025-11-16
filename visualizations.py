"""
Plotly-based visualization utilities for historical prices and model forecasts.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_historical_prices(df: pd.DataFrame, bank: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name="Close",
                line=dict(color="#1f77b4"),
            )
        ]
    )
    fig.update_layout(
        title=f"{bank} Historical Closing Prices",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        height=500,
    )
    return fig


def plot_model_backtests(predictions: Dict[str, pd.DataFrame], bank: str) -> go.Figure:
    fig = go.Figure()
    reference_model = next(iter(predictions.values()))
    fig.add_trace(
        go.Scatter(
            x=reference_model.index,
            y=reference_model["actual"],
            mode="lines",
            name="Actual",
            line=dict(color="#2ca02c"),
        )
    )
    colors = {
        "LSTM": "#1f77b4",
        "ARIMA": "#ff7f0e",
        "XGBoost": "#9467bd",
        "Hybrid": "#d62728",
    }
    for name, df_pred in predictions.items():
        fig.add_trace(
            go.Scatter(
                x=df_pred.index,
                y=df_pred["prediction"],
                mode="lines",
                name=f"{name} Prediction",
                line=dict(color=colors.get(name, None)),
            )
        )
    fig.update_layout(
        title=f"{bank} Actual vs. Predicted Closing Prices",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        height=500,
    )
    return fig


def plot_future_forecast(future_df: pd.DataFrame, bank: str) -> go.Figure:
    fig = go.Figure()
    for column in future_df.columns:
        fig.add_trace(
            go.Scatter(
                x=future_df.index,
                y=future_df[column],
                mode="lines+markers",
                name=column,
            )
        )
    fig.update_layout(
        title=f"{bank} Forecast - Next {len(future_df)} Business Days",
        xaxis_title="Date",
        yaxis_title="Predicted Close",
        template="plotly_white",
        height=500,
    )
    return fig


def plot_model_metrics(metrics: Dict[str, Dict[str, float]], metric_name: str = "RMSE") -> go.Figure:
    values = []
    labels = []
    for model, metric_dict in metrics.items():
        if model == "Hybrid":
            continue
        score = metric_dict.get(metric_name)
        if score is not None:
            values.append(score)
            labels.append(model)
    fig = px.bar(
        x=labels,
        y=values,
        labels={"x": "Model", "y": metric_name},
        title=f"Model Comparison ({metric_name})",
        text_auto=".2f",
        template="plotly_white",
    )
    fig.update_traces(marker_color=["#1f77b4", "#ff7f0e", "#9467bd"])
    fig.update_layout(height=400)
    return fig

