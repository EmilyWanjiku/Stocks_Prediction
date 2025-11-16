"""
Hybrid forecasting helpers to combine LSTM, ARIMA, and XGBoost models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from pandas.tseries.offsets import BDay

from utils import (
    TRAIN_RATIO,
    compute_inverse_rmse_weights,
    engineer_features,
    forecast_arima_future,
    forecast_lstm_future,
    forecast_xgb_future,
    load_arima_model,
    load_bank_dataset,
    load_lstm_artifacts,
    load_metrics,
    load_xgb_model,
    lstm_predict_series,
    next_business_days,
    TARGET_COLUMN,
)

MODEL_NAMES = ("LSTM", "ARIMA", "XGBoost")


@dataclass
class ForecastOutputs:
    historical: Dict[str, pd.DataFrame]
    hybrid_historical: pd.DataFrame
    future: pd.DataFrame
    weights: Dict[str, float]


def _align_and_weight(prediction_frames: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> pd.DataFrame:
    if not prediction_frames:
        raise ValueError("No prediction frames supplied for hybrid computation.")
    common_index = set.intersection(*(set(df.index) for df in prediction_frames.values()))
    if not common_index:
        raise ValueError("Models do not share a common prediction horizon for hybrid computation.")
    common_index = sorted(common_index)

    first_model = next(iter(prediction_frames.values()))
    combined = pd.DataFrame(index=common_index)
    combined["actual"] = first_model.loc[common_index, "actual"].values

    combined_pred = pd.Series(0.0, index=common_index, dtype=float)
    for name, df_pred in prediction_frames.items():
        weight = weights.get(name, 0.0)
        combined_pred += weight * df_pred.loc[common_index, "prediction"]
        combined[f"{name.lower()}_pred"] = df_pred.loc[common_index, "prediction"].values

    combined["prediction"] = combined_pred.values
    return combined


def _select_hybrid_weights(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    hybrid_metrics = metrics.get("Hybrid", {})
    stored_weights = hybrid_metrics.get("weights")
    if stored_weights:
        total = sum(stored_weights.values())
        if total > 0:
            return {name: weight / total for name, weight in stored_weights.items()}
    return compute_inverse_rmse_weights(metrics, MODEL_NAMES)


def _build_historical_predictions(bank: str) -> Dict[str, pd.DataFrame]:
    df = load_bank_dataset(bank)
    close_series = df["Close"]

    lstm_model, scaler, lookback = load_lstm_artifacts(bank)
    lstm_actual, lstm_pred = lstm_predict_series(close_series, lstm_model, scaler, lookback)
    lstm_index = close_series.index[lookback:]
    lstm_df = pd.DataFrame({"actual": lstm_actual, "prediction": lstm_pred}, index=lstm_index)

    arima_model = load_arima_model(bank)
    train_size = int(len(df) * TRAIN_RATIO)
    arima_forecast = arima_model.forecast(steps=len(df) - train_size)
    arima_index = close_series.index[train_size:]
    arima_actual = close_series.iloc[train_size:]
    arima_df = pd.DataFrame({"actual": arima_actual.values, "prediction": arima_forecast.values}, index=arima_index)

    xgb_model, feature_columns, required_history = load_xgb_model(bank)
    features = engineer_features(df)
    X = features[feature_columns]
    y = features[TARGET_COLUMN]
    xgb_pred = xgb_model.predict(X)
    xgb_index = features.index + BDay(1)
    xgb_df = pd.DataFrame({"actual": y.values, "prediction": xgb_pred}, index=xgb_index)

    return {"LSTM": lstm_df, "ARIMA": arima_df, "XGBoost": xgb_df}


def _build_future_predictions(bank: str, steps: int, weights: Dict[str, float]) -> pd.DataFrame:
    df = load_bank_dataset(bank)
    close_series = df["Close"]
    future_index = next_business_days(close_series.index[-1], steps)

    lstm_model, scaler, lookback = load_lstm_artifacts(bank)
    lstm_future = forecast_lstm_future(close_series, lstm_model, scaler, lookback, steps)

    arima_model = load_arima_model(bank)
    arima_future = forecast_arima_future(arima_model, steps)

    xgb_model, feature_columns, required_history = load_xgb_model(bank)
    xgb_future = forecast_xgb_future(close_series, xgb_model, feature_columns, steps, required_history)

    future_df = pd.DataFrame(
        {
            "LSTM": lstm_future,
            "ARIMA": arima_future,
            "XGBoost": xgb_future,
        },
        index=future_index,
    )
    future_df["Hybrid"] = sum(
        weights.get(name, 0.0) * future_df[name] for name in MODEL_NAMES
    )
    return future_df


def generate_forecasts(bank: str, steps: int = 30) -> ForecastOutputs:
    metrics = load_metrics(bank)
    weights = _select_hybrid_weights(metrics)
    historical_predictions = _build_historical_predictions(bank)
    hybrid_historical = _align_and_weight(historical_predictions, weights)
    future_predictions = _build_future_predictions(bank, steps, weights)
    return ForecastOutputs(
        historical=historical_predictions,
        hybrid_historical=hybrid_historical,
        future=future_predictions,
        weights=weights,
    )

