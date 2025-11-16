"""
Shared utilities for data loading, feature engineering, model
retrieval, and forecasting helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from statsmodels.tsa.arima.model import ARIMAResults
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "app" / "models"
TARGET_COLUMN = "Close_target"
TRAIN_RATIO = 0.8


def ensure_business_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe uses a business-day index and forward-fill gaps.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must be indexed by dates before calling ensure_business_frequency.")
    if df.index.freq is None or df.index.freqstr != "B":
        df = df.asfreq("B")
    return df.ffill()


def discover_datasets(data_dir: Path | None = None) -> Dict[str, Path]:
    root = data_dir or DATA_DIR
    datasets: Dict[str, Path] = {}
    for csv_path in root.glob("*_stocks.csv"):
        if csv_path.name.lower() == "cleaned_bank_stocks.csv":
            continue
        bank = csv_path.stem.replace("_stocks", "").upper()
        datasets[bank] = csv_path
    if not datasets:
        raise FileNotFoundError(f"No bank datasets found in {root.resolve()}")
    return datasets


def load_bank_dataset(bank: str) -> pd.DataFrame:
    csv_path = DATA_DIR / f"{bank}_stocks.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset for bank '{bank}' not found at {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    df = df.dropna(subset=["Close"])
    df = ensure_business_frequency(df)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features[TARGET_COLUMN] = df["Close"].shift(-1)
    features["Close"] = df["Close"]
    features["Close_lag_1"] = df["Close"].shift(1)
    features["Close_lag_2"] = df["Close"].shift(2)
    features["Close_lag_3"] = df["Close"].shift(3)
    features["Return_1d"] = df["Close"].pct_change()
    features["MA_5"] = df["Close"].rolling(window=5).mean()
    features["MA_10"] = df["Close"].rolling(window=10).mean()
    features["STD_5"] = df["Close"].rolling(window=5).std()
    return features.dropna()


def load_metrics(bank: str) -> Dict[str, Any]:
    metrics_path = MODEL_DIR / f"{bank}_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found for bank '{bank}' at {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_lstm_artifacts(bank: str):
    model_path = MODEL_DIR / f"{bank}_lstm.keras"
    meta_path = MODEL_DIR / f"{bank}_lstm_meta.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"LSTM model not found for bank '{bank}' at {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"LSTM metadata not found for bank '{bank}' at {meta_path}")

    model = load_model(model_path)
    meta = joblib.load(meta_path)
    scaler = meta["scaler"]
    lookback = meta["lookback"]
    return model, scaler, lookback


def load_arima_model(bank: str):
    model_path = MODEL_DIR / f"{bank}_arima.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"ARIMA model not found for bank '{bank}' at {model_path}")
    return ARIMAResults.load(str(model_path))


def load_xgb_model(bank: str):
    model_path = MODEL_DIR / f"{bank}_xgb.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost model not found for bank '{bank}' at {model_path}")
    payload = joblib.load(model_path)
    return (
        payload["model"],
        payload["feature_columns"],
        payload.get("required_history", 10),
    )


def lstm_predict_series(
    close_values: Iterable[float],
    model,
    scaler,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.array(list(close_values), dtype=np.float32).reshape(-1, 1)
    scaled = scaler.transform(values)
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback : i])
        y.append(scaled[i])
    if not X:
        raise ValueError("Not enough data to build LSTM sequences.")
    X_arr = np.array(X)
    preds_scaled = model.predict(X_arr, verbose=0)
    y_actual = scaler.inverse_transform(np.array(y).reshape(-1, 1)).ravel()
    y_pred = scaler.inverse_transform(preds_scaled).ravel()
    return y_actual, y_pred


def forecast_lstm_future(
    close_series: pd.Series,
    model,
    scaler,
    lookback: int,
    steps: int,
) -> List[float]:
    if len(close_series) < lookback:
        raise ValueError("Not enough history to forecast with the LSTM model.")
    history_scaled = scaler.transform(close_series.values.reshape(-1, 1)).flatten().tolist()
    predictions: List[float] = []
    for _ in range(steps):
        window = np.array(history_scaled[-lookback:]).reshape(1, lookback, 1)
        pred_scaled = model.predict(window, verbose=0)[0][0]
        history_scaled.append(pred_scaled)
        pred = scaler.inverse_transform(np.array([[pred_scaled]])).ravel()[0]
        predictions.append(float(pred))
    return predictions


def _compute_xgb_feature_row(history: List[float], required_history: int) -> Dict[str, float]:
    if len(history) < required_history:
        raise ValueError(
            f"At least {required_history} historical close values are required for XGBoost features."
        )
    close_series = pd.Series(history)
    return {
        "Close": close_series.iloc[-1],
        "Close_lag_1": close_series.iloc[-2],
        "Close_lag_2": close_series.iloc[-3],
        "Close_lag_3": close_series.iloc[-4],
        "Return_1d": (close_series.iloc[-1] / close_series.iloc[-2] - 1) if close_series.iloc[-2] != 0 else 0.0,
        "MA_5": close_series.iloc[-5:].mean(),
        "MA_10": close_series.iloc[-10:].mean(),
        "STD_5": close_series.iloc[-5:].std(),
    }


def forecast_xgb_future(
    close_series: pd.Series,
    model,
    feature_columns: List[str],
    steps: int,
    required_history: int,
) -> List[float]:
    history = close_series.tolist()
    if len(history) < required_history:
        raise ValueError(f"Not enough history to forecast with the XGBoost model (need {required_history}).")
    predictions: List[float] = []
    for _ in range(steps):
        feature_row = _compute_xgb_feature_row(history, required_history)
        feature_vector = np.array([[feature_row[col] for col in feature_columns]], dtype=np.float32)
        pred = float(model.predict(feature_vector)[0])
        predictions.append(pred)
        history.append(pred)
    return predictions


def forecast_arima_future(arima_results, steps: int) -> List[float]:
    forecast = arima_results.forecast(steps=steps)
    return [float(value) for value in forecast]


def next_business_days(start_date: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start_date + BDay(1), periods=steps, freq="B")


def compute_inverse_rmse_weights(metrics: Dict[str, Dict[str, Any]], model_names: Iterable[str]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    total = 0.0
    for name in model_names:
        model_metric = metrics.get(name, {})
        rmse = model_metric.get("RMSE")
        if rmse and rmse > 0:
            weight = 1.0 / rmse
            weights[name] = weight
            total += weight
    if total == 0:
        model_names = list(model_names)
        equal = 1.0 / len(model_names)
        return {name: equal for name in model_names}
    return {name: weight / total for name, weight in weights.items()}

