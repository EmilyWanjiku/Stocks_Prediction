"""
Training pipeline for forecasting Kenyan bank stock prices using
LSTM, ARIMA, and XGBoost models. Each trained model is persisted to
`app/models/` and evaluation metrics are printed and saved per bank.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor

from utils import (
    DATA_DIR,
    MODEL_DIR,
    TARGET_COLUMN,
    TRAIN_RATIO,
    discover_datasets,
    engineer_features,
    ensure_business_frequency,
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LOOKBACK = 60
RANDOM_STATE = 42


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    metrics: Dict[str, float]
    predictions: pd.DataFrame  # Columns: actual, prediction
    extras: Dict[str, Any] | None = None


def evaluate_predictions(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=np.float32)
    y_pred = np.asarray(list(y_pred), dtype=np.float32)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def prepare_lstm_data(series: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback : i])
        y.append(series[i])
    X_arr = np.array(X)
    y_arr = np.array(y)
    return X_arr, y_arr


def train_lstm(df: pd.DataFrame, bank: str) -> ModelResult:
    close_values = df[["Close"]].values.astype(np.float32)
    lookback = min(DEFAULT_LOOKBACK, max(5, len(df) // 5))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(close_values)
    X, y = prepare_lstm_data(scaled_close, lookback)
    if len(X) < 10:
        raise ValueError("Insufficient data to train LSTM model.")

    dates = df.index[lookback:]
    train_size = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    test_dates = dates[train_size:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = build_lstm_model((X_train.shape[1], 1))
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )
    model.fit(
        X_train,
        y_train,
        epochs=75,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0,
    )

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred_inv = scaler.inverse_transform(y_pred_scaled).ravel()

    metrics = evaluate_predictions(y_test_inv, y_pred_inv)

    model_path = MODEL_DIR / f"{bank}_lstm.keras"
    model.save(model_path)
    meta_path = MODEL_DIR / f"{bank}_lstm_meta.pkl"
    joblib.dump(
        {
            "scaler": scaler,
            "lookback": lookback,
        },
        meta_path,
    )

    predictions = pd.DataFrame(
        {"actual": y_test_inv, "prediction": y_pred_inv},
        index=test_dates,
    )
    return ModelResult("LSTM", metrics, predictions)


def train_arima(df: pd.DataFrame, bank: str) -> ModelResult:
    split_idx = int(len(df) * TRAIN_RATIO)
    train_close = df["Close"].iloc[:split_idx]
    test_close = df["Close"].iloc[split_idx:]
    test_dates = df.index[split_idx:]

    if len(test_close) == 0:
        raise ValueError("Insufficient test data for ARIMA evaluation.")

    # Simple ARIMA configuration; can be tuned later or replaced with auto-ARIMA.
    order = (5, 1, 0)
    model = ARIMA(train_close, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test_close))

    metrics = evaluate_predictions(test_close.values, forecast.values)

    model_path = MODEL_DIR / f"{bank}_arima.pkl"
    model_fit.save(model_path)

    predictions = pd.DataFrame(
        {"actual": test_close.values, "prediction": forecast.values},
        index=test_dates,
    )
    return ModelResult("ARIMA", metrics, predictions)


def train_xgboost(df: pd.DataFrame, bank: str) -> ModelResult:
    features = engineer_features(df)
    feature_cols = [col for col in features.columns if col != TARGET_COLUMN]
    X = features[feature_cols]
    y = features[TARGET_COLUMN]

    train_size = int(len(X) * TRAIN_RATIO)
    if train_size == 0 or train_size == len(X):
        raise ValueError("Insufficient data to split for XGBoost training.")

    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    test_dates = y_test.index + BDay(1)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate_predictions(y_test.values, y_pred)

    model_path = MODEL_DIR / f"{bank}_xgb.pkl"
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "required_history": 10,
            "target_column": TARGET_COLUMN,
        },
        model_path,
    )

    predictions = pd.DataFrame(
        {"actual": y_test.values, "prediction": y_pred},
        index=test_dates,
    )
    return ModelResult("XGBoost", metrics, predictions)


def compute_hybrid_result(bank: str, results: List[ModelResult]) -> ModelResult | None:
    if len(results) < 3:
        return None

    prediction_frames = {res.name: res.predictions for res in results}

    common_index = set.intersection(*(set(df.index) for df in prediction_frames.values()))
    if not common_index:
        logger.warning("No overlapping prediction horizon for hybrid model.")
        return None
    common_index = sorted(common_index)

    combined = pd.DataFrame(index=common_index)
    first_model = next(iter(prediction_frames.values()))
    combined["actual"] = first_model.loc[common_index, "actual"].values

    weights: Dict[str, float] = {}
    weight_sum = 0.0
    for res in results:
        rmse = res.metrics.get("RMSE")
        if rmse and rmse > 0:
            weight = 1.0 / rmse
            weights[res.name] = weight
            weight_sum += weight
        else:
            weights[res.name] = 0.0

    if weight_sum == 0:
        equal_weight = 1.0 / len(results)
        weights = {res.name: equal_weight for res in results}
    else:
        weights = {name: weight / weight_sum for name, weight in weights.items()}

    for name, df_pred in prediction_frames.items():
        combined[f"{name.lower()}_pred"] = df_pred.loc[common_index, "prediction"].values

    combined["prediction"] = 0.0
    for name, df_pred in prediction_frames.items():
        weight = weights.get(name, 0.0)
        combined["prediction"] += weight * df_pred.loc[common_index, "prediction"].values

    metrics = evaluate_predictions(combined["actual"].values, combined["prediction"].values)
    return ModelResult("Hybrid", metrics, combined, extras={"weights": weights})


def persist_metrics(bank: str, results: List[ModelResult]) -> None:
    metrics_payload: Dict[str, Dict[str, Any]] = {}
    for res in results:
        payload: Dict[str, Any] = dict(res.metrics)
        if res.extras:
            payload.update(res.extras)
        metrics_payload[res.name] = payload
    metrics_path = MODEL_DIR / f"{bank}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=4)


def train_models_for_bank(bank: str, csv_path: Path) -> None:
    logger.info("Processing %s from %s", bank, csv_path.name)
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    df = df.dropna(subset=["Close"])
    df = ensure_business_frequency(df)

    if len(df) < 100:
        logger.warning("Dataset for %s is small (%d rows); results may be unstable.", bank, len(df))

    results: List[ModelResult] = []

    try:
        lstm_result = train_lstm(df.copy(), bank)
        results.append(lstm_result)
        logger.info("%s LSTM metrics: %s", bank, lstm_result.metrics)
    except Exception as exc:
        logger.exception("LSTM training failed for %s: %s", bank, exc)

    try:
        arima_result = train_arima(df.copy(), bank)
        results.append(arima_result)
        logger.info("%s ARIMA metrics: %s", bank, arima_result.metrics)
    except Exception as exc:
        logger.exception("ARIMA training failed for %s: %s", bank, exc)

    try:
        xgb_result = train_xgboost(df.copy(), bank)
        results.append(xgb_result)
        logger.info("%s XGBoost metrics: %s", bank, xgb_result.metrics)
    except Exception as exc:
        logger.exception("XGBoost training failed for %s: %s", bank, exc)

    hybrid_result = compute_hybrid_result(bank, results)
    if hybrid_result:
        results.append(hybrid_result)
        logger.info("%s Hybrid metrics: %s", bank, hybrid_result.metrics)

    if results:
        persist_metrics(bank, results)
    else:
        logger.error("No models were successfully trained for %s.", bank)


def main() -> None:
    datasets = discover_datasets(DATA_DIR)
    for bank, csv_path in datasets.items():
        train_models_for_bank(bank, csv_path)
    logger.info("Training run complete.")


if __name__ == "__main__":
    main()

