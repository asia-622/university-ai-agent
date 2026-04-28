"""
model.py — Linear Regression model for student performance prediction.
Trains on any detected subject columns; target = Average or specified col.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("university_agent.model")


def train_model(meta: dict, target_col: str | None = None) -> dict | None:
    """
    Train a Linear Regression model.

    Parameters
    ----------
    meta        preprocessed metadata dict
    target_col  column to predict (default: 'Average' if exists, else last subject col)

    Returns
    -------
    dict with keys: estimator, scaler, feature_cols, target_col, metrics
    or None if training is not possible.
    """
    df = meta["df"]
    subject_cols = meta.get("subject_cols", [])

    if not subject_cols:
        logger.warning("No subject columns found – cannot train model.")
        return None

    # Choose target
    if target_col is None:
        if "Average" in df.columns:
            target_col = "Average"
        elif len(subject_cols) >= 2:
            target_col = subject_cols[-1]   # predict last subject
        else:
            logger.warning("Not enough columns to train a regression model.")
            return None

    feature_cols = [c for c in subject_cols if c != target_col]
    if meta.get("attend_col") and meta["attend_col"] in df.columns:
        feature_cols.append(meta["attend_col"])

    if len(feature_cols) < 1:
        logger.warning("Not enough feature columns.")
        return None

    data = df[feature_cols + [target_col]].dropna()
    if len(data) < 20:
        logger.warning("Too few samples (%d) to train.", len(data))
        return None

    X = data[feature_cols].values
    y = data[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    y_pred = lr.predict(X_test_s)

    metrics = {
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    logger.info("Model trained — R²=%.4f  MAE=%.4f", metrics["r2"], metrics["mae"])

    return {
        "estimator": lr,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "metrics": metrics,
    }


def predict_batch(model: dict, df: pd.DataFrame) -> pd.Series:
    """Generate predictions for a whole DataFrame."""
    X = df[model["feature_cols"]].fillna(0).values
    X_s = model["scaler"].transform(X)
    preds = model["estimator"].predict(X_s)
    return pd.Series(preds, index=df.index, name="Predicted_" + model["target_col"])
