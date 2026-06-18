"""Model training and prediction utilities for the project."""

from typing import Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[LogisticRegression, float]:
    """Train a logistic regression classifier.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.

    Returns:
        tuple: A 2‑item tuple containing:
            - LogisticRegression: Trained model.
            - float: Training accuracy score.
    """
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    return model, acc


def predict(
    model: LogisticRegression,
    X: pd.DataFrame,
) -> pd.Series:
    """Generate predictions using a trained model.

    Args:
        model (LogisticRegression): Trained classifier.
        X (pd.DataFrame): Feature matrix.

    Returns:
        pandas.Series: Predicted labels.
    """
    preds = model.predict(X)
    return pd.Series(preds)
