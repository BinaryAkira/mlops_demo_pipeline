"""Orchestration layer for model training."""

import logging
from typing import Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression

from .model import train_model

LOGGER = logging.getLogger(__name__)


def run(
    config: dict,
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[LogisticRegression, float]:
    """Execute the model training step.

    Args:
        config (dict): Pipeline configuration dictionary.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.

    Returns:
        tuple: Trained model and training accuracy.
    """
    LOGGER.info("Starting model training step")
    model, acc = train_model(X, y)
    LOGGER.info("Model training complete: accuracy=%.4f", acc)
    return model, acc
