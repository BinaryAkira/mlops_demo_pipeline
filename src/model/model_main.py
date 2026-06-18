"""Model training orchestration."""

import logging
from typing import Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression

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

    model_type = config["model"]["type"]
    max_iter = config["model"]["max_iter"]

    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=max_iter)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X, y)
    acc = model.score(X, y)

    LOGGER.info("Model training complete: accuracy=%.4f", acc)
    return model, acc
