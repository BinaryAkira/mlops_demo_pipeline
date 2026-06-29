"""Model training orchestration."""

import logging
from typing import Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

LOGGER = logging.getLogger(__name__)

# Point MLflow to a local folder
mlflow.set_tracking_uri("file:///app/mlruns")
mlflow.set_experiment("training")


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

    # Mlflow logging
    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("max_iter", max_iter)

        # Use the current MLflow API to name the logged model artifact.
        mlflow.sklearn.log_model(model, name="model")

    return model, acc
