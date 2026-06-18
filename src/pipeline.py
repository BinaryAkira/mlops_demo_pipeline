"""End‑to‑end pipeline orchestration."""

import logging
from typing import Tuple

from sklearn.linear_model import LogisticRegression

from src.data.data_main import run as run_data
from src.features.features_main import run as run_features
from src.model.model_main import run as run_model

LOGGER = logging.getLogger(__name__)


def run_pipeline(config) -> Tuple[LogisticRegression, float]:
    """Execute the full ML pipeline.

    Returns:
        tuple: Trained model and training accuracy.
    """
    LOGGER.info("Pipeline started")

    # Step 1: Load data
    X, y = run_data(config)

    # Step 2: Feature engineering
    X_processed, _ = run_features(config, X)

    # Step 3: Model training
    model, acc = run_model(config, X_processed, y)

    LOGGER.info("Pipeline complete: accuracy=%.4f", acc)
    return model, acc
