"""Orchestration layer for feature engineering."""

import logging
from typing import Tuple

import pandas as pd

from src.features.features import scale_features

LOGGER = logging.getLogger(__name__)


def run(config: dict, X: pd.DataFrame) -> Tuple[pd.DataFrame, object]:
    """Execute the feature engineering step.

    Args:
        config (dict): Pipeline configuration dictionary.
        X (pd.DataFrame): Raw feature matrix.

    Returns:
        tuple: Scaled features and fitted scaler.
    """
    LOGGER.info("Starting feature engineering step")
    X_scaled, scaler = scale_features(X)
    LOGGER.info("Feature engineering complete: X_scaled=%s", X_scaled.shape)
    return X_scaled, scaler
