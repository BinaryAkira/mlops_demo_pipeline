"""Orchestration layer for data loading."""

import logging
from typing import Tuple

import pandas as pd

from src.data.data import load_data

LOGGER = logging.getLogger(__name__)


def run() -> Tuple[pd.DataFrame, pd.Series]:
    """Execute the data loading step.

    Returns:
        tuple: Feature matrix and target labels.
    """
    LOGGER.info("Starting data load step")
    X, y = load_data()
    LOGGER.info("Data load complete: X=%s, y=%s", X.shape, y.shape)
    return X, y
