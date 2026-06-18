"""Data loading orchestration."""

import logging
from typing import Tuple

import pandas as pd

from src.data.data import load_data

LOGGER = logging.getLogger(__name__)


def run(config: dict) -> Tuple[pd.DataFrame, pd.Series]:
    """Execute the data loading step.

    Args:
        config (dict): Pipeline configuration dictionary.

    Returns:
        tuple: Feature matrix X and target vector y.
    """
    LOGGER.info("Starting data load step")

    dataset_name = config["data"]["dataset"]
    as_frame = config["data"]["as_frame"]

    X, y = load_data(dataset_name=dataset_name, as_frame=as_frame)

    LOGGER.info("Data load complete: X=%s, y=%s", X.shape, y.shape)
    return X, y
