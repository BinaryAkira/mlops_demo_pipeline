"""Pure data loading logic."""

from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris


def load_data(
    dataset_name: str = "iris",
    as_frame: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset based on configuration.

    Args:
        dataset_name (str): Name of dataset to load.
        as_frame (bool): Whether to return pandas DataFrame/Series.

    Returns:
        tuple: Feature matrix X and target vector y.
    """
    if dataset_name == "iris":
        data = load_iris(as_frame=as_frame)
        return data.data, data.target

    raise ValueError(f"Unsupported dataset: {dataset_name}")
