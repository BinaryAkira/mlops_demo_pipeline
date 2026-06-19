"""Data loading logic."""

from typing import Callable, Dict, Tuple

import pandas as pd
from sklearn.datasets import load_iris
from src.data.validate import validate_data


def _load_iris(as_frame: bool) -> Tuple[pd.DataFrame, pd.Series]:
    data = load_iris(as_frame=as_frame)
    return data.data, data.target


_DATASET_LOADERS: Dict[str, Callable[[bool],
                                     Tuple[pd.DataFrame, pd.Series]]] = {
    "iris": _load_iris,
    # "wine": _load_wine,
}


def load_data(
    dataset_name: str,
    as_frame: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset based on configuration.

    Args:
        dataset_name (str): Name of dataset to load.
        as_frame (bool): Whether to return pandas DataFrame/Series.

    Returns:
        tuple: Feature matrix X and target vector y.
    """
    if dataset_name not in _DATASET_LOADERS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    loader = _DATASET_LOADERS[dataset_name]
    X, y = loader(as_frame)

    validate_data(X, dataset_name)

    return X, y
