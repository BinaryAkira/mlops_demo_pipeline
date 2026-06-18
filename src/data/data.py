"""Data loading utilities for the project."""

import logging

from sklearn.datasets import load_iris


LOGGER = logging.getLogger(__name__)


def load_data():
    """Load the Iris dataset as feature and target objects.

    Returns:
        tuple: A 2-item tuple containing:
            - pandas.DataFrame: Feature matrix.
            - pandas.Series: Target labels.

    Raises:
        RuntimeError: If the Iris dataset cannot be loaded.
    """
    try:
        # Use the sklearn frame representation to keep tabular metadata.
        data = load_iris(as_frame=True)
        return data.data, data.target
    except Exception as exc:
        LOGGER.exception("Failed to load Iris dataset")
        raise RuntimeError("Unable to load Iris dataset") from exc
