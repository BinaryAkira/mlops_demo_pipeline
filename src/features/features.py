"""Feature engineering utilities for the project."""

from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_features(
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale numerical features using StandardScaler.

    Args:
        X (pd.DataFrame): Input feature matrix.

    Returns:
        tuple: A 2‑item tuple containing:
            - pandas.DataFrame: Scaled feature matrix.
            - StandardScaler: Fitted scaler instance.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled_df, scaler
