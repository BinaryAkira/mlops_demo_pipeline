"""Pure feature engineering logic."""

from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale_features(
    X: pd.DataFrame,
    method: str = "standard",
) -> Tuple[pd.DataFrame, object]:
    """Scale features using the specified method.

    Args:
        X (pd.DataFrame): Input feature matrix.
        method (str): Scaling method.

    Returns:
        tuple: Scaled DataFrame and fitted scaler.
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "none":
        return X.copy(), None
    else:
        raise ValueError(f"Unsupported scaling method: {method}")

    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    return X_scaled, scaler
