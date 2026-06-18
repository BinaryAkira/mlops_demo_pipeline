"""Tests for the feature engineering module."""

from src.features.features import scale_features
from src.data.data import load_data
import pandas as pd


def test_scale_features_returns_dataframe_and_scaler():
    # Arrange
    X, _ = load_data()

    # Act
    X_scaled, scaler = scale_features(X)

    # Assert
    assert isinstance(X_scaled, pd.DataFrame)
    assert hasattr(scaler, "transform")


def test_scale_features_preserves_shape():
    # Arrange
    X, _ = load_data()

    # Act
    X_scaled, _ = scale_features(X)

    # Assert
    assert X_scaled.shape == X.shape
