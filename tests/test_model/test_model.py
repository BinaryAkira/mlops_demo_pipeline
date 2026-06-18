"""Tests for the model training module."""

from src.model.model import train_model, predict
from src.data.data import load_data
from src.features.features import scale_features
import pandas as pd


def test_train_model_returns_model_and_accuracy():
    # Arrange
    X, y = load_data("iris")
    X_scaled, _ = scale_features(X)

    # Act
    model, acc = train_model(X_scaled, y)

    # Assert
    assert hasattr(model, "predict")
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_predict_returns_series():
    # Arrange
    X, y = load_data("iris")
    X_scaled, _ = scale_features(X)
    model, _ = train_model(X_scaled, y)

    # Act
    preds = predict(model, X_scaled)

    # Assert
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(X_scaled)
