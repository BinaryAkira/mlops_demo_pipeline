"""Tests for the data loading module."""

from src.data.data import load_data
import pandas as pd


def test_load_data_returns_expected_types():
    # Arrange & Act
    X, y = load_data("iris")

    # Assert
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_load_data_shapes_are_correct():
    # Arrange & Act
    X, y = load_data("iris")

    # Assert
    assert X.shape == (150, 4)
    assert y.shape == (150,)
