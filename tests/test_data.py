"""
Tests for data loading and cleaning modules.
"""
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from housing.data.load_data import validate_schema
from housing.data.clean_data import handle_missing_values, remove_outliers_iqr


@pytest.fixture
def sample_config():
    """Minimal config for testing."""
    return {
        "data": {
            "expected_columns": ["square_feet", "bedrooms", "bathrooms", "age", "location", "price"],
            "column_types": {
                "square_feet": "int",
                "bedrooms": "int",
                "bathrooms": "float",
                "age": "float",
                "location": "str",
                "price": "float"
            },
            "missing_threshold": 0.05,
            "valid_ranges": {
                "square_feet": [500, 5000],
                "bedrooms": [1, 10],
                "bathrooms": [0.5, 6.0],
                "age": [0, 200],
                "price": [50000, 2000000]
            },
            "categorical_values": {
                "location": ["Downtown", "Suburbs", "Rural"]
            }
        }
    }


@pytest.fixture
def valid_df():
    """A valid DataFrame for testing."""
    return pd.DataFrame({
        "square_feet": [1500, 2000, 2500],
        "bedrooms": [2, 3, 4],
        "bathrooms": [1.5, 2.0, 2.5],
        "age": [10.0, 20.0, 30.0],
        "location": ["Downtown", "Suburbs", "Rural"],
        "price": [300000.0, 400000.0, 350000.0]
    })


class TestValidateSchema:
    def test_valid_data_passes(self, valid_df, sample_config):
        is_valid, errors = validate_schema(valid_df, sample_config)
        assert is_valid is True
        assert len(errors) == 0

    def test_missing_column_fails(self, valid_df, sample_config):
        df = valid_df.drop(columns=["price"])
        is_valid, errors = validate_schema(df, sample_config)
        assert is_valid is False
        assert any("Missing columns" in e for e in errors)

    def test_invalid_location_fails(self, valid_df, sample_config):
        df = valid_df.copy()
        df.loc[0, "location"] = "Beachfront"
        is_valid, errors = validate_schema(df, sample_config)
        assert is_valid is False
        assert any("invalid values" in e for e in errors)

    def test_out_of_range_fails(self, valid_df, sample_config):
        df = valid_df.copy()
        df.loc[0, "price"] = 5000
        is_valid, errors = validate_schema(df, sample_config)
        assert is_valid is False
        assert any("below" in e for e in errors)

    def test_high_missing_values_flagged(self, valid_df, sample_config):
        df = valid_df.copy()
        df.loc[:2, "age"] = np.nan
        is_valid, errors = validate_schema(df, sample_config)
        assert is_valid is False
        assert any("missing values" in e for e in errors)


class TestHandleMissingValues:
    def test_median_imputation(self):
        df = pd.DataFrame({
            "age": [10.0, 20.0, np.nan, 30.0],
            "price": [100.0, 200.0, 300.0, 400.0]
        })
        result = handle_missing_values(df, strategy="median")
        assert result["age"].isnull().sum() == 0
        assert result.loc[2, "age"] == 20.0

    def test_no_missing_returns_unchanged(self):
        df = pd.DataFrame({"age": [10.0, 20.0, 30.0]})
        result = handle_missing_values(df, strategy="median")
        pd.testing.assert_frame_equal(result, df)


class TestRemoveOutliers:
    def test_removes_extreme_outliers(self):
        df = pd.DataFrame({"price": [100, 105, 110, 115, 10000]})
        result = remove_outliers_iqr(df, ["price"], multiplier=3.0)
        assert len(result) == 4
        assert 10000 not in result["price"].values

    def test_keeps_normal_data(self):
        df = pd.DataFrame({"price": [100, 105, 110, 115, 120]})
        result = remove_outliers_iqr(df, ["price"], multiplier=3.0)
        assert len(result) == 5
