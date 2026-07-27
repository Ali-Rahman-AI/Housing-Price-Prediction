"""
Tests for feature engineering module.
"""
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from housing.features.build_features import FeatureEngineer, engineer_features


@pytest.fixture
def clean_df():
    return pd.DataFrame({
        "square_feet": [1500, 2000, 2500, 800],
        "bedrooms": [2, 3, 4, 1],
        "bathrooms": [1.5, 2.0, 2.5, 1.0],
        "age": [5.0, 15.0, 25.0, 35.0],
        "location": ["Downtown", "Suburbs", "Rural", "Downtown"],
        "price": [400000.0, 350000.0, 300000.0, 250000.0]
    })


class TestFeatureEngineer:
    def test_creates_all_features(self, clean_df):
        transformer = FeatureEngineer()
        result = transformer.transform(clean_df)
        expected_features = [
            "sqft_per_room", "total_rooms", "bed_bath_ratio",
            "is_new", "size_category", "age_category"
        ]
        for feat in expected_features:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_sqft_per_room_calculation(self, clean_df):
        transformer = FeatureEngineer()
        result = transformer.transform(clean_df)
        expected = 1500 / 3.5
        assert abs(result.loc[0, "sqft_per_room"] - expected) < 0.01

    def test_is_new_flag(self, clean_df):
        transformer = FeatureEngineer()
        result = transformer.transform(clean_df)
        assert result.loc[0, "is_new"] == 1
        assert result.loc[1, "is_new"] == 0

    def test_size_category_bins(self, clean_df):
        transformer = FeatureEngineer()
        result = transformer.transform(clean_df)
        assert result.loc[0, "size_category"] == "Medium"
        assert result.loc[3, "size_category"] == "Small"

    def test_age_category_bins(self, clean_df):
        transformer = FeatureEngineer()
        result = transformer.transform(clean_df)
        assert result.loc[0, "age_category"] == "New"
        assert result.loc[1, "age_category"] == "Recent"
        assert result.loc[2, "age_category"] == "Mature"
        assert result.loc[3, "age_category"] == "Old"

    def test_does_not_modify_original(self, clean_df):
        original_cols = list(clean_df.columns)
        transformer = FeatureEngineer()
        transformer.transform(clean_df)
        assert list(clean_df.columns) == original_cols


class TestEngineerFeatures:
    def test_adds_price_per_sqft(self, clean_df):
        result = engineer_features(clean_df)
        assert "price_per_sqft" in result.columns
        assert abs(result.loc[0, "price_per_sqft"] - 400000/1500) < 0.01
