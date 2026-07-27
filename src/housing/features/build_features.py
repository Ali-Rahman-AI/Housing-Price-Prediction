"""
Feature engineering for housing price prediction.

All transformations are based on insights from EDA:
- sqft_per_room: captures spaciousness beyond raw square footage
- total_rooms: reduces multicollinearity between bedrooms/bathrooms
- is_new: captures the slight premium for homes <= 10 years old
- size_category: bins square footage for non-linear effects
"""
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom sklearn-compatible transformer for housing feature engineering.
    Can be dropped directly into a Pipeline.
    """

    def __init__(self, create_interactions: bool = True):
        self.create_interactions = create_interactions
        self.new_features = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 1. Spaciousness: square feet per room
        X["sqft_per_room"] = X["square_feet"] / (X["bedrooms"] + X["bathrooms"])

        # 2. Total rooms (reduces bedroom/bathroom multicollinearity)
        X["total_rooms"] = X["bedrooms"] + X["bathrooms"]

        # 3. Bed-to-bath ratio (luxury indicator)
        X["bed_bath_ratio"] = X["bedrooms"] / X["bathrooms"]

        # 4. Is new? (0-10 years showed slight premium in EDA)
        X["is_new"] = (X["age"] <= 10).astype(int)

        # 5. Size category
        X["size_category"] = pd.cut(
            X["square_feet"],
            bins=[0, 1500, 2500, 3500, 5000],
            labels=["Small", "Medium", "Large", "XL"],
            right=False,
            include_lowest=True
        )

        # 6. Age category (non-linear relationship discovered)
        X["age_category"] = pd.cut(
            X["age"],
            bins=[0, 10, 20, 30, 40, 50],
            labels=["New", "Recent", "Mature", "Old", "VeryOld"]
        )

        self.new_features = [
            "sqft_per_room", "total_rooms", "bed_bath_ratio",
            "is_new", "size_category", "age_category"
        ]

        return X


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline to clean data.

    Args:
        df: Clean DataFrame from clean_data.py.

    Returns:
        DataFrame with engineered features added.
    """
    logger.info("Starting feature engineering...")
    df = df.copy()

    # Price per sqft (for EDA only — data leakage risk if used in modeling)
    if "price" in df.columns:
        df["price_per_sqft"] = df["price"] / df["square_feet"]

    # Apply the transformer
    transformer = FeatureEngineer()
    df = transformer.transform(df)

    logger.info(f"Created features: {transformer.new_features}")
    logger.info(f"Final shape: {df.shape}")
    return df


def get_feature_columns(config: dict) -> Tuple[List[str], List[str], str]:
    """
    Get lists of feature columns for modeling.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (numeric_features, categorical_features, target_column).
    """
    numeric = (
        config["features"]["numeric_features"] +
        config["features"]["engineered_features"]
    )
    categorical = config["features"]["categorical_features"]
    target = config["features"]["target"]

    return numeric, categorical, target


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from housing.data.load_data import load_raw_data, load_config
    from housing.data.clean_data import clean_data

    config = load_config()
    df = load_raw_data(config["paths"]["raw_data"])
    df_clean = clean_data(df)
    df_features = engineer_features(df_clean)
    print(df_features.head())
