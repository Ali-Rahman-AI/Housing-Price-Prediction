"""
Data cleaning utilities.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.

    Args:
        df: Input DataFrame.
        strategy: Imputation strategy ('median', 'mean', 'mode', 'drop').

    Returns:
        DataFrame with missing values handled.
    """
    df = df.copy()
    missing_before = df.isnull().sum().sum()

    if missing_before == 0:
        logger.info("No missing values found.")
        return df

    logger.info(f"Handling {missing_before} missing values with strategy: {strategy}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue

        if strategy == "median" and col in numeric_cols:
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            logger.info(f"  Filled {col} with median: {fill_val}")
        elif strategy == "mean" and col in numeric_cols:
            fill_val = df[col].mean()
            df[col] = df[col].fillna(fill_val)
            logger.info(f"  Filled {col} with mean: {fill_val:.2f}")
        elif strategy == "mode":
            fill_val = df[col].mode()[0]
            df[col] = df[col].fillna(fill_val)
            logger.info(f"  Filled {col} with mode: {fill_val}")
        elif strategy == "drop":
            df = df.dropna(subset=[col])
            logger.info(f"  Dropped rows with missing {col}")

    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values after cleaning: {missing_after}")
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Remove extreme outliers using the IQR method.
    Uses a high multiplier (3.0) to only catch genuine errors, not natural variance.

    Args:
        df: Input DataFrame.
        columns: Numeric columns to check.
        multiplier: IQR multiplier (default 3.0 for extreme outliers only).

    Returns:
        DataFrame with extreme outliers removed.
    """
    df = df.copy()
    initial_rows = len(df)

    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

        mask = (df[col] >= lower) & (df[col] <= upper)
        removed = (~mask).sum()
        if removed > 0:
            logger.warning(
                f"Removed {removed} outliers from '{col}' "
                f"(outside [{lower:.0f}, {upper:.0f}])"
            )
        df = df[mask]

    final_rows = len(df)
    if final_rows < initial_rows:
        logger.info(f"Removed {initial_rows - final_rows} total outlier rows.")

    return df.reset_index(drop=True)


def clean_data(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Full cleaning pipeline.

    Args:
        df: Raw DataFrame.
        config: Optional configuration dict.

    Returns:
        Clean DataFrame ready for feature engineering.
    """
    logger.info("Starting data cleaning pipeline...")
    df = df.copy()

    # 1. Handle missing values
    df = handle_missing_values(df, strategy="median")

    # 2. Remove extreme outliers (only on price and square_feet)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = remove_outliers_iqr(df, numeric_cols, multiplier=3.0)

    # 3. Reset index
    df = df.reset_index(drop=True)

    logger.info(f"Cleaning complete. Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    from housing.data.load_data import load_raw_data, load_config

    config = load_config()
    df = load_raw_data(config["paths"]["raw_data"])
    df_clean = clean_data(df, config)
    print(f"Cleaned data shape: {df_clean.shape}")
