"""
Data loading and validation utilities.
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    Load raw housing data from CSV.

    Args:
        data_path: Path to the raw CSV file.

    Returns:
        DataFrame with raw data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    return df


def validate_schema(df: pd.DataFrame, config: dict) -> Tuple[bool, List[str]]:
    """
    Validate that the DataFrame matches the expected schema.

    Args:
        df: Input DataFrame.
        config: Configuration dictionary with schema rules.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    errors = []
    schema = config["data"]

    # Check columns
    expected_cols = set(schema["expected_columns"])
    actual_cols = set(df.columns)

    missing_cols = expected_cols - actual_cols
    extra_cols = actual_cols - expected_cols

    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    if extra_cols:
        logger.warning(f"Extra columns found (will be ignored): {extra_cols}")

    # Check types
    type_map = {
        "int": ["int64", "int32", "Int64"],
        "float": ["float64", "float32", "Float64"],
        "str": ["object", "string", "str"],
    }

    for col, expected_type in schema["column_types"].items():
        if col not in df.columns:
            continue
        actual_type = str(df[col].dtype)
        valid_types = type_map.get(expected_type, [expected_type])

        # Special case: pandas 3.0+ may report strings as "str"
        # Also accept any pandas StringDtype variant
        is_valid_type = (
            actual_type in valid_types
            or (expected_type == "float" and actual_type.startswith("int"))
            or (expected_type == "str" and ("str" in actual_type.lower() or actual_type == "object"))
        )

        if not is_valid_type:
            errors.append(
                f"Column '{col}' has type {actual_type}, expected {expected_type}"
            )

    # Check missing values threshold
    missing_pct = df.isnull().mean()
    flagged = missing_pct[missing_pct > schema["missing_threshold"]]
    if not flagged.empty:
        for col, pct in flagged.items():
            errors.append(
                f"Column '{col}' has {pct:.1%} missing values (threshold: {schema['missing_threshold']:.1%})"
            )

    # Check valid ranges
    for col, (min_val, max_val) in schema["valid_ranges"].items():
        if col not in df.columns:
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min < min_val:
            errors.append(f"Column '{col}' has values below {min_val} (min: {col_min})")
        if col_max > max_val:
            errors.append(f"Column '{col}' has values above {max_val} (max: {col_max})")

    # Check categorical values
    for col, valid_values in schema["categorical_values"].items():
        if col not in df.columns:
            continue
        invalid = set(df[col].dropna().unique()) - set(valid_values)
        if invalid:
            errors.append(f"Column '{col}' has invalid values: {invalid}")

    is_valid = len(errors) == 0
    if is_valid:
        logger.info("Schema validation passed.")
    else:
        logger.error(f"Schema validation failed with {len(errors)} errors.")

    return is_valid, errors


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check and report duplicate rows.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with duplicates removed.
    """
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        logger.warning(f"Found {n_dups} duplicate rows. Removing them.")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        logger.info("No duplicate rows found.")
    return df


if __name__ == "__main__":
    config = load_config()
    df = load_raw_data(config["paths"]["raw_data"])
    is_valid, errors = validate_schema(df, config)
    if not is_valid:
        for e in errors:
            print(f"  ❌ {e}")
    else:
        print("✅ All validation checks passed.")
