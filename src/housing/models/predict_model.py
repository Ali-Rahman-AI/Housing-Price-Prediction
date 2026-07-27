"""
Model inference utilities.

Load a trained model and make predictions on new data.
"""
import argparse
import logging
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from housing.features.build_features import engineer_features, get_feature_columns
from housing.data.load_data import load_config

logger = logging.getLogger(__name__)


def load_model(model_path: str) -> Pipeline:
    """
    Load a serialized model pipeline.

    Args:
        model_path: Path to the .pkl or .joblib file.

    Returns:
        Loaded sklearn Pipeline.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    pipeline = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    return pipeline


def predict(pipeline: Pipeline, df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Make predictions on new data.

    Args:
        pipeline: Trained sklearn Pipeline.
        df: DataFrame with raw or engineered features.
        config: Configuration dictionary.

    Returns:
        DataFrame with original columns + 'predicted_price' column.
    """
    df = df.copy()

    # Engineer features if not already present
    required_engineered = config["features"]["engineered_features"]
    if not all(col in df.columns for col in required_engineered):
        logger.info("Engineering features for prediction data...")
        df = engineer_features(df)

    # Select features used by the model
    numeric_features, categorical_features, target = get_feature_columns(config)
    feature_cols = numeric_features + categorical_features

    # Ensure all required columns exist
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required features for prediction: {missing}")

    X = df[feature_cols]
    predictions = pipeline.predict(X)

    df["predicted_price"] = predictions
    df["prediction_confidence"] = "medium"  # Placeholder for future uncertainty quantification

    logger.info(f"Generated predictions for {len(df)} rows")
    return df


def predict_single(pipeline: Pipeline, features: dict, config: dict) -> float:
    """
    Predict price for a single house.

    Args:
        pipeline: Trained sklearn Pipeline.
        features: Dictionary of house features.
        config: Configuration dictionary.

    Returns:
        Predicted price as float.
    """
    df = pd.DataFrame([features])
    df = engineer_features(df)

    numeric_features, categorical_features, target = get_feature_columns(config)
    feature_cols = numeric_features + categorical_features
    X = df[feature_cols]

    pred = pipeline.predict(X)[0]
    return float(pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make housing price predictions")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--model", default="models/gradient_boosting_latest.pkl", help="Path to model")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    config = load_config()
    pipeline = load_model(args.model)

    input_df = pd.read_csv(args.input)
    results = predict(pipeline, input_df, config)
    results.to_csv(args.output, index=False)

    print(f"Predictions saved to {args.output}")
    print(results[["square_feet", "location", "predicted_price"]].head())
