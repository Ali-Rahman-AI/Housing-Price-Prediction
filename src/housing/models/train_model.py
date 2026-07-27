"""
Model training pipeline.

Follows the principle: start simple, then add complexity only if it helps.
We compare Linear Regression (baseline), Ridge, Random Forest, and Gradient Boosting.
The best model is selected based on cross-validated RMSE.
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from housing.features.build_features import get_feature_columns

logger = logging.getLogger(__name__)


def build_pipeline(model, numeric_features: list, categorical_features: list) -> Pipeline:
    """
    Build a sklearn Pipeline with preprocessing + model.

    Args:
        model: Unfitted sklearn estimator.
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.

    Returns:
        sklearn Pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features)
        ],
        remainder="drop"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    return pipeline


def evaluate_model(pipeline, X_train, y_train, X_test, y_test, cv_folds: int = 5) -> Dict:
    """
    Train and evaluate a model pipeline.

    Args:
        pipeline: sklearn Pipeline.
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        cv_folds: Number of cross-validation folds.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Cross-validation
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv_folds, scoring="neg_mean_squared_error"
    )
    cv_rmse = np.sqrt(-cv_scores.mean())
    cv_rmse_std = np.sqrt(cv_scores.std())

    # Fit on full training set
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # Metrics
    metrics = {
        "cv_rmse_mean": float(cv_rmse),
        "cv_rmse_std": float(cv_rmse_std),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
        "test_r2": float(r2_score(y_test, y_pred_test)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return metrics, y_pred_test


def train_and_evaluate(df: pd.DataFrame, config: dict) -> Tuple[Pipeline, Dict, pd.DataFrame, pd.DataFrame]:
    """
    Full training and evaluation workflow.

    Args:
        df: Feature-engineered DataFrame.
        config: Configuration dictionary.

    Returns:
        Tuple of (best_pipeline, all_results_dict, X_test, y_test_with_preds).
    """
    numeric_features, categorical_features, target = get_feature_columns(config)

    # Prepare features
    feature_cols = numeric_features + categorical_features
    X = df[feature_cols].copy()
    y = df[target].copy()

    # Stratified split by location to ensure all locations in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["project"]["random_state"],
        stratify=X["location"]
    )

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Define models
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=config["project"]["random_state"], n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, random_state=config["project"]["random_state"]
        )
    }

    # Train and evaluate each model
    all_results = {}
    best_model_name = None
    best_cv_rmse = float("inf")
    best_pipeline = None
    best_predictions = None

    for name, model in models.items():
        logger.info(f"Training {name}...")
        pipeline = build_pipeline(model, numeric_features, categorical_features)
        metrics, y_pred = evaluate_model(
            pipeline, X_train, y_train, X_test, y_test,
            cv_folds=config["model"]["cv_folds"]
        )
        all_results[name] = metrics

        logger.info(
            f"  {name}: CV RMSE=${metrics['cv_rmse_mean']:,.0f}, "
            f"Test R²={metrics['test_r2']:.4f}, Test MAE=${metrics['test_mae']:,.0f}"
        )

        if metrics["cv_rmse_mean"] < best_cv_rmse:
            best_cv_rmse = metrics["cv_rmse_mean"]
            best_model_name = name
            best_pipeline = pipeline
            best_predictions = y_pred

    logger.info(f"Best model: {best_model_name} (CV RMSE=${best_cv_rmse:,.0f})")

    # Create results DataFrame
    results_df = pd.DataFrame(all_results).T
    results_df = results_df.round(2)

    # Add predictions to test set for analysis
    y_test_with_preds = pd.DataFrame({
        "actual": y_test.values,
        "predicted": best_predictions,
        "residual": y_test.values - best_predictions,
        "abs_error": np.abs(y_test.values - best_predictions),
        "pct_error": np.abs(y_test.values - best_predictions) / y_test.values * 100
    })
    y_test_with_preds["location"] = X_test["location"].values

    return best_pipeline, all_results, X_test, y_test_with_preds


def save_model(pipeline, model_name: str, model_dir: str = "models/") -> str:
    """Save trained pipeline to disk."""
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name.lower()}_{timestamp}.pkl"
    filepath = os.path.join(model_dir, filename)
    joblib.dump(pipeline, filepath)
    logger.info(f"Model saved to {filepath}")
    return filepath


def save_results(results: Dict, filepath: str = "reports/metrics.json"):
    """Save evaluation results to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {filepath}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")

    from housing.data.load_data import load_raw_data, load_config
    from housing.data.clean_data import clean_data
    from housing.features.build_features import engineer_features

    config = load_config()

    # Load and prepare data
    df = load_raw_data(config["paths"]["raw_data"])
    is_valid, errors = validate_schema(df, config)
    if not is_valid:
        for e in errors:
            print(f"❌ {e}")
        raise ValueError("Data validation failed")

    df_clean = clean_data(df)
    df_features = engineer_features(df_clean)

    # Train
    best_pipeline, results, X_test, y_test_preds = train_and_evaluate(df_features, config)

    # Save
    model_path = save_model(best_pipeline, "gradient_boosting")
    save_results(results)

    print("\n🏆 Final Results:")
    print(pd.DataFrame(results).T)
