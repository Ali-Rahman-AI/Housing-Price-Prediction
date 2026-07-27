#!/usr/bin/env python3
"""
Main pipeline runner.

Executes the full workflow: load → clean → engineer → train → evaluate → save.
"""
import logging
import os
import sys

# Add src/ to path so the 'housing' package is discoverable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from housing.data.load_data import load_config, load_raw_data, validate_schema
from housing.data.clean_data import clean_data
from housing.features.build_features import engineer_features, get_feature_columns
from housing.models.train_model import train_and_evaluate, save_model, save_results
from housing.models.predict_model import load_model
from housing.visualization.visualize import (
    plot_distributions, plot_correlation_matrix,
    plot_model_comparison, plot_residual_analysis, plot_feature_importance
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("HOUSING PRICE PREDICTION PIPELINE")
    logger.info("=" * 60)

    # 1. Load config
    config = load_config("config/config.yaml")
    logger.info("Configuration loaded")

    # 2. Load raw data
    df = load_raw_data(config["paths"]["raw_data"])

    # 3. Validate schema
    is_valid, errors = validate_schema(df, config)
    if not is_valid:
        for e in errors:
            logger.error(e)
        raise ValueError("Data validation failed")

    # 4. Clean data
    df_clean = clean_data(df)

    # 5. Engineer features
    df_features = engineer_features(df_clean)
    df_features.to_csv(config["paths"]["processed_data"], index=False)
    logger.info(f"Processed data saved to {config['paths']['processed_data']}")

    # 6. Generate EDA plots
    logger.info("Generating EDA visualizations...")
    os.makedirs("reports/figures", exist_ok=True)
    plot_distributions(df, save_path="reports/figures/eda_distributions.png")
    plot_correlation_matrix(df, save_path="reports/figures/eda_correlations.png")

    # 7. Train and evaluate
    logger.info("Training models...")
    best_pipeline, results, X_test, y_test_preds = train_and_evaluate(df_features, config)

    # 8. Save model and results
    model_path = save_model(best_pipeline, "gradient_boosting", model_dir="models/")
    save_results(results, filepath="reports/metrics.json")

    # 9. Generate evaluation plots
    logger.info("Generating evaluation visualizations...")
    plot_model_comparison(results, save_path="reports/figures/model_comparison.png")

    # Reconstruct predictions for residual analysis
    numeric_features, categorical_features, target = get_feature_columns(config)
    feature_cols = numeric_features + categorical_features
    X = df_features[feature_cols]
    y = df_features[target]
    X_train, X_test_split, y_train, y_test_split = train_test_split(
        X, y, test_size=config["model"]["test_size"],
        random_state=config["project"]["random_state"],
        stratify=X["location"]
    )
    y_pred = best_pipeline.predict(X_test_split)

    plot_residual_analysis(
        y_test_split.values, y_pred,
        locations=X_test_split["location"].values,
        save_path="reports/figures/residual_analysis.png"
    )

    # Feature importance
    preprocessor = best_pipeline.named_steps["preprocessor"]
    feature_names = (
        numeric_features +
        list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features))
    )
    plot_feature_importance(
        best_pipeline, feature_names,
        save_path="reports/figures/feature_importance.png"
    )

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Best model: Gradient Boosting (R² = {results['GradientBoosting']['test_r2']:.4f})")
    logger.info(f"Model saved: {model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
