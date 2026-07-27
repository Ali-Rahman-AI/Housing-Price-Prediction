"""
Reusable visualization utilities for the housing project.

All functions return matplotlib Figure objects so they can be saved
or displayed by the caller.
"""
import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Set consistent style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_distributions(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distributions of key numeric variables.
    """
    numeric_cols = ["square_feet", "bedrooms", "bathrooms", "age", "price"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Housing Data Distributions", fontsize=16, fontweight="bold", y=1.02)

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx // 3, idx % 3]
        ax.hist(df[col].dropna(), bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        ax.set_title(f"{col.replace('_', ' ').title()}")
        ax.set_xlabel(col.replace('_', ' ').title())
        ax.set_ylabel("Count")

        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean_val:,.0f}")
        ax.axvline(median_val, color="green", linestyle="--", alpha=0.7, label=f"Median: {median_val:,.0f}")
        ax.legend(fontsize=8)

    # Remove empty subplot
    axes[1, 2].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved distribution plot to {save_path}")
    return fig


def plot_correlation_matrix(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot correlation heatmap for numeric features.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f",
        square=True, ax=ax, vmin=-1, vmax=1,
        linewidths=0.5
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved correlation plot to {save_path}")
    return fig


def plot_price_by_location(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Boxplot of price by location.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df.boxplot(column="price", by="location", ax=axes[0])
    axes[0].set_title("Price by Location")
    axes[0].set_xlabel("Location")
    axes[0].set_ylabel("Price ($)")

    # Price per sqft by location
    df["price_per_sqft"] = df["price"] / df["square_feet"]
    df.boxplot(column="price_per_sqft", by="location", ax=axes[1])
    axes[1].set_title("Price per Sq Ft by Location")
    axes[1].set_xlabel("Location")
    axes[1].set_ylabel("Price per Sq Ft ($)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(pipeline: Pipeline, feature_names: list, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importance from tree-based model.
    """
    model = pipeline.named_steps["regressor"]

    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not have feature_importances_ attribute.")
        return None

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feat_imp["feature"], feat_imp["importance"], color="goldenrod", edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance", fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved feature importance plot to {save_path}")
    return fig


def plot_residual_analysis(y_true: np.ndarray, y_pred: np.ndarray, 
                            locations: Optional[np.ndarray] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Comprehensive residual analysis plot.
    """
    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Model Evaluation: Residual Analysis", fontsize=16, fontweight="bold", y=1.02)

    # 1. Predicted vs Actual
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.6, color="steelblue", edgecolor="white", s=50)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title("Predicted vs Actual")
    ax.legend()

    # 2. Residuals vs Predicted
    ax = axes[0, 1]
    ax.scatter(y_pred, residuals, alpha=0.6, color="coral", edgecolor="white", s=50)
    ax.axhline(y=0, color="black", linestyle="--", lw=1)
    ax.set_xlabel("Predicted Price ($)")
    ax.set_ylabel("Residual ($)")
    ax.set_title("Residuals vs Predicted")

    # 3. Residual distribution
    ax = axes[0, 2]
    ax.hist(residuals, bins=30, color="mediumseagreen", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="red", linestyle="--", lw=2)
    ax.axvline(x=residuals.mean(), color="blue", linestyle="-", lw=1, label=f"Mean: ${residuals.mean():,.0f}")
    ax.set_xlabel("Residual ($)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.legend()

    # 4. Residuals by location
    ax = axes[1, 0]
    if locations is not None:
        residual_df = pd.DataFrame({
            "location": locations,
            "residual": residuals
        })
        residual_df.boxplot(column="residual", by="location", ax=ax)
        ax.set_title("Residuals by Location")
        ax.set_xlabel("Location")
        ax.set_ylabel("Residual ($)")
    else:
        ax.text(0.5, 0.5, "Location data not provided", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Residuals by Location")

    # 5. Absolute error vs actual price
    ax = axes[1, 1]
    ax.scatter(y_true, abs_errors, alpha=0.6, color="mediumpurple", edgecolor="white", s=50)
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Absolute Error ($)")
    ax.set_title("Absolute Error vs Actual Price")

    # 6. Percentage error distribution
    ax = axes[1, 2]
    pct_errors = (abs_errors / y_true) * 100
    ax.hist(pct_errors, bins=30, color="teal", edgecolor="white", alpha=0.8)
    ax.axvline(x=pct_errors.mean(), color="red", linestyle="--", lw=2, 
               label=f"Mean: {pct_errors.mean():.1f}%")
    ax.set_xlabel("Percentage Error (%)")
    ax.set_ylabel("Count")
    ax.set_title("Percentage Error Distribution")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved residual analysis plot to {save_path}")
    return fig


def plot_model_comparison(results: dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare multiple models side by side.

    Args:
        results: Dict of {model_name: metrics_dict}.
    """
    results_df = pd.DataFrame(results).T

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R² comparison
    axes[0].bar(results_df.index, results_df["test_r2"], color="steelblue", edgecolor="white")
    axes[0].set_ylabel("R² Score")
    axes[0].set_title("Model Comparison: R² Score")
    axes[0].set_ylim(0.95, 0.985)
    for i, v in enumerate(results_df["test_r2"]):
        axes[0].text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=9)

    # RMSE comparison
    axes[1].bar(results_df.index, results_df["test_rmse"], color="coral", edgecolor="white")
    axes[1].set_ylabel("Test RMSE ($)")
    axes[1].set_title("Model Comparison: Test RMSE")
    for i, v in enumerate(results_df["test_rmse"]):
        axes[1].text(i, v + 500, f"${v:,.0f}", ha="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from housing.data.load_data import load_raw_data

    df = load_raw_data("data/raw/housing.csv")
    fig = plot_distributions(df, save_path="reports/figures/distributions.png")
    plt.show()
