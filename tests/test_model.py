"""
Tests for model training and prediction modules.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from housing.models.train_model import build_pipeline, evaluate_model
from housing.models.predict_model import predict_single


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "square_feet": np.random.randint(800, 3500, n),
        "bedrooms": np.random.randint(1, 5, n),
        "bathrooms": np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0], n),
        "age": np.random.randint(1, 50, n),
        "location": np.random.choice(["Downtown", "Suburbs", "Rural"], n),
        "price": np.random.randint(100000, 800000, n)
    })


@pytest.fixture
def config():
    return {
        "features": {
            "numeric_features": ["square_feet", "bedrooms", "bathrooms", "age"],
            "engineered_features": ["sqft_per_room", "total_rooms", "bed_bath_ratio", "is_new"],
            "categorical_features": ["location"],
            "target": "price"
        },
        "model": {
            "test_size": 0.2,
            "cv_folds": 3
        },
        "project": {
            "random_state": 42
        }
    }


class TestBuildPipeline:
    def test_pipeline_has_two_steps(self):
        pipeline = build_pipeline(
            LinearRegression(),
            numeric_features=["square_feet"],
            categorical_features=["location"]
        )
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "preprocessor"
        assert pipeline.steps[1][0] == "regressor"

    def test_pipeline_can_fit_and_predict(self, sample_data, config):
        from housing.features.build_features import engineer_features

        df = engineer_features(sample_data)
        numeric = config["features"]["numeric_features"] + config["features"]["engineered_features"]
        categorical = config["features"]["categorical_features"]

        pipeline = build_pipeline(LinearRegression(), numeric, categorical)
        X = df[numeric + categorical]
        y = df["price"]

        pipeline.fit(X, y)
        preds = pipeline.predict(X)

        assert len(preds) == len(y)
        assert not np.isnan(preds).any()


class TestEvaluateModel:
    def test_returns_expected_metrics(self, sample_data, config):
        from housing.features.build_features import engineer_features
        from sklearn.model_selection import train_test_split

        df = engineer_features(sample_data)
        numeric = config["features"]["numeric_features"] + config["features"]["engineered_features"]
        categorical = config["features"]["categorical_features"]

        X = df[numeric + categorical]
        y = df["price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline = build_pipeline(LinearRegression(), numeric, categorical)
        metrics, preds = evaluate_model(pipeline, X_train, y_train, X_test, y_test, cv_folds=3)

        assert "cv_rmse_mean" in metrics
        assert "test_r2" in metrics
        assert "test_mae" in metrics
        assert isinstance(metrics["test_r2"], float)  # R² can be negative with random data
        assert metrics["test_rmse"] >= 0


class TestPredictSingle:
    def test_predicts_positive_price(self, sample_data, config):
        from housing.features.build_features import engineer_features

        df = engineer_features(sample_data)
        numeric = config["features"]["numeric_features"] + config["features"]["engineered_features"]
        categorical = config["features"]["categorical_features"]

        pipeline = build_pipeline(LinearRegression(), numeric, categorical)
        X = df[numeric + categorical]
        y = df["price"]
        pipeline.fit(X, y)

        features = {
            "square_feet": 2000,
            "bedrooms": 3,
            "bathrooms": 2.0,
            "age": 15,
            "location": "Suburbs"
        }

        pred = predict_single(pipeline, features, config)
        assert pred > 0
        assert isinstance(pred, float)
