# 🏠 Housing Price Prediction

> **End-to-end machine learning project:** Predicting home prices from property characteristics using a production-grade Python pipeline.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Why This Dataset?](#-why-this-dataset)
3. [Data Collection & Source](#-data-collection--source)
4. [Project Structure](#-project-structure)
5. [Understanding the Data](#-understanding-the-data)
6. [Data Loading & Validation](#-data-loading--validation)
7. [Data Cleaning](#-data-cleaning)
8. [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
9. [Feature Engineering](#-feature-engineering)
10. [Modeling Strategy](#-modeling-strategy)
11. [Why These Models?](#-why-these-models)
12. [Results & Evaluation](#-results--evaluation)
13. [Key Findings](#-key-findings)
14. [Limitations & Future Work](#-limitations--future-work)
15. [How to Run](#-how-to-run)
16. [Testing](#-testing)

---

## 🎯 Problem Statement

**Real estate decisions involve hundreds of thousands of dollars.** Buyers fear overpaying. Sellers fear under-pricing. Investors need to spot undervalued properties. The core question is:

> *Can we predict a home's sale price from its observable characteristics, and what actually drives value in this market?*

This is a **supervised regression problem** — we predict a continuous target (price) from a set of input features (square footage, location, bedrooms, etc.). The real value is not just the prediction number, but understanding **which features matter most** and **where the model gets it wrong**.

---

## 📊 Why This Dataset?

We chose this housing dataset because it strikes the right balance for a learning project:

| Criterion | How This Dataset Delivers |
|-----------|--------------------------|
| **Real-world relevance** | Housing is the largest asset class most people ever own. The problem is universally understood. |
| **Clean but not toy** | 500 rows, 6 features — small enough to iterate quickly, but rich enough to require real decisions. |
| **Mixed feature types** | Numeric (sqft, age) + Categorical (location) forces proper preprocessing. |
| **Missing values** | 3% missing in `age` — realistic enough to practice imputation strategies. |
| **Non-linear patterns** | Location interacts with size in complex ways — linear models alone won't suffice. |
| **Actionable insights** | The output directly answers "Is this house fairly priced?" — a real business question. |

**Why not Kaggle/Hugging Face?** We used the provided `housing.csv` because it is pre-scoped for this exact problem, eliminating hours of data wrangling so we can focus on the *methodology* — how to think about features, select models, and evaluate honestly.

---

## 🔍 Data Collection & Source

| Attribute | Detail |
|-----------|--------|
| **Source** | Provided dataset (`housing.csv`) |
| **Size** | 500 rows × 6 columns |
| **Target** | `price` — continuous sale price in USD |
| **Features** | `square_feet`, `bedrooms`, `bathrooms`, `age`, `location` |
| **Locations** | Downtown (38%), Suburbs (48%), Rural (14%) |

The data represents a snapshot of home sales across three distinct market segments. No temporal component is included, so we treat this as a cross-sectional analysis rather than a time-series forecast.

---

## 🗂️ Project Structure

```
housing-price-prediction/
├── config/
│   └── config.yaml              # Central configuration (paths, ranges, hyperparameters)
├── data/
│   ├── raw/
│   │   └── housing.csv          # Original data — never modified
│   └── processed/
│       └── housing_clean.csv    # Cleaned + engineered features
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   └── 00_master_notebook.ipynb # All steps in one place
├── src/housing/                 # Production Python package
│   ├── data/                    # load_data.py, clean_data.py
│   ├── features/                # build_features.py
│   ├── models/                  # train_model.py, predict_model.py
│   └── visualization/           # visualize.py
├── models/
│   └── gradient_boosting_*.pkl  # Serialized best model
├── reports/
│   ├── figures/                 # All generated plots
│   ├── metrics.json             # Model comparison results
│   └── final_report.md          # Executive summary
├── tests/                       # pytest suite (20 tests)
├── requirements.txt
├── setup.py
└── run_pipeline.py              # One-command full execution
```

---

## 📖 Understanding the Data

Before writing any model code, we sat with the data to understand its personality:

| Feature | Type | Range | Missing | Initial Observation |
|---------|------|-------|---------|---------------------|
| `square_feet` | int | 701 – 3,499 | 0% | Strong positive skew; likely dominant driver |
| `bedrooms` | int | 1 – 4 | 0% | Discrete, low cardinality |
| `bathrooms` | float | 0.75 – 3.5 | 0% | Fractions indicate half-baths |
| `age` | float | 1 – 49 | 3% | Uniform distribution; surprisingly weak initial signal |
| `location` | categorical | Downtown, Suburbs, Rural | 0% | Imbalanced; Suburbs dominates |
| `price` | float | $87,400 – $851,900 | 0% | Right-skewed target; wide range |

**No duplicates. No negative values. No impossible entries** (e.g., 0 bedrooms). The data is clean enough that we can focus on modeling rather than firefighting.

---

## 📥 Data Loading & Validation

We do not trust data blindly. Every load operation runs through a **schema validator** that enforces:

1. **Column presence** — Are all expected columns present?
2. **Type correctness** — Do columns match their declared types?
3. **Range checks** — Are values within business-logic bounds? (e.g., price > $50K)
4. **Categorical integrity** — Are location values only from the allowed set?
5. **Missingness threshold** — Is missing data below an acceptable %?

```python
from housing.data.load_data import load_raw_data, validate_schema

config = load_config("config/config.yaml")
df = load_raw_data(config["paths"]["raw_data"])
is_valid, errors = validate_schema(df, config)
```

If validation fails, the pipeline halts immediately with a descriptive error. This prevents silent data corruption from propagating into model training.

---

## 🧹 Data Cleaning

| Issue | Action | Rationale |
|-------|--------|-----------|
| **15 missing `age` values** | Median imputation (age = 23) | Only 3% missing; median is robust to outliers; age distribution is roughly uniform |
| **Extreme outliers** | IQR × 3.0 filter on numeric columns | Conservative multiplier — removes genuine data entry errors without discarding natural variance |
| **Type consistency** | Pandas 3.0 compatibility | String columns may report as `"str"` instead of `"object"`; validator accepts both |

We deliberately **did not** drop the missing rows. With only 500 observations, every row matters. Median imputation preserves the distribution without introducing bias.

---

## 🔬 Exploratory Data Analysis (EDA)

EDA answers: *What story does the data tell?*

### 1. Correlation Analysis

| Feature | Correlation with Price |
|---------|------------------------|
| `square_feet` | **0.80** |
| `bedrooms` | 0.13 |
| `bathrooms` | 0.12 |
| `age` | **-0.09** |

**Insight:** `square_feet` is the dominant driver. Age is surprisingly weak — a 40-year-old house is not systematically cheaper than a 5-year-old one in this market.

### 2. Location Creates Three Markets

| Location | Median Price | $/SqFt |
|----------|-------------|--------|
| Downtown | $504,100 | $243 |
| Suburbs | $374,000 | $175 |
| Rural | $284,200 | $136 |

**Insight:** Downtown commands an **80% premium per square foot** over Rural. Location is not just a feature — it is a market segment.

### 3. Multicollinearity Alert

`bedrooms` and `bathrooms` are **0.93 correlated**. They measure the same underlying concept (home size/complexity). Including both in a linear model causes unstable coefficients.

### 4. Interaction Effects

Plotting `square_feet` vs `price` by location reveals **different intercepts but similar slopes**. This means location modifies the baseline price, but the value of an extra square foot is roughly consistent across locations. Tree-based models capture this naturally.

---

## ⚙️ Feature Engineering

**Principle:** Every new feature must be justified by EDA evidence, not guesswork.

| Feature | Formula | Rationale | Evidence |
|---------|---------|-----------|----------|
| `sqft_per_room` | `square_feet / (bedrooms + bathrooms)` | Measures spaciousness beyond raw size | Correlation with price: **0.38** |
| `total_rooms` | `bedrooms + bathrooms` | Reduces bedroom/bathroom multicollinearity | Replaces two correlated features with one |
| `bed_bath_ratio` | `bedrooms / bathrooms` | Luxury homes often have more baths per bed | Marginal signal (-0.01); kept for tree models |
| `is_new` | `age <= 10` | EDA showed slight premium for newest homes | Binary flag captures non-linear threshold |
| `size_category` | Binned sqft | Captures non-linear size effects | Small/Medium/Large/XL bins |
| `age_category` | Binned age | Captures non-linear age effects | Weak overall, but tree models may use it |

**What we deliberately did NOT engineer:**
- `price_per_sqft` as a model feature — this is **data leakage** (uses the target to create a predictor)
- Polynomial age features — EDA showed age barely matters; forcing complexity is "feature engineering theater"

---

## 🤖 Modeling Strategy

We follow a **progressive complexity** approach:

1. **Start with a strong baseline** (Linear Regression)
2. **Add regularization** (Ridge)
3. **Add non-linearity** (Random Forest)
4. **Add boosting** (Gradient Boosting)
5. **Pick the best via cross-validation**, not test-set peeking

All models use the same preprocessing pipeline:

```python
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="first"), categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", model)
])
```

**Train/test split:** Stratified by location (80/20) to ensure all three markets appear in both sets.

---

## 🧠 Why These Models?

| Model | Why We Included It | Why We Did (Not) Pick It |
|-------|---------------------|--------------------------|
| **Linear Regression** | Fast, interpretable, strong baseline. Coefficients tell you exactly how much each feature moves price. | **Not picked** — misses interaction effects (location × sqft) |
| **Ridge** | L2 regularization handles multicollinearity between bedrooms/bathrooms. | **Not picked** — No improvement over plain linear; data is not high-dimensional enough for regularization to matter. |
| **Random Forest** | Bagging trees capture non-linear interactions natively. Feature importance reveals what the model actually uses. | **Not picked** — Slightly higher variance than Gradient Boosting; more prone to overfitting on this dataset size. |
| **Gradient Boosting** | Sequential learning corrects previous trees' errors. Lower variance than Random Forest. Best CV RMSE. | **✅ PICKED** — Best balance of bias and variance. Handles interactions without explicit engineering. |

**Why not XGBoost / LightGBM / Neural Networks?**
- With only 500 rows and 6 original features, deep learning would overfit immediately.
- XGBoost and LightGBM are excellent but add dependency complexity without meaningful gain on this data scale.
- Gradient Boosting from `sklearn` is sufficient, portable, and requires no extra installs.

---

## 📈 Results & Evaluation

### Model Comparison

| Model | CV RMSE | Test R² | Test RMSE | Test MAE |
|-------|---------|---------|-----------|----------|
| Linear Regression | $33,072 | 0.963 | $29,900 | $23,117 |
| Ridge | $33,152 | 0.963 | $30,088 | $22,951 |
| Random Forest | $27,607 | 0.969 | $27,434 | $20,635 |
| **Gradient Boosting** | **$23,188** | **0.980** | **$22,325** | **$17,123** |

### Feature Importance (Gradient Boosting)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `square_feet` | **68.1%** |
| 2 | `location_Rural` | 16.5% |
| 3 | `location_Suburbs` | 10.8% |
| 4-8 | All others combined | ~4.6% |

**Interpretation:** Square footage and location explain **95% of the model's decision-making**. Everything else is marginal.

### Residual Analysis

| Metric | Value | Meaning |
|--------|-------|---------|
| Mean residual | ~$0 | Little overall bias |
| Downtown bias | +$7,460 | Model slightly **under-predicts** Downtown |
| Suburbs bias | -$9,595 | Model slightly **over-predicts** Suburbs |
| Rural bias | -$1,084 | Nearly unbiased |

### Error by Price Range

| Price Range | Mean % Error | What It Means |
|-------------|-------------|---------------|
| <$250K | 9.6% | Hardest to predict — distressed/unique sales |
| $250-550K | 3.5-5.0% | **Sweet spot** — most common, most predictable |
| >$550K | 3.5% | Low % error but ~$23K absolute dollars |

---

## 🔑 Key Findings

1. **Square footage is king.** It alone explains ~64% of price variance (r = 0.80).
2. **Location is not just a feature — it is a market.** Downtown homes trade at an 80% per-sqft premium over Rural.
3. **Age does not matter in this dataset.** Correlation with price is -0.09. Do not pay a "new home premium" unless it captures something else (construction quality).
4. **The model is a pricing guide, not a gospel.** Average error is $17K — real money. Always flag predictions that deviate >$40K from the listing price for manual review.
5. **The model has location bias.** It under-predicts Downtown and over-predicts Suburbs, suggesting unmeasured amenities (transit, walkability, views) inflate Downtown prices beyond what sqft and room counts capture.

---

## ⚠️ Limitations & Future Work

### Current Limitations

| Limitation | Impact |
|------------|--------|
| **No lot size** | Critical for Rural/Suburbs valuation; a 3,000 sqft house on 0.25 acres vs 5 acres is priced very differently |
| **No school district ratings** | Major price driver for family buyers; completely absent |
| **No renovation/condition data** | Explains why some old homes sell high and some new homes sell low |
| **No temporal dimension** | Housing markets move in cycles; this is a static snapshot |
| **Small sample** | 500 rows limits model complexity and generalization confidence |

### What Would Improve This

- **Lot size** (especially for Rural/Suburbs)
- **School district ratings** (GreatSchools API)
- **Renovation year / condition score** (from listing descriptions or inspection reports)
- **Distance to city center / transit** (Google Maps API)
- **More data** (5,000+ rows to support more complex models)

---

## 🚀 How to Run

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
cd housing-price-prediction
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python run_pipeline.py
```

This single command executes:
1. Data loading & schema validation
2. Data cleaning (missing values, outliers)
3. Feature engineering
4. EDA plot generation
5. Model training & comparison
6. Best model serialization
7. Residual analysis & evaluation plots

### Run Individual Notebooks

```bash
jupyter notebook notebooks/
```

| Notebook | Purpose |
|----------|---------|
| `01_data_understanding.ipynb` | Load, validate, profile |
| `02_eda.ipynb` | Visualize distributions, correlations, location effects |
| `03_feature_engineering.ipynb` | Create features, validate correlations |
| `04_modeling.ipynb` | Train 4 models, compare metrics, save best |
| `05_evaluation.ipynb` | Residual analysis, feature importance, bias check |
| `00_master_notebook.ipynb` | All steps in one notebook |

### Make Predictions on New Data

```python
from housing.models.predict_model import load_model, predict_single

pipeline = load_model("models/gradient_boosting_latest.pkl")

house = {
    "square_feet": 2500,
    "bedrooms": 3,
    "bathrooms": 2.5,
    "age": 10,
    "location": "Downtown"
}

price = predict_single(pipeline, house, config)
print(f"Predicted price: ${price:,.0f}")
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

**Coverage:**
- ✅ Schema validation (missing columns, invalid types, out-of-range values)
- ✅ Missing value handling (median imputation)
- ✅ Outlier removal (IQR method)
- ✅ Feature engineering (correct calculations, no mutation of original data)
- ✅ Pipeline construction and fitting
- ✅ Prediction sanity checks

**Result:** 20 tests, all passing.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgments

Built with:
- [pandas](https://pandas.pydata.org/) for data manipulation
- [scikit-learn](https://scikit-learn.org/) for modeling
- [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/) for visualization
- [pytest](https://docs.pytest.org/) for testing

---

> *"The model is good. But the 2.1% of variance it doesn't explain is where the real estate deals live."*
