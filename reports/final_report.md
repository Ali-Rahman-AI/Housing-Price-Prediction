# Housing Price Prediction: Final Report

## Executive Summary

We built a Gradient Boosting Regressor that predicts home prices with **97.9% accuracy** (R²) and an average error of **$17,655**. The model is most reliable for mid-range homes ($250K–$550K) and struggles with extreme outliers.

## Business Impact

- **Buyers:** Can identify if a listing is fairly priced relative to its specs
- **Sellers:** Understand which features actually drive value
- **Investors:** Spot undervalued properties (e.g., Suburbs homes priced below model prediction)

## Key Drivers of Price

| Rank | Feature | Importance | Insight |
|------|---------|-----------|---------|
| 1 | Square footage | 68.1% | The single most important factor |
| 2 | Location (Rural) | 16.5% | Rural homes trade at significant discount |
| 3 | Location (Suburbs) | 10.8% | Suburbs priced between Rural and Downtown |
| 4–8 | Bedrooms, bathrooms, age, etc. | ~4.6% combined | Marginal after controlling for size + location |

## Model Performance

| Metric | Value |
|--------|-------|
| Cross-Validation RMSE | $23,297 |
| Test RMSE | $22,635 |
| Test MAE | $17,655 |
| Test R² | 0.979 |

## Known Limitations

1. **Age is underweighted.** The data shows age has almost no correlation with price, which contradicts intuition. This may be because "age" in this dataset doesn't capture renovation quality.

2. **Location bias exists.** The model under-predicts Downtown by ~$7,500 and over-predicts Suburbs by ~$9,600 on average. This suggests unmeasured Downtown amenities (transit, walkability, views) that inflate prices.

3. **Outlier sensitivity.** Homes priced below $250K or above $550K have higher relative error, likely due to unique circumstances (distressed sales, luxury upgrades) not captured in the data.

## Recommendations

### For Model Improvement
- Add **lot size** — critical for Rural/Suburbs valuation
- Add **school district ratings** — major price driver for families
- Add **renovation year / condition score** — explains why some old homes sell high
- Add **distance to city center / transit** — refines the blunt "location" variable

### For Business Use
- **Use the model as a pricing guide, not a gospel.** The $17K average error is real money.
- **Flag properties where actual price deviates >$40K from prediction** — these warrant manual review.
- **Apply location-specific adjustments** for Downtown listings to compensate for known under-prediction bias.

## Technical Notes

- **Data:** 500 homes, 6 features, 3% missing age values (median-imputed)
- **Validation:** 5-fold cross-validation + stratified train/test split by location
- **Models compared:** Linear Regression, Ridge, Random Forest, Gradient Boosting
- **Best model:** Gradient Boosting (lowest CV RMSE, best generalization)
- **Feature engineering:** `sqft_per_room`, `total_rooms`, `is_new`, `size_category`, `age_category`

## Appendix: Worst Predictions

| Actual | Predicted | Error | Location | Likely Cause |
|--------|-----------|-------|----------|--------------|
| $325,800 | $385,492 | -$59,692 | Suburbs | Fixer-upper or motivated seller |
| $587,600 | $644,343 | -$56,743 | Downtown | Underpriced prime location |
| $627,300 | $574,562 | +$52,738 | Downtown | Premium features not in data |

---

*Report generated: 2026-07-24*  
*Model version: 1.0.0*  
*Contact: [your-email@example.com]*
