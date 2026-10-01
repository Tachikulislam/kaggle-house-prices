# House Prices - Advanced Regression Techniques

Kaggle competition solution for predicting house sale prices using regression and gradient boosting.

**Competition:** [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
**Leaderboard Score:** `0.12089` (RMSE, log scale)

## Overview

Predicts `SalePrice` for houses in Ames, Iowa using the Ames Housing dataset (1460 train / 1459 test rows, 79 features).

## Approach

1. **Target transform** — log1p(SalePrice) to correct right-skew
2. **Missing values** — domain-aware handling (NaN = "None"/0 for features like PoolQC, GarageType; neighborhood-median for LotFrontage)
3. **Feature engineering** — TotalSF, TotalBath, HouseAge, RemodAge, and binary flags (HasPool, HasGarage, etc.)
4. **Skew correction** — Box-Cox transform applied only to continuous features used by Ridge (tree models use raw features, which performed better untransformed)
5. **Encoding** — ordinal mapping for quality features, one-hot encoding for nominal categoricals
6. **Outlier removal** — dropped high GrLivArea / low SalePrice outliers
7. **Models** — Ridge, Random Forest, XGBoost, LightGBM, evaluated via 5-fold CV
8. **Final prediction** — weighted blend: `0.45 * Ridge + 0.30 * XGBoost + 0.25 * LightGBM`

## Results (5-fold CV, log-scale RMSE)

| Model | RMSE |
|---|---|
| Ridge | 0.1135 |
| XGBoost | 0.1182 |
| LightGBM | 0.1216 |
| Random Forest | 0.1366 |
| **Blend (Kaggle leaderboard)** | **0.12089** |

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, Matplotlib/Seaborn

## Usage

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn scipy
jupyter notebook house_prices-advanced_regression_techniques.ipynb
```

Place `train.csv`, `test.csv`, and `sample_submission.csv` (from Kaggle) in the same directory, then run all cells. Output: `submission.csv`.
