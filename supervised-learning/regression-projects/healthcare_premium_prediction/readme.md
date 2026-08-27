# Annual Health Insurance Premium Prediction

Predicting annual health insurance premiums from demographic, financial, and lifestyle data using regression models.

## 📌 Overview

Insurance premiums typically depend on a mix of demographic, financial, and health-related factors. This project builds a regression pipeline to predict an individual's **annual health insurance premium** using attributes like age, income, medical history, and lifestyle habits (physical activity, stress level, smoking status, etc.).

Two models were trained and compared — a baseline **Linear Regression** and a **XGBoost Regressor** — with the latter significantly outperforming the baseline.

## 📊 Dataset

- **Size:** 10,000 records × 15 columns (raw)
- **Target variable:** `annual_premium_amount`
- **Features:** age, gender, region, marital status, physical activity, stress level, number of dependants, BMI category, smoking status, employment status, income level, income (lakhs), medical history, insurance plan

## 🧹 Data Cleaning & Preprocessing

| Step | Detail |
|---|---|
| Missing values | Dropped rows with nulls (9,973 rows remaining) — nulls were a very small fraction of the dataset |
| Duplicates | None found |
| Column naming | Standardized to lowercase, underscore-separated |
| Outlier removal | Removed unrealistic ages (`age < 100`); capped `number_of_dependants` and `income_lakhs` using the IQR method |
| Final shape | 9,828 rows × 15 columns after cleaning |

## 🔍 Exploratory Data Analysis

- Boxplots & histograms for numerical features (age, dependants, income) before and after outlier treatment
- Scatter plots of each numerical feature against the target to inspect relationships
- Distribution plots for all categorical features (gender, region, smoking status, BMI category, etc.)
- Correlation heatmap and **Variance Inflation Factor (VIF)** analysis for multicollinearity checks

## 🛠️ Feature Engineering

- **`lifestyle_score`** — combined score derived from mapped `physical_activity` and `stress_level` levels
- **`total_risk_score`** — medical risk score built by parsing `medical_history` (which could contain up to two conditions, e.g. "Diabetes & Heart Disease"), mapping each diagnosis to a severity weight (e.g., heart disease = 8, diabetes = 6, thyroid = 5), and summing them

## 🔢 Encoding & Scaling

- **Label/Ordinal encoding:** `income_level` and `insurance_plan` (Bronze/Silver/Gold), since these have a natural order
- **One-hot encoding:** `gender`, `region`, `marital_status`, `bmi_category`, `smoking_status`, `employment_status`
- **MinMax scaling** applied to numeric and ordinal columns before modeling

## 🎯 Feature Selection

VIF analysis revealed strong multicollinearity between `income_level` and `income_lakhs` (VIF ≈ 12.8). Dropping `income_lakhs` brought `income_level`'s VIF down to ≈ 2.8, so it was removed from the final feature set.

## 🤖 Modeling & Results

Train/test split: 75% / 25%

| Model | Train R² | Test R² | MSE |
|---|---|---|---|
| Linear Regression | 0.958 | 0.958 | 3,230,590 |
| **XGBoost Regressor** (`n_estimators=50, max_depth=5`) | **0.996** | **0.994** | **428,130** |

XGBoost outperforms the linear baseline substantially, capturing non-linear relationships in the data (e.g., interactions between age, smoking, and medical risk) that a linear model can't.

**Top drivers of premium** (by feature importance): smoking status, insurance plan tier, age, and medical risk score consistently rank as the strongest predictors across both models.

## 🔬 Error Analysis

- Residuals (actual − predicted) were converted to percentage error for interpretability
- **~5.7%** of test predictions had an absolute error greater than 10%
- These higher-error cases skewed toward **younger customers** (mean age ≈ 22), suggesting the model is slightly less reliable for younger applicants — likely due to smaller premiums making percentage errors more sensitive, and/or less representation of this segment in training data

## 🧰 Tech Stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `statsmodels` · `seaborn` · `matplotlib`

## 🚀 Potential Improvements

- Hyperparameter tuning (GridSearch/Optuna) for XGBoost
- Cross-validation instead of a single train/test split for more robust performance estimates
- Try other models (LightGBM, CatBoost, Random Forest) for comparison
- Investigate and address the elevated error rate among younger applicants
- SHAP values for more rigorous feature importance / explainability

## 📁 Repository Structure

```
├── premiums_with_life_style.xlsx      # Raw dataset
├── ml_premium_prediction_lifestyle.ipynb  # Full analysis & modeling notebook
└── README.md
```

---
*This project demonstrates an end-to-end regression workflow: data cleaning, outlier treatment, EDA, feature engineering, encoding, multicollinearity checks, model comparison, and error analysis.*