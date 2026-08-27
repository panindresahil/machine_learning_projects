# Credit Risk Default Prediction — Project Summary

A binary classification model that predicts whether a loan applicant will **default**, built on 50,000 customer records combining demographic, credit bureau, and loan data.

---

## 1. Problem & Data

- **Goal:** Predict `default` (0/1) at the time of loan application, using customer, bureau, and loan datasets.
- **Data sources:** `customers.csv`, `bureau_data.csv`, `loans.csv`, merged on `cust_id` into a single table of **50,000 rows × 33 columns**.
- **Split strategy:** An early, **stratified 75/25 train-test split** (37,500 / 12,500 rows) was performed *before* any cleaning or feature engineering, specifically to prevent data leakage from the test set into the training pipeline.

## 2. Data Cleaning & Sanity Checks

- Filled missing `residence_type` values (~62 records) with the training-set mode.
- Corrected a categorical typo (`"Personaal"` → `"Personal"` in `loan_purpose`).
- Verified referential/business-rule consistency between related fields, e.g.:
  - `processing_fee` should be ≤ 3% of `loan_amount`
  - `gst` should be ≤ 18% of `loan_amount`
- Checked and confirmed no duplicate records in either split.

## 3. Exploratory Data Analysis

- **Univariate:** Boxplots and histograms across 18 numerical features to inspect distributions and outliers.
- **Bivariate:** KDE plots of every numerical feature split by `default` status to visually assess separability.
- **Categorical:** Value-count review across `gender`, `marital_status`, `employment_status`, `residence_type`, `city`, `state`, `loan_purpose`, `loan_type`.

## 4. Feature Engineering

Four domain-driven ratio features were engineered to capture credit risk signal beyond raw fields:

| Feature | Definition |
|---|---|
| **Loan-to-Income (LTI)** | `loan_amount / income` |
| **Delinquency Ratio** | `delinquent_months / total_loan_months` (%) |
| **Avg. DPD per Delinquency** | `total_dpd / delinquent_months` |
| **Credit Utilization per Income** | `credit_utilization_ratio / loan_to_income` |

## 5. Feature Selection

A two-stage selection process was used to arrive at a lean, non-redundant feature set:

1. **Multicollinearity (VIF):** Dropped highly collinear numeric fields (`processing_fee`, `gst`, `net_disbursement`, `principal_outstanding`, `sanction_amount`), while deliberately retaining `credit_utilization_ratio` for its business relevance despite mild collinearity.
2. **Information Value (WOE/IV):** Computed Weight-of-Evidence and Information Value for all remaining features (binning continuous variables into deciles) and kept only features with **IV > 0.02**, discarding weak predictors.

Categorical features were then **one-hot encoded** (`drop_first=True`) and numeric features **min-max scaled**.

## 6. Modeling Approach — Three Iterations

**Attempt 1 — Baseline models** (Logistic Regression, Random Forest, XGBoost) on the encoded feature set:

| Model | Precision (default) | Recall (default) | F1 (default) |
|---|---|---|---|
| Logistic Regression | 0.85 | 0.72 | 0.78 |
| Random Forest | 0.85 | 0.72 | 0.78 |
| XGBoost | 0.83 | 0.75 | 0.79 |

All three performed comparably, with XGBoost slightly ahead on recall.

**Attempt 2 — Hyperparameter tuning** via `RandomizedSearchCV` (optimizing for recall):

| Model | Best CV Recall | Test Precision | Test Recall |
|---|---|---|---|
| Logistic Regression | — | 0.83 | 0.74 |
| XGBoost | 0.756 | 0.82 | 0.76 |

Since the tuned XGBoost offered only a marginal recall gain, **Logistic Regression was chosen as the working model for its superior interpretability** — an important factor in credit risk, where regulators and stakeholders expect explainable decisions.

**Attempt 3 — Class imbalance handling + Optuna tuning:** The target is imbalanced (~9.5% defaults), so `RandomUnderSampler` and `SMOTETomek` were tested on the training data, followed by Optuna-based hyperparameter search (50 trials, optimizing F1) on the SMOTETomek-resampled data.

| Approach | Precision (default) | Recall (default) | F1 (default) |
|---|---|---|---|
| Logistic Regression (no resampling, tuned) | 0.83 | 0.74 | 0.78 |
| + RandomUnderSampler | — | — | — |
| **+ SMOTETomek + Optuna (Final Model)** | **0.56** | **0.95** | **0.70** |

The final model trades some precision for a large recall gain — catching 95% of true defaulters, which is typically the priority in credit risk (the cost of missing a defaulter far outweighs the cost of a false alarm).

## 7. Final Model Evaluation

The final model is a **SMOTETomek-resampled, Optuna-tuned Logistic Regression**, evaluated with credit-risk-specific diagnostics:

- **ROC-AUC: 0.984**
- **Gini Coefficient: 0.968**
- **KS Statistic: ~86%** (max separation observed at the top deciles)
- **Rank ordering:** Decile-wise analysis confirms strong discrimination — the top 2 deciles alone capture **~98.6% of all actual defaults**, and the bottom 6 deciles contain effectively zero defaults, indicating the model ranks risk very cleanly.
- **Feature importance:** Coefficient-based importance plot generated for the final model to support explainability.

## 8. Key Takeaways

- Careful, leakage-free pipeline design (early train/test split, fit-on-train-only scaling/encoding) is central to a trustworthy credit risk model.
- Domain-driven feature engineering (LTI, delinquency ratio, DPD, utilization-per-income) added meaningful predictive signal beyond raw bureau/loan fields.
- VIF + IV-based feature selection reduced redundancy while preserving business-relevant variables.
- Model choice balanced **performance vs. interpretability** — Logistic Regression was preferred over tree ensembles despite comparable accuracy, given the regulatory need for explainable credit decisions.
- Addressing class imbalance (SMOTETomek) was the single biggest lever for recall, at an acceptable precision trade-off, and the final model shows excellent rank-ordering power (AUC 0.98, KS ~86%) — strong indicators for a production-grade credit scoring model.

## 9. Tech Stack

`pandas`, `numpy`, `scikit-learn`, `XGBoost`, `imbalanced-learn` (SMOTETomek, RandomUnderSampler), `statsmodels` (VIF), `Optuna`, `seaborn`/`matplotlib`.

---

*Note: This summary is based on the accompanying Jupyter notebook (`credit_risk_classification_exercise.ipynb`), which contains the full EDA, code, and visualizations.*