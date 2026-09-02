# Food Delivery Order ETA Prediction

A regression project predicting **actual food delivery time (in minutes)** using order, restaurant, driver, and environmental features. Built end-to-end: EDA → leakage-aware feature selection → multicollinearity diagnostics → model comparison → hyperparameter tuning → SHAP interpretability → error analysis.

## Problem Statement

Given the details of a food delivery order at the time it's placed (restaurant, distance, weather, traffic, driver, customer info), predict `actual_delivery_time_minutes` — the true end-to-end delivery time. This mirrors the ETA prediction problem used by platforms like Swiggy, Zomato, DoorDash, and Uber Eats.

## Dataset

- **50,000 orders**, 37 raw columns, target: `actual_delivery_time_minutes`
- Features span order metadata, restaurant attributes, driver attributes, pricing, and environmental conditions (weather, traffic)

## Project Workflow

### 1. Data Cleaning & Leakage Prevention
Train/test split was performed **before** any cleaning to prevent test-set information from influencing preprocessing decisions.

Several columns were identified and removed because they would not be known at prediction time (order placement) and would leak target information:
- `tip_amount`, `cancellation_reason`, `customer_rating` — only available post-delivery
- `late_delivery`, `order_status` — derived from/dependent on the outcome
- `estimated_delivery_time_minutes` — the platform's own heuristic estimate, excluded as a feature (kept conceptually as a benchmark, not used as model input)

Rows with a missing target (`actual_delivery_time_minutes`) were **dropped**, not imputed — these corresponded to incomplete/cancelled orders with no true delivery outcome to learn from.

### 2. Exploratory Data Analysis
- Univariate and bivariate analysis across all numerical features
- Target distribution found to be right-skewed (mean ≈ 35 min, long tail toward 170+ min), consistent with real-world delivery delay patterns
- High-traffic conditions confirmed to dominate the upper tail of delivery times

### 3. Feature Selection (data-driven, not assumption-driven)
Every drop decision was validated statistically rather than by intuition:

| Feature | Decision | Evidence |
|---|---|---|
| `delivery_area` (~40 categories) | **Dropped** | Area-level average delivery time spread of only ~4 min (≈1× standard error) — statistically indistinguishable from noise; also uncorrelated with area-level distance (r=0.32, p=0.10) |
| `is_weekend` | **Dropped in favor of `day_of_week`** | `day_of_week` showed a statistically significant spread (≈17× standard error) that `is_weekend` couldn't capture — e.g. Monday was the slowest day, and Saturday behaved unlike Sunday despite both being "weekend" |
| Raw ID columns (`order_id`, `customer_id`, `restaurant_id`, `driver_id`), `order_timestamp`, `order_date` | **Dropped** | High-cardinality identifiers with no direct predictive value in raw form |

### 4. Multicollinearity Diagnostics (VIF)
Variance Inflation Factor analysis revealed severe multicollinearity among the financial columns (`subtotal`, `tax_amount`, `service_fee`, `delivery_fee`, `order_total`) — `order_total` is a near-exact linear combination of the others, producing VIF values in the millions. These redundant components were dropped, keeping `order_total` as the single representative feature. A follow-up correlation check on the remaining moderate-VIF features (`order_total`, `items_count`, `restaurant_preparation_time_minutes`) confirmed a real, explainable "order size/complexity" relationship rather than a data artifact.

### 5. Preprocessing
- **One-hot encoding** (`drop_first=True`) for categorical features
- **Min-Max scaling** fit strictly on training data and applied (not refit) to the test set, avoiding train/test scale inconsistency

### 6. Model Comparison

| Model | Test R² | Test MAE (min) | Test MSE |
|---|---|---|---|
| Linear Regression | 0.9346 | 2.76 | 13.61 |
| Random Forest Regressor | 0.9602 | 2.25 | 8.29 |
| XGBoost Regressor (default) | 0.9641 | 2.18 | 7.47 |
| **XGBoost (tuned via RandomizedSearchCV)** | **0.9662** | **2.12** | **7.03** |

The tuned XGBoost model was selected as the final model. Given how close Linear Regression performs to the tree-based models, the relationships in this dataset are largely additive — a useful finding in its own right, since it suggests a simpler model captures most of the signal.

### 7. Hyperparameter Tuning
`RandomizedSearchCV` (20 candidate combinations, 3-fold CV) was used to tune `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `gamma`, `subsample`, `colsample_bytree`, `reg_alpha`, and `reg_lambda`. Best cross-validated R²: **0.9671**.

### 8. Interpretability
- **Linear Regression coefficients** were plotted and sanity-checked against domain expectations: `distance_km` and `restaurant_preparation_time_minutes` emerged as the largest legitimate positive drivers, and weather severity showed a clean, monotonic ordering (`Storm` > `Rain` > `Cloudy`)
- **SHAP values** (computed via XGBoost's native `pred_contribs`) were used to explain both global feature importance and individual predictions, cross-validating the findings from the linear coefficients and tree feature importances

### 9. Error Analysis
Beyond aggregate metrics, predictions were analyzed for structural patterns:
- Residual plots (predicted vs. actual, residuals vs. predicted, residual distribution) to check for heteroscedasticity or systematic bias
- Error broken down by delivery-time quintile to check whether the model underperforms specifically on long-tail deliveries
- Error broken down by categorical segments (weather, traffic, city, restaurant type) and compared against baseline distribution to surface any blind spots
- Manual inspection of the highest-error predictions

## Key Findings

- **Distance and restaurant preparation time** are the strongest, most consistent predictors of delivery time across all three modeling approaches (linear coefficients, tree feature importances, and SHAP).
- **Weather and traffic conditions** have a clear, monotonic effect on delivery time (worse conditions → longer, more variable delivery times).
- **`day_of_week` carries real signal beyond a simple weekend/weekday split** — Monday orders are notably slower than any other day.
- Tree-based ensembles outperform linear regression, but only modestly (ΔR² ≈ 0.03), indicating the underlying relationships are largely linear/additive.

## Tech Stack

- **Data manipulation:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn (LinearRegression, RandomForestRegressor), XGBoost
- **Statistics:** statsmodels (VIF)
- **Interpretability:** SHAP (via XGBoost native `pred_contribs`)

## Project Structure

```
├── data/
│   └── food_delivery_orders_dataset.csv
├── food_order_delivery_prediction.ipynb   # Main analysis notebook
└── README.md
```

## Possible Future Work

- Log-transform the target to address right-skew and evaluate impact on long-tail prediction accuracy
- K-fold cross-validation across all three model types for more robust performance estimates
- Feature interactions (e.g., `distance_km × traffic_level`) as engineered inputs
- Quantile regression to predict a P90 ETA in addition to a point estimate, which better reflects how ETAs are used in production delivery apps

## Author

Feel free to reach out with questions, suggestions, or feedback on this project.