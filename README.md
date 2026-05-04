# Customer Revenue Forecasting

A comparative machine learning study to determine the best-performing model for predicting customer salary outcomes, evaluated across classification, regression, and clustering techniques.

---

## Overview

This project benchmarks multiple machine learning algorithms across two supervised tasks: predicting whether a customer earns above £35,000 (binary classification), and predicting their exact annual salary (regression). A third unsupervised track groups customers into behavioural segments.

Three parallel analytical tracks are explored:

| Track | Type | Goal |
|---|---|---|
| Binary Classification | Supervised | Predict if salary > £35K |
| Salary Regression | Supervised | Predict exact salary value |
| Customer Segmentation | Unsupervised | Group customers by behaviour |

---

## Dataset

**Records:** 1,000 customers

| Feature | Type | Description |
|---|---|---|
| `Age` | Numerical | Customer age (years) |
| `SiteSpending` | Numerical | Total spend on site (£) |
| `SiteTime` | Numerical | Time on site (minutes) |
| `RecommendImpression` | Numerical | Recommendation impressions count |
| `Education` | Categorical | GCSE / Degree / Masters / PhD / Other |
| `WorkType` | Categorical | Private sector / Public sector |
| `Sex` | Categorical | Customer gender |
| `Region` | Categorical | UK region |
| `Salary` | Numerical | Annual salary (£), target variable |

**Salary Distribution:**

```
Min:    £12,441     25th Pct: £27,678
Mean:   £46,823     75th Pct: £60,967
Max:   £145,225     Std Dev:  £22,595
```

---

## Approach

### Preprocessing

Applied consistently across all experiments:

- **Label Encoding:** Categorical features (`Education`, `WorkType`, `Sex`, `Region`) converted to numeric integers via `LabelEncoder`
- **Min-Max Scaling:** All features normalised to [0, 1] to prevent magnitude dominance
- **Binary Target:** Derived column `£35K+ Salary` created for classification:
  ```python
  data['£35K+ Salary'] = np.where(data['Salary'] > 35000, True, False)
  ```
- **Train/Test Split:** 80/20 for classification, 70/30 for regression

---

### Binary Classification

Classifiers trained to predict `£35K+ Salary` (True/False):

- **Decision Tree:** Tuned via `GridSearchCV` with PCA pipeline; best: `criterion=gini`, `max_depth=12`
- **Logistic Regression:** Tuned via `RepeatedStratifiedKFold`; best: `C=1.0`, `solver=lbfgs`
- **Random Forest:** Tuned via 5-fold CV; best: `max_features=3`, `n_estimators=40`
- **Neural Network (Keras):** `Dense(128, relu) → Dense(1, sigmoid)`, 250 epochs, Adam optimizer; best kernel initialiser: `he_normal`

---

### Salary Regression

Regressors trained to predict exact salary value. Evaluated on R², Adjusted R², MAE, MSE, and RMSE:

- **Support Vector Regressor:** RBF kernel
- **Decision Tree Regressor:** Default parameters
- **Random Forest Regressor:** Default parameters

---

### Customer Segmentation

Features used: `Age`, `SiteSpending`, `SiteTime`, `RecommendImpression`, `Education`, `WorkType`

- **DBSCAN:** `eps=1`, `min_samples=8`, producing 3 clusters and a noise class
- **KMeans:** Optimal `k=3` selected via Elbow Method (inertia plot)

---

## Models & Results

### Binary Classification: £35K+ Salary

| Model | Accuracy | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression | 62.0% | 0.68 | 0.78 | 0.72 |
| Neural Network | ~91.0% | N/A | N/A | N/A |
| Decision Tree | 92.0% | 0.93 | 0.95 | 0.94 |
| **Random Forest** | **93.0%** | **0.97** | **0.92** | **0.94** |

> Confusion matrix (Random Forest): **72 TN · 114 TP · 4 FP · 10 FN** out of 200 test samples.

---

### Salary Regression

| Model | R² | Adjusted R² | MAE | RMSE |
|---|:---:|:---:|:---:|:---:|
| SVR (RBF) | 0.644 | 0.634 | 0.082 | 0.099 |
| Decision Tree Regressor | 0.728 | 0.720 | 0.060 | 0.087 |
| **Random Forest Regressor** | **0.898** | **0.895** | **0.038** | **0.053** |

---

### Customer Segmentation

| Method | Clusters Found | Notes |
|---|:---:|---|
| DBSCAN | 3 + noise | ~33% of points flagged as noise, suggesting diverse profiles |
| KMeans | 3 | Clean groupings; k=3 confirmed by Elbow Method |

---

## Conclusion

**Random Forest is the top-performing model across both supervised tasks.**

The 62% accuracy of Logistic Regression, compared to 93% for Random Forest, indicates the salary decision boundary is non-linear, a characteristic that tree-based ensemble methods handle significantly better through variance reduction via bagging.

On the regression task, Random Forest explains **89.8% of salary variance** (R² = 0.898), nearly 25 points higher than SVR, with the lowest error across all metrics.

The high DBSCAN noise proportion (~33%) reveals that customer profiles are diverse and not naturally density-concentrated. KMeans with k=3 offers a more interpretable segmentation for downstream use.

**Future directions:**
- Feature importance analysis to identify the strongest salary predictors
- Evaluation of gradient boosting methods (XGBoost, LightGBM) as potential improvements
- SHAP value analysis for model explainability
