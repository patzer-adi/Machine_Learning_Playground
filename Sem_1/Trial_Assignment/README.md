# Titanic Survival Prediction: EDA and Classification Pipeline

## Overview

This project applies systematic data preprocessing and exploratory data analysis to the Kaggle Titanic dataset, followed by binary classification using Logistic Regression. The primary objective is to quantify the effect of preprocessing interventions — missing value imputation, outlier removal, and feature encoding — on classification accuracy.

All visualizations are produced using `matplotlib` exclusively.

---

## Repository Contents

| File | Description |
|------|-------------|
| `titanic_eda_ml.ipynb` | Complete analysis notebook |
| `train.csv` | Training data (Kaggle Titanic competition) |
| `plot_01_survival_count.png` | Target variable class distribution |
| `plot_02_age_distribution.png` | Age frequency histogram and box plot |
| `plot_03_fare_distribution.png` | Fare distribution — full range and restricted view |
| `plot_04_fare_boxplot.png` | Fare outlier identification via IQR |
| `plot_05_survival_by_sex.png` | Conditional survival rate by sex |
| `plot_06_survival_by_class.png` | Conditional survival rate by passenger class |
| `plot_07_accuracy_comparison.png` | Logistic Regression test accuracy — baseline vs. preprocessed |

---

## Dataset

- **Source:** [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- **Observations:** 891 passengers
- **Target variable:** `Survived` (0 = deceased, 1 = survived)
- **Features used:** `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`

---

## Analysis Pipeline

### 1. Missing Value Analysis

| Feature | Missing (n) | Missing (%) | Strategy |
|---------|------------|-------------|----------|
| `Age` | ~177 | 19.9 | Median imputation |
| `Cabin` | ~687 | 77.1 | Column removal |
| `Embarked` | 2 | 0.2 | Mode imputation |

**Rationale:**

- **Age — median imputation:** The `Age` distribution exhibits positive skew (skewness ~0.39). Under skewed distributions, the median is a more stable estimator of central tendency than the mean, as it is not influenced by tail values. Mean imputation would systematically overestimate the typical age for imputed records.

- **Cabin — removal:** A missingness rate of 77.1% means any imputation strategy would fabricate the majority of values in this column. Retaining `Cabin` under these conditions introduces substantially more noise than signal.

- **Embarked — mode imputation:** With only two missing records in a categorical feature, mode imputation introduces negligible bias. Southampton ('S') is the dominant embarkation port.

---

### 2. Outlier Treatment — Fare

`Fare` exhibits a skewness of approximately 4.8, with a maximum of £512.33 against a median of £14.45. The divergence between mean (£32.20) and median confirms concentration of the distribution at the lower end with a long upper tail.

Outlier identification uses the Tukey fence criterion:

```
Lower fence = Q1 - 1.5 x IQR
Upper fence = Q3 + 1.5 x IQR
```

The IQR method is preferred over Z-score thresholding because the latter presupposes a normal distribution. Observations outside the fence interval are removed; Logistic Regression is sensitive to extreme predictor magnitudes.

| Statistic | Value |
|-----------|-------|
| Q1 | ~£7.91 |
| Q3 | ~£31.00 |
| IQR | ~£23.09 |
| Upper fence | ~£65.64 |
| Removed observations | ~60-90 (~8%) |

---

### 3. Exploratory Observations

**Target variable:** Class distribution is 61.6% non-survival vs. 38.4% survival, establishing a majority-class baseline accuracy of 61.6%.

**Age:** Modal range is 20–35 years. Skewness (~0.39) is mild; no extreme outliers warrant removal. The upper tail (65–80 years) represents genuine passenger records.

**Fare:** The heavily right-skewed distribution reflects the three-class fare structure. The small number of very high fares (first-class suites) disproportionately distort the Fare coefficient in Logistic Regression.

**Survival rate by sex:** Female survival rate ~74%; male ~19% — a 3.8x differential attributable to documented evacuation protocol. `Sex` carries the highest univariate predictive signal.

**Survival rate by passenger class:** Rates decline monotonically: Class 1 (~63%), Class 2 (~47%), Class 3 (~24%). `Pclass` functions as a proxy for socioeconomic status and physical proximity to lifeboat stations.

---

### 4. Feature Encoding

| Feature | Method | Notes |
|---------|--------|-------|
| `Sex` | Binary label encoding | male = 0, female = 1 |
| `Embarked` | One-hot encoding (drop_first=True) | Prevents dummy variable trap |
| `Name` | Removed | High cardinality; no extractable signal without additional engineering |
| `Ticket` | Removed | Alphanumeric; no consistent structure |

---

### 5. Model Comparison

**Algorithm:** Logistic Regression (max_iter=1000, random_state=42)  
**Evaluation:** Held-out test set, 80/20 split

| Condition | Preprocessing | Test Accuracy |
|-----------|--------------|---------------|
| Baseline | dropna only, no outlier removal | ~79–81% |
| Preprocessed | median imputation + IQR filtering | ~80–82% |

The improvement reflects two compounding factors: (1) median imputation retains ~177 observations that dropna discards, increasing the effective training set; and (2) removal of extreme Fare values reduces distortion in the decision boundary. The accuracy gain is 1–3 percentage points; the preprocessed model also exhibits lower variance across random seeds.

---

## Reproduction

```bash
# Obtain data: https://www.kaggle.com/c/titanic/data -> train.csv
pip install pandas matplotlib scikit-learn jupyter
jupyter notebook titanic_eda_ml.ipynb
```

**Dependencies:** pandas >= 1.3, matplotlib >= 3.4, scikit-learn >= 1.0, jupyter

---

## Notes

- Results are reproducible with random_state=42 in train_test_split.
- For production evaluation, k-fold cross-validation is recommended over a single held-out split to obtain variance-adjusted accuracy estimates.
