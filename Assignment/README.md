# Titanic Survival Prediction: Data Preprocessing and Logistic Regression Analysis

## Abstract

This project examines the effect of systematic data preprocessing on binary classification performance using the Titanic passenger survival dataset. The pipeline covers missing value treatment, duplicate removal, outlier detection via the Interquartile Range (IQR) method, and categorical feature encoding. A Logistic Regression classifier is trained under two conditions — minimally preprocessed (baseline) and fully cleaned — to quantify the impact of each preprocessing step. All visualizations are produced exclusively with Matplotlib.

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Missing Value Analysis](#2-missing-value-analysis)
3. [Duplicate Records](#3-duplicate-records)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Missing Value Treatment](#5-missing-value-treatment)
6. [Outlier Treatment](#6-outlier-treatment)
7. [Feature Encoding](#7-feature-encoding)
8. [Model Training and Evaluation](#8-model-training-and-evaluation)
9. [Comparative Results](#9-comparative-results)
10. [Discussion](#10-discussion)
11. [References](#11-references)

---

## Prerequisites

```bash
pip install pandas matplotlib scikit-learn
```

**Data source:** Download `train.csv` from the [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data) and place it in this directory.

---

## 1. Dataset Overview

The Titanic training set consists of 891 passenger records across 12 features:

| Feature | Type | Description |
|---------|------|-------------|
| `PassengerId` | Integer | Unique identifier |
| `Survived` | Binary (0/1) | Target variable |
| `Pclass` | Ordinal (1/2/3) | Passenger class |
| `Name` | String | Passenger name |
| `Sex` | Categorical | Gender |
| `Age` | Continuous | Age in years |
| `SibSp` | Integer | Number of siblings/spouses aboard |
| `Parch` | Integer | Number of parents/children aboard |
| `Ticket` | String | Ticket number |
| `Fare` | Continuous | Ticket price |
| `Cabin` | Categorical | Cabin number |
| `Embarked` | Categorical (C/Q/S) | Port of embarkation |

The target variable `Survived` is binary: 0 indicates the passenger perished, 1 indicates survival.

---

## 2. Missing Value Analysis

| Column | Missing Count | Percentage | Classification |
|--------|--------------|------------|----------------|
| `Age` | ~177 | 19.9% | Moderate — imputation feasible |
| `Cabin` | ~687 | 77.1% | Severe — imputation unreliable |
| `Embarked` | 2 | 0.2% | Trivial — mode imputation |

**Key considerations:**
- `Age` is a potentially significant predictor of survival (children were prioritized during evacuation). Dropping 177 records would reduce the training set by nearly 20%, making imputation the preferred strategy.
- `Cabin` missingness exceeds 77%. Any imputation at this level would introduce substantial noise with minimal information gain. Column removal is the appropriate action.
- `Embarked` has only 2 missing entries, making mode imputation statistically sound.

---

## 3. Duplicate Records

The dataset was checked for exact duplicate rows using `df.duplicated()`. In the standard Kaggle Titanic dataset, the duplicate count is typically zero. Nonetheless, deduplication is performed as a standard preprocessing safeguard.

---

## 4. Exploratory Data Analysis

All plots use Matplotlib exclusively.

### 4.1 Target Variable Distribution

The survival classes are imbalanced: approximately 549 passengers (61.6%) did not survive versus 342 (38.4%) who did. This reflects the historical reality of the disaster where survival was the minority outcome. The imbalance is acknowledged but not explicitly addressed (e.g., via SMOTE or class weighting), as the focus of this analysis is on the impact of data cleaning rather than class-imbalance mitigation.

### 4.2 Age Distribution

The age distribution is moderately right-skewed with a primary concentration between 20 and 40 years. A secondary peak is observed in the 0–5 age range (infants and young children). The skewness justifies the use of the median over the mean for imputation, as the mean would be disproportionately influenced by the right tail.

### 4.3 Fare Distribution

The fare distribution is heavily right-skewed. The majority of fares fall below 50 currency units, corresponding to second- and third-class passengers. A small number of first-class fares exceed 200, with the maximum reaching approximately 512. This extreme skewness signals the presence of outliers that could distort model coefficients.

### 4.4 Fare Boxplot

The boxplot confirms multiple data points beyond the upper whisker:
- **Median fare:** approximately 14.45
- **Interquartile range (IQR):** approximately 7.91 to 31.00
- **Upper whisker extent:** approximately 65.63

Points above the upper whisker represent statistical outliers — predominantly first-class luxury tickets.

---

## 5. Missing Value Treatment

| Column | Strategy | Justification |
|--------|----------|---------------|
| `Age` | Median imputation | The distribution is right-skewed; the median is robust to asymmetry and outliers, unlike the mean which would overestimate the central tendency. |
| `Embarked` | Mode imputation | As a categorical variable with only 2 missing entries, the most frequent category (Southampton, 'S') provides a statistically defensible replacement. |
| `Cabin` | Column removal | At 77.1% missingness, imputation would generate unreliable synthetic values. The information loss from dropping a heavily sparse column is outweighed by the noise reduction. |

---

## 6. Outlier Treatment

The Interquartile Range (IQR) method was applied to the `Fare` column:

```
Q1 = 25th percentile of Fare
Q3 = 75th percentile of Fare
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 * IQR
Upper Bound = Q3 + 1.5 * IQR
```

**Computed values (approximate):**

| Statistic | Value |
|-----------|-------|
| Q1 | 7.91 |
| Q3 | 31.00 |
| IQR | 23.09 |
| Lower Bound | -26.72 (effectively 0) |
| Upper Bound | 65.63 |

Records with `Fare > 65.63` are classified as outliers and removed. This typically eliminates approximately 100 records corresponding to luxury first-class tickets.

**Rationale for the IQR method:**
- It is non-parametric and does not assume normality.
- The 1.5 * IQR threshold is a widely accepted convention for identifying mild outliers (Tukey, 1977).
- It is more robust than standard deviation-based methods for skewed distributions.

**Trade-off:** Outlier removal reduces the dataset size but produces a more homogeneous distribution, which can improve the stability of linear model coefficients.

---

## 7. Feature Encoding

| Feature | Method | Details |
|---------|--------|---------|
| `Sex` | Label encoding | `male` mapped to 0, `female` mapped to 1 |
| `Embarked` | One-hot encoding | `drop_first=True` to eliminate multicollinearity |

**One-hot encoding with `drop_first=True`:**
The `Embarked` column has three categories (C, Q, S). Encoding all three as binary columns introduces perfect multicollinearity (the dummy variable trap), which is problematic for Logistic Regression. By dropping the first category (C = Cherbourg), it serves as the implicit reference level — when both `Embarked_Q` and `Embarked_S` are zero, the model infers embarkation from Cherbourg.

**Dropped columns:** `Name`, `Ticket`, and `PassengerId` are identifiers with no predictive value and are excluded from the feature set.

---

## 8. Model Training and Evaluation

### 8.1 Classifier

**Logistic Regression** (scikit-learn) with `max_iter=1000` was selected for its interpretability and suitability for binary classification tasks. The solver converges within the iteration limit for this dataset size.

### 8.2 Evaluation Protocol

- 80/20 train-test split with `random_state=42` for reproducibility
- Accuracy as the primary evaluation metric

### 8.3 Baseline Pipeline (Raw Data)

The baseline model applies only the transformations strictly necessary to run the classifier:
1. Drop `Cabin`, `Name`, `Ticket` columns
2. Label-encode `Sex`
3. Mode-impute `Embarked`
4. One-hot encode `Embarked`
5. Drop all remaining rows containing NaN (primarily those with missing `Age`)

This approach discards approximately 177 rows with missing `Age` values rather than imputing them.

### 8.4 Cleaned Pipeline

The cleaned model applies the full preprocessing pipeline:
1. Median imputation for `Age` (preserves rows)
2. Mode imputation for `Embarked`
3. Drop `Cabin` column
4. IQR-based outlier removal on `Fare`
5. Label-encode `Sex`, one-hot encode `Embarked`
6. Drop `Name`, `Ticket`, `PassengerId`

---

## 9. Comparative Results

| Metric | Baseline (Raw) | Cleaned |
|--------|---------------|---------|
| Missing value strategy | Row deletion | Imputation (median/mode) |
| Outlier treatment | None | IQR filtering |
| Approximate training set size | ~571 | ~624 |
| Expected accuracy range | 78–80% | 79–82% |

The exact values are computed at runtime and displayed in the notebook. In general, the cleaned pipeline achieves equal or improved accuracy due to two factors:

1. **Data preservation:** Imputation retains rows that would otherwise be discarded, providing the model with more training examples.
2. **Noise reduction:** Outlier removal prevents extreme fare values from disproportionately influencing model coefficients.

---

## 10. Discussion

### Why the cleaned model typically performs better

- Median imputation for `Age` preserves approximately 177 additional training examples compared to row deletion. For a dataset of this size, the additional data contributes meaningfully to model generalization.
- IQR-based outlier removal on `Fare` eliminates extreme values that could bias the logistic regression coefficients, particularly since fare is correlated with `Pclass` and indirectly with survival.

### Potential limitations

- **Imputation bias:** Filling missing `Age` values with the global median assumes that missingness is random (MCAR). If missingness is related to other variables (e.g., third-class passengers were less likely to have recorded ages), this assumption may not hold.
- **Outlier removal as data loss:** Removing high-fare passengers eliminates potentially informative first-class records. An alternative approach would be log-transformation of `Fare` to compress the distribution while retaining all records.
- **Single metric evaluation:** Accuracy alone may not fully characterize model performance on imbalanced data. Precision, recall, F1-score, and AUC-ROC would provide a more complete evaluation.

### Directions for further analysis

- Investigate feature interactions (e.g., `Sex * Pclass`).
- Apply cross-validation instead of a single train-test split for more robust accuracy estimates.
- Compare against non-linear classifiers (Random Forest, Gradient Boosting) to assess whether the preprocessing effects generalize across model families.

---

## 11. References

1. Kaggle. *Titanic: Machine Learning from Disaster.* https://www.kaggle.com/c/titanic
2. Tukey, J. W. (1977). *Exploratory Data Analysis.* Addison-Wesley.
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 51–56.
5. Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*, 9(3), 90–95.
