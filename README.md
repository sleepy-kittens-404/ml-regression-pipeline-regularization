# California Housing Price Prediction

A complete machine learning pipeline built on the California Housing dataset.It covers data corruption simulation, exploratory data analysis, preprocessing, and regression model training and evaluation.

---

## Project Structure

```
├── dataset_nanvalues.ipynb      # Loads clean dataset and injects missing values
├── data_analysis.ipynb          # EDA, cleaning, preprocessing, exports cleaned CSV
├── model_training.ipynb         # Model training and evaluation
├── california_housing_dataset.csv   # Dataset with injected missing values (input)
└── cleaned_housing_dataset.csv      # Preprocessed dataset ready for training (output)
```

---

## Objective

The goal of this project is to simulate a realistic, messy dataset and build a full preprocessing and modeling pipeline from scratch. The dataset is intentionally corrupted with missing values, then cleaned and used to train and compare multiple linear regression models.

---

## Dataset

**Source:** [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) from `sklearn.datasets`

**Original shape:** 20,640 rows × 9 columns

**Features:**

| Feature | Description |
|---|---|
| MedInc | Median income in block group |
| HouseAge | Median house age in block group |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | Block group population |
| AveOccup | Average number of household members |
| Latitude | Block group latitude |
| Longitude | Block group longitude |
| target | Median house value (in $100,000s) |

---

## Pipeline Overview

### 1. Data Corruption — `dataset_nanvalues.ipynb`

The clean sklearn dataset is loaded and 10% of values are randomly set to `NaN` across all columns, simulating real-world missing data. The corrupted dataset is saved as `california_housing_dataset.csv`.

```python
def add_missing_values(df, frac=0.1):
    df_copy = df.copy()
    for col in df_copy.columns:
        df_copy.loc[df_copy.sample(frac=frac).index, col] = np.nan
    return df_copy
```

---

### 2. EDA & Preprocessing — `data_analysis.ipynb`

#### Outlier Removal
IQR-based clipping is applied to all numeric features before imputation to prevent outliers from corrupting the imputed values. Geographic columns (Latitude, Longitude) are included in clipping since the dataset also contained artificially extreme values.

```python
for col in df.columns:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df_clean[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
```

#### Missing Value Imputation
Median imputation is used via `sklearn.impute.SimpleImputer`. Median is preferred over mean for robustness against skewed distributions.

```python
imputer = SimpleImputer(strategy='median')
df = pd.DataFrame(imputer.fit_transform(df_clean), columns=df_clean.columns)
```

#### Target Distribution Cleaning
The original California Housing dataset contains artificial price caps — a large cluster of values at ~$179,500 and an upper cap at ~$500,000. Both clusters are removed as they distort the target distribution and mislead the model.

```python
df = df[np.abs(df['target'] - 1.795) > 0.05]
df = df[df['target'] < 4.8]
```

#### Skewness Correction
Features with skewness above 0.5 are log-transformed to reduce skew and improve linear model assumptions. The target is also log-transformed to achieve a more symmetric distribution.

```python
for col in ['Population', 'AveBedrms', 'AveOccup']:
    df[col] = np.log1p(df[col])

df['target'] = np.log1p(df['target'])
```

`MedInc` was tested for log transformation but showed no meaningful improvement in its linear relationship with the target, so it was kept in its original form.

#### Multicollinearity Check
A correlation heatmap was generated. Latitude and Longitude showed a correlation of -0.80 with each other but both were retained after feature importance analysis confirmed their predictive value (~13% each via Random Forest).

`AveBedrms` was dropped in the final training run due to low feature importance (~3%).

#### Cleaned data exported
```python
df.to_csv("cleaned_housing_dataset.csv", index=False)
```

---

### 3. Model Training & Evaluation — `model_training.ipynb`

#### Train/Test Split & Scaling

```python
X = df.drop(columns=['target', 'AveBedrms'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Note: The scaler is fit only on training data and applied to the test set to prevent data leakage.

#### Models Trained

Four linear models were trained and compared:

| Model | R² Score | MSE |
|---|---|---|
| Linear Regression | 0.5355 | 0.0501 |
| Ridge (CV) | 0.5355 | 0.0501 |
| Lasso (CV) | 0.5354 | 0.0501 |
| ElasticNet (CV) | 0.5354 | 0.0501 |

All regularized variants (Ridge, Lasso, ElasticNet) produced nearly identical results to plain Linear Regression, confirming that overfitting is not the bottleneck — the linear assumption itself is.

#### Feature Importance (Random Forest)

| Feature | Importance |
|---|---|
| MedInc | 43.7% |
| Longitude | 13.0% |
| Latitude | 12.7% |
| AveOccup | 11.4% |
| AveRooms | 7.5% |
| HouseAge | 4.5% |
| Population | 4.1% |
| AveBedrms | 3.2% |

---

## Results & Discussion

The best R² achieved with linear regression is **~0.535**, which is consistent with the known ceiling for linear models on this dataset (the unmodified dataset benchmarks at ~0.60 with linear regression).

The performance gap is explained by two factors:

1. **Non-linear relationships** — `MedInc` (the strongest predictor at ~44% importance) has a curved, non-linear relationship with house prices that a straight-line model cannot fully capture.
2. **Geographic clustering** — Latitude and Longitude encode regional price patterns (e.g. Bay Area vs. inland) that are spatial and non-linear in nature.

Since regularization did not improve performance, the limitation lies entirely in the linear assumption and not in overfitting. Tree-based models such as Random Forest or Gradient Boosting are expected to perform significantly better on this dataset.

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## How to Run

1. Run `dataset_nanvalues.ipynb` to generate `california_housing_dataset.csv`
2. Run `data_analysis.ipynb` to clean and export `cleaned_housing_dataset.csv`
3. Run `model_training.ipynb` to train and evaluate models

---

## Author

Muhammad Sami — GIK Institute
