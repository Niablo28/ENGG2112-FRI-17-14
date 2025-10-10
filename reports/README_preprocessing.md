# SleepQualityProject – Preprocessing Report

This document summarises the **data preprocessing pipeline** built for the Kaggle Sleep Health dataset.

## 1. Project Directory Structure
```
SleepQualityProject/
├─ figures/
├─ Kaggle_SleepHealth/
├─ notebooks/
├─ reports/
├─ Sleep_EDF/
```

## 2. Environment
- Python 3.13.x
- pandas, numpy, scikit-learn, joblib, pyarrow, matplotlib

Install:
```bash
pip install pandas numpy scikit-learn joblib pyarrow matplotlib
```

## 3. Data Sources
- Kaggle: Sleep health and lifestyle dataset (`/Kaggle_SleepHealth/`)
- (Optional) Sleep-EDF (.edf files) for potential feature extension

## 4. Preprocessing Pipeline Overview
Implemented using **Pandas + Scikit-learn ColumnTransformer**.
- Numeric: `SimpleImputer(strategy='median')` → `StandardScaler()`
- Categorical: `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')`

Artifacts (saved in `/reports/`):
- `preprocessor.joblib` → reusable pipeline object
- `preprocess_feature_names.csv` → encoded feature names
- `X_train_proc.parquet`, `X_test_proc.parquet` → processed features
- `y_train.csv`, `y_test.csv` → target data
- `data_dictionary.csv`, `data_dictionary.md` → data dictionary
- Normalisation results: `norm_stats_standard.csv`, `norm_stats_minmax.csv`, `norm_stats_robust.csv`

## 5. Usage Example
```python
import pandas as pd
from joblib import load

df = pd.read_csv('reports/kaggle_clean_winsorized.csv')  # or new data with same schema
preprocessor = load('reports/preprocessor.joblib')
X = preprocessor.transform(df)
print(X.shape)
```

## 6. Notes
- Target column: **quality_of_sleep**
- Train/Test shape: train (299, 50), test (75, 50)
- No missing or infinite values remain after processing.

## 7. Reproducibility
All steps documented in `notebooks/02_preprocess_template.ipynb` (Cells 1–6).
