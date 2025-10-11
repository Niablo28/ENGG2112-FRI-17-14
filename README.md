# ENGG2112 – Sleep Quality Detector

This project predicts a person's sleep quality (good or bad) based on their daily lifestyle data such as step count, heart rate, caffeine intake, and screen time.

---

## 📊 Data Handoff Summary (Week 09)

This section explains which files are required for modeling and which are for documentation.

### Core Modeling Files (for ML Engineer)
- `reports/preprocessor.joblib` — Preprocessing pipeline (imputer, encoder, scaler).
- `reports/X_train_proc.parquet`, `reports/y_train.csv` — Processed training data.
- `reports/X_test_proc.parquet`, `reports/y_test.csv` — Processed test data.
- `reports/preprocess_feature_names.csv` — Feature name mapping.
- `reports/data_dictionary.md` — Description of variables and meanings.
- `reports/README_preprocessing.md` — Step-by-step preprocessing documentation.

### Supporting Files
- `reports/kaggle_cleaned_snapshot.csv`
- `reports/kaggle_clean_drop_outliers.csv`
- `reports/kaggle_clean_winsorized.csv`

### Report-Only Files
- `reports/kaggle_missingness.csv`
- `reports/kaggle_numeric_summary.csv`
- `reports/norm_stats_standard.csv`
- `reports/norm_stats_minmax.csv`
- `reports/norm_stats_robust.csv`
- `reports/outcome_week9.txt`

---

### 📄 Full Data Description
For a detailed explanation of all processed data files, please refer to  
[`reports/Brief_description_of_the_data.pdf`](reports/Brief_description_of_the_data.pdf).
