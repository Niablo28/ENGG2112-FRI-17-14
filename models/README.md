# Models Directory

This directory contains trained machine learning models and preprocessing pipelines.

## Files

### ✅ `sleep_quality_model.joblib`
**Production Model** - Baseline logistic regression (Kaggle dataset only)
- **Type:** Sklearn Pipeline with preprocessor + classifier
- **Training Data:** 374 subjects from Kaggle
- **Performance:** ROC-AUC = 0.999, Test Accuracy = 100%
- **Features:** 14 lifestyle and health features
- **Status:** ✅ **In Production** - Used by UI (`ui/app.py`)

### `sleep_quality_model_hybrid.joblib`
**Hybrid Model** - Logistic regression (Kaggle + EDF datasets)
- **Type:** Sklearn Pipeline with preprocessor + classifier
- **Training Data:** 376 subjects (374 Kaggle + 2 EDF)
- **Performance:** ROC-AUC = 0.998, Test Accuracy = 98.7%
- **Features:** 43 features (14 lifestyle + 29 physiological)
- **Status:** Not deployed (EDF data too sparse)

### `preprocessor.joblib`
**Preprocessing Pipeline** - Data transformation pipeline
- **Type:** Sklearn ColumnTransformer
- **Components:**
  - Numeric: Median imputation + Standard scaling
  - Categorical: Most frequent imputation + One-hot encoding
- **Status:** Used by training scripts



