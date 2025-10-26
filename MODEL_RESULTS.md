# Sleep Quality Model Results

**Project:** ENGG2112 Sleep Quality Detector  
**Date:** October 26, 2025  
**Task:** Compare Baseline (Kaggle-only) vs. Hybrid (Kaggle+EDF) vs. EDF-only models

---

## Model Performance Summary

| Model | Dataset | Subjects | Features | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | CV ROC-AUC |
|-------|---------|----------|----------|---------------|----------------|-------------|---------|--------------|------------|
| **Baseline** | Kaggle | 374 | 14 → 50* | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 ± 0.004 |
| **Hybrid** | Kaggle+EDF | 376 | 43 → 79* | 0.987 | 0.981 | 1.000 | 0.991 | 0.978 | 0.998 ± 0.003 |
| **EDF-Only** | EDF | 2 | 29 | — | — | — | — | — | Training Failed** |

\* After preprocessing (one-hot encoding)  
** Both subjects have same label (poor sleep) - cannot train classifier

**Winner: Baseline Model** 

---

## Dataset Details

### Baseline (Kaggle-Only)
- **Subjects:** 374
- **Features:** age, gender, occupation, sleep_duration, physical_activity_level, stress_level, bmi_category, blood_pressure, heart_rate, daily_steps, sleep_disorder, sleep_disorder_missing
- **Target Distribution:** 119 poor (<7), 255 good (≥7)
- **Train/Test Split:** 299 train, 75 test

### Hybrid (Kaggle + EDF)
- **Subjects:** 376 (374 Kaggle + 2 EDF: SC4001, SC4011)
- **Features:** 14 lifestyle + 29 physiological = 43 total
  - Lifestyle: same as baseline
  - Physiological: sleep architecture (W, N1, N2, N3, REM %), EEG/EOG/EMG statistics, sleep efficiency, latency, WASO
- **Feature Sparsity:** 99.5% missing (only 2/376 have physiological data)
- **Target Distribution:** 119 poor, 257 good
- **Train/Test Split:** 300 train, 76 test

### EDF-Only
- **Subjects:** 2 (SC4001, SC4011)
- **Features:** 29 physiological only
- **Sleep Quality:** SC4001=5, SC4011=5 (both poor)
- **Status:** Training failed - single class problem

---

## Feature Importance (Top 10)

### Baseline Model
| Feature | Coefficient | Effect |
|---------|-------------|--------|
| sleep_duration | +2.200 | Strong positive |
| stress_level | -1.970 | Strong negative |
| occupation_Salesperson | -1.280 | Negative |
| occupation_Accountant | +1.080 | Positive |
| occupation_Lawyer | +0.580 | Moderate positive |
| occupation_Nurse | -0.500 | Negative |
| heart_rate | (not in top) | — |

### Hybrid Model
| Feature | Coefficient | Effect |
|---------|-------------|--------|
| sleep_duration | +2.514 | Strong positive |
| stress_level | -1.782 | Strong negative |
| occupation_Salesperson | -1.359 | Negative |
| occupation_Accountant | +0.972 | Positive |
| heart_rate | -0.675 | Moderate negative |
| age | +0.633 | Moderate positive |
| gender_Female | +0.597 | Moderate positive |
| occupation_Lawyer | +0.533 | Moderate positive |
| physical_activity_level | +0.509 | Moderate positive |
| occupation_Nurse | -0.477 | Negative |

**Note:** No physiological features appear in top 15 (ignored due to 99.5% sparsity)

---

## Cross-Validation Results (5-Fold)

### Baseline
| Fold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|----------|-----------|--------|----|----|
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 0.983 | 0.977 | 1.000 | 0.988 | 0.999 |
| 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 5 | 0.983 | 1.000 | 0.976 | 0.988 | 0.997 |
| **Mean** | **0.993** | **0.995** | **0.995** | **0.995** | **0.999** |
| **Std** | 0.010 | 0.011 | 0.011 | 0.006 | 0.001 |

### Hybrid
| Metric | Mean | Std Dev |
|--------|------|---------|
| Accuracy | 0.980 | 0.036 |
| Precision | 0.995 | 0.011 |
| Recall | 0.976 | 0.055 |
| F1 | 0.985 | 0.028 |
| ROC-AUC | 0.998 | 0.003 |

---

## Key Findings

### Performance
- **Baseline achieves near-perfect metrics** (ROC-AUC = 0.999)
- **Hybrid shows no improvement** over baseline (ROC-AUC = 0.998)
- Adding 2 EDF subjects provides **no statistical benefit**

### Feature Analysis
- **Lifestyle features dominate:** sleep_duration and stress_level are strongest predictors
- **Physiological features ignored:** extreme sparsity (99.5% missing) makes them ineffective
- **Occupation matters:** Salesperson/Nurse negative, Accountant/Lawyer positive

### Data Quality Issues
- **EDF sample size too small:** 2 subjects vs. 374 insufficient
- **Lack of class diversity:** Both EDF subjects have poor sleep (label=0)
- **Synthetic labels:** EDF labels created from heuristics, not validated

---

## Recommendations

### For Deployment
✅ **Use Baseline Model** (`reports/sleep_quality_model.joblib`)
- Simpler (14 features vs. 43)
- Better performance
- No missing data issues
- Faster inference

### For Reporting
- Document hybrid integration as **proof-of-concept**
- Highlight limitations: sample size, feature sparsity, label quality
- Reference this file for all metrics

### For Future Work
- Collect ≥50 EDF subjects with diverse sleep quality
- Obtain paired data (same subjects with both lifestyle + physiological)
- Validate synthetic labels with clinician ratings
- Test regularization (L1/L2) for high-dimensional features

---

## Model Files

| Model | File | Size | Use Case |
|-------|------|------|----------|
| Baseline | `reports/sleep_quality_model.joblib` | — | **Production** ✅ |
| Hybrid | `reports/sleep_quality_model_hybrid.joblib` | — | Research only |
| Preprocessor | `reports/preprocessor.joblib` | — | Feature engineering |

---

## Detailed Results Location

- **Baseline:** `reports/logistic_cv/`
  - `logreg_metrics_test.csv` - test set performance
  - `logreg_coefficients.csv` - feature importance
  - `cv_results.csv` - cross-validation metrics
  - `roc_logreg.png`, `cm_logreg.png` - visualizations

- **Hybrid:** `reports/hybrid_model/`
  - `hybrid_metrics_test.csv` - test set performance
  - `hybrid_coefficients.csv` - feature importance
  - `hybrid_cv_results.csv` - cross-validation metrics
  - `roc_hybrid.png`, `cm_hybrid.png` - visualizations

- **EDF-Only:** `reports/edf_only_model/`
  - `edf_only_summary.csv` - diagnostic info (training failed)

---

## Conclusion

**Baseline model wins** due to superior performance, simplicity, and no missing data. Hybrid integration successfully demonstrates multi-source data engineering but provides no practical benefit with current dataset (n=2 EDF subjects, 99.5% feature sparsity, single class).

**Status:** Baseline model ready for production deployment ✅

