# ENGG2112 – Sleep Quality Detector

This project predicts a person's sleep quality (good or bad) based on their daily lifestyle data such as step count, heart rate, caffeine intake, and screen time.

---

## 📊 Kaggle Sleep Health Dataset – Preprocessing & Model-Ready Outputs (Week 09)

This section explains which files are required for modeling and which are for documentation.

### Core Modeling Files (for ML Engineer)
- `reports/preprocessor.joblib` — Preprocessing pipeline (imputer, encoder, scaler).
- `reports/X_train_proc.parquet`, `reports/y_train.csv` — Processed training data.
- `reports/X_test_proc.parquet`, `reports/y_test.csv` — Processed test data.
- `reports/preprocess_feature_names.csv` — Feature name mapping.
- `reports/data_dictionary.md` — Description of variables and meanings.

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

---

### 📄 Full Data Description
For a detailed explanation of all processed data files, please refer to  
[`reports/Brief_description_of_the_data.pdf`](reports/Brief_description_of_the_data.pdf).


## ⭐️ Sleep-EDF Dataset (Week 11 Data Engineering Deliverables)

This section adds a second dataset used in the project — **Sleep-EDF** — which contains physiological sleep recordings (EEG, EOG, EMG) and corresponding hypnograms. These data provide detailed insights into real sleep stages and complement the Kaggle lifestyle dataset for hybrid model development.

### Extracted Features and Outputs
The raw `.edf` files were processed and analyzed in the notebook [`03_sleepedf_feature_extraction.ipynb`](notebooks/03_sleepedf_feature_extraction.ipynb) using **MNE**, **YASA**, and **SciPy**. The workflow involved signal loading, hypnogram alignment, 30-second epoch segmentation, and computation of EEG bandpower (δ, θ, α, β), EOG variance, and EMG RMS amplitude.

After cleaning and normalization, the following outputs were generated:

#### 📊 Processed Reports (`/reports/`)
- `sleepedf_features.csv` — Extracted epoch-level features with subject and stage labels (≈ 5450 rows).  
- `sleepedf_numeric_summary.csv` — Statistical summary of each feature (mean, std, skewness, kurtosis).  
- `sleepedf_missingness.csv` — Missing-value overview (all <1%).  
- `sleepedf_iqr_outliers.csv` — Outlier proportions per feature using IQR detection.  
- `norm_stats_standard_sleepedf.csv` — Z-score normalization parameters for reproducibility.  

#### 🖼 Figures (`/figures/`)
- `sleepedf_heatmap.png` — Correlation heatmap showing relationships among EEG/EOG/EMG features.  
- `sleepedf_histograms.png` — Feature distributions across sleep stages.  
- `sleepedf_boxplots.png` — Outlier and variability visualization for major numerical features.  

#### 🧾 Notebook (`/notebooks/`)
- `03_sleepedf_feature_extraction.ipynb` — Full end-to-end pipeline for EDF loading, feature computation, and export.  

### Integration Note for ML Engineers
The Sleep-EDF dataset expands the project beyond lifestyle indicators by adding **true physiological measures** of sleep.  
ML engineers can now train models that combine both behavioral (Kaggle) and physiological (EDF) features to improve generalization and stage-specific prediction.  
The feature table (`sleepedf_features.csv`) is ready for merging or standalone modeling.

**Commit reference:**  
`Add Sleep-EDF extracted features, numeric summaries, and figures (Week 11 Data Engineering deliverables)`

---

## 🔬 Hybrid Model: Kaggle + EDF Integration (Final Deliverable)

### Overview
We integrated the Sleep-EDF physiological data with the Kaggle lifestyle dataset to create a **hybrid model** that combines both feature types. The goal was to assess whether adding physiological measurements (EEG, EOG, EMG, sleep architecture) improves prediction accuracy over lifestyle features alone.

### Approach: Two Strategies Tested

#### **Strategy A: Baseline Model (Kaggle-Only)**
- **Dataset:** 374 subjects with lifestyle features
- **Features:** Age, gender, occupation, sleep duration, activity level, stress, BMI, blood pressure, heart rate, daily steps, sleep disorder
- **Performance:** ROC-AUC = 0.999, F1 = 1.000 (near-perfect)

#### **Strategy B: Hybrid Model (Kaggle + EDF)**
- **Dataset:** 376 subjects (374 Kaggle + 2 EDF)
- **Features:** 14 lifestyle + 29 physiological (43 total)
- **Data Engineering Steps:**
  1. Aggregated 5,450 EDF epochs → 2 subject-level records with sleep architecture metrics
  2. Created synthetic sleep quality labels based on sleep efficiency heuristics
  3. Generated lifestyle proxy features for EDF subjects (using dataset medians)
  4. Merged datasets with NaN imputation for missing physiological features
- **Performance:** ROC-AUC = 0.998, F1 = 0.991

#### **Strategy C: EDF-Only Model (Physiological Features)**
- **Dataset:** 2 EDF subjects with only physiological features
- **Result:** ❌ **Training Failed** — Both subjects had same label (poor sleep), preventing classifier training
- **Conclusion:** Need larger EDF dataset with diverse sleep quality levels

### Key Findings

1. **No Performance Improvement:** Hybrid model performs similarly to baseline (AUC: 0.998 vs. 0.999)
2. **Feature Sparsity Issue:** 99.5% of physiological features are missing (only 2/376 subjects have values)
3. **Lifestyle Features Dominate:** Top predictors remain sleep duration (+2.5), stress level (-1.8), occupation
4. **Sample Size Limitation:** Adding only 2 EDF subjects provides no statistical benefit

### Model Comparison Summary

| Model | Subjects | Features | Test ROC-AUC | Test F1 | CV ROC-AUC |
|-------|----------|----------|--------------|---------|------------|
| **Baseline (Kaggle)** | 374 | 14 | 1.000 | 1.000 | 0.999 ± 0.004 |
| **Hybrid (Kaggle+EDF)** | 376 | 43 | 0.978 | 0.991 | 0.998 ± 0.003 |
| **EDF-Only** | 2 | 29 | N/A | N/A | Training failed |

**Winner:** **Baseline Model** (simpler, no missing data, same performance)

### Deliverables

#### Data Files
- `reports/sleepedf_subject_aggregated.csv` — 2 EDF subjects with physiological + lifestyle features
- `reports/kaggle_edf_merged.csv` — Combined 376-subject dataset

#### Models
- `models/sleep_quality_model.joblib` — **Production model** (Baseline, Kaggle-only)
- `models/sleep_quality_model_hybrid.joblib` — Hybrid model (for future use if more EDF data available)

#### Analysis Reports
- `reports/MODEL_RESULTS.md` — **Comprehensive results** for all 3 models with metrics and recommendations
- `reports/hybrid_model/hybrid_metrics_test.csv` — Test performance metrics
- `reports/hybrid_model/hybrid_coefficients.csv` — Feature importance rankings
- `reports/hybrid_model/hybrid_cv_results.csv` — Cross-validation results
- `reports/hybrid_model/roc_hybrid.png` — ROC curve visualization
- `reports/hybrid_model/cm_hybrid.png` — Confusion matrix
- `reports/edf_only_model/edf_only_summary.csv` — EDF-only diagnostic info

#### Code
- `scripts/merge_edf_kaggle.py` — Data merging pipeline
- `scripts/train_hybrid_model.py` — Hybrid model training script
- `scripts/train_edf_only.py` — EDF-only model training script

### Recommendations

**For This Project:**
- ✅ Use **baseline model** for deployment (already in production at `models/sleep_quality_model.joblib`)
- ✅ Document EDF integration as **proof-of-concept** showing feasibility
- ✅ Highlight limitations: small EDF sample (n=2), synthetic labels, extreme sparsity

**For Future Work:**
- Expand EDF dataset to ≥50 subjects with diverse sleep quality
- Collect paired data (same subjects providing both lifestyle AND physiological data)
- Validate synthetic labels against clinician ratings
- Test regularization techniques (L1/L2) for high-dimensional physiological features

### Conclusion

While the hybrid model integration was **technically successful** (demonstrating end-to-end data engineering), it provided **no practical benefit** due to:
1. Insufficient EDF sample size (2 subjects vs. 374)
2. Lack of class diversity in EDF labels (both poor sleep)
3. Extreme feature sparsity (99.5% missing physiological data)

The exercise validates that **lifestyle features alone are sufficient** for accurate sleep quality prediction in this dataset. Physiological features could become valuable with a larger, more diverse EDF dataset collected alongside lifestyle measurements.

**Full analysis:** See `reports/MODEL_RESULTS.md`