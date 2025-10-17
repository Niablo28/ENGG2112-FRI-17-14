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

### Stuff for the report (Temporarily in readme)
#### Things to mention in report
- small dataset
- mention data leakage and overfitting
- Methods – Logistic Regression Model
  - " Logistic Regression was implemented using scikit-learn (LogisticRegression, class_weight = “balanced”) following the probabilistic framework described in Week 5 lectures.
The model assumes a linear relationship between input features and the log-odds of achieving good sleep quality.
Training used 5-fold Stratified Cross-Validation to reduce sampling bias.
Continuous variables were scaled and categorical features one-hot encoded through the pre-processing pipeline (preprocessor.joblib).
Model parameters (coefficients β) were estimated via maximum-likelihood, optimised using the LBFGS solver until convergence (max_iter = 1000).
Evaluation metrics included accuracy, precision, recall, F1-score, and ROC-AUC, in accordance with Week 2 guidelines on classification performance. "
- Findings – Logistic Regression and Trade-off Analysis
  - "Logistic Regression achieved the best overall performance among baseline classifiers, with mean cross-validation accuracy ≈ 0.993 and ROC-AUC ≈ 0.999.
The most influential predictors were sleep duration (+2.20) and stress level (–1.97), confirming that longer rest and lower stress are strong indicators of high sleep quality.
Occupational categories such as Salesperson (–1.28) and Nurse (–0.50) were negatively correlated with good sleep, while Accountant (+1.08) and Lawyer (+0.58) showed positive effects, suggesting lifestyle regularity impacts outcomes.
The ROC curve indicated near-perfect separability of the two classes at a default threshold (0.5).
Given the project’s recall-priority objective (avoiding false “good” predictions), subsequent threshold sweeps will confirm the optimal operating point (likely ≈ 0.45).
Overall, the model meets and exceeds the acceptance target (AUC ≥ 0.75, F1 ≥ 0.70) defined in the proposal."
- Cross-validation and shuffle-split results show minimal variance (AUC ≈ 0.998 ± 0.0035), with the learning curve demonstrating convergence between training and validation F1 scores. These patterns indicate that the logistic regression model generalises well and is not overfitting to the small dataset. Nevertheless, given the limited sample size (n ≈ 374), external validation on unseen data is recommended before deployment.
- Discussion – Reflection
  - The exceptionally high metrics imply potential overfitting or feature redundancy due to small sample size (~374 records).
Future iterations will test regularisation strength (C parameter) and repeat validation with the winsorised and drop-outlier datasets to verify robustness.
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
