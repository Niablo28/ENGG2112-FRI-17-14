# Model Statistics

Trained variants kept for the final submission can be rebuilt via:

```
py -3 scripts/train_core_models.py
```

That command refreshes the artefacts in `models/` and updates
`reports/core_model_metrics.json`, which feeds the table below.

| Model | Artefact | Training data | Samples (train/test) | Pos/Neg | Best threshold | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kaggle Logistic Baseline | `model_kaggle_baseline.joblib` | 374 Kaggle rows (`kaggle_clean_winsorized.csv`) | 299 / 75 | 257 / 117 | 0.55 | 0.9733 | 0.9630 | 1.0000 | 0.9811 | 0.9967 |
| Kaggle KNN (k=5) | `model_kaggle_knn.joblib` | same as above | 299 / 75 | 257 / 117 | 0.80 | 0.9867 | 1.0000 | 0.9808 | 0.9903 | 0.9992 |
| Kaggle Gaussian NB | `model_kaggle_nb.joblib` | same as above | 299 / 75 | 257 / 117 | 0.80 | 0.8533 | 0.8254 | 1.0000 | 0.9043 | 0.9983 |
| Augmented Logistic *(also saved as `sleep_quality_model.joblib`)* | `model_augmented_latest.joblib` | 744 rows (374 Kaggle + 370 synthetic in `kaggle_augmented.csv`) | 595 / 149 | 287 / 457 | 0.80 | 0.9866 | 0.9825 | 0.9825 | 0.9825 | 0.9977 |
| Hybrid Logistic (Kaggle + EDF) | `model_hybrid.joblib` | 376 merged rows (`kaggle_edf_merged.csv`) | 300 / 76 | 257 / 119 | 0.65 | 0.9868 | 0.9811 | 1.0000 | 0.9905 | 0.9760 |
| EDF Only *(fallback)* | `model_edf_only.joblib` | 2 aggregated EDF rows (`sleepedf_subject_aggregated.csv`) | 2 / 0 | 0 / 2 | 0.80 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | N/A |

- The augmented logistic variant is duplicated as `sleep_quality_model.joblib`
  so the UI and tests keep working without modification.
- The three Kaggle baselines (logistic, KNN, Gaussian NB) are included as final
  deliverables to showcase model diversity on a consistent feature space.
- `model_edf_only.joblib` uses a dummy classifier because the EDF aggregate file
  contains only two negative samples; its metrics are placeholders.
# Model Statistics

This project keeps four trained variants for reporting. Rebuild them with:

```
py -3 scripts/train_core_models.py
```

That script retrains each pipeline, refreshes the artefacts in `models/`, and
updates `reports/core_model_metrics.json` (source for the figures below).

| Model | Artefact | Training data | Samples (train/test) | Pos/Neg | Best threshold | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kaggle Logistic Baseline | `model_kaggle_baseline.joblib` | 374 Kaggle rows (`kaggle_clean_winsorized.csv`) | 299 / 75 | 257 / 117 | 0.55 | 0.9733 | 0.9630 | 1.0000 | 0.9811 | 0.9967 |
| Kaggle KNN (k=5) | `model_kaggle_knn.joblib` | same as above | 299 / 75 | 257 / 117 | 0.80 | 0.9867 | 1.0000 | 0.9808 | 0.9903 | 0.9992 |
| Kaggle Gaussian NB | `model_kaggle_nb.joblib` | same as above | 299 / 75 | 257 / 117 | 0.80 | 0.8533 | 0.8254 | 1.0000 | 0.9043 | 0.9983 |
| Augmented Logistic *(also saved as `sleep_quality_model.joblib`)* | `model_augmented_latest.joblib` | 744 rows (374 Kaggle + 370 synthetic in `kaggle_augmented.csv`) | 595 / 149 | 287 / 457 | 0.80 | 0.9866 | 0.9825 | 0.9825 | 0.9825 | 0.9977 |
| Hybrid Logistic (Kaggle + EDF) | `model_hybrid.joblib` | 376 merged rows (`kaggle_edf_merged.csv`) | 300 / 76 | 257 / 119 | 0.65 | 0.9868 | 0.9811 | 1.0000 | 0.9905 | 0.9760 |
| EDF Only *(fallback)* | `model_edf_only.joblib` | 2 aggregated EDF rows (`sleepedf_subject_aggregated.csv`) | 2 / 0 | 0 / 2 | 0.80 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | N/A |

- The augmented logistic variant is duplicated as `sleep_quality_model.joblib`
  so the UI and tests keep working without modification.
- The Kaggle baseline, KNN, and Gaussian NB models are packaged as part of the
  final deliverables for head-to-head comparison on the same feature space.
- `model_edf_only.joblib` uses a dummy classifier because the EDF aggregate file
  contains only two negative samples; its metrics are placeholders.

