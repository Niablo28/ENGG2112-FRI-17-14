"""Trained EDF-only model"""

import pandas as pd
import numpy as np
import pathlib
import joblib
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_edf_data(repo_root):
    root = pathlib.Path(repo_root) / "reports"
    edf = pd.read_csv(root / "sleepedf_subject_aggregated.csv")
    
    exclude_cols = [
        'subject', 'person_id', 'quality_of_sleep',
        'gender', 'age', 'occupation', 'sleep_duration', 
        'physical_activity_level', 'stress_level', 'bmi_category',
        'blood_pressure', 'heart_rate', 'daily_steps', 
        'sleep_disorder', 'sleep_disorder_missing'
    ]
    
    X = edf.drop(columns=[col for col in exclude_cols if col in edf.columns], errors='ignore')
    y = edf['quality_of_sleep']
    return X, y, edf

def make_binary(y, cutoff=7):
    return (y >= cutoff).astype(int)

def run(repo_root=".", cutoff=7, out_dir="reports/edf_only_model"):
    out = pathlib.Path(repo_root) / out_dir
    out.mkdir(parents=True, exist_ok=True)
    
    print("Training EDF-only model...")
    X, y_continuous, edf_df = load_edf_data(repo_root)
    y = make_binary(y_continuous, cutoff)
    
    if len(X) < 2:
        print("Error: Need at least 2 samples")
        return
    
    if (y == 0).sum() == 0 or (y == 1).sum() == 0:
        print(f"Training failed: All samples have same label {y.tolist()}")
        summary = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'subjects': edf_df['subject'].tolist(),
            'quality_scores': y_continuous.tolist(),
            'binary_labels': y.tolist(),
            'status': 'insufficient_class_diversity',
            'message': 'All samples belong to same class'
        }
        pd.DataFrame([summary]).to_csv(out / "edf_only_summary.csv", index=False)
        return
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, solver='lbfgs'))
    ])
    
    loo = LeaveOneOut()
    predictions, actual = [], []
    
    for train_idx, test_idx in loo.split(X):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        pipeline.fit(X_train_fold, y_train_fold)
        y_pred = pipeline.predict(X_test_fold)[0]
        predictions.append(y_pred)
        actual.append(y_test_fold.iloc[0])
    
    accuracy = np.mean([p == a for p, a in zip(predictions, actual)])
    
    pipeline.fit(X, y)
    model_path = pathlib.Path(repo_root) / "reports" / "sleep_quality_model_edf_only.joblib"
    joblib.dump(pipeline, model_path)
    
    coefs = pd.Series(pipeline.named_steps['classifier'].coef_[0], index=X.columns)
    coefs = coefs.sort_values(key=abs, ascending=False)
    coefs.to_csv(out / "edf_only_coefficients.csv")
    
    results = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'loo_accuracy': accuracy,
        'cutoff': cutoff,
        'subjects': ', '.join(edf_df['subject'].tolist()),
        'quality_scores': ', '.join(map(str, y_continuous.tolist())),
        'binary_labels': ', '.join(map(str, y.tolist()))
    }
    pd.DataFrame([results]).to_csv(out / "edf_only_results.csv", index=False)
    
    print(f"Complete: LOO accuracy={accuracy:.2f} (n={len(X)} subjects)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=7)
    ap.add_argument("--repo_root", default=".")
    args = ap.parse_args()
    run(args.repo_root, args.cutoff)
