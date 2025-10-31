"""
DEPRECATED: This script preprocesses data before CV which may cause alignment issues.
Use scripts/retrain_end_to_end.py instead for proper end-to-end pipeline training.
"""
raise RuntimeError("This script is deprecated and may cause data leakage. Use scripts/retrain_end_to_end.py instead.")

import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, FunctionTransformer
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).parent))
from shared_transforms import _bin_sleep_duration
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, roc_curve
)
import matplotlib.pyplot as plt
from pathlib import Path

def load_augmented_data(repo_root):
    """Load augmented dataset"""
    data = pd.read_csv(repo_root / "reports" / "kaggle_augmented.csv")
    
    # Prepare target
    y = (data['quality_of_sleep'] >= 7).astype(int)
    
    # Drop target and ID
    X = data.drop(columns=['quality_of_sleep', 'person_id'])
    
    return X, y

def create_preprocessing_pipeline():
    """Create preprocessing pipeline"""
    numeric_features = ['age', 'sleep_duration', 'physical_activity_level', 
                        'stress_level', 'heart_rate', 'daily_steps', 'sleep_disorder_missing']
    categorical_features = ['gender', 'occupation', 'bmi_category', 'blood_pressure', 'sleep_disorder']
    
    # Separate transformer for sleep_duration engineering (quadratic + bins)
    duration_quad = ('sleepdur_quad', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=2, include_bias=False))
    ]), ['sleep_duration'])

    duration_bins = ('sleepdur_bins', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('bins', FunctionTransformer(_bin_sleep_duration))
    ]), ['sleep_duration'])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            duration_quad,
            duration_bins,
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    
    return preprocessor

def metrics(y_true, y_prob):
    """Calculate metrics"""
    y_pred = (y_prob >= 0.5).astype(int)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }

def main():
    repo_root = Path(__file__).parent.parent
    
    # Load data
    print("Loading augmented dataset...")
    X, y = load_augmented_data(repo_root)
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} (Good: {sum(y_train)}, Poor: {len(y_train)-sum(y_train)})")
    print(f"Test set: {len(X_test)} (Good: {sum(y_test)}, Poor: {len(y_test)-sum(y_test)})")
    
    # Create preprocessing pipeline
    print("\nCreating preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(X_train)
    
    # Save preprocessor
    joblib.dump(preprocessor, repo_root / "models" / "preprocessor_augmented.joblib")
    print("Saved preprocessor to: models/preprocessor_augmented.joblib")
    
    # Transform data
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    print(f"\nPreprocessed train shape: {X_train_proc.shape}")
    print(f"Preprocessed test shape: {X_test_proc.shape}")
    
    # Train logistic regression
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs'
    )
    model.fit(X_train_proc, y_train)
    
    # Cross-validation
    print("\nRunning 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_proc, y_train, cv=cv, 
                                scoring='roc_auc', n_jobs=-1)
    
    print(f"CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_test_prob = model.predict_proba(X_test_proc)[:, 1]
    test_metrics = metrics(y_test, y_test_prob)
    
    print("\nTest Set Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # Create full pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Save model
    model_path = repo_root / "models" / "model_augmented_latest.joblib"
    joblib.dump(full_pipeline, model_path)
    print(f"\nSaved model to: {model_path}")
    
    # Save test metrics
    metrics_df = pd.DataFrame([test_metrics])
    metrics_df.to_csv(repo_root / "reports" / "augmented_model_test_metrics.csv", index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("AUGMENTED MODEL TRAINING SUMMARY")
    print("="*60)
    print(f"\nDataset: Augmented (200 synthetic + 374 original)")
    print(f"Total subjects: {len(X)}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"\nCross-Validation ROC-AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.3f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"\nModel saved: {model_path}")
    print("="*60)

if __name__ == "__main__":
    main()

