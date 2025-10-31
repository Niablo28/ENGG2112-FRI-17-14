"""
DEPRECATED: This script is for legacy hybrid model training.
Use scripts/retrain_end_to_end.py instead for proper end-to-end pipeline training.
"""
raise RuntimeError("This script is deprecated. Use scripts/retrain_end_to_end.py instead.")

import pandas as pd
import numpy as np
import pathlib
import joblib
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

def load_merged_data(repo_root):
    root = pathlib.Path(repo_root) / "reports"
    merged = pd.read_csv(root / "kaggle_edf_merged.csv")
    X = merged.drop(columns=['quality_of_sleep', 'person_id', 'subject'], errors='ignore')
    y = merged['quality_of_sleep']
    return X, y, merged

def make_binary(y, cutoff=7):
    return (y >= cutoff).astype(int)

def create_preprocessing_pipeline(X):
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if 'id' not in col.lower()]
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    return preprocessor, numeric_cols, categorical_cols

def metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob)
    acc = (y_pred == y_true).mean()
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=auc)

def run(repo_root=".", cutoff=7, out_dir="reports/hybrid_model"):
    out = pathlib.Path(repo_root) / out_dir
    out.mkdir(parents=True, exist_ok=True)
    
    print("Training hybrid model...")
    X, y_continuous, merged_df = load_merged_data(repo_root)
    y = make_binary(y_continuous, cutoff)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor, numeric_cols, categorical_cols = create_preprocessing_pipeline(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=42)
    model.fit(X_train_proc, y_train)
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    model_path = pathlib.Path(repo_root) / "models" / "sleep_quality_model_hybrid.joblib"
    joblib.dump(full_pipeline, model_path)
    
    y_prob_test = model.predict_proba(X_test_proc)[:, 1]
    test_metrics = metrics(y_test, y_prob_test)
    pd.DataFrame([test_metrics]).to_csv(out / "hybrid_metrics_test.csv", index=False)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        full_pipeline, X_train, y_train, cv=skf,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        return_train_score=False
    )
    pd.DataFrame(cv_results).to_csv(out / "hybrid_cv_results.csv", index=False)
    
    feature_names = list(numeric_cols)
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        feature_names.extend(cat_encoder.get_feature_names_out(categorical_cols))
    
    coefs = pd.Series(model.coef_[0], index=feature_names).sort_values(key=abs, ascending=False)
    coefs.to_csv(out / "hybrid_coefficients.csv")
    
    RocCurveDisplay.from_predictions(y_test, y_prob_test)
    plt.title("Hybrid Model ROC Curve")
    plt.savefig(out / "roc_hybrid.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    y_pred_test = (y_prob_test >= 0.5).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
    plt.title("Hybrid Model Confusion Matrix")
    plt.savefig(out / "cm_hybrid.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Complete: ROC-AUC={test_metrics['roc_auc']:.3f}, F1={test_metrics['f1']:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=7)
    ap.add_argument("--repo_root", default=".")
    args = ap.parse_args()
    run(args.repo_root, args.cutoff)
