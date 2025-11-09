"""
End-to-end retrain script with proper CV threshold selection and honest test evaluation.
Fixes data leakage by selecting threshold from train OOF predictions only.
"""
import argparse
import pathlib
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    cross_validate,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import sklearn
import sys
sys.path.append(str(pathlib.Path(__file__).parent))
from shared_transforms import (
    _bin_sleep_duration,
)


## moved to shared_transforms for stable import when unpickling

REPO = pathlib.Path(__file__).resolve().parents[1]
RAW = REPO / "reports" / "kaggle_augmented.csv"
OUT_MODELS = REPO / "models"
OUT_REPORTS = REPO / "reports"
OUT_MODELS.mkdir(exist_ok=True, parents=True)
OUT_REPORTS.mkdir(exist_ok=True, parents=True)


def binarise(y_continuous, cutoff=7):
    return (y_continuous >= cutoff).astype(int)


def f1_at_threshold(y_true, p, t):
    yhat = (p >= t).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, yhat, average="binary", zero_division=0)
    return prec, rec, f1


def create_preprocessing_pipeline():
    # OneHotEncoder backward compatibility
    kw = {"drop": "first", "handle_unknown": "ignore"}
    if tuple(map(int, sklearn.__version__.split(".")[:2])) >= (1, 2):
        enc = OneHotEncoder(sparse_output=False, **kw)
    else:
        enc = OneHotEncoder(sparse=False, **kw)
    
    # Blood pressure is now numeric (bp_sys, bp_dia) after _split_bp transformer
    numeric_features = ['age', 'sleep_duration', 'physical_activity_level', 
                        'stress_level', 'heart_rate', 'daily_steps', 'sleep_disorder_missing',
                        'bp_sys', 'bp_dia']
    categorical_features = ['gender', 'occupation', 'bmi_category', 'sleep_disorder']  # removed 'blood_pressure'
    
    duration_quad = ('sleepdur_quad', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=2, include_bias=False))
    ]), ['sleep_duration'])

    duration_bins = ('sleepdur_bins', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('bins', FunctionTransformer(_bin_sleep_duration))
    ]), ['sleep_duration'])

    ct = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            duration_quad,
            duration_bins,
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', enc)
            ]), categorical_features)
        ],
        remainder="drop"
    )
    return ct


def main(cutoff=7, random_state=42, cv=5, out_model=None):
    df = pd.read_csv(RAW)
    target = "quality_of_sleep"
    groups = df["person_id"].copy() if "person_id" in df.columns else None
    X = df.drop(columns=[target, "person_id"], errors='ignore')
    # Pre-split blood pressure into numeric columns to avoid transformer naming issues
    if 'blood_pressure' in X.columns:
        bp = X['blood_pressure'].astype(str).str.extract(r'(?P<bp_sys>\d{2,3})/(?P<bp_dia>\d{2,3})').astype(float)
        X = X.drop(columns=['blood_pressure']).join(bp)
    y = df[target].astype(int)
    y_bin = binarise(y, cutoff=cutoff)

    if groups is not None:
        holdout_splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(holdout_splitter.split(X, y_bin, groups=groups))
        groups_tr = groups.iloc[train_idx].reset_index(drop=True)
    else:
        holdout_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(holdout_splitter.split(X, y_bin))
        groups_tr = None

    Xtr = X.iloc[train_idx].reset_index(drop=True)
    Xte = X.iloc[test_idx].reset_index(drop=True)
    ytr = y_bin.iloc[train_idx].reset_index(drop=True)
    yte = y_bin.iloc[test_idx].reset_index(drop=True)

    preprocessor = create_preprocessing_pipeline()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=random_state)
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])

    # Cross-validated threshold selection (on TRAIN ONLY)
    if groups_tr is not None:
        skf = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=random_state)
        split_generator = skf.split(Xtr, ytr, groups=groups_tr)
    else:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        split_generator = skf.split(Xtr, ytr)
    oof_prob = np.zeros(len(Xtr))
    for train_idx, val_idx in split_generator:
        X_tr, X_val = Xtr.iloc[train_idx], Xtr.iloc[val_idx]
        y_tr, y_val = ytr.iloc[train_idx], ytr.iloc[val_idx]
        pipe.fit(X_tr, y_tr)
        oof_prob[val_idx] = pipe.predict_proba(X_val)[:, 1]

    # Choose threshold that maximises F1 on OOF preds
    grid = np.linspace(0.1, 0.9, 33)
    th, best = 0.5, -1
    for t in grid:
        _, _, f1 = f1_at_threshold(ytr.values, oof_prob, t)
        if f1 > best:
            best, th = f1, t

    # Save OOF probabilities for threshold sweep analysis
    pd.DataFrame({
        "y_true": ytr.values,
        "y_prob": oof_prob
    }).to_csv(OUT_REPORTS / "oof_probs_logreg.csv", index=False)

    # 5-fold CV reporting on TRAIN set with preprocessing inside the pipeline
    cv_splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=random_state) if groups_tr is not None else StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_scores = cross_validate(
        pipe,
        Xtr,
        ytr,
        cv=cv_splitter,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        return_train_score=False,
        n_jobs=None,
        groups=groups_tr if groups_tr is not None else None
    )
    cv_df = pd.DataFrame(cv_scores)
    cv_df.to_csv(OUT_REPORTS / "cv_results_end_to_end.csv", index=False)

    # Final fit on full TRAIN and honest TEST evaluation
    pipe.fit(Xtr, ytr)
    p_test = pipe.predict_proba(Xte)[:, 1]
    yhat_test = (p_test >= th).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(yte, yhat_test, average="binary", zero_division=0)
    auc = roc_auc_score(yte, p_test)
    acc = (yhat_test == yte).mean()

    # === Save final pipeline (standardised name + legacy copy) ===
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    final_path = OUT_MODELS / "model_augmented_latest.joblib"
    joblib.dump(pipe, final_path)

    # (optional) keep a legacy copy so older scripts still work
    legacy_path = OUT_MODELS / "sleep_quality_model.joblib"
    try:
        if legacy_path.exists():
            legacy_path.unlink()
    except Exception:
        pass
    joblib.dump(pipe, legacy_path)
    print(f"[OK] Saved model to: {final_path} (and legacy copy to {legacy_path})")
    pd.DataFrame({
        "metric": ["accuracy", "precision", "recall", "f1", "roc_auc", "threshold"],
        "value": [acc, prec, rec, f1, auc, th]
    }).to_csv(OUT_REPORTS / "logreg_metrics_test.csv", index=False)
    # Save CV summary (mean/std) for quick reference
    summary = pd.DataFrame({
        "metric": [
            "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"
        ],
        "mean": [
            cv_df["test_accuracy"].mean(),
            cv_df["test_precision"].mean(),
            cv_df["test_recall"].mean(),
            cv_df["test_f1"].mean(),
            cv_df["test_roc_auc"].mean()
        ],
        "std": [
            cv_df["test_accuracy"].std(),
            cv_df["test_precision"].std(),
            cv_df["test_recall"].std(),
            cv_df["test_f1"].std(),
            cv_df["test_roc_auc"].std()
        ]
    })
    summary.to_csv(OUT_REPORTS / "cv_results_end_to_end_summary.csv", index=False)

    print(f"Saved models/sleep_quality_model.joblib")
    print(f"CV (mean±std) — auc={cv_df['test_roc_auc'].mean():.3f}±{cv_df['test_roc_auc'].std():.3f} f1={cv_df['test_f1'].mean():.3f}±{cv_df['test_f1'].std():.3f}")
    print(f"TEST — acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} auc={auc:.3f} (th={th:.2f})")
    print(f"OOF F1 at selected threshold: {best:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train end-to-end sleep quality model with proper CV and OOF threshold selection")
    ap.add_argument("--cutoff", type=int, default=7, help="Threshold for binary classification (default: 7)")
    ap.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    ap.add_argument("--cv", type=int, default=5, help="Number of CV folds (default: 5)")
    ap.add_argument("--out_model", type=str, default=None, help="Output model path (default: auto-generated with timestamp)")
    args = ap.parse_args()
    
    # Generate timestamped model name if not provided
    if args.out_model is None:
        import datetime
        import subprocess
        try:
            git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        except:
            git_sha = "0000000"
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        args.out_model = f"models/model_{date_str}_{git_sha}.joblib"
    
    main(cutoff=args.cutoff, random_state=args.random_state, cv=args.cv, out_model=args.out_model)



