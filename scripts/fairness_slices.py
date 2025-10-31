"""
Compute basic performance slices by sensitive attributes (e.g., gender, age buckets)
using the trained end-to-end pipeline and the augmented Kaggle dataset.
"""
from pathlib import Path
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score


def binarise(y_continuous: pd.Series, cutoff: int = 7) -> np.ndarray:
    return (y_continuous.astype(int) >= cutoff).astype(int).to_numpy()


def evaluate_slice(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan
    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}


def main(model_path: Path, data_csv: Path, out_csv: Path, threshold: float = 0.5):
    df = pd.read_csv(data_csv)
    y = binarise(df["quality_of_sleep"], cutoff=7)
    X = df.drop(columns=["quality_of_sleep", "person_id"], errors='ignore')

    pipe = joblib.load(model_path)
    p = pipe.predict_proba(X)[:, 1]

    rows = []
    # Gender slice
    if "gender" in df.columns:
        for g, sub in df.groupby("gender"):
            idx = sub.index.to_numpy()
            rows.append({"slice": f"gender={g}", **evaluate_slice(y[idx], p[idx], threshold)})

    # Age buckets
    if "age" in df.columns:
        bins = [18, 30, 45, 60, 100]
        labels = ["18-29", "30-44", "45-59", "60+"]
        age_bucket = pd.cut(df["age"], bins=bins, labels=labels, right=False, include_lowest=True)
        for b in age_bucket.unique():
            idx = age_bucket[age_bucket == b].index.to_numpy()
            if len(idx) == 0:
                continue
            rows.append({"slice": f"age={b}", **evaluate_slice(y[idx], p[idx], threshold)})

    pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(repo_root / "models" / "sleep_quality_model.joblib"))
    ap.add_argument("--data", default=str(repo_root / "reports" / "kaggle_augmented.csv"))
    ap.add_argument("--out", default=str(repo_root / "reports" / "fairness_slices.csv"))
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    main(Path(args.model), Path(args.data), Path(args.out), threshold=args.threshold)


