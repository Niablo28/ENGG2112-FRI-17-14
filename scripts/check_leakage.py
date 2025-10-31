"""
Leakage sentinel: shuffle-label control test.
If AUC >> 0.5 after shuffling labels, investigate leakage or proxy features.
"""
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = pathlib.Path(__file__).resolve().parents[1]
RAW = REPO / "reports" / "kaggle_augmented.csv"


def run_once(seed=42):
    df = pd.read_csv(RAW)
    y = (df["quality_of_sleep"].astype(int) >= 7).astype(int)
    X = df.drop(columns=["quality_of_sleep", "person_id"], errors='ignore')
    
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    
    # Shuffle-label control (should score ~0.5 AUC if no leakage)
    ytr_shuf = ytr.sample(frac=1.0, replace=False, random_state=seed).reset_index(drop=True)
    
    # Simple one-hot encoding for control test
    Xtr_oh = pd.get_dummies(Xtr, drop_first=True)
    Xte_oh = pd.get_dummies(Xte, drop_first=True).reindex(columns=Xtr_oh.columns, fill_value=0)
    
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    model.fit(Xtr_oh, ytr_shuf)
    p = model.predict_proba(Xte_oh)[:, 1]
    
    return roc_auc_score(yte, p)


if __name__ == "__main__":
    auc = run_once()
    print(f"Shuffle-label AUC = {auc:.3f} (expect ~0.5)")
    if auc > 0.6:
        print("⚠️  WARNING: AUC >> 0.5 suggests leakage or proxy features. Investigate.")
    else:
        print("✓ No obvious leakage detected.")



