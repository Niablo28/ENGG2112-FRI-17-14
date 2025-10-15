import pandas as pd, numpy as np, pathlib, joblib, argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, RocCurveDisplay, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

def load_xy(repo_root):
    root = pathlib.Path(repo_root) / "reports"
    Xtr = pd.read_parquet(root / "X_train_proc.parquet")
    Xte = pd.read_parquet(root / "X_test_proc.parquet")
    ytr = pd.read_csv(root / "y_train.csv").squeeze()
    yte = pd.read_csv(root / "y_test.csv").squeeze()
    return Xtr, Xte, ytr, yte

def make_binary(y, cutoff): return (y >= cutoff).astype(int)

def metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob)
    return dict(precision=prec, recall=rec, f1=f1, roc_auc=auc)

def run(repo_root=".", cutoff=7, out_dir="reports/logistic_cv"):
    out = pathlib.Path(repo_root) / out_dir; out.mkdir(parents=True, exist_ok=True)
    Xtr, Xte, ytr_c, yte_c = load_xy(repo_root)
    ytr, yte = make_binary(ytr_c, cutoff), make_binary(yte_c, cutoff)

    # fit logistic regression with balanced class weight
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    model.fit(Xtr, ytr)

    # evaluating on test set
    yprob = model.predict_proba(Xte)[:, 1]
    m = metrics(yte, yprob)
    pd.DataFrame([m]).to_csv(out/"logreg_metrics_test.csv", index=False)

    # coefficients
    coefs = pd.Series(model.coef_[0], index=Xtr.columns).sort_values(key=abs, ascending=False)
    coefs.to_csv(out/"logreg_coefficients.csv")
    print("Top coefficients:\n", coefs.head(10))

    # ROC and confusion plots
    RocCurveDisplay.from_predictions(yte, yprob)
    plt.title("Logistic Regression ROC")
    plt.savefig(out/"roc_logreg.png", bbox_inches='tight'); plt.close()
    ConfusionMatrixDisplay.from_predictions(yte, (yprob>=0.5).astype(int))
    plt.title("Confusion (th=0.5)"); plt.savefig(out/"cm_logreg.png", bbox_inches='tight'); plt.close()

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cvres = cross_validate(model, Xtr, ytr, cv=skf, scoring=['accuracy','precision','recall','f1','roc_auc'], return_train_score=False)
    pd.DataFrame(cvres).to_csv(out/"cv_results.csv", index=False)
    print("Mean CV metrics:\n", pd.DataFrame(cvres).mean())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=7)
    ap.add_argument("--repo_root", default=".")
    args = ap.parse_args()
    run(args.repo_root, args.cutoff)