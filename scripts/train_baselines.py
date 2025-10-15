import argparse, pathlib, joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (precision_recall_fscore_support,roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

def load_xy(repo_root):
    root = pathlib.Path(repo_root) / "reports"
    Xtr = pd.read_parquet(root/"X_train_proc.parquet")
    Xte = pd.read_parquet(root/"X_test_proc.parquet")
    ytr = pd.read_csv(root/"y_train.csv").squeeze()
    yte = pd.read_csv(root/"y_test.csv").squeeze()
    return Xtr, Xte, ytr, yte

def make_binary(y_continuous, cutoff):
    return (y_continuous >= cutoff).astype(int)

def metrics_dict(y_true, y_prob, y_pred):
    acc = (y_true==y_pred).mean()
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan
    return {"accuracy":acc, "precision":prec, "recall":rec, "f1":f1, "roc_auc":auc}

def plot_roc(y_true, y_prob, title, out_png):
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(title)
    plt.savefig(out_png, bbox_inches='tight'); plt.close()

def plot_cm(y_true, y_pred, title, out_png):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(title)
    plt.savefig(out_png, bbox_inches='tight'); plt.close()

def run(repo_root, cutoff, out_dir):
    Xtr, Xte, ytr_cont, yte_cont = load_xy(repo_root)
    ytr = make_binary(ytr_cont, cutoff)
    yte = make_binary(yte_cont, cutoff)

    results=[]
    models={
        "logreg_balanced": LogisticRegression(max_iter=500, class_weight="balanced"),
        "gaussian_nb": GaussianNB(),
        "knn5": KNeighborsClassifier(n_neighbors=5)
    }

    out_path = pathlib.Path(repo_root)/out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        clf = model.fit(Xtr, ytr)
        if hasattr(clf, "predict_proba"):
            yprob = clf.predict_proba(Xte)[:,1]
        elif hasattr(clf, "decision_function"):
            s = clf.decision_function(Xte)
            yprob = (s - s.min())/(s.max()-s.min()+1e-9)
        else:
            yprob = clf.predict(Xte)

        ypred = (yprob >= 0.5).astype(int)
        m = metrics_dict(yte, yprob, ypred); m["model"]=name
        results.append(m)

        # Save probabilities for threshold tuning
        pd.DataFrame({"y_true":yte, "y_prob":yprob, "y_pred":ypred}).to_csv(out_path/f"probas_{name}.csv", index=False)

        # Plots
        plot_roc(yte, yprob, f"ROC – {name} (cutoff≥{cutoff})", out_path/f"roc_{name}.png")
        plot_cm(yte, ypred, f"Confusion – {name} (threshold=0.5)", out_path/f"cm_{name}.png")

        # Save model
        joblib.dump(clf, out_path/f"{name}.joblib")

    pd.DataFrame(results).to_csv(out_path/"baseline_metrics.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".", help="path to repo root")
    ap.add_argument("--cutoff", type=int, default=7, help="good sleep if score≥cutoff")
    ap.add_argument("--out", default="reports/baselines", help="output directory")
    args=ap.parse_args()
    run(args.repo_root, args.cutoff, args.out)
