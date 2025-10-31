"""
Export feature coefficients from the trained end-to-end Logistic Regression pipeline.
Produces a CSV and a horizontal bar plot of top features.
"""
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def get_feature_names_from_column_transformer(ct: ColumnTransformer) -> list:
    feature_names = []
    for name, transformer, columns in ct.transformers_:
        if name == 'remainder' and transformer == 'drop':
            continue
        if hasattr(transformer, 'named_steps'):
            # Attempt to get inner most transformer producing names
            inner = list(transformer.named_steps.values())[-1]
        else:
            inner = transformer
        if isinstance(inner, OneHotEncoder):
            if isinstance(columns, list):
                cats = inner.get_feature_names_out(columns)
            else:
                cats = inner.get_feature_names_out([columns])
            feature_names.extend(cats.tolist())
        elif hasattr(inner, 'get_feature_names_out'):
            names = inner.get_feature_names_out(columns)
            feature_names.extend(names.tolist())
        else:
            # Fallback to column names length
            if isinstance(columns, list):
                feature_names.extend([str(c) for c in columns])
            else:
                feature_names.append(str(columns))
    return feature_names


def main(model_path: Path, out_dir: Path, top_k: int = 20):
    pipe: Pipeline = joblib.load(model_path)
    prep: ColumnTransformer = pipe.named_steps['prep'] if 'prep' in pipe.named_steps else pipe.named_steps['preprocessor']
    clf = pipe.named_steps['clf'] if 'clf' in pipe.named_steps else pipe.named_steps['classifier']

    if not hasattr(clf, 'coef_'):
        raise ValueError("Classifier does not expose coef_. Use a linear model or implement tree importances instead.")

    feature_names = get_feature_names_from_column_transformer(prep)
    coefs = clf.coef_.ravel()
    if len(feature_names) != len(coefs):
        # Length mismatch guard
        feature_names = [f"f{i}" for i in range(len(coefs))]

    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    coef_df["abs"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs", ascending=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    coef_csv = out_dir / "coefficients_end_to_end.csv"
    coef_df.drop(columns=["abs"]).to_csv(coef_csv, index=False)

    # Plot top features
    top = coef_df.head(top_k).iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.35 * len(top))))
    colors = ["#00B894" if v > 0 else "#FF6B6B" for v in top["coefficient"]]
    plt.barh(top["feature"], top["coefficient"], color=colors)
    plt.xlabel("Coefficient")
    plt.title("Top feature coefficients (end-to-end pipeline)")
    plt.tight_layout()
    plt.savefig(out_dir / "coefficients_end_to_end.png", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(repo_root / "models" / "sleep_quality_model.joblib"))
    ap.add_argument("--out", default=str(repo_root / "reports"))
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()
    main(Path(args.model), Path(args.out), top_k=args.top_k)


