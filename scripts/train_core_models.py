from __future__ import annotations

"""Train and refresh the core sleep-quality models in one pass.

This utility rebuilds the primary pipelines we keep in the repository:

1. Kaggle logistic baseline (``model_kaggle_baseline.joblib``).
2. Augmented logistic with synthetic cases (``model_augmented_latest.joblib``)
   plus a compatibility copy ``sleep_quality_model.joblib`` for the UI/tests.
3. Kaggle KNN baseline (``model_kaggle_knn.joblib``).
4. Kaggle Gaussian NB baseline (``model_kaggle_nb.joblib``).
5. Hybrid logistic that mixes Kaggle lifestyle + EDF physiology
   (``model_hybrid.joblib``).
6. EDF-only fallback model (``model_edf_only.joblib``).

The script stores concise evaluation metrics for each model in
``reports/core_model_metrics.json`` so they can be surfaced in
``models_statisitics.md``.

Run from the repository root:

    py -3 scripts/train_core_models.py
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from shared_transforms import _bin_sleep_duration, _split_bp

CUTOFF = 7
RANDOM_STATE = 42
THRESHOLD_GRID = np.linspace(0.2, 0.8, 13)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"
METRICS_PATH = REPORTS_DIR / "core_model_metrics.json"


@dataclass
class ModelStats:
    name: str
    artefact: str
    dataset: str
    n_samples: int
    n_features: int
    train_samples: int
    test_samples: int
    positive_cases: int
    negative_cases: int
    metrics: Dict[str, float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "artefact": self.artefact,
            "dataset": self.dataset,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "positive_cases": self.positive_cases,
            "negative_cases": self.negative_cases,
            "metrics": self.metrics,
        }


def _classification_summary(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: Iterable[float] = THRESHOLD_GRID,
) -> Dict[str, float]:
    """Return best-threshold metrics and a 0.5 reference."""

    summary: Dict[str, float] = {}
    best = {
        "threshold": 0.5,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    for t in thresholds:
        preds = (proba >= t).astype(int)
        acc = accuracy_score(y_true, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )
        if f1 >= best["f1"]:
            best = {
                "threshold": float(t),
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            }

    default_preds = (proba >= 0.5).astype(int)
    def_acc = accuracy_score(y_true, default_preds)
    def_prec, def_rec, def_f1, _ = precision_recall_fscore_support(
        y_true, default_preds, average="binary", zero_division=0
    )

    try:
        roc_auc = float(roc_auc_score(y_true, proba))
    except ValueError:
        roc_auc = None

    summary.update(best)
    summary.update(
        {
            "roc_auc": roc_auc,
            "threshold_0_5_accuracy": float(def_acc),
            "threshold_0_5_precision": float(def_prec),
            "threshold_0_5_recall": float(def_rec),
            "threshold_0_5_f1": float(def_f1),
        }
    )
    return summary


def _kaggle_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "age",
        "sleep_duration",
        "physical_activity_level",
        "stress_level",
        "heart_rate",
        "daily_steps",
        "sleep_disorder_missing",
        "bp_sys",
        "bp_dia",
    ]
    categorical_features = ["gender", "occupation", "bmi_category", "sleep_disorder"]

    duration_quad = (
        "sleepdur_quad",
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ]
        ),
        ["sleep_duration"],
    )

    duration_bins = (
        "sleepdur_bins",
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("bins", FunctionTransformer(_bin_sleep_duration)),
            ]
        ),
        ["sleep_duration"],
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            duration_quad,
            duration_bins,
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def _prepare_kaggle_frame(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    y = (df["quality_of_sleep"] >= CUTOFF).astype(int)
    features = df.drop(columns=["quality_of_sleep", "person_id"], errors="ignore")
    features = _split_bp(features)
    return features, y


def _train_kaggle_variant(
    *,
    data_path: Path,
    artefact_name: str,
    display_name: str,
    classifier,
    random_state: int = RANDOM_STATE,
) -> ModelStats:
    X, y = _prepare_kaggle_frame(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("prep", _kaggle_preprocessor()),
            ("clf", classifier),
        ]
    )

    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = _classification_summary(y_test.values, proba)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / artefact_name)

    return ModelStats(
        name=display_name,
        artefact=artefact_name,
        dataset=data_path.name,
        n_samples=len(y),
        n_features=X.shape[1],
        train_samples=len(y_train),
        test_samples=len(y_test),
        positive_cases=int(y.sum()),
        negative_cases=int((1 - y).sum()),
        metrics=metrics,
    )


def _train_hybrid(random_state: int = RANDOM_STATE) -> ModelStats:
    path = REPORTS_DIR / "kaggle_edf_merged.csv"
    df = pd.read_csv(path)
    y = (df["quality_of_sleep"] >= CUTOFF).astype(int)

    features = df.drop(columns=["quality_of_sleep", "person_id", "subject"], errors="ignore")
    features = _split_bp(features)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = _classification_summary(y_test.values, proba)

    artefact = "model_hybrid.joblib"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / artefact)

    return ModelStats(
        name="Hybrid Logistic (Kaggle + EDF)",
        artefact=artefact,
        dataset=path.name,
        n_samples=len(y),
        n_features=features.shape[1],
        train_samples=len(y_train),
        test_samples=len(y_test),
        positive_cases=int(y.sum()),
        negative_cases=int((1 - y).sum()),
        metrics=metrics,
    )


def _train_edf_only() -> ModelStats:
    path = REPORTS_DIR / "sleepedf_subject_aggregated.csv"
    df = pd.read_csv(path)

    y = (df["quality_of_sleep"] >= CUTOFF).astype(int)
    features = df.drop(
        columns=[
            "quality_of_sleep",
            "subject",
            "person_id",
            "gender",
            "occupation",
            "bmi_category",
            "sleep_disorder",
        ],
        errors="ignore",
    )

    numeric_cols = features.select_dtypes(include=["number"]).columns.tolist()
    X = features[numeric_cols]

    if len(np.unique(y)) < 2:
        clf = DummyClassifier(strategy="constant", constant=int(y.iloc[0]))
    else:
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )

    pipeline.fit(X, y)

    clf_step = pipeline.named_steps["clf"]
    if hasattr(clf_step, "predict_proba") and getattr(clf_step, "classes_", np.array([0])).size > 1:
        proba = pipeline.predict_proba(X)[:, 1]
    else:
        proba = np.full(len(y), float(y.iloc[0]))

    metrics = _classification_summary(y.values, proba)

    artefact = "model_edf_only.joblib"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / artefact)

    return ModelStats(
        name="EDF Only (Dummy when single class)",
        artefact=artefact,
        dataset=path.name,
        n_samples=len(y),
        n_features=X.shape[1],
        train_samples=len(y),
        test_samples=0,
        positive_cases=int(y.sum()),
        negative_cases=int((1 - y).sum()),
        metrics=metrics,
    )


def run(random_state: int = RANDOM_STATE) -> List[ModelStats]:
    kaggle_clean = REPORTS_DIR / "kaggle_clean_winsorized.csv"
    kaggle_augmented = REPORTS_DIR / "kaggle_augmented.csv"

    stats: List[ModelStats] = []

    stats.append(
        _train_kaggle_variant(
            data_path=kaggle_clean,
            artefact_name="model_kaggle_baseline.joblib",
            display_name="Kaggle Logistic Baseline",
            classifier=LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_state,
            ),
            random_state=random_state,
        )
    )

    augmented_stats = _train_kaggle_variant(
        data_path=kaggle_augmented,
        artefact_name="model_augmented_latest.joblib",
        display_name="Augmented Logistic",
        classifier=LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        ),
        random_state=random_state,
    )
    stats.append(augmented_stats)

    joblib.dump(
        joblib.load(MODELS_DIR / "model_augmented_latest.joblib"),
        MODELS_DIR / "sleep_quality_model.joblib",
    )

    stats.append(
        _train_kaggle_variant(
            data_path=kaggle_clean,
            artefact_name="model_kaggle_knn.joblib",
            display_name="Kaggle KNN (k=5)",
            classifier=KNeighborsClassifier(n_neighbors=5),
            random_state=random_state,
        )
    )

    stats.append(
        _train_kaggle_variant(
            data_path=kaggle_clean,
            artefact_name="model_kaggle_nb.joblib",
            display_name="Kaggle Gaussian NB",
            classifier=GaussianNB(),
            random_state=random_state,
        )
    )

    stats.append(_train_hybrid(random_state=random_state))
    stats.append(_train_edf_only())

    return stats


def main(random_state: int = RANDOM_STATE, save_metrics: bool = True) -> None:
    stats = run(random_state=random_state)

    if save_metrics:
        payload = [item.as_dict() for item in stats]
        with METRICS_PATH.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[OK] Metrics saved to {METRICS_PATH.relative_to(REPO_ROOT)}")

    for item in stats:
        print(f"\n=== {item.name} ({item.artefact}) ===")
        for key, value in item.metrics.items():
            if isinstance(value, float):
                print(f"{key:>28s}: {value:.4f}")
            else:
                print(f"{key:>28s}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the core sleep-quality models")
    parser.add_argument("--random_state", type=int, default=RANDOM_STATE, help="Random seed for splits")
    parser.add_argument(
        "--no-save",
        dest="save_metrics",
        action="store_false",
        help="Skip writing metrics JSON (prints to stdout only)",
    )

    args = parser.parse_args()
    main(random_state=args.random_state, save_metrics=args.save_metrics)


