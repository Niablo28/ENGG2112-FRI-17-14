import numpy as np
import pandas as pd

def _bin_sleep_duration(X):
    """Return three indicator columns for sleep duration buckets: <6, 6-9, >9 hours.
    X: 2D numpy array with one column (sleep_duration).
    """
    sd = X[:, 0]
    lt6 = (sd < 6.0).astype(int)
    between = ((sd >= 6.0) & (sd <= 9.0)).astype(int)
    gt9 = (sd > 9.0).astype(int)
    return np.c_[lt6, between, gt9]


def _split_bp(df: pd.DataFrame) -> pd.DataFrame:
    """Split blood pressure string (e.g., '126/83') into numeric systolic and diastolic columns.
    This function is intentionally defined in a shared module so that pickled
    pipelines can import it reliably at inference time.
    """
    out = df.copy()
    if "blood_pressure" in out.columns:
        bp = out["blood_pressure"].astype(str).str.extract(r'(?P<bp_sys>\d{2,3})/(?P<bp_dia>\d{2,3})').astype(float)
        out = out.drop(columns=["blood_pressure"]).join(bp)
    return out


def _get_bp_split_feature_names_out(transformer, input_features):
    """Provide feature names after _split_bp: drop 'blood_pressure', append 'bp_sys','bp_dia'."""
    features = list(input_features)
    if 'blood_pressure' in features:
        features.remove('blood_pressure')
        features.append('bp_sys')
        features.append('bp_dia')
    return np.array(features, dtype=object)


