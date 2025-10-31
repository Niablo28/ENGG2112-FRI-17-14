"""
Calibration evaluation using OOF probabilities.
Outputs: ECE, Brier score, and reliability diagram.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        p_bin = y_prob[mask]
        y_bin = y_true[mask]
        conf = p_bin.mean()
        acc = y_bin.mean()
        ece += p_bin.size * abs(acc - conf)
    return ece / y_true.size


def main(proba_csv: Path, out_prefix: Path, n_bins: int = 10):
    df = pd.read_csv(proba_csv)
    y = df["y_true"].to_numpy()
    p = df["y_prob"].to_numpy()

    # Metrics
    ece = expected_calibration_error(y, p, n_bins=n_bins)
    brier = brier_score_loss(y, p)

    metrics_path = out_prefix.with_suffix("")  # drop .png if present
    metrics_csv = Path(str(metrics_path) + "_calibration_metrics.csv")
    pd.DataFrame({"metric": ["ECE", "Brier"], "value": [ece, brier]}).to_csv(metrics_csv, index=False)

    # Reliability diagram
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy='uniform')
    plt.figure(figsize=(4.5, 4.5))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.plot(mean_pred, frac_pos, marker='o', linewidth=2, label='Model')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Reliability Diagram (ECE={ece:.3f}, Brier={brier:.3f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    ap.add_argument("--proba_csv", default=str(repo_root / "reports" / "oof_probs_logreg.csv"))
    ap.add_argument("--out_png", default=str(repo_root / "reports" / "calibration_reliability.png"))
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()
    main(Path(args.proba_csv), Path(args.out_png), n_bins=args.bins)


