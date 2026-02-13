"""
I evaluate how well the model's 95% confidence intervals are calibrated.

The current CI approach (as implemented in train.py) computes per-line
empirical residual quantiles from the test set and stores them in
feature_metadata.pkl.  At inference time, FutureDelayPredictor looks up the
q025/q975 residuals for the requested line and adds them to the point
prediction.  For the saved run_20260210_153030 model, residual_quantiles were
not yet stored, so I compute them here from the training data.

I evaluate the achieved coverage at six nominal confidence levels
[0.50, 0.60, 0.70, 0.80, 0.90, 0.95] and plot a calibration curve showing
whether the intervals are over- or under-confident.

Two concrete improvements I suggest in comments below:
  1. Conformal prediction — splits calibration and test sets to give
     distribution-free coverage guarantees without distributional assumptions.
  2. Bootstrapped prediction intervals — resamples the training set many times
     to build an empirical distribution of predictions, giving wider and more
     honest intervals for extrapolation regions.

Run from the project root:
    python analysis/confidence_interval_calibration.py
"""

import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import get_config
from data import load_data, get_train_test_split
from features import engineer_features, prepare_features_for_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CONFIDENCE_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def _coverage(y_true: np.ndarray, y_pred: np.ndarray,
              residuals_train: np.ndarray, level: float) -> float:
    """
    I compute empirical coverage for a given nominal confidence level.

    Interval boundaries are derived from the corresponding quantiles of the
    training residuals, mirroring a production system where interval widths
    are calibrated on training data and evaluated on unseen observations.

    Args:
        y_true:          True test target values.
        y_pred:          Point predictions on the test set.
        residuals_train: Residuals (y_true - y_pred) on the training set.
        level:           Nominal confidence level (e.g. 0.95).

    Returns:
        Fraction of test samples whose true value falls within the interval.
    """
    alpha      = 1.0 - level
    q_lower    = np.percentile(residuals_train, 100 * alpha / 2)
    q_upper    = np.percentile(residuals_train, 100 * (1 - alpha / 2))

    lb = y_pred + q_lower
    ub = y_pred + q_upper

    covered = np.mean((y_true >= lb) & (y_true <= ub))
    return float(covered)


def run() -> None:
    """I run the full calibration evaluation and persist the results."""
    output_dir = ROOT / "analysis" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data and replicate training split ---
    config = get_config()
    df, mode = load_data(config)
    logger.info("Data mode: %s | Shape: %s", mode, df.shape)

    df = engineer_features(df, config, is_training=True)
    train_df, test_df = get_train_test_split(df, config)

    # --- Load model ---
    model_path = ROOT / "artifacts" / "run_20260210_153030" / "best_model.pkl"
    model = joblib.load(model_path)
    logger.info("Loaded model from %s", model_path)

    # I extract the feature list directly from the saved model's ColumnTransformer
    # to avoid any risk of divergence from the training configuration.
    preprocessor  = model.named_steps["preprocessor"]
    numeric_feats  = preprocessor.transformers_[0][2]
    cat_feats      = preprocessor.transformers_[1][2]
    all_feats      = numeric_feats + cat_feats

    X_train, y_train = prepare_features_for_model(
        train_df, all_feats, config.features.target_column
    )
    X_test, y_test = prepare_features_for_model(
        test_df, all_feats, config.features.target_column
    )

    # --- Compute residuals on training set ---
    y_pred_train = model.predict(X_train)
    train_residuals = y_train.values - y_pred_train

    logger.info(
        "Training residuals: mean=%.3f  std=%.3f  q2.5=%.3f  q97.5=%.3f",
        train_residuals.mean(),
        train_residuals.std(),
        np.percentile(train_residuals, 2.5),
        np.percentile(train_residuals, 97.5),
    )

    # --- Compute test predictions ---
    y_pred_test = model.predict(X_test)
    y_true_test = y_test.values

    # --- Evaluate coverage at each confidence level ---
    records = []
    for level in CONFIDENCE_LEVELS:
        achieved = _coverage(y_true_test, y_pred_test, train_residuals, level)
        records.append({"nominal": level, "achieved": achieved})
        logger.info(
            "Nominal %.2f → Achieved coverage %.4f (%s-confident)",
            level, achieved,
            "OVER" if achieved < level else "UNDER",
        )

    results = pd.DataFrame(records)

    # --- Determine overall calibration direction ---
    mean_gap = (results["achieved"] - results["nominal"]).mean()
    if mean_gap < -0.05:
        direction = (
            "OVER-CONFIDENT: actual coverage is consistently below the nominal level. "
            "The intervals are too narrow — more true values fall outside them than expected."
        )
    elif mean_gap > 0.05:
        direction = (
            "UNDER-CONFIDENT: actual coverage exceeds the nominal level. "
            "The intervals are too wide — conservative but suboptimal for operational use."
        )
    else:
        direction = (
            "WELL-CALIBRATED: actual coverage is close to the nominal level across "
            "all confidence levels (mean gap = {:.3f}).".format(mean_gap)
        )
    print("\n## Calibration verdict")
    print(direction)

    # --- Calibration suggestions ---
    # Improvement 1: Conformal prediction intervals
    # Rather than using training residuals, a held-out calibration split would
    # yield non-conformity scores (|y - ŷ|) whose (1-α)(1+1/n) empirical
    # quantile serves as the interval half-width, guaranteeing marginal
    # coverage without distributional assumptions about residuals.
    #
    # Improvement 2: Bootstrapped prediction intervals
    # Resampling the training set B=500 times and aggregating the resulting
    # prediction distribution would capture both model uncertainty (variance
    # across bootstraps) and aleatoric noise (residual variance), producing
    # wider and more honest intervals in sparse or extrapolation regions.

    # --- Print table ---
    print("\n| Nominal level | Achieved coverage | Gap |")
    print("|---------------|-------------------|-----|")
    for _, row in results.iterrows():
        gap  = row["achieved"] - row["nominal"]
        sign = "+" if gap >= 0 else ""
        print(
            f"| {row['nominal']:.2f} | {row['achieved']:.4f} | "
            f"{sign}{gap:.4f} |"
        )

    # --- Calibration plot ---
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(
        results["nominal"], results["achieved"],
        marker="o", color="#d62728", linewidth=2, label="Actual coverage",
    )

    ax.fill_between(
        results["nominal"], results["achieved"], results["nominal"],
        where=results["achieved"] < results["nominal"],
        alpha=0.15, color="#d62728", label="Over-confident region",
    )
    ax.fill_between(
        results["nominal"], results["achieved"], results["nominal"],
        where=results["achieved"] >= results["nominal"],
        alpha=0.15, color="#1f77b4", label="Under-confident region",
    )

    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Nominal confidence level")
    ax.set_ylabel("Achieved coverage")
    ax.set_title("Confidence Interval Calibration\n(residuals from training set)")
    ax.legend()
    plt.tight_layout()

    out_path = output_dir / "ci_calibration.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved chart to %s", out_path)


if __name__ == "__main__":
    run()
