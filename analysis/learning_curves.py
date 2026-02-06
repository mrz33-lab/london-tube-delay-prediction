"""
I generate learning curves to understand how model performance scales with
training data size and whether collecting more data would improve accuracy.

I train LightGBM on increasing proportions of the training set, recording
train MAE, validation MAE, and wall-clock time at each increment.  I plot
MAE on the left y-axis and training time on the right to give an at-a-glance
cost/benefit view of additional data collection.

Bias-variance diagnosis: if train MAE and val MAE both plateau early and remain
close together at full training size, the model is in a high-bias (underfitting)
regime and more data will not help — I should add features or increase model
capacity.  If val MAE is substantially higher than train MAE at full size, the
model is in a high-variance (overfitting) regime and more data WOULD help.

Run from the project root:
    python analysis/learning_curves.py
"""

import sys
import logging
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error

from config import get_config, RANDOM_SEED
from data import load_data, get_train_test_split
from features import engineer_features, prepare_features_for_model, get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Production hyperparameters are matched exactly so any observed bias/variance
# behaviour reflects the deployed configuration rather than an experimental one.
LGBM_PARAMS = dict(
    max_depth=7,
    n_estimators=50,
    num_leaves=15,
    reg_alpha=0.1,
    reg_lambda=1.0,
    subsample=0.7,
    random_state=RANDOM_SEED,
    verbosity=-1,
    force_col_wise=True,
)

PROPORTIONS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def _build_pipeline(numeric_cols: list, cat_cols: list) -> Pipeline:
    """I construct a stateless preprocessing + LightGBM pipeline for a single training run."""
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if cat_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols,
        ))
    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers)),
        ("regressor",    lgb.LGBMRegressor(**LGBM_PARAMS)),
    ])


def run() -> None:
    """I run the full learning-curve experiment and emit a bias/variance diagnosis."""
    output_dir = ROOT / "analysis" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_config()
    df, mode = load_data(config)
    logger.info("Data mode: %s | Shape: %s", mode, df.shape)

    df = engineer_features(df, config, is_training=True)
    train_df, test_df = get_train_test_split(df, config)
    numeric_feats, cat_feats, all_feats = get_feature_columns(train_df, config)

    X_train_full, y_train_full = prepare_features_for_model(
        train_df, all_feats, config.features.target_column
    )
    X_test, y_test = prepare_features_for_model(
        test_df, all_feats, config.features.target_column
    )

    logger.info(
        "Full training set: %d rows | Test set: %d rows",
        len(X_train_full), len(X_test),
    )

    records = []
    for prop in PROPORTIONS:
        n = max(10, int(len(X_train_full) * prop))  # floor at 10 to ensure a valid fit
        X_tr = X_train_full.iloc[:n]
        y_tr = y_train_full.iloc[:n]

        logger.info("Training on %.0f%% of data (%d rows)...", prop * 100, n)

        t0       = time.time()
        pipeline = _build_pipeline(numeric_feats, cat_feats)
        pipeline.fit(X_tr, y_tr)
        elapsed  = time.time() - t0

        train_mae = mean_absolute_error(y_tr,    pipeline.predict(X_tr))
        val_mae   = mean_absolute_error(y_test,  pipeline.predict(X_test))

        records.append({
            "proportion":   prop,
            "n_train":      n,
            "train_mae":    train_mae,
            "val_mae":      val_mae,
            "train_time_s": elapsed,
        })
        logger.info(
            "  prop=%.2f  n=%d  train_MAE=%.3f  val_MAE=%.3f  time=%.2fs",
            prop, n, train_mae, val_mae, elapsed,
        )

    results = pd.DataFrame(records)

    # I define convergence as the first proportion at which val MAE improvement
    # drops below 0.01 min between consecutive increments.
    convergence_idx = None
    for i in range(1, len(results)):
        delta = results["val_mae"].iloc[i - 1] - results["val_mae"].iloc[i]
        if abs(delta) < 0.01:
            convergence_idx = i
            break
    conv_prop = results["proportion"].iloc[convergence_idx] if convergence_idx else None

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(
        results["proportion"] * 100, results["train_mae"],
        marker="o", color="#1f77b4", label="Train MAE",
    )
    ax1.plot(
        results["proportion"] * 100, results["val_mae"],
        marker="s", color="#d62728", label="Validation MAE",
    )

    if conv_prop is not None:
        ax1.axvline(
            conv_prop * 100, color="grey", linestyle=":", linewidth=1.5,
            label=f"Approx. convergence (~{int(conv_prop*100)}%)",
        )

    ax1.set_xlabel("Training data used (%)")
    ax1.set_ylabel("MAE (minutes)")
    ax1.legend(loc="upper right")
    ax1.set_title("Learning Curves — LightGBM\n(same hyperparameters as production model)")

    # Training time is overlaid on a secondary axis to provide cost/benefit context.
    ax2 = ax1.twinx()
    ax2.plot(
        results["proportion"] * 100, results["train_time_s"],
        marker="^", color="#2ca02c", linestyle="--", alpha=0.6, label="Train time (s)",
    )
    ax2.set_ylabel("Training time (seconds)", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.legend(loc="center right")

    plt.tight_layout()
    out_path = output_dir / "learning_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved chart to %s", out_path)

    # --- Bias / variance diagnosis ---
    final  = results.iloc[-1]
    gap    = final["val_mae"] - final["train_mae"]
    logger.info(
        "At full training size: train_MAE=%.3f  val_MAE=%.3f  gap=%.3f",
        final["train_mae"], final["val_mae"], gap,
    )

    # The diagnosis is logged so it appears in the run record and can be
    # cited directly in the dissertation without re-running the script.
    if gap > 0.5:
        diagnosis = (
            "HIGH VARIANCE (overfitting): validation MAE is substantially higher than "
            "train MAE at full data size.  Collecting more data would likely improve "
            "generalisation.  Alternatively, increase regularisation or reduce model capacity."
        )
    elif final["val_mae"] > 3.0:
        diagnosis = (
            "HIGH BIAS (underfitting): both train and val MAE are elevated at full data "
            "size.  More data alone will not help — the model needs richer features or "
            "a higher-capacity architecture."
        )
    else:
        diagnosis = (
            "BALANCED regime: train and val MAE are close and relatively low at full "
            "data size.  The model is neither heavily over- nor underfitting.  Marginal "
            "gains from additional data are expected to be small."
        )
    print("\n## Bias-variance diagnosis")
    print(diagnosis)

    # --- Print table ---
    print("\n| Proportion | N train | Train MAE | Val MAE | Train time (s) |")
    print("|------------|---------|-----------|---------|----------------|")
    for _, row in results.iterrows():
        print(
            f"| {row['proportion']:.0%} | {int(row['n_train'])} | "
            f"{row['train_mae']:.3f} | {row['val_mae']:.3f} | "
            f"{row['train_time_s']:.2f} |"
        )


if __name__ == "__main__":
    run()
