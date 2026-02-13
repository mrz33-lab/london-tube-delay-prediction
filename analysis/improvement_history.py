"""
"""
Compares the original 21-feature model against the extended 37-feature
model on an identical temporal test set.

The original feature set has basic temporal flags, raw weather readings,
lag/rolling delay stats, and line identity. The extended set adds network
effects, cyclical encodings, line topology metadata, and train-frequency
features.

Trains a fresh LightGBM with the same hyperparameters on each feature set
and reports the MAE delta.

Run from the project root:
    python analysis/improvement_history.py
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
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import get_config, RANDOM_SEED
from data import load_data, get_train_test_split
from features import engineer_features, prepare_features_for_model, get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Same hyperparameters as saved model so performance delta reflects features only
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

# Feature list pulled from the saved model rather than hard-coded
def _get_original_features() -> list:
    """Get the original 21-feature list from the saved model's preprocessor."""
    model_path = ROOT / "artifacts" / "run_20260210_153030" / "best_model.pkl"
    model = joblib.load(model_path)
    preprocessor  = model.named_steps["preprocessor"]
    numeric_feats  = preprocessor.transformers_[0][2]
    cat_feats      = preprocessor.transformers_[1][2]
    return numeric_feats + cat_feats  # 20 numeric + ['line']


def _build_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
) -> tuple:
    """Build, fit and evaluate a fresh LightGBM pipeline."""
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if cat_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols,
        ))

    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer(transformers)),
        ("regressor",    lgb.LGBMRegressor(**LGBM_PARAMS)),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = r2_score(y_test, y_pred)
    return mae, rmse, r2


def run() -> None:
    """Compare reduced (21-feature) vs full (37-feature) model."""
    output_dir = ROOT / "analysis" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_config()
    df, mode = load_data(config)
    logger.info("Data mode: %s | Shape: %s", mode, df.shape)

    # All 37 features are engineered upfront; the two model variants differ only
    # in which subset of those columns they are allowed to observe.
    df = engineer_features(df, config, is_training=True)
    train_df, test_df = get_train_test_split(df, config)

    original_feats = _get_original_features()
    logger.info("Original feature set (%d): %s", len(original_feats), original_feats)

    _, _, full_feats = get_feature_columns(train_df, config)
    logger.info("Full feature set (%d)", len(full_feats))

    # Check that all original features still exist in the current pipeline
    missing = [f for f in original_feats if f not in full_feats]
    if missing:
        logger.warning("Original features missing from current pipeline: %s", missing)
        original_feats = [f for f in original_feats if f in full_feats]

    target = config.features.target_column

    # --- Reduced model (original 21 features) ---
    X_train_red, y_train = prepare_features_for_model(train_df, original_feats, target)
    X_test_red,  y_test  = prepare_features_for_model(test_df,  original_feats, target)

    logger.info("Training reduced model (%d features)...", len(original_feats))
    red_mae, red_rmse, red_r2 = _build_and_evaluate(
        X_train_red, y_train, X_test_red, y_test
    )
    logger.info("Reduced  → MAE=%.3f  RMSE=%.3f  R²=%.3f", red_mae, red_rmse, red_r2)

    # --- Full model (all 37 features) ---
    X_train_full, y_train_full = prepare_features_for_model(train_df, full_feats, target)
    X_test_full,  y_test_full  = prepare_features_for_model(test_df,  full_feats, target)

    logger.info("Training full model (%d features)...", len(full_feats))
    full_mae, full_rmse, full_r2 = _build_and_evaluate(
        X_train_full, y_train_full, X_test_full, y_test_full
    )
    logger.info("Full     → MAE=%.3f  RMSE=%.3f  R²=%.3f", full_mae, full_rmse, full_r2)

    # Published metrics from the saved run are included as a third reference point.
    saved_mae  = 2.01
    saved_rmse = 4.70
    saved_r2   = 0.194

    delta_from_reduced = full_mae - red_mae  # negative = full model is better
    pct_improvement    = -100.0 * delta_from_reduced / red_mae

    # --- Markdown table ---
    print("\n| Feature Set | N Features | MAE | RMSE | R2 | Delta MAE vs Reduced |")
    print("|-------------|------------|-----|------|----|----------------------|")
    print(
        f"| Reduced (original) | {len(original_feats)} | "
        f"{red_mae:.3f} | {red_rmse:.3f} | {red_r2:.3f} | 0.000 |"
    )
    sign = "+" if delta_from_reduced >= 0 else ""
    print(
        f"| Full (extended) | {len(full_feats)} | "
        f"{full_mae:.3f} | {full_rmse:.3f} | {full_r2:.3f} | "
        f"{sign}{delta_from_reduced:.3f} |"
    )
    print(
        f"| Saved model (published) | {len(original_feats)} | "
        f"{saved_mae:.3f} | {saved_rmse:.3f} | {saved_r2:.3f} | n/a |"
    )

    print(f"\n**Feature expansion improvement: {pct_improvement:+.1f}% MAE reduction**")

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [
        f"Reduced\n({len(original_feats)} features)",
        f"Full\n({len(full_feats)} features)",
    ]
    maes   = [red_mae, full_mae]
    colours = ["#ff7f0e", "#1f77b4"]

    bars = ax.bar(labels, maes, color=colours, width=0.4, edgecolor="white")
    for bar, mae in zip(bars, maes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mae:.3f} min",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    if pct_improvement > 0:
        ax.annotate(
            f"−{pct_improvement:.1f}% MAE",
            xy=(1, full_mae),
            xytext=(0.5, max(maes) * 1.05),
            ha="center",
            fontsize=10,
            color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4"),
        )

    ax.set_ylabel("Test MAE (minutes)")
    ax.set_title(
        "Feature Engineering Impact: 21 → 37 Features\n"
        "(same LightGBM hyperparameters, identical train/test split)"
    )
    ax.set_ylim(0, max(maes) * 1.25)
    plt.tight_layout()

    out_path = output_dir / "improvement_history.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved chart to %s", out_path)

    # Each extended feature was motivated by a specific operational hypothesis
    # and added only after the 21-feature baseline was established.


if __name__ == "__main__":
    run()
