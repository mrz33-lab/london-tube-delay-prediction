"""
"""
Ablation study to measure each feature group's contribution to
LightGBM performance.

Trains a fresh model with each group removed in turn and records
the MAE delta vs the full-feature model.

Feature groups (derived from the 37-column engineered feature set):
  temporal     — hour/day/month flags + cyclical encodings
  weather      — temperature, humidity, precipitation and their interactions
  network      — leave-one-out cross-line disruption signals
  historical   — lag delays and rolling statistics
  line_metadata — line identity, topology and capacity metadata

Run from the project root:
    python analysis/ablation_study.py
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import time

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
from features import engineer_features, prepare_features_for_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Hyperparameters from the saved model so every ablation uses the same capacity.
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

# Features are partitioned into five semantically coherent groups.
# crowding_index and interaction terms belong to line_metadata because
# they characterise operational load rather than meteorology or time.
# 'line' is placed in line_metadata so its removal answers the question:
# how much predictive value does line identity alone contribute?
FEATURE_GROUPS: Dict[str, List[str]] = {
    "temporal": [
        "hour", "day_of_week", "month", "is_weekend", "is_holiday", "peak_time",
        "hour_sin", "hour_cos", "is_late_night", "is_early_morning",
    ],
    "weather": [
        "temp_c", "precipitation_mm", "humidity",
        "temp_delta_1h", "precipitation_delta_1h", "precipitation_x_temp",
    ],
    "network": [
        "network_avg_delay", "network_delay_volatility",
        "lines_disrupted_ratio", "is_network_wide_disruption",
    ],
    "historical": [
        "lag_delay_1", "lag_delay_3",
        "rolling_mean_delay_3", "rolling_mean_delay_12", "rolling_std_delay_12",
        "recent_disruption_rate",
    ],
    "line_metadata": [
        "line",  # removing this tests how much line identity alone contributes
        "line_length_km", "n_stations", "n_interchange_stations",
        "is_deep_tube", "zone_coverage",
        "trains_per_hour", "service_headway_min", "capacity_pressure",
        "crowding_index", "crowding_x_peak",
    ],
}


def _build_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[float, float, float]:
    """Build a fresh LightGBM pipeline, fit it, and return test MAE/RMSE/R²."""
    numeric_cols     = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_cols,
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
    """Run the full ablation experiment and save results."""
    output_dir = ROOT / "analysis" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Prepare data ---
    config = get_config()
    df, mode = load_data(config)
    logger.info("Data mode: %s | Shape: %s", mode, df.shape)

    df = engineer_features(df, config, is_training=True)
    train_df, test_df = get_train_test_split(df, config)

    # Collect every feature assigned to a group, plus any extras not explicitly grouped
    all_group_features = [f for grp in FEATURE_GROUPS.values() for f in grp]
    target = config.features.target_column
    exclude = set(config.features.exclude_columns + [target])
    all_available = [c for c in train_df.columns if c not in exclude]
    unassigned = [c for c in all_available if c not in all_group_features]
    if unassigned:
        logger.info("Unassigned features (added to full model only): %s", unassigned)

    full_features = all_available

    X_train_full, y_train = prepare_features_for_model(train_df, full_features, target)
    X_test_full,  y_test  = prepare_features_for_model(test_df,  full_features, target)

    # --- Naive mean baseline ---
    logger.info("Training naive mean baseline...")
    naive_pred = np.full(len(y_test), y_train.mean())
    naive_mae  = mean_absolute_error(y_test, naive_pred)
    logger.info("Naive MAE: %.3f", naive_mae)

    # --- Full model ---
    logger.info("Training full-feature model...")
    t0 = time.time()
    full_mae, full_rmse, full_r2 = _build_and_evaluate(
        X_train_full, y_train, X_test_full, y_test
    )
    logger.info("Full model MAE: %.3f  RMSE: %.3f  R²: %.3f  [%.1fs]",
                full_mae, full_rmse, full_r2, time.time() - t0)

    # --- Ablation: remove one group at a time ---
    records = []
    for group_name, group_cols in FEATURE_GROUPS.items():
        # Skip groups that reference columns not in the current data
        cols_to_remove = [c for c in group_cols if c in full_features]
        if not cols_to_remove:
            logger.warning("Group '%s' has no matching columns — skipping", group_name)
            continue

        reduced = [c for c in full_features if c not in cols_to_remove]
        logger.info(
            "Ablating '%s' (%d features removed, %d remaining)...",
            group_name, len(cols_to_remove), len(reduced),
        )

        X_tr = X_train_full[reduced]
        X_te = X_test_full[reduced]

        t0  = time.time()
        mae, rmse, r2 = _build_and_evaluate(X_tr, y_train, X_te, y_test)
        elapsed = time.time() - t0

        delta     = mae - full_mae
        pct_change = 100.0 * delta / full_mae
        records.append({
            "group":      group_name,
            "mae":        mae,
            "delta":      delta,
            "pct_change": pct_change,
            "rmse":       rmse,
            "r2":         r2,
            "n_removed":  len(cols_to_remove),
            "elapsed_s":  elapsed,
        })
        logger.info(
            "  → MAE=%.3f  Δ=+%.3f (%.1f%%)  [%.1fs]",
            mae, delta, pct_change, elapsed,
        )

    results = pd.DataFrame(records).sort_values("delta", ascending=False)

    # --- Markdown table ---
    print("\n| Removed Group | MAE | Delta vs Full Model | % Change |")
    print("|---------------|-----|---------------------|----------|")
    print(f"| *(naive mean baseline)* | {naive_mae:.3f} | "
          f"+{naive_mae - full_mae:.3f} | "
          f"+{100*(naive_mae - full_mae)/full_mae:.1f}% |")
    print(f"| *(full model - all features)* | {full_mae:.3f} | 0.000 | 0.0% |")
    for _, row in results.iterrows():
        sign = "+" if row["delta"] >= 0 else ""
        print(
            f"| {row['group']} | {row['mae']:.3f} | "
            f"{sign}{row['delta']:.3f} | {sign}{row['pct_change']:.1f}% |"
        )

    # --- Horizontal bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))

    colours = ["#d62728" if d > 0 else "#1f77b4" for d in results["delta"]]
    bars = ax.barh(results["group"], results["delta"], color=colours, edgecolor="white")

    for bar, delta in zip(bars, results["delta"]):
        sign = "+" if delta >= 0 else ""
        ax.text(
            bar.get_width() + (0.005 if delta >= 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{delta:.3f} min",
            va="center", ha="left" if delta >= 0 else "right", fontsize=9,
        )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("MAE delta vs full model (minutes)\nPositive = group was helpful")
    ax.set_title(
        "Ablation Study — Feature Group Contribution\n"
        "(Full model MAE: {:.3f} min)".format(full_mae)
    )
    plt.tight_layout()

    out_path = output_dir / "ablation_results.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved chart to %s", out_path)


if __name__ == "__main__":
    run()
