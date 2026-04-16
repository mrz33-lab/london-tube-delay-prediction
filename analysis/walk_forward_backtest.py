"""
Walk-forward backtesting for the London Underground delay prediction model.

Simulates real operational conditions: the model is trained only on data that
would have been available *before* each evaluation window, then tested on the
next unseen window.  This differs from a single train/test split by proving
the model generalises across multiple, non-overlapping future periods rather
than just the final 20%.

Protocol (expanding-window):
  1.  Sort the full dataset chronologically.
  2.  Use the first ``initial_train_frac`` of rows as the seed training set.
  3.  Divide the remaining rows into ``n_folds`` equal-length test windows.
  4.  For fold k:  train on all data before window k, test on window k.
  5.  Record MAE, RMSE, R² and the timestamp range of each fold.
  6.  Save results to JSON and a diagnostic plot to PNG.

Run from the project root:
    python analysis/walk_forward_backtest.py
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import get_config, RANDOM_SEED
from data import load_data, generate_synthetic_data
from features import engineer_features, prepare_features_for_model
from utils import get_latest_run_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Backtesting hyper-parameters ────────────────────────────────────────────
INITIAL_TRAIN_FRAC: float = 0.50   # seed window: first 50 % of all data
N_FOLDS:            int   = 5      # number of forward evaluation windows

# Fallback LGBM params used when no trained model is found in artifacts
_LGBM_FALLBACK = dict(
    max_depth=7,
    n_estimators=100,
    num_leaves=31,
    learning_rate=0.05,
    reg_alpha=0.1,
    reg_lambda=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    verbosity=-1,
    force_col_wise=True,
)

_LGBM_KEEP = {
    "num_leaves", "max_depth", "learning_rate", "n_estimators",
    "min_child_samples", "subsample", "colsample_bytree",
    "reg_alpha", "reg_lambda",
}


def _load_lgbm_params() -> dict:
    """Re-use Optuna-tuned params from the latest training run when available."""
    run_id = get_latest_run_id(ROOT / "artifacts")
    if run_id is None:
        logger.warning("No training artifacts found — using fallback LightGBM params")
        return dict(_LGBM_FALLBACK)

    model_path = ROOT / "artifacts" / run_id / "best_model.pkl"
    try:
        import joblib
        model = joblib.load(model_path)
        reg = model.named_steps.get("regressor")
        if reg is not None and hasattr(reg, "get_params"):
            raw = reg.get_params()
            params = {k: v for k, v in raw.items() if k in _LGBM_KEEP}
            params.update(random_state=RANDOM_SEED, verbosity=-1, force_col_wise=True)
            logger.info("Loaded LGBM params from %s", run_id)
            return params
    except Exception as exc:
        logger.warning("Could not load model params (%s) — using fallback", exc)

    return dict(_LGBM_FALLBACK)


def _build_pipeline(params: dict, X: pd.DataFrame) -> Pipeline:
    """Build a fresh LightGBM sklearn Pipeline for the given feature matrix."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols,
        ))

    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers, remainder="drop")),
        ("regressor",    lgb.LGBMRegressor(**params)),
    ])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return mae, rmse, r2


def run(
    initial_train_frac: float = INITIAL_TRAIN_FRAC,
    n_folds: int = N_FOLDS,
) -> List[Dict]:
    """Execute walk-forward backtesting and return per-fold result dicts."""
    output_dir = ROOT / "analysis" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and engineer features ────────────────────────────────────────────
    config = get_config()
    df, data_mode = load_data(config)
    logger.info("Data mode: %s | Shape: %s", data_mode, df.shape)

    df = engineer_features(df, config, is_training=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Full engineered dataset: %d rows (%s → %s)",
                len(df), df["timestamp"].min(), df["timestamp"].max())

    # ── Feature columns ───────────────────────────────────────────────────────
    target  = config.features.target_column
    exclude = set(config.features.exclude_columns + [target, "timestamp"])
    all_features = [c for c in df.columns if c not in exclude]
    logger.info("Feature count: %d", len(all_features))

    lgbm_params = _load_lgbm_params()

    # ── Build fold boundaries ─────────────────────────────────────────────────
    n = len(df)
    seed_end = int(n * initial_train_frac)   # last row of the initial training set
    remaining = n - seed_end                  # rows available for evaluation
    fold_size = remaining // n_folds

    if fold_size < 50:
        logger.warning(
            "Fold size is very small (%d rows). Consider reducing n_folds or "
            "using a larger dataset for reliable backtest estimates.", fold_size
        )

    # ── Walk-forward evaluation ───────────────────────────────────────────────
    fold_records: List[Dict] = []

    for fold in range(n_folds):
        test_start = seed_end + fold * fold_size
        # Last fold absorbs any remainder so we never waste rows
        test_end   = (seed_end + (fold + 1) * fold_size
                      if fold < n_folds - 1 else n)

        train_df = df.iloc[:test_start]
        test_df  = df.iloc[test_start:test_end]

        X_train, y_train = prepare_features_for_model(train_df, all_features, target)
        X_test,  y_test  = prepare_features_for_model(test_df,  all_features, target)

        ts_start = test_df["timestamp"].min()
        ts_end   = test_df["timestamp"].max()

        logger.info(
            "Fold %d/%d | train rows=%d | test rows=%d | test window=%s → %s",
            fold + 1, n_folds, len(train_df), len(test_df),
            ts_start.date(), ts_end.date(),
        )

        t0 = time.time()
        pipeline = _build_pipeline(lgbm_params, X_train)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Naive baseline for this fold: mean of the training target
        naive_pred = np.full(len(y_test), float(y_train.mean()))

        mae,  rmse,  r2   = _metrics(y_test.values, y_pred)
        nmae, nrmse, nr2  = _metrics(y_test.values, naive_pred)
        elapsed = time.time() - t0

        improvement = (1 - mae / nmae) * 100 if nmae > 0 else 0.0

        record = {
            "fold":          fold + 1,
            "train_rows":    int(len(train_df)),
            "test_rows":     int(len(test_df)),
            "test_start":    ts_start.isoformat(),
            "test_end":      ts_end.isoformat(),
            "mae":           round(mae,  4),
            "rmse":          round(rmse, 4),
            "r2":            round(r2,   4),
            "naive_mae":     round(nmae, 4),
            "improvement_pct": round(improvement, 2),
            "elapsed_s":     round(elapsed, 2),
        }
        fold_records.append(record)

        logger.info(
            "  → MAE=%.4f  RMSE=%.4f  R²=%.4f  Naive MAE=%.4f  "
            "Improvement=%.1f%%  [%.1fs]",
            mae, rmse, r2, nmae, improvement, elapsed,
        )

    # ── Summary statistics ────────────────────────────────────────────────────
    maes  = [r["mae"]  for r in fold_records]
    rmses = [r["rmse"] for r in fold_records]
    r2s   = [r["r2"]   for r in fold_records]
    imps  = [r["improvement_pct"] for r in fold_records]

    summary = {
        "n_folds":              n_folds,
        "initial_train_frac":   initial_train_frac,
        "data_mode":            data_mode,
        "mae_mean":             round(float(np.mean(maes)),  4),
        "mae_std":              round(float(np.std(maes)),   4),
        "mae_min":              round(float(np.min(maes)),   4),
        "mae_max":              round(float(np.max(maes)),   4),
        "rmse_mean":            round(float(np.mean(rmses)), 4),
        "r2_mean":              round(float(np.mean(r2s)),   4),
        "improvement_pct_mean": round(float(np.mean(imps)),  2),
    }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {"folds": fold_records, "summary": summary}
    json_path = output_dir / "walk_forward_backtest.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved backtest results → %s", json_path)

    # ── Diagnostic plot ───────────────────────────────────────────────────────
    x = np.arange(1, n_folds + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: MAE per fold with naive baseline
    ax = axes[0]
    ax.bar(x - 0.18, maes,                  width=0.35, color="#003688", label="LightGBM MAE")
    ax.bar(x + 0.18, [r["naive_mae"] for r in fold_records],
                                             width=0.35, color="#d62728", alpha=0.65,
                                             label="Naive Baseline MAE")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in x])
    ax.set_ylabel("MAE (severity units)")
    ax.set_title("Walk-Forward MAE per Fold\nvs Naive Baseline")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Annotate MAE values
    for i, (m, nm) in enumerate(zip(maes, [r["naive_mae"] for r in fold_records])):
        ax.text(i + 1 - 0.18, m + 0.01, f"{m:.3f}", ha="center", va="bottom",
                fontsize=8, color="#003688", fontweight="bold")
        ax.text(i + 1 + 0.18, nm + 0.01, f"{nm:.3f}", ha="center", va="bottom",
                fontsize=8, color="#d62728")

    # Right panel: improvement % and R²
    ax2 = axes[1]
    colour_bars = ["#2ca02c" if v >= 0 else "#d62728" for v in imps]
    ax2.bar(x, imps, color=colour_bars, width=0.5, label="MAE improvement over naive (%)")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Fold {i}" for i in x])
    ax2.set_ylabel("Improvement over Naive (%)")
    ax2.set_title(
        f"Walk-Forward Improvement per Fold\n"
        f"Mean MAE: {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}  "
        f"|  Mean R²: {summary['r2_mean']:.4f}"
    )
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(imps):
        ax2.text(i + 1, v + (0.5 if v >= 0 else -1.5), f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle(
        f"Walk-Forward Backtesting  —  {n_folds} folds  "
        f"(initial train: {initial_train_frac*100:.0f}% of data)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plot_path = output_dir / "walk_forward_backtest.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Saved backtest plot → %s", plot_path)

    # ── Console summary table ─────────────────────────────────────────────────
    print(f"\nWalk-Forward Backtest — {n_folds} folds  "
          f"(initial train: {initial_train_frac*100:.0f}%)")
    print("-" * 72)
    print(f"{'Fold':<6} {'Test window':<24} {'Rows':>6} {'MAE':>8} "
          f"{'RMSE':>8} {'R²':>7} {'vs Naive':>9}")
    print("-" * 72)
    for r in fold_records:
        ts = r["test_start"][:10]
        te = r["test_end"][:10]
        print(f"{r['fold']:<6} {ts+' → '+te:<24} {r['test_rows']:>6} "
              f"{r['mae']:>8.4f} {r['rmse']:>8.4f} {r['r2']:>7.4f} "
              f"{r['improvement_pct']:>+8.1f}%")
    print("-" * 72)
    print(f"{'Mean':<31} {summary['mae_mean']:>8.4f} "
          f"{summary['rmse_mean']:>8.4f} {summary['r2_mean']:>7.4f} "
          f"{summary['improvement_pct_mean']:>+8.1f}%")
    print(f"{'Std':<31} {summary['mae_std']:>8.4f}")

    return fold_records


if __name__ == "__main__":
    run()
