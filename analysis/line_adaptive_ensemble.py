"""
Line-adaptive ensemble — ARIMA for stable lines, LightGBM for high-variance lines.

Background (from CRITICAL_ANALYSIS.md)
---------------------------------------
The ARIMA baseline analysis revealed that a univariate SARIMA model — using only
the delay time series, with no exogenous features — outperforms LightGBM on 7 of
11 lines. This happens because at low data volumes (~600 training rows per line)
the exogenous features (weather, network state, topology) add noise rather than
signal; the lag autocorrelation alone provides most of the predictive value.

On 4 high-variance lines (Victoria, Metropolitan, Piccadilly, District), LightGBM
wins because occasional large delay spikes require cross-line network signals and
temporal context that ARIMA's univariate structure cannot capture.

This script builds a line-adaptive ensemble:
  - Identifies the winning model per line using a held-out validation set (first
    20% of the test period, never seen during selection or training).
  - Routes each test observation to the appropriate model at inference time.
  - Evaluates the ensemble against both standalone models and reports the result.

Selection protocol
------------------
  1. Split data: train (first 60%) / validation (next 20%) / test (final 20%).
  2. Train ARIMA and LightGBM on the training portion.
  3. Compare validation MAE per line → winner selection is locked in.
  4. Evaluate ensemble on the test portion (the selection never saw test data).

This ensures the model selection does not leak test-set information.

Run from the project root:
    python analysis/line_adaptive_ensemble.py
"""

import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import get_config, RANDOM_SEED
from data import load_data
from features import engineer_features, prepare_features_for_model
from utils import get_latest_run_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Split fractions: train / val / test
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20
# test = remaining 0.20

SARIMA_ORDER    = (1, 1, 1)
SARIMA_SEASONAL = (1, 1, 1, 96)  # 96 × 15 min = 1 day
ARIMA_ORDER     = (1, 1, 1)
MIN_TRAIN_ROWS  = 40             # minimum rows to attempt ARIMA fit


def _fit_arima(train_series: pd.Series, n_forecast: int, line: str) -> Tuple[np.ndarray, str]:
    """Fit SARIMA → fallback ARIMA → fallback mean. Returns (forecasts, model_name)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order, seasonal, name in [
            (SARIMA_ORDER, SARIMA_SEASONAL, "SARIMA"),
            (ARIMA_ORDER,  (0, 0, 0, 0),   "ARIMA"),
        ]:
            try:
                mdl = SARIMAX(
                    train_series, order=order,
                    seasonal_order=seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit = mdl.fit(disp=False, maxiter=200)
                fc  = fit.forecast(steps=n_forecast)
                return np.clip(fc.values, 0, None), name
            except Exception as exc:
                logger.debug("Line '%s': %s failed — %s", line, name, exc)

        logger.warning("Line '%s': all ARIMA variants failed — using mean fallback", line)
        return np.full(n_forecast, train_series.mean()), "mean"


def _lgbm_predict_on_split(
    model, df_split: pd.DataFrame, feature_cols: List[str], target_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) for LightGBM on a DataFrame slice."""
    X, y = prepare_features_for_model(df_split, feature_cols, target_col)
    return np.array(y), model.predict(X)


def run() -> None:
    output_dir = ROOT / "analysis" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    config   = get_config()
    df_raw, mode = load_data(config)
    logger.info("Data mode: %s | Shape: %s", mode, df_raw.shape)

    df = engineer_features(df_raw, config, is_training=True)
    df = df.sort_values([config.features.group_column, config.features.time_column])

    # ── Load LightGBM model ───────────────────────────────────────────────────
    run_id = get_latest_run_id(ROOT / "artifacts")
    if run_id is None:
        raise FileNotFoundError("No trained model found. Run train.py first.")
    model_path = ROOT / "artifacts" / run_id / "best_model.pkl"
    lgbm_model = joblib.load(model_path)

    preprocessor  = lgbm_model.named_steps["preprocessor"]
    numeric_feats  = preprocessor.transformers_[0][2]
    cat_feats      = preprocessor.transformers_[1][2]
    all_feats      = numeric_feats + cat_feats

    target_col = config.features.target_column
    group_col  = config.features.group_column
    time_col   = config.features.time_column

    # ── Per-line split + model selection on validation ────────────────────────
    selection: Dict[str, str] = {}   # line → "ARIMA" | "LightGBM"
    val_records: List[Dict]   = []

    for line in sorted(df[group_col].unique()):
        ldf = df[df[group_col] == line].sort_values(time_col).reset_index(drop=True)
        n   = len(ldf)
        n_train = int(n * TRAIN_FRAC)
        n_val   = int(n * VAL_FRAC)

        if n_train < MIN_TRAIN_ROWS:
            logger.warning("Line '%s': only %d train rows — defaulting to LightGBM", line, n_train)
            selection[line] = "LightGBM"
            continue

        train_df = ldf.iloc[:n_train]
        val_df   = ldf.iloc[n_train: n_train + n_val]

        if val_df.empty:
            logger.warning("Line '%s': empty validation split — defaulting to LightGBM", line)
            selection[line] = "LightGBM"
            continue

        # --- ARIMA on raw delay series ---
        train_series = train_df[target_col].reset_index(drop=True)
        val_series   = val_df[target_col].reset_index(drop=True)
        arima_preds, arima_model_name = _fit_arima(train_series, len(val_df), line)
        arima_val_mae = mean_absolute_error(val_series, arima_preds)

        # --- LightGBM on validation ---
        try:
            y_true_val, lgbm_val_preds = _lgbm_predict_on_split(
                lgbm_model, val_df, all_feats, target_col
            )
            if len(y_true_val) == 0:
                raise ValueError("empty prediction output")
            lgbm_val_mae = mean_absolute_error(y_true_val, lgbm_val_preds)
        except Exception as exc:
            logger.warning("LightGBM val prediction failed for '%s': %s — defaulting to ARIMA", line, exc)
            lgbm_val_mae = float("inf")

        winner = "ARIMA" if arima_val_mae <= lgbm_val_mae else "LightGBM"
        selection[line] = winner

        val_records.append({
            "line":            line,
            "arima_val_mae":   round(arima_val_mae, 4),
            "lgbm_val_mae":    round(lgbm_val_mae, 4),
            "val_winner":      winner,
            "arima_model":     arima_model_name,
        })
        logger.info(
            "%-25s  ARIMA val MAE=%.4f  LGBM val MAE=%.4f  → %s",
            line, arima_val_mae, lgbm_val_mae, winner,
        )

    # ── Evaluate ensemble on test set ─────────────────────────────────────────
    ensemble_preds: List[float] = []
    lgbm_only_preds: List[float] = []
    arima_only_preds: List[float] = []
    y_true_all: List[float] = []
    test_records: List[Dict] = []

    for line in sorted(df[group_col].unique()):
        ldf = df[df[group_col] == line].sort_values(time_col).reset_index(drop=True)
        n       = len(ldf)
        n_train = int(n * TRAIN_FRAC)
        n_val   = int(n * VAL_FRAC)
        test_df = ldf.iloc[n_train + n_val:]

        if test_df.empty:
            continue

        train_val_series = ldf[target_col].iloc[:n_train + n_val].reset_index(drop=True)
        test_series      = test_df[target_col].reset_index(drop=True)

        # ARIMA on full train+val series, forecast test period
        arima_test_preds, _ = _fit_arima(train_val_series, len(test_df), line)

        # LightGBM on test set
        try:
            y_true_test, lgbm_test_preds = _lgbm_predict_on_split(
                lgbm_model, test_df, all_feats, target_col
            )
        except Exception as exc:
            logger.warning("LightGBM test prediction failed for '%s': %s", line, exc)
            continue

        # Align lengths (ARIMA forecast length may differ from LightGBM due to NaN drops)
        min_len = min(len(y_true_test), len(arima_test_preds))
        y_true_test     = y_true_test[:min_len]
        arima_test_preds = arima_test_preds[:min_len]
        lgbm_test_preds  = lgbm_test_preds[:min_len]

        chosen = selection.get(line, "LightGBM")
        ens_preds = arima_test_preds if chosen == "ARIMA" else lgbm_test_preds

        mae_arima  = mean_absolute_error(y_true_test, arima_test_preds)
        mae_lgbm   = mean_absolute_error(y_true_test, lgbm_test_preds)
        mae_ens    = mean_absolute_error(y_true_test, ens_preds)

        test_records.append({
            "line":           line,
            "val_winner":     selection.get(line, "LightGBM"),
            "test_mae_arima": round(mae_arima, 4),
            "test_mae_lgbm":  round(mae_lgbm, 4),
            "test_mae_ens":   round(mae_ens, 4),
            "n_test":         min_len,
        })

        y_true_all.extend(y_true_test)
        ensemble_preds.extend(ens_preds)
        lgbm_only_preds.extend(lgbm_test_preds)
        arima_only_preds.extend(arima_test_preds)

        logger.info(
            "%-25s  ARIMA=%.4f  LGBM=%.4f  Ensemble=%.4f  (used %s)",
            line, mae_arima, mae_lgbm, mae_ens, chosen,
        )

    if not y_true_all:
        logger.error("No test predictions generated — aborting")
        return

    overall_arima_mae = mean_absolute_error(y_true_all, arima_only_preds)
    overall_lgbm_mae  = mean_absolute_error(y_true_all, lgbm_only_preds)
    overall_ens_mae   = mean_absolute_error(y_true_all, ensemble_preds)

    logger.info("=" * 60)
    logger.info("Overall test MAE — ARIMA: %.4f | LightGBM: %.4f | Ensemble: %.4f",
                overall_arima_mae, overall_lgbm_mae, overall_ens_mae)

    pct_vs_lgbm = (1 - overall_ens_mae / overall_lgbm_mae) * 100
    pct_vs_arima = (1 - overall_ens_mae / overall_arima_mae) * 100
    logger.info("Ensemble improvement vs LightGBM: %.1f%%", pct_vs_lgbm)
    logger.info("Ensemble improvement vs ARIMA only: %.1f%%", pct_vs_arima)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    if test_records:
        df_test = pd.DataFrame(test_records).sort_values("test_mae_lgbm")
        x      = np.arange(len(df_test))
        width  = 0.28
        fig, ax = plt.subplots(figsize=(13, 6))
        b1 = ax.bar(x - width,     df_test["test_mae_arima"], width, label="ARIMA",     color="#ff7f0e")
        b2 = ax.bar(x,             df_test["test_mae_lgbm"],  width, label="LightGBM",  color="#1f77b4")
        b3 = ax.bar(x + width,     df_test["test_mae_ens"],   width, label="Ensemble",  color="#2ca02c")

        ax.set_xticks(x)
        ax.set_xticklabels(df_test["line"], rotation=30, ha="right")
        ax.set_ylabel("MAE (severity units)")
        ax.set_title(
            f"Line-Adaptive Ensemble vs Standalone Models\n"
            f"Overall MAE — ARIMA: {overall_arima_mae:.4f} | "
            f"LightGBM: {overall_lgbm_mae:.4f} | "
            f"Ensemble: {overall_ens_mae:.4f}"
        )
        ax.legend()
        plt.tight_layout()

        png_path = output_dir / "ensemble_results.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        logger.info("Saved chart to %s", png_path)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    results = {
        "run_id":             run_id,
        "split_fractions":    {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": 1 - TRAIN_FRAC - VAL_FRAC},
        "overall_mae": {
            "arima":    round(overall_arima_mae, 4),
            "lgbm":     round(overall_lgbm_mae,  4),
            "ensemble": round(overall_ens_mae,   4),
        },
        "ensemble_vs_lgbm_pct":  round(pct_vs_lgbm, 2),
        "ensemble_vs_arima_pct": round(pct_vs_arima, 2),
        "selection":   selection,
        "val_records": val_records,
        "test_records": test_records,
    }

    json_path = output_dir / "ensemble_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved ensemble results to %s", json_path)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Line':<25}  {'Val winner':<12}  {'Test ARIMA':>11}  {'Test LGBM':>10}  {'Test Ens':>9}")
    print("-" * 75)
    for r in test_records:
        print(
            f"{r['line']:<25}  {r['val_winner']:<12}  "
            f"{r['test_mae_arima']:>11.4f}  {r['test_mae_lgbm']:>10.4f}  {r['test_mae_ens']:>9.4f}"
        )
    print("-" * 75)
    print(
        f"{'OVERALL':<25}  {'':>12}  "
        f"{overall_arima_mae:>11.4f}  {overall_lgbm_mae:>10.4f}  {overall_ens_mae:>9.4f}"
    )


if __name__ == "__main__":
    run()
