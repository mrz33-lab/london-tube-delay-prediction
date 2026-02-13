"""
"""
Per-line ARIMA/SARIMA baseline comparison against LightGBM.

ARIMA uses only the delay time series with no exogenous features, so
it tests whether weather/network/temporal features actually add value
beyond simple delay autocorrelation.

Attempts SARIMA(1,1,1)(1,1,1,24) first and falls back to ARIMA(1,1,1)
if convergence fails.

Run from the project root:
    python analysis/arima_baseline.py
"""

import sys
import subprocess
import logging
from pathlib import Path
import warnings

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Install statsmodels on demand so the script works in fresh environments
try:
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    logger_bootstrap = logging.getLogger("bootstrap")
    logger_bootstrap.warning("statsmodels not found — installing via pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import get_config
from data import load_data, get_train_test_split
from features import engineer_features, prepare_features_for_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MIN_SAMPLES = 50          # Minimum training observations required for a stable ARIMA fit.
SARIMA_ORDER    = (1, 1, 1)
SARIMA_SEASONAL = (1, 1, 1, 24)
ARIMA_ORDER     = (1, 1, 1)


def _fit_arima_line(
    train_series: pd.Series,
    test_series:  pd.Series,
    line:         str,
) -> tuple:
    """Try SARIMA, fall back to ARIMA for a single line. Returns (y_pred, model_name)."""
    # Convergence warnings are suppressed here to keep output readable;
    # failures are surfaced through the logging system instead.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = SARIMAX(
                train_series,
                order=SARIMA_ORDER,
                seasonal_order=SARIMA_SEASONAL,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=200)
            if not fit.mle_retvals.get("converged", True):
                raise RuntimeError("SARIMA did not converge")

            forecast = fit.forecast(steps=len(test_series))
            return np.clip(forecast.values, 0, None), "SARIMA"

        except Exception as sarima_exc:
            logger.warning(
                "Line '%s': SARIMA failed (%s) — falling back to ARIMA(1,1,1)",
                line, sarima_exc,
            )
            try:
                model = SARIMAX(
                    train_series,
                    order=ARIMA_ORDER,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit = model.fit(disp=False, maxiter=200)
                forecast = fit.forecast(steps=len(test_series))
                return np.clip(forecast.values, 0, None), "ARIMA"
            except Exception as arima_exc:
                logger.error(
                    "Line '%s': ARIMA also failed (%s) — using mean prediction",
                    line, arima_exc,
                )
                return np.full(len(test_series), train_series.mean()), "mean"


def run() -> None:
    """Run ARIMA vs LightGBM comparison and save results."""
    output_dir = ROOT / "analysis" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data and replicate training split ---
    config = get_config()
    df, mode = load_data(config)
    logger.info("Data mode: %s | Shape: %s", mode, df.shape)

    # Feature engineering just to reproduce the temporal split boundary;
    # ARIMA operates on the raw delay series.
    df_eng = engineer_features(df, config, is_training=True)
    train_df, test_df = get_train_test_split(df_eng, config)

    split_ts = test_df["timestamp"].min()
    logger.info("Split boundary timestamp: %s", split_ts)

    # --- Load LightGBM model for comparison ---
    model_path = ROOT / "artifacts" / "run_20260210_153030" / "best_model.pkl"
    lgbm_model = joblib.load(model_path)
    preprocessor  = lgbm_model.named_steps["preprocessor"]
    numeric_feats  = preprocessor.transformers_[0][2]
    cat_feats      = preprocessor.transformers_[1][2]
    all_feats      = numeric_feats + cat_feats

    X_test_lgbm, y_test_lgbm = prepare_features_for_model(
        test_df, all_feats, config.features.target_column
    )
    X_test_lgbm["_line"] = test_df.loc[X_test_lgbm.index, "line"].values
    lgbm_preds_all = lgbm_model.predict(X_test_lgbm.drop(columns=["_line"]))
    X_test_lgbm["_lgbm_pred"] = lgbm_preds_all
    X_test_lgbm["_true"]      = y_test_lgbm.values

    lines   = sorted(df["line"].unique())
    records = []

    for line in lines:
        # Raw (un-engineered) delay series per line in chronological order
        line_df    = df[df["line"] == line].sort_values("timestamp")
        all_delays = line_df["delay_minutes"].values
        n_train    = int(len(line_df) * config.models.train_ratio)

        train_series = pd.Series(all_delays[:n_train])
        test_series  = pd.Series(all_delays[n_train:])

        if len(train_series) < MIN_SAMPLES:
            logger.warning(
                "Line '%s' has only %d training samples (< %d) — skipping ARIMA",
                line, len(train_series), MIN_SAMPLES,
            )
            continue

        logger.info(
            "Line '%s': fitting SARIMA on %d train / %d test obs...",
            line, len(train_series), len(test_series),
        )

        arima_preds, model_used = _fit_arima_line(train_series, test_series, line)

        arima_mae  = mean_absolute_error(test_series, arima_preds)
        arima_rmse = float(np.sqrt(mean_squared_error(test_series, arima_preds)))

        # LightGBM metrics on the same test rows for fair comparison
        line_mask     = X_test_lgbm["_line"] == line
        lgbm_true     = X_test_lgbm.loc[line_mask, "_true"].values
        lgbm_pred_line = X_test_lgbm.loc[line_mask, "_lgbm_pred"].values

        if len(lgbm_true) == 0:
            lgbm_mae = lgbm_rmse = float("nan")
        else:
            lgbm_mae  = mean_absolute_error(lgbm_true, lgbm_pred_line)
            lgbm_rmse = float(np.sqrt(mean_squared_error(lgbm_true, lgbm_pred_line)))

        improvement = lgbm_mae - arima_mae  # negative when LightGBM wins

        records.append({
            "line":        line,
            "arima_mae":   arima_mae,
            "arima_rmse":  arima_rmse,
            "lgbm_mae":    lgbm_mae,
            "lgbm_rmse":   lgbm_rmse,
            "improvement": improvement,
            "model_used":  model_used,
        })
        logger.info(
            "%-25s  ARIMA MAE=%.3f  LightGBM MAE=%.3f  Δ=%.3f  [%s]",
            line, arima_mae, lgbm_mae, improvement, model_used,
        )

    results = pd.DataFrame(records)

    # --- Markdown table ---
    print("\n| Line | ARIMA MAE | LightGBM MAE | Improvement | Model used |")
    print("|------|-----------|--------------|-------------|------------|")
    for _, row in results.iterrows():
        lgbm_str = f"{row['lgbm_mae']:.3f}" if not np.isnan(row["lgbm_mae"]) else "N/A"
        impr_str = f"{row['improvement']:+.3f}" if not np.isnan(row["improvement"]) else "N/A"
        print(
            f"| {row['line']} | {row['arima_mae']:.3f} | "
            f"{lgbm_str} | {impr_str} | {row['model_used']} |"
        )

    # Flag any lines that needed the ARIMA(1,1,1) fallback
    fallback_lines = results[results["model_used"] != "SARIMA"]["line"].tolist()
    if fallback_lines:
        print(f"\n*ARIMA(1,1,1) fallback used for: {', '.join(fallback_lines)}*")
    else:
        print("\n*All lines used SARIMA(1,1,1)(1,1,1,24) successfully.*")

    # --- Grouped bar chart ---
    valid = results.dropna(subset=["lgbm_mae"])
    x     = np.arange(len(valid))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width / 2, valid["arima_mae"], width, label="ARIMA/SARIMA", color="#ff7f0e")
    ax.bar(x + width / 2, valid["lgbm_mae"],  width, label="LightGBM",     color="#1f77b4")

    ax.set_xticks(x)
    ax.set_xticklabels(valid["line"], rotation=30, ha="right")
    ax.set_ylabel("MAE (minutes)")
    ax.set_title("ARIMA vs LightGBM MAE per Tube Line")
    ax.legend()
    plt.tight_layout()

    out_path = output_dir / "arima_vs_lgbm.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved chart to %s", out_path)


if __name__ == "__main__":
    run()
