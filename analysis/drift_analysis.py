"""Data drift analysis — time-windowed MAE to detect model accuracy degradation.

Groups test-set predictions into time windows and computes MAE per window.
A monotonically rising MAE trend suggests distributional drift: the data
generating process has shifted away from the training distribution.

Drift is measured two ways:
  1. Spearman rank correlation between window index and MAE (monotonic trend).
  2. Relative MAE increase: (last-window MAE) / (first-window MAE) - 1.

Run from the project root:
    python analysis/drift_analysis.py
"""

import sys
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from utils import get_latest_run_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def run() -> None:
    """Load test predictions and analyse MAE trend over time."""
    output_dir = ROOT / "analysis" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load predictions ─────────────────────────────────────────────────────
    run_id = get_latest_run_id(ROOT / "artifacts")
    if run_id is None:
        raise FileNotFoundError("No training run found. Run train.py first.")

    preds_path = ROOT / "artifacts" / run_id / "test_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"test_predictions.csv not found: {preds_path}")

    df = pd.read_csv(preds_path, parse_dates=["timestamp"])
    logger.info("Loaded %d predictions from run %s", len(df), run_id)

    # ── Choose window granularity based on data span ──────────────────────────
    df = df.sort_values("timestamp").reset_index(drop=True)
    span_days = (df["timestamp"].max() - df["timestamp"].min()).days

    if span_days >= 14:
        df["window"] = df["timestamp"].dt.to_period("W")
        window_label = "weekly"
    elif span_days >= 4:
        df["window"] = df["timestamp"].dt.to_period("D")
        window_label = "daily"
    else:
        # Fewer than 4 days: split into 8 equal-width buckets
        df["window"] = pd.cut(df.index, bins=8, labels=False)
        window_label = "equal-width"

    # ── Compute per-window metrics ────────────────────────────────────────────
    records = []
    for window, grp in df.groupby("window", sort=True):
        y_true = grp["actual"].values
        mae_best  = _mae(y_true, grp["pred_best"].values)
        mae_naive = _mae(y_true, grp["pred_naive"].values)
        records.append({
            "window":       str(window),
            "window_start": grp["timestamp"].min().isoformat(),
            "n":            int(len(grp)),
            "mae_best":     round(mae_best,  4),
            "mae_naive":    round(mae_naive, 4),
        })

    results = pd.DataFrame(records)
    logger.info(
        "Per-%s MAE:\n%s", window_label,
        results[["window", "n", "mae_best", "mae_naive"]].to_string(index=False),
    )

    # ── Drift statistics ─────────────────────────────────────────────────────
    x = np.arange(len(results))
    rho, p_val = spearmanr(x, results["mae_best"])

    first_mae = results["mae_best"].iloc[0]
    last_mae  = results["mae_best"].iloc[-1]
    relative_increase = (last_mae / first_mae - 1) if first_mae > 0 else 0.0

    drift_detected = bool(p_val < 0.05 and rho > 0)
    interpretation = (
        f"Positive monotonic trend detected (rho={rho:.3f}, p={p_val:.4f}) — "
        f"MAE grew {relative_increase*100:.1f}% from first to last window. "
        "Possible distributional drift."
        if drift_detected
        else
        f"No significant upward trend (rho={rho:.3f}, p={p_val:.4f}) — "
        "model accuracy is stable across the test period."
    )
    logger.info("Drift test: %s", interpretation)

    trend_summary = {
        "window_type":      window_label,
        "n_windows":        int(len(results)),
        "spearman_rho":     round(float(rho), 4),
        "p_value":          round(float(p_val), 6),
        "relative_increase": round(float(relative_increase), 4),
        "drift_detected":   drift_detected,
        "interpretation":   interpretation,
    }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "run_id":  run_id,
        "windows": records,
        "trend":   trend_summary,
    }
    json_path = output_dir / "drift_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved drift results → %s", json_path)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, results["mae_best"],  marker="o", linewidth=2,
            label="LightGBM MAE", color="#003688")
    ax.plot(x, results["mae_naive"], marker="s", linewidth=1.5, linestyle="--",
            label="Naive MAE", color="#d62728", alpha=0.7)

    # OLS trend line for visual reference
    z = np.polyfit(x, results["mae_best"], 1)
    ax.plot(x, np.poly1d(z)(x), linestyle=":", color="#888888",
            linewidth=1.5, label=f"Trend (slope={z[0]:+.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels(results["window"], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel(f"Time Window ({window_label})")
    ax.set_ylabel("Mean Absolute Error (minutes)")
    ax.set_title(
        f"Model MAE Over Time — Drift Analysis ({run_id})\n"
        f"Spearman rho={rho:.3f}  p={p_val:.4f}  "
        f"{'DRIFT DETECTED' if drift_detected else 'No drift'}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / "drift_mae_over_time.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Saved drift plot → %s", plot_path)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\nDrift analysis — {run_id}")
    print(f"  Window type : {window_label} ({len(results)} windows)")
    print(f"  Spearman rho: {rho:+.4f}  p-value: {p_val:.4f}")
    print(f"  MAE change  : {first_mae:.3f} -> {last_mae:.3f} "
          f"({relative_increase*100:+.1f}%)")
    print(f"  Verdict     : {interpretation}")


if __name__ == "__main__":
    run()
