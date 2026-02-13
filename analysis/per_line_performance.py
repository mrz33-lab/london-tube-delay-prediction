"""
"""
Per-line prediction performance analysis.

Replicates the 80/20 temporal split from training, then computes
MAE, RMSE, and R² for each tube line on the test set.

Run from the project root:
    python analysis/per_line_performance.py
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import get_config
from data import load_data, get_train_test_split
from features import engineer_features, prepare_features_for_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Results are reported against this published baseline so the reader can
# immediately identify which lines beat or miss the overall target.
OVERALL_BASELINE_MAE = 2.01

# Each note captures the structural characteristic that drives prediction
# difficulty, as documented in TfL reports and academic literature.
LINE_NOTES = {
    "Central":            "complex: branching eastern / western arms, 49 stations",
    "Northern":           "complex: two southern branches merge at Camden Town",
    "District":           "complex: three western branches (Wimbledon/Ealing/Richmond)",
    "Circle":             "moderate: no terminus; disruptions recirculate around the loop",
    "Hammersmith & City": "moderate: shared infrastructure with Circle and Metropolitan",
    "Metropolitan":       "moderate: longest line, semi-fast services mix with stopping",
    "Jubilee":            "moderate: signalled for high frequency; disruptions cascade fast",
    "Piccadilly":         "moderate: long route with Heathrow branch causing schedule variance",
    "Bakerloo":           "simple: single trunk route, 25 stations, limited branching",
    "Victoria":           "simple: straight trunk, highest frequency, delays clear quickly",
    "Waterloo & City":    "simplest: 2-station shuttle, operates only peak hours",
}


def run() -> None:
    """Replicate train/test split and compute per-line metrics."""
    output_dir = ROOT / "analysis" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Same preprocessing as training so test features match exactly
    config = get_config()
    df, mode = load_data(config)
    logger.info("Data mode: %s, shape: %s", mode, df.shape)

    # is_training=True reproduces the exact lag/rolling features from the original run
    df = engineer_features(df, config, is_training=True)
    train_df, test_df = get_train_test_split(df, config)
    logger.info("Train rows: %d | Test rows: %d", len(train_df), len(test_df))

    # --- Load the saved model ---
    model_path = ROOT / "artifacts" / "run_20260210_153030" / "best_model.pkl"
    model = joblib.load(model_path)
    logger.info("Loaded model from %s", model_path)

    # The feature list is recovered by inspecting the saved model's
    # ColumnTransformer rather than hard-coding column names.
    preprocessor = model.named_steps["preprocessor"]
    numeric_feats = preprocessor.transformers_[0][2]   # 'num' branch
    cat_feats     = preprocessor.transformers_[1][2]   # 'cat' branch
    all_feats     = numeric_feats + cat_feats
    logger.info("Model expects %d features: %s", len(all_feats), all_feats)

    # Restrict to model's expected columns to avoid dimension mismatches
    X_test, y_test = prepare_features_for_model(
        test_df, all_feats, config.features.target_column
    )

    X_test_with_line = X_test.copy()
    if "line" not in X_test_with_line.columns:
        X_test_with_line["line"] = test_df.loc[X_test.index, "line"].values

    y_pred_all = model.predict(X_test)

    # --- Per-line metrics ---
    lines = sorted(X_test_with_line["line"].unique())
    records = []

    for line in lines:
        mask = X_test_with_line["line"] == line
        y_true_line = y_test[mask]
        y_pred_line = y_pred_all[mask]
        n = mask.sum()

        if n < 2:
            logger.warning("Line %s has only %d test samples — skipping R²", line, n)
            r2 = float("nan")
        else:
            r2 = r2_score(y_true_line, y_pred_line)

        records.append({
            "line":      line,
            "mae":       mean_absolute_error(y_true_line, y_pred_line),
            "rmse":      np.sqrt(mean_squared_error(y_true_line, y_pred_line)),
            "r2":        r2,
            "n_samples": int(n),
        })
        logger.info(
            "%-25s  MAE=%.3f  RMSE=%.3f  R²=%.3f  N=%d",
            line, records[-1]["mae"], records[-1]["rmse"], r2, n,
        )

    results = pd.DataFrame(records).sort_values("mae").reset_index(drop=True)

    # --- Markdown table ---
    print("\n| Line | MAE | RMSE | R² | N samples |")
    print("|------|-----|------|-----|-----------|")
    for _, row in results.iterrows():
        r2_str = f"{row['r2']:.3f}" if not np.isnan(row["r2"]) else "N/A"
        print(
            f"| {row['line']} | {row['mae']:.3f} | {row['rmse']:.3f} | "
            f"{r2_str} | {row['n_samples']} |"
        )

    # --- Horizontal bar chart ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # Bars below the overall MAE are coloured blue; those above are red,
    # providing an immediate visual signal of which lines underperform.
    colours = [
        "#d62728" if mae > OVERALL_BASELINE_MAE else "#1f77b4"
        for mae in results["mae"]
    ]
    bars = ax.barh(results["line"], results["mae"], color=colours, edgecolor="white")

    for bar, mae in zip(bars, results["mae"]):
        ax.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{mae:.2f}", va="center", fontsize=9,
        )

    ax.axvline(
        OVERALL_BASELINE_MAE, color="black", linestyle="--", linewidth=1.5,
        label=f"Overall MAE ({OVERALL_BASELINE_MAE} min)",
    )

    ax.set_xlabel("Mean Absolute Error (minutes)")
    ax.set_title("Per-Line MAE on Test Set — LightGBM (run_20260210_153030)")
    ax.legend(loc="lower right")
    plt.tight_layout()

    out_path = output_dir / "per_line_mae.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved chart to %s", out_path)

    # The Central and Northern lines are expected to show the highest MAE:
    # both have branching geometries that introduce schedule variance that
    # propagates along routes in ways difficult to capture in a single feature.
    # The Waterloo & City shuttle, by contrast, is expected to show the lowest
    # MAE — its delay pattern is almost entirely determined by off-peak closure
    # and the residual variance is well-explained by peak_time and crowding.
    print("\n## Structural difficulty notes")
    for _, row in results.iterrows():
        note = LINE_NOTES.get(row["line"], "")
        print(f"- **{row['line']}** (MAE={row['mae']:.2f}): {note}")


if __name__ == "__main__":
    run()
