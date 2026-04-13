"""
Inference lag-feature gap analysis.

Quantifies the MAE penalty incurred when a cold-start inference call must use
LINE_BASE_DELAYS defaults instead of computed lag/rolling features.

Usage
-----
    python analysis/inference_lag_gap.py [--artifact-dir artifacts/<run>]

If --artifact-dir is omitted the most-recently modified artifacts/ subdirectory
is used (matching the behaviour of the other analysis scripts).

Outputs
-------
    analysis/outputs/inference_lag_gap.json   — per-line + overall MAE deltas
    analysis/outputs/inference_lag_gap.png    — bar chart
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from line_metadata import LINE_BASE_DELAYS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Features whose NaN means "no prior history" — the ones that default to
# LINE_BASE_DELAYS at cold-start inference.
_LAG_PREFIXES = ("lag_", "rolling_mean_delay_", "rolling_std_delay_")
_SCALAR_LAG_FEATURES = ("recent_disruption_rate",)
_RECENT_DISRUPTION_DEFAULT = 0.2   # matches FutureDelayPredictor default


# ---------------------------------------------------------------------------

def _latest_artifact_dir() -> Path:
    base = ROOT / "artifacts"
    if not base.is_dir():
        raise FileNotFoundError(f"artifacts/ directory not found at {base}")
    candidates = sorted(
        [p for p in base.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No subdirectories found inside artifacts/")
    return candidates[0]


def _load_artifacts(artifact_dir: Path):
    model_path = artifact_dir / "best_model.pkl"
    metadata_path = artifact_dir / "feature_metadata.pkl"
    x_test_path = artifact_dir / "X_test.parquet"
    y_test_path = artifact_dir / "y_test.parquet"

    for p in (model_path, metadata_path, x_test_path, y_test_path):
        if not p.exists():
            raise FileNotFoundError(f"Required artifact missing: {p}")

    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    X_test = pd.read_parquet(x_test_path)
    y_test = pd.read_parquet(y_test_path).squeeze()

    return model, metadata, X_test, y_test


def _inject_cold_start_defaults(
    X: pd.DataFrame, metadata: dict, line_col: str = "line"
) -> pd.DataFrame:
    """Return a copy of X with lag/rolling features replaced by cold-start defaults.

    This mirrors the logic in FutureDelayPredictor._engineer_features() when
    recent_delays is None — i.e. the first prediction after deployment.
    """
    X_cold = X.copy()

    lag_cols = [
        c for c in X_cold.columns
        if any(c.startswith(p) for p in _LAG_PREFIXES)
        or c in _SCALAR_LAG_FEATURES
    ]

    for col in lag_cols:
        if line_col in X_cold.columns:
            # Use per-line baseline where available (matches FutureDelayPredictor)
            X_cold[col] = X_cold[line_col].map(
                lambda ln: LINE_BASE_DELAYS.get(ln, 3.0)
            )
        else:
            X_cold[col] = 3.0  # global fallback

    # recent_disruption_rate has a separate default
    if "recent_disruption_rate" in X_cold.columns:
        X_cold["recent_disruption_rate"] = _RECENT_DISRUPTION_DEFAULT

    return X_cold


def _mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


# ---------------------------------------------------------------------------

def run(artifact_dir: Path) -> dict:
    log.info("Loading artifacts from %s", artifact_dir)
    model, metadata, X_test, y_test = _load_artifacts(artifact_dir)

    log.info("Test-set shape: X=%s  y=%s", X_test.shape, y_test.shape)

    # Align index
    common_idx = X_test.index.intersection(y_test.index)
    X_test = X_test.loc[common_idx]
    y_test = y_test.loc[common_idx]

    # --- Scenario A: correct history-backed features ---
    log.info("Predicting with history-backed features …")
    y_pred_history = model.predict(X_test)

    # --- Scenario B: cold-start defaults ---
    log.info("Injecting cold-start defaults …")
    X_cold = _inject_cold_start_defaults(X_test, metadata)
    log.info("Predicting with cold-start defaults …")
    y_pred_cold = model.predict(X_cold)

    # --- Overall MAE ---
    mae_history = _mae(y_test, y_pred_history)
    mae_cold    = _mae(y_test, y_pred_cold)
    delta_overall = mae_cold - mae_history

    log.info("Overall MAE — history: %.4f  cold-start: %.4f  Δ: %+.4f",
             mae_history, mae_cold, delta_overall)

    # --- Per-line MAE ---
    line_col = "line" if "line" in X_test.columns else None
    per_line: dict[str, dict] = {}

    if line_col:
        for line in sorted(X_test[line_col].unique()):
            mask = X_test[line_col] == line
            if mask.sum() < 10:
                continue
            m_hist  = _mae(y_test[mask], y_pred_history[mask])
            m_cold  = _mae(y_test[mask], y_pred_cold[mask])
            per_line[line] = {
                "mae_history":    round(m_hist, 4),
                "mae_cold_start": round(m_cold, 4),
                "delta":          round(m_cold - m_hist, 4),
                "n_samples":      int(mask.sum()),
            }
            log.info("  %-25s  hist=%.4f  cold=%.4f  Δ=%+.4f  (n=%d)",
                     line, m_hist, m_cold, m_cold - m_hist, mask.sum())

    results = {
        "artifact_dir": str(artifact_dir),
        "overall": {
            "mae_history":    round(mae_history, 4),
            "mae_cold_start": round(mae_cold, 4),
            "delta":          round(delta_overall, 4),
            "delta_pct":      round(100 * delta_overall / (mae_history + 1e-9), 2),
            "n_samples":      int(len(y_test)),
        },
        "per_line": per_line,
        "cold_start_features_replaced": [
            c for c in X_test.columns
            if any(c.startswith(p) for p in _LAG_PREFIXES)
            or c in _SCALAR_LAG_FEATURES
        ],
    }

    return results


def _save_outputs(results: dict) -> None:
    out_dir = ROOT / "analysis" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / "inference_lag_gap.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved JSON → %s", json_path)

    # Bar chart
    per_line = results["per_line"]
    if not per_line:
        log.warning("No per-line data — skipping chart")
        return

    lines   = list(per_line.keys())
    deltas  = [per_line[l]["delta"] for l in lines]
    overall = results["overall"]["delta"]

    fig, ax = plt.subplots(figsize=(max(8, len(lines) * 0.8), 5))

    colours = ["#d32f2f" if d > 0 else "#388e3c" for d in deltas]
    bars = ax.bar(lines, deltas, color=colours, edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(overall, color="#1565c0", linewidth=1.5, linestyle="--",
               label=f"Overall Δ MAE = {overall:+.3f} min")

    # Value labels
    for bar, d in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            d + (0.02 if d >= 0 else -0.04),
            f"{d:+.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel("Line")
    ax.set_ylabel("ΔMAE (cold-start − history-backed)  [minutes]")
    ax.set_title(
        "Cold-Start Inference Penalty: MAE increase when lag features\n"
        "use LINE_BASE_DELAYS defaults instead of computed history"
    )
    ax.legend(loc="upper right")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    img_path = out_dir / "inference_lag_gap.png"
    fig.savefig(img_path, dpi=150)
    plt.close(fig)
    log.info("Saved chart  → %s", img_path)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir", type=Path, default=None,
        help="Path to artifact run directory (default: most-recently modified)",
    )
    args = parser.parse_args()

    artifact_dir = args.artifact_dir or _latest_artifact_dir()
    results = run(artifact_dir)
    _save_outputs(results)

    overall = results["overall"]
    print(
        f"\nInference lag-feature gap summary\n"
        f"  History-backed MAE : {overall['mae_history']:.4f} min\n"
        f"  Cold-start MAE     : {overall['mae_cold_start']:.4f} min\n"
        f"  Delta MAE          : {overall['delta']:+.4f} min  "
        f"({overall['delta_pct']:+.1f}%)\n"
        f"  n samples          : {overall['n_samples']:,}\n"
    )


if __name__ == "__main__":
    main()
