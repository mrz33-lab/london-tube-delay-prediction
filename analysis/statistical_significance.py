"""
Statistical significance testing for model comparisons.

The model comparison table in train.py reports descriptive MAE/RMSE values
but provides no inferential evidence that differences are significant.  This
script applies Wilcoxon signed-rank tests (chosen over the paired t-test
because prediction errors on delay data are right-skewed and violate the
t-test's normality assumption) and reports p-values with effect sizes.

Tests performed:
  - Naive  vs Ridge    (one-sided: naive errors > ridge errors?)
  - Naive  vs LightGBM (one-sided: naive errors > lgbm errors?)
  - Ridge  vs LightGBM (one-sided: ridge errors > lgbm errors?)

Each test uses absolute errors |y_true - y_pred| as paired observations,
so the null hypothesis is H0: median(|e_A|) = median(|e_B|).

Effect sizes are reported as the matched-pairs rank-biserial correlation r,
which is interpretable as Cohen (1988): r ≈ 0.1 small, 0.3 medium, 0.5 large.

Run from the project root:
    python analysis/statistical_significance.py
"""

import sys
import json
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
from scipy.stats import wilcoxon, ttest_rel

from config import get_config
from data import load_data, get_train_test_split
from features import engineer_features, prepare_features_for_model
from utils import get_latest_run_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _rank_biserial_correlation(n: int, wilcoxon_stat: float) -> float:
    """Effect size r for Wilcoxon signed-rank test (matched-pairs).

    r = 1 - 4W / (n(n+1))  where W is the Wilcoxon test statistic.
    Ranges from -1 to +1; |r| ≈ 0.1 small, 0.3 medium, 0.5 large.
    """
    max_w = n * (n + 1) / 2
    return float(1 - (2 * wilcoxon_stat) / max_w) if max_w > 0 else 0.0


def run() -> None:
    """Load test-set predictions from the latest run and run significance tests."""
    output_dir = ROOT / "analysis" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = get_latest_run_id(ROOT / "artifacts")
    if run_id is None:
        raise FileNotFoundError("No trained model found in artifacts/. Run train.py first.")

    artifact_dir = ROOT / "artifacts" / run_id
    preds_csv = artifact_dir / "test_predictions.csv"
    if not preds_csv.exists():
        raise FileNotFoundError(f"test_predictions.csv not found in {artifact_dir}")

    preds_df = pd.read_csv(preds_csv)
    logger.info("Loaded %d test predictions from %s", len(preds_df), run_id)

    y_true = preds_df["actual"].values
    pred_cols = {
        "naive":  preds_df["pred_naive"].values,
        "ridge":  preds_df["pred_ridge"].values,
        "best":   preds_df["pred_best"].values,
    }

    abs_errors = {name: np.abs(y_true - preds) for name, preds in pred_cols.items()}

    pairs = [
        ("naive", "ridge"),
        ("naive", "best"),
        ("ridge", "best"),
    ]

    records = []

    print("\n## Pairwise Wilcoxon Signed-Rank Tests")
    print("H0: median(|e_A|) = median(|e_B|)  (one-sided: A errors > B errors?)")
    print()
    print(f"{'Comparison':<25} {'MAE_A':>7} {'MAE_B':>7} {'W stat':>10} {'p-value':>10} {'r':>6} {'sig?':>6}")
    print("-" * 80)

    for m1, m2 in pairs:
        if m1 not in abs_errors or m2 not in abs_errors:
            continue

        ae1, ae2 = abs_errors[m1], abs_errors[m2]
        mae1 = float(np.mean(ae1))
        mae2 = float(np.mean(ae2))
        n    = len(ae1)

        try:
            # one-sided: H1: m1 errors > m2 errors (i.e. m2 is better)
            w_stat, p_wilcoxon = wilcoxon(ae1, ae2, alternative="greater")
            r_effect = _rank_biserial_correlation(n, w_stat)

            sig = "YES" if p_wilcoxon < 0.05 else "no"
            label = f"{m1} vs {m2}"
            print(
                f"{label:<25} {mae1:>7.4f} {mae2:>7.4f} "
                f"{w_stat:>10.1f} {p_wilcoxon:>10.4f} {r_effect:>6.3f} {sig:>6}"
            )

            records.append({
                "comparison":        label,
                "model_A":           m1,
                "model_B":           m2,
                "mae_A":             round(mae1, 4),
                "mae_B":             round(mae2, 4),
                "wilcoxon_stat":     round(float(w_stat), 2),
                "p_value":           round(float(p_wilcoxon), 6),
                "effect_size_r":     round(r_effect, 4),
                "significant_0.05":  bool(p_wilcoxon < 0.05),
                "conclusion":        (
                    f"{m2} significantly outperforms {m1} (p={p_wilcoxon:.4f}, r={r_effect:.3f})"
                    if p_wilcoxon < 0.05
                    else f"no significant difference between {m1} and {m2} (p={p_wilcoxon:.4f})"
                ),
            })

        except Exception as exc:
            logger.error("Wilcoxon test failed for %s vs %s: %s", m1, m2, exc)

    results_path = output_dir / "statistical_tests.json"
    with open(results_path, "w") as f:
        json.dump(records, f, indent=2)
    logger.info("Saved results to %s", results_path)

    # --- Effect-size bar chart ---
    if records:
        labels  = [r["comparison"] for r in records]
        r_vals  = [r["effect_size_r"] for r in records]
        p_vals  = [r["p_value"] for r in records]
        colours = ["#1f77b4" if p < 0.05 else "#aec7e8" for p in p_vals]

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(labels, r_vals, color=colours, edgecolor="white")

        for bar, rv, pv in zip(bars, r_vals, p_vals):
            annot = f"r={rv:.3f}\n{'p=' + f'{pv:.3f}' if pv >= 0.001 else 'p<0.001'}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                annot, ha="center", va="bottom", fontsize=9,
            )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Rank-biserial correlation r\n(effect size: 0.1=small, 0.3=medium, 0.5=large)")
        ax.set_title(
            "Wilcoxon Signed-Rank Test Effect Sizes\n"
            "(blue = significant at α=0.05; grey = not significant)"
        )
        ax.set_ylim(0, max(r_vals) * 1.4 if r_vals else 1)
        plt.tight_layout()

        fig.savefig(output_dir / "statistical_significance.png", dpi=150)
        plt.close(fig)
        logger.info("Saved chart to %s", output_dir / "statistical_significance.png")

    print(f"\nFull results saved to {results_path}")


if __name__ == "__main__":
    run()
