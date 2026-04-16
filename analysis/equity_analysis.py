"""
Equity analysis — does prediction error correlate with community deprivation?

Each tube line serves a different socioeconomic cross-section of London.
If the model systematically underperforms on lines serving deprived communities,
it could bias resource allocation against passengers who depend most on public
transport.

Method
------
1. Assign each line a catchment-area IMD score: the weighted average of the
   English Indices of Deprivation 2019 (% of LSOAs in the most deprived 20%
   nationally, from MHCLG) across the London boroughs the line passes through,
   weighted by the number of stations in each borough.
2. Load per-line MAE from the latest test_predictions.csv artifact.
3. Compute Spearman rank correlation (ρ) between IMD score and MAE.
4. Generate a scatter plot and save equity_results.json.

Source: English Indices of Deprivation 2019, Ministry of Housing, Communities
and Local Government (MHCLG). Published 26 September 2019.
https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019

Run from the project root:
    python analysis/equity_analysis.py
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


# ---------------------------------------------------------------------------
# IMD scores per line
# ---------------------------------------------------------------------------
# Each score is the weighted average % of LSOAs in the most deprived 20%
# nationally (IMD 2019) across the boroughs the line passes through.
# Weights = number of stations in each borough.
#
# Borough scores used (IMD 2019, % most deprived 20%):
#   Tower Hamlets 53.3 | Newham 58.7 | Hackney 48.8 | Brent 35.2
#   Waltham Forest 38.1 | Islington 37.3 | Haringey 36.2 | Lambeth 30.0
#   Southwark 28.5 | Enfield 26.9 | Camden 23.3 | Westminster 24.2
#   Ealing 20.7 | Hounslow 20.8 | Hammersmith & Fulham 19.8
#   Havering 16.6 | Hillingdon 16.1 | Harrow 13.7 | Merton 12.0
#   Kensington & Chelsea 10.2 | Barnet 9.8 | Richmond 2.5 | City of London 5.0

LINE_IMD_SCORES = {
    # Harrow (14) × 2, Brent (35) × 5, Westminster (24) × 7, Lambeth (30) × 2
    "Bakerloo":           27.1,

    # Hillingdon (16) × 7, Ealing (21) × 2, Hammersmith (20) × 3,
    # Kensington (10) × 3, Westminster (24) × 4, City (5) × 3,
    # Tower Hamlets (53) × 3, Newham (59) × 5, Barking (57) × 1
    "Central":            27.9,

    # Westminster (24) × 15, Kensington (10) × 5, City (5) × 5,
    # Hammersmith (20) × 3, Islington (37) × 2, Tower Hamlets (53) × 2
    "Circle":             21.2,

    # Richmond (3) × 6, Ealing (21) × 6, Hammersmith (20) × 6,
    # Kensington (10) × 5, Westminster (24) × 6, City (5) × 2,
    # Tower Hamlets (53) × 4, Newham (59) × 2, Barking (57) × 1, Havering (17) × 3
    "District":           21.2,

    # Hammersmith (20) × 5, Westminster (24) × 5, City (5) × 5,
    # Tower Hamlets (53) × 8, Barking (57) × 1
    "Hammersmith & City": 30.3,

    # Harrow (14) × 1, Brent (35) × 4, Camden (23) × 4, Westminster (24) × 5,
    # Southwark (29) × 5, Tower Hamlets (53) × 1, Greenwich (26) × 1, Newham (59) × 3
    "Jubilee":            31.9,

    # Hillingdon (16) × 8, Harrow (14) × 8, Barnet (10) × 2,
    # Westminster (24) × 6, City (5) × 3
    "Metropolitan":       15.5,

    # Barnet (10) × 8, Camden (23) × 6, Islington (37) × 4,
    # Westminster (24) × 5, Lambeth (30) × 4, Merton (12) × 3
    "Northern":           21.5,

    # Hillingdon (16) × 8, Ealing (21) × 2, Hammersmith (20) × 4,
    # Kensington (10) × 3, Westminster (24) × 5, Islington (37) × 5,
    # Haringey (36) × 5, Enfield (27) × 2
    "Piccadilly":         24.2,

    # Waltham Forest (38) × 3, Haringey (36) × 3, Islington (37) × 2,
    # Westminster (24) × 5, Lambeth (30) × 4, Merton (12) × 2
    "Victoria":           29.6,

    # Westminster (24) × 1, City (5) × 1
    "Waterloo & City":    14.6,
}

# Structural complexity note for each line (for dashboard annotation)
LINE_COMPLEXITY = {
    "Bakerloo":           "Single trunk, 25 stations",
    "Central":            "Branching east/west arms, 49 stations",
    "Circle":             "Loop — disruptions recirculate",
    "District":           "Three western branches, 60 stations",
    "Hammersmith & City": "Shared infrastructure with Circle",
    "Jubilee":            "High frequency — cascades fast",
    "Metropolitan":       "Longest line, semi-fast/stopping mix",
    "Northern":           "Two southern + two northern branches",
    "Piccadilly":         "Heathrow branch + long NE spine",
    "Victoria":           "Straight, highest ridership",
    "Waterloo & City":    "2-station shuttle, peak-only",
}


def _load_per_line_mae(artifact_dir: Path) -> dict:
    """Compute per-line MAE from test_predictions.csv."""
    pred_path = artifact_dir / "test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"test_predictions.csv not found in {artifact_dir}")

    df = pd.read_csv(pred_path)
    if "pred_best" not in df.columns:
        raise ValueError("test_predictions.csv must contain 'pred_best' column")

    mae_per_line = (
        df.groupby("line")
        .apply(lambda g: float(np.abs(g["actual"] - g["pred_best"]).mean()))
        .to_dict()
    )
    return mae_per_line


def run() -> None:
    output_dir = ROOT / "analysis" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load latest artifact ──────────────────────────────────────────────────
    run_id = get_latest_run_id(ROOT / "artifacts")
    if run_id is None:
        raise FileNotFoundError("No trained model found in artifacts/. Run train.py first.")
    artifact_dir = ROOT / "artifacts" / run_id
    logger.info("Using artifact run: %s", run_id)

    mae_per_line = _load_per_line_mae(artifact_dir)
    logger.info("Loaded per-line MAE for %d lines", len(mae_per_line))

    # ── Build comparison table ────────────────────────────────────────────────
    records = []
    for line, imd in LINE_IMD_SCORES.items():
        if line not in mae_per_line:
            logger.warning("Line '%s' not in test predictions — skipping", line)
            continue
        records.append({
            "line":       line,
            "imd_score":  imd,
            "mae":        mae_per_line[line],
            "complexity": LINE_COMPLEXITY.get(line, ""),
        })

    if len(records) < 4:
        logger.error("Too few lines (%d) for meaningful correlation", len(records))
        return

    df = pd.DataFrame(records)
    imd_arr = df["imd_score"].values
    mae_arr = df["mae"].values

    rho, p_val = spearmanr(imd_arr, mae_arr)
    logger.info(
        "Spearman ρ = %.4f  p = %.4f  (n = %d lines)",
        rho, p_val, len(df),
    )

    # Interpret result
    if p_val < 0.05 and rho > 0.3:
        interpretation = (
            f"Significant positive correlation (ρ={rho:.3f}, p={p_val:.3f}): "
            "the model performs WORSE on lines serving more deprived communities — "
            "potential equity concern."
        )
    elif p_val < 0.05 and rho < -0.3:
        interpretation = (
            f"Significant negative correlation (ρ={rho:.3f}, p={p_val:.3f}): "
            "the model performs BETTER on lines serving more deprived communities."
        )
    else:
        interpretation = (
            f"No significant correlation between deprivation and prediction error "
            f"(ρ={rho:.3f}, p={p_val:.3f}). Prediction accuracy appears driven by "
            "operational complexity (route branching, service frequency) rather than "
            "the socioeconomic profile of communities served."
        )

    logger.info("Interpretation: %s", interpretation)

    # ── Scatter plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(
        df["imd_score"], df["mae"],
        s=120, zorder=3,
        c=df["mae"], cmap="RdYlGn_r",
        edgecolors="white", linewidths=0.8,
    )
    plt.colorbar(scatter, ax=ax, label="MAE (severity units)")

    for _, row in df.iterrows():
        ax.annotate(
            row["line"],
            (row["imd_score"], row["mae"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8.5,
        )

    # Trend line
    z = np.polyfit(imd_arr, mae_arr, 1)
    p = np.poly1d(z)
    x_line = np.linspace(imd_arr.min(), imd_arr.max(), 100)
    ax.plot(x_line, p(x_line), "k--", alpha=0.4, linewidth=1.2, label="Trend")

    ax.set_xlabel("Catchment-area deprivation score (% LSOAs in most deprived 20%, IMD 2019)", fontsize=10)
    ax.set_ylabel("Test MAE (severity units)", fontsize=10)
    ax.set_title(
        f"Equity Analysis: Prediction Error vs Community Deprivation\n"
        f"Spearman ρ = {rho:.3f}   p = {p_val:.3f}",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    png_path = output_dir / "equity_analysis.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info("Saved scatter plot to %s", png_path)

    # ── Save JSON results ─────────────────────────────────────────────────────
    results = {
        "run_id":          run_id,
        "spearman_rho":    round(float(rho),   4),
        "p_value":         round(float(p_val), 4),
        "n_lines":         len(df),
        "significant":     bool(p_val < 0.05),
        "interpretation":  interpretation,
        "imd_source":      "English Indices of Deprivation 2019 (MHCLG)",
        "records": [
            {
                "line":       row["line"],
                "imd_score":  row["imd_score"],
                "mae":        round(row["mae"], 4),
                "complexity": row["complexity"],
            }
            for _, row in df.sort_values("imd_score").iterrows()
        ],
    }

    json_path = output_dir / "equity_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved equity results to %s", json_path)

    # Summary print
    print(f"\nSpearman rho = {rho:.4f}  p = {p_val:.4f}")
    print(interpretation)
    print("\n| Line | IMD score | MAE |")
    print("|------|-----------|-----|")
    for _, row in df.sort_values("imd_score").iterrows():
        print(f"| {row['line']} | {row['imd_score']:.1f} | {row['mae']:.4f} |")


if __name__ == "__main__":
    run()
