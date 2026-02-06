"""
Analysis scripts for in-depth model evaluation.

Each script loads artifacts from a specific training run and produces
diagnostic plots and CSVs saved to ``analysis/outputs/``.

Scripts:
    ablation_study.py                  -- Feature group importance via leave-one-out ablation.
    arima_baseline.py                  -- SARIMA vs LightGBM head-to-head comparison.
    learning_curves.py                 -- Bias/variance diagnosis with sample-size sweeps.
    per_line_performance.py            -- Per-line MAE/RMSE error breakdown.
    confidence_interval_calibration.py -- Empirical CI reliability check.
    improvement_history.py             -- 21-feature vs 37-feature metric comparison.
"""

__all__ = [
    "ablation_study",
    "arima_baseline",
    "learning_curves",
    "per_line_performance",
    "confidence_interval_calibration",
    "improvement_history",
]
