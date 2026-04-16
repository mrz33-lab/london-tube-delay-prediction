"""
SHAP-guided feature selection — reduce overfitting by keeping only the top-N
most important features.

The ablation study (CRITICAL_ANALYSIS.md §1) showed that adding 16 features
beyond the original 21 actually *increased* MAE at fixed hyperparameters.
The learning curves showed a persistent train/val gap (1.73 vs 4.64 at full
data) suggesting the model overfits the exogenous features at this data volume.

This script tests whether trimming to the top-10 SHAP features narrows the
overfitting gap, providing concrete evidence that deliberate feature selection
improves generalisation.

Method
------
1. Load feature importance from the latest artifact run (feature_importance.csv
   produced by explain.py). Falls back to computing SHAP if the CSV is missing.
2. Identify the top-N features by mean |SHAP| importance.
3. Retrain LightGBM with only those features using TimeSeriesSplit CV.
4. Compare train MAE and CV-val MAE for full model vs reduced model.
5. The "overfitting gap" = val_MAE - train_MAE. A narrower gap = less overfitting.
6. Save results to shap_feature_selection.json.

Run from the project root:
    python analysis/shap_feature_selection.py
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from config import get_config, RANDOM_SEED
from data import load_data, get_train_test_split
from features import (
    engineer_features, prepare_features_for_model,
    create_preprocessing_pipeline,
)
from utils import get_latest_run_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TOP_N_OPTIONS = [5, 10, 15]      # feature counts to evaluate
CV_SPLITS     = 5


def _load_feature_importance(artifact_dir: Path) -> pd.DataFrame:
    """Load SHAP feature importance from artifact. Falls back to the most recent
    run that has feature_importance.csv if the latest run lacks it."""
    feat_imp_path = artifact_dir / "feature_importance.csv"
    if feat_imp_path.exists():
        df = pd.read_csv(feat_imp_path)
        logger.info("Loaded feature_importance.csv from %s", artifact_dir)
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    # Fallback: search all artifact runs in reverse-chronological order
    artifacts_root = artifact_dir.parent
    for run_dir in sorted(artifacts_root.iterdir(), reverse=True):
        candidate = run_dir / "feature_importance.csv"
        if candidate.exists():
            df = pd.read_csv(candidate)
            logger.info(
                "feature_importance.csv not in latest run — using %s", run_dir.name
            )
            return df.sort_values("importance", ascending=False).reset_index(drop=True)

    raise FileNotFoundError(
        f"feature_importance.csv not found in any artifact run under {artifacts_root}. "
        "Run `python explain.py` first to generate SHAP values."
    )


def _get_lgbm_params(model) -> dict:
    """Extract LightGBM hyperparameters from a trained sklearn Pipeline."""
    keep = {
        "num_leaves", "max_depth", "learning_rate", "n_estimators",
        "min_child_samples", "subsample", "colsample_bytree",
        "reg_alpha", "reg_lambda",
    }
    regressor = model.named_steps.get("regressor")
    if regressor is None or not hasattr(regressor, "get_params"):
        return {}
    raw = regressor.get_params()
    return {k: v for k, v in raw.items() if k in keep}


def _train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: List[str],
    lgbm_params: dict,
    label: str,
) -> dict:
    """Retrain LightGBM on feature_cols and return train/val/test MAE."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM required. Install with: pip install lightgbm")

    X_tr = X_train[feature_cols].copy()
    X_te = X_test[feature_cols].copy()

    num_feats = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = X_tr.select_dtypes(include=["object", "category"]).columns.tolist()

    pipeline = Pipeline([
        ("preprocessor", create_preprocessing_pipeline(num_feats, cat_feats)),
        ("regressor",    lgb.LGBMRegressor(
            random_state=RANDOM_SEED,
            verbosity=-1,
            force_col_wise=True,
            **lgbm_params,
        )),
    ])

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    cv_scores = cross_val_score(
        pipeline, X_tr, y_train,
        cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    cv_mae_mean = float(np.mean(-cv_scores))
    cv_mae_std  = float(np.std(-cv_scores))

    # Refit on full training set for test evaluation
    pipeline.fit(X_tr, y_train)
    train_mae = mean_absolute_error(y_train, pipeline.predict(X_tr))
    test_mae  = mean_absolute_error(y_test,  pipeline.predict(X_te))
    overfit_gap = cv_mae_mean - train_mae

    logger.info(
        "%s — train MAE=%.4f  CV val MAE=%.4f±%.4f  test MAE=%.4f  gap=%.4f",
        label, train_mae, cv_mae_mean, cv_mae_std, test_mae, overfit_gap,
    )

    return {
        "label":        label,
        "n_features":   len(feature_cols),
        "train_mae":    round(train_mae,    4),
        "cv_val_mae":   round(cv_mae_mean,  4),
        "cv_val_std":   round(cv_mae_std,   4),
        "test_mae":     round(test_mae,     4),
        "overfit_gap":  round(overfit_gap,  4),
    }


def run() -> None:
    output_dir = ROOT / "analysis" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_config()

    # ── Load artifact ─────────────────────────────────────────────────────────
    run_id = get_latest_run_id(ROOT / "artifacts")
    if run_id is None:
        raise FileNotFoundError("No artifact found. Run train.py first.")
    artifact_dir = ROOT / "artifacts" / run_id
    logger.info("Using artifact run: %s", run_id)

    # Load feature importance
    try:
        feat_imp_df = _load_feature_importance(artifact_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return

    # ── Load data and prepare features ────────────────────────────────────────
    df_raw, mode = load_data(config)
    logger.info("Data mode: %s | Shape: %s", mode, df_raw.shape)
    df = engineer_features(df_raw, config, is_training=True)
    train_df, test_df = get_train_test_split(df, config)

    # Load the full model to get feature list and hyperparams
    model_path = artifact_dir / "best_model.pkl"
    full_model  = joblib.load(model_path)
    lgbm_params = _get_lgbm_params(full_model)

    preprocessor   = full_model.named_steps["preprocessor"]
    num_feats_full  = preprocessor.transformers_[0][2]
    cat_feats_full  = preprocessor.transformers_[1][2]
    all_feats_full  = num_feats_full + cat_feats_full

    X_train, y_train = prepare_features_for_model(
        train_df, all_feats_full, config.features.target_column
    )
    X_test, y_test = prepare_features_for_model(
        test_df, all_feats_full, config.features.target_column
    )

    # ── Baseline: full feature set ────────────────────────────────────────────
    results = []

    logger.info("Evaluating full model (%d features)...", len(all_feats_full))
    full_result = _train_and_evaluate(
        X_train, y_train, X_test, y_test,
        all_feats_full, lgbm_params,
        label=f"Full model ({len(all_feats_full)} features)",
    )
    results.append(full_result)

    # ── Reduced models at each top-N threshold ────────────────────────────────
    available_features = set(all_feats_full)
    ranked_features = [
        f for f in feat_imp_df["feature"].tolist()
        if f in available_features
    ]

    for top_n in TOP_N_OPTIONS:
        top_feats = ranked_features[:top_n]
        if len(top_feats) < 3:
            logger.warning("Only %d ranked features available for top-%d — skipping", len(top_feats), top_n)
            continue

        logger.info("Evaluating top-%d features: %s", top_n, top_feats)
        result = _train_and_evaluate(
            X_train, y_train, X_test, y_test,
            top_feats, lgbm_params,
            label=f"Top-{top_n} features",
        )
        result["features"] = top_feats
        results.append(result)

    # ── Bar chart: overfitting gap ────────────────────────────────────────────
    labels      = [r["label"] for r in results]
    train_maes  = [r["train_mae"] for r in results]
    val_maes    = [r["cv_val_mae"] for r in results]
    test_maes   = [r["test_mae"] for r in results]
    gaps        = [r["overfit_gap"] for r in results]

    x     = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.bar(x - width, train_maes, width, label="Train MAE",      color="#1f77b4", alpha=0.85)
    ax.bar(x,         val_maes,   width, label="CV-Val MAE",     color="#ff7f0e", alpha=0.85)
    ax.bar(x + width, test_maes,  width, label="Test MAE",       color="#2ca02c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("MAE (severity units)")
    ax.set_title("Train / Val / Test MAE by Feature Count")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    colours = ["#DC241F" if g > 0 else "#00B140" for g in gaps]
    ax2.bar(x, gaps, color=colours, alpha=0.85, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Overfitting gap (CV-Val MAE − Train MAE)")
    ax2.set_title("Overfitting Gap: smaller is better")
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("SHAP-Guided Feature Selection: Overfitting Analysis", fontsize=13)
    plt.tight_layout()

    png_path = output_dir / "shap_feature_selection.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info("Saved chart to %s", png_path)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    best_reduced = min(
        (r for r in results if r["label"] != results[0]["label"]),
        key=lambda r: r["overfit_gap"],
        default=None,
    )

    json_out = {
        "run_id":        run_id,
        "full_n_features": len(all_feats_full),
        "ranked_features": ranked_features,
        "results":       results,
        "best_reduced":  best_reduced,
        "conclusion": (
            f"Reducing from {len(all_feats_full)} to "
            f"{best_reduced['n_features'] if best_reduced else 'N/A'} features "
            f"{'narrows' if best_reduced and best_reduced['overfit_gap'] < results[0]['overfit_gap'] else 'does not narrow'} "
            "the train/val overfitting gap."
        ) if best_reduced else "No reduced model evaluated.",
    }

    json_path = output_dir / "shap_feature_selection.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    logger.info("Saved feature selection results to %s", json_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n| Model | Features | Train MAE | CV-Val MAE | Test MAE | Gap |")
    print("|-------|----------|-----------|------------|----------|-----|")
    for r in results:
        print(
            f"| {r['label']} | {r['n_features']} | {r['train_mae']:.4f} | "
            f"{r['cv_val_mae']:.4f} | {r['test_mae']:.4f} | {r['overfit_gap']:+.4f} |"
        )

    if json_out["conclusion"]:
        print(f"\n{json_out['conclusion']}")


if __name__ == "__main__":
    run()
