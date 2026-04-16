"""
Model training pipeline — naive baseline, Ridge, and LightGBM.
"""

import logging
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import time
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import optuna
from scipy.stats import wilcoxon as _wilcoxon_test
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config, get_config, RANDOM_SEED
from utils import (
    setup_logging, generate_run_id, save_config, save_metrics,
    set_random_seeds, format_duration, evaluate_model
)
from data import load_data, get_train_test_split, save_data
from features import (
    engineer_features, get_feature_columns, create_preprocessing_pipeline,
    save_feature_metadata, prepare_features_for_model
)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available")


logger = logging.getLogger(__name__)


def _get_git_hash() -> str:
    """Return the short git commit hash, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def _save_model_info(artifact_dir: Path, best_model_name: str, data_mode: str,
                     metrics: Dict[str, Any]) -> None:
    """Write model_info.json with reproducibility metadata."""
    import sklearn
    try:
        import lightgbm as lgb
        lgb_version = lgb.__version__
    except ImportError:
        lgb_version = 'not installed'

    info = {
        'git_commit': _get_git_hash(),
        'trained_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'python_version': sys.version.split()[0],
        'sklearn_version': sklearn.__version__,
        'lightgbm_version': lgb_version,
        'best_model': best_model_name,
        'data_mode': data_mode,
        'target': 'delay_severity (0=Good, 1=Minor, 2=Severe — real TfL label)',
        'test_mae': round(metrics.get('test_mae', float('nan')), 4),
        'test_rmse': round(metrics.get('test_rmse', float('nan')), 4),
    }
    with open(artifact_dir / 'model_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    logger.info("Saved model_info.json (git=%s)", info['git_commit'])


def _compare_models_statistically(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifact_dir: Path,
) -> Dict[str, Any]:
    """Wilcoxon signed-rank tests for pairwise model comparison.

    Uses absolute errors |y_true - y_pred| as paired observations.
    The one-sided alternative='greater' answers: "are model A's errors
    significantly *larger* than model B's?" — i.e. is B significantly better?

    Why Wilcoxon rather than a paired t-test?
      The t-test assumes normally distributed differences.  Prediction errors
      on delay data are right-skewed (exponential tails from rare severe events),
      violating that assumption.  The Wilcoxon signed-rank test is
      distribution-free and remains valid under heavy-tailed residuals.

    Results are saved to statistical_tests.json for dissertation reporting.
    """
    predictions = {name: model.predict(X_test) for name, model in models.items()}
    abs_errors = {
        name: np.abs(np.array(y_test) - preds)
        for name, preds in predictions.items()
    }

    pairs = [
        ('naive', 'ridge'), ('naive', 'xgboost'), ('naive', 'best'),
        ('ridge', 'xgboost'), ('ridge', 'best'), ('xgboost', 'best'),
    ]
    results: Dict[str, Any] = {}

    for m1, m2 in pairs:
        if m1 not in abs_errors or m2 not in abs_errors:
            continue
        try:
            # one-sided: does m1 have significantly greater error than m2?
            stat, p_val = _wilcoxon_test(
                abs_errors[m1], abs_errors[m2], alternative='greater'
            )
            results[f'{m1}_vs_{m2}'] = {
                'statistic': round(float(stat), 4),
                'p_value': round(float(p_val), 6),
                'significant_at_0.05': bool(p_val < 0.05),
                'conclusion': (
                    f'{m2} significantly outperforms {m1} (p={p_val:.4f})'
                    if p_val < 0.05
                    else f'no significant difference between {m1} and {m2} (p={p_val:.4f})'
                ),
            }
            logger.info(
                "Wilcoxon %s vs %s: stat=%.1f  p=%.4f  %s",
                m1, m2, stat, p_val,
                "(significant)" if p_val < 0.05 else "(not significant)",
            )
        except Exception as exc:
            logger.warning("Wilcoxon test failed for %s vs %s: %s", m1, m2, exc)

    with open(artifact_dir / 'statistical_tests.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("Saved pairwise Wilcoxon test results to statistical_tests.json")
    return results


class NaiveBaselineModel:
    """Predicts the last observed delay for each line.
    Anything that can't beat this is useless.
    """

    def __init__(self):
        self.last_delays_: Dict[str, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, X, y):
        if 'line' not in X.columns:
            raise ValueError("X must contain 'line' column for naive baseline")

        data = X.copy()
        data['delay'] = y.values

        for line in data['line'].unique():
            line_data = data[data['line'] == line]
            self.last_delays_[line] = line_data['delay'].iloc[-1]

        self.global_mean_ = y.mean()
        return self

    def predict(self, X):
        # Vectorised: map each line to its stored last delay, fall back to
        # global mean for unseen lines.  10–100x faster than .iterrows().
        return X['line'].map(
            lambda line: self.last_delays_.get(line, self.global_mean_)
        ).to_numpy(dtype=float)


def train_naive_baseline(X_train, y_train, X_test, y_test):
    logger.info("Training naive baseline model...")

    model = NaiveBaselineModel()
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    logger.info("Naive baseline - Test MAE: %.3f, Test RMSE: %.3f",
                metrics['test_mae'], metrics['test_rmse'])
    return model, metrics


def train_ridge_baseline(X_train, y_train, X_test, y_test,
                         numeric_features, categorical_features, config):
    """Ridge regression with TimeSeriesSplit CV."""
    logger.info("Training Ridge regression baseline...")

    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(random_state=RANDOM_SEED))
    ])

    param_grid = {
        'regressor__alpha': config.models.ridge_params['alpha']
    }

    tscv = TimeSeriesSplit(n_splits=config.models.cv_splits)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=min(config.models.n_iter_search, len(param_grid['regressor__alpha'])),
        cv=tscv,
        scoring=config.models.scoring,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    logger.info("Best Ridge alpha: %.4f", search.best_params_['regressor__alpha'])

    best_model = search.best_estimator_
    metrics = evaluate_model(best_model, X_train, y_train, X_test, y_test)
    metrics['best_params'] = search.best_params_

    logger.info("Ridge - Test MAE: %.3f, Test RMSE: %.3f",
                metrics['test_mae'], metrics['test_rmse'])
    return best_model, metrics


def train_lightgbm(X_train, y_train, X_test, y_test,
                   numeric_features, categorical_features, config):
    """LightGBM with TimeSeriesSplit CV."""
    logger.info("Training LightGBM model...")

    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(
            random_state=RANDOM_SEED,
            verbosity=-1,
            force_col_wise=True
        ))
    ])

    tscv = TimeSeriesSplit(n_splits=config.models.cv_splits)

    def objective(trial):
        space = config.models.lightgbm_optuna_space
        params = {
            'num_leaves': trial.suggest_int('num_leaves', space['num_leaves']['low'], space['num_leaves']['high']),
            'max_depth': trial.suggest_int('max_depth', space['max_depth']['low'], space['max_depth']['high']),
            'learning_rate': trial.suggest_float('learning_rate', space['learning_rate']['low'], space['learning_rate']['high'], log=True),
            'n_estimators': trial.suggest_int('n_estimators', space['n_estimators']['low'], space['n_estimators']['high']),
            'min_child_samples': trial.suggest_int('min_child_samples', space['min_child_samples']['low'], space['min_child_samples']['high']),
            'subsample': trial.suggest_float('subsample', space['subsample']['low'], space['subsample']['high']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', space['colsample_bytree']['low'], space['colsample_bytree']['high']),
            'reg_alpha': trial.suggest_float('reg_alpha', space['reg_alpha']['low'], space['reg_alpha']['high']),
            'reg_lambda': trial.suggest_float('reg_lambda', space['reg_lambda']['low'], space['reg_lambda']['high'])
        }

        # Wrap preprocessing + estimator in a single Pipeline so cross_val_score
        # refits the ColumnTransformer on each CV fold's training portion.
        # Previously the preprocessor was fit once on the full X_train and passed
        # as a pre-processed array, causing train-set statistics (mean, std) to
        # leak into every validation fold.
        trial_pipeline = Pipeline([
            ('preprocessor', create_preprocessing_pipeline(numeric_features, categorical_features)),
            ('regressor', lgb.LGBMRegressor(
                random_state=RANDOM_SEED,
                verbosity=-1,
                force_col_wise=True,
                **params
            ))
        ])

        scores = cross_val_score(
            trial_pipeline, X_train, y_train, cv=tscv,
            scoring=config.models.scoring, n_jobs=-1
        )

        # scoring="neg_mean_absolute_error" returns values in (-∞, 0].
        # Maximising neg-MAE is equivalent to minimising MAE — a score
        # closer to 0 is better, so direction="maximize" is correct.
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        storage=f"sqlite:///{config.paths.artifacts_dir}/optuna_study.db",
        study_name="lgbm_tube_delay",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=config.models.optuna_n_trials, n_jobs=1)

    logger.info("Best LightGBM params: %s", study.best_params)

    best_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(
            random_state=RANDOM_SEED,
            verbosity=-1,
            force_col_wise=True,
            **study.best_params
        ))
    ])
    best_model.fit(X_train, y_train)

    metrics = evaluate_model(best_model, X_train, y_train, X_test, y_test)
    metrics['best_params'] = study.best_params

    logger.info("LightGBM - Test MAE: %.3f, Test RMSE: %.3f",
                metrics['test_mae'], metrics['test_rmse'])
    return best_model, metrics


def train_xgboost(X_train, y_train, X_test, y_test,
                  numeric_features, categorical_features, config):
    """XGBoost with RandomizedSearchCV — primary advanced model alongside LightGBM."""
    if not XGBOOST_AVAILABLE:
        return None, None

    logger.info("Training XGBoost model...")

    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(random_state=RANDOM_SEED, verbosity=0))
    ])

    tscv = TimeSeriesSplit(n_splits=config.models.cv_splits)

    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.05, 0.1, 0.2],
        'regressor__subsample': [0.7, 0.8, 1.0],
        'regressor__colsample_bytree': [0.7, 0.8, 1.0],
        'regressor__reg_alpha': [0.0, 0.1, 1.0],
        'regressor__reg_lambda': [1.0, 5.0, 10.0],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=config.models.n_iter_search,
        cv=tscv,
        scoring=config.models.scoring,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    logger.info("Best XGBoost params: %s", search.best_params_)

    best_model = search.best_estimator_
    metrics = evaluate_model(best_model, X_train, y_train, X_test, y_test)
    metrics['best_params'] = search.best_params_

    logger.info(
        "XGBoost - Test MAE: %.3f, Test RMSE: %.3f",
        metrics['test_mae'], metrics['test_rmse'],
    )
    return best_model, metrics


def train_fallback_model(X_train, y_train, X_test, y_test,
                         numeric_features, categorical_features, config):
    """XGBoost or RandomForest fallback when LightGBM isn't installed.

    Delegates to train_xgboost when XGBoost is available so the two code
    paths stay in sync.  Only owns the RandomForest branch.
    """
    if XGBOOST_AVAILABLE:
        logger.info("Training XGBoost (LightGBM fallback)...")
        model, metrics = train_xgboost(
            X_train, y_train, X_test, y_test,
            numeric_features, categorical_features, config,
        )
        return model, metrics, "xgboost"

    logger.info("Training RandomForest (LightGBM/XGBoost fallback)...")
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1))
    ])

    param_grid = {
        'regressor__%s' % k: v for k, v in config.models.rf_params.items()
    }

    tscv = TimeSeriesSplit(n_splits=config.models.cv_splits)
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_grid,
        n_iter=config.models.n_iter_search,
        cv=tscv, scoring=config.models.scoring,
        random_state=RANDOM_SEED, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    logger.info("Best randomforest params: %s", search.best_params_)

    best_model = search.best_estimator_
    metrics = evaluate_model(best_model, X_train, y_train, X_test, y_test)
    metrics['best_params'] = search.best_params_

    logger.info("RandomForest - Test MAE: %.3f, Test RMSE: %.3f",
                metrics['test_mae'], metrics['test_rmse'])
    return best_model, metrics, "randomforest"


def create_diagnostic_plots(y_test, y_pred, model_name, output_dir):
    """Residual histogram, pred vs actual, residual vs predicted."""
    sns.set_style('whitegrid')
    residuals = y_test - y_pred

    # residual histogram
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Residual (minutes)')
    plt.ylabel('Frequency')
    plt.title(f'Residual Distribution - {model_name}')
    plt.axvline(x=0, color='r', linestyle='--', label='Zero residual')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_residual_hist.png', dpi=150)
    plt.close()

    # pred vs actual scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    plt.xlabel('Actual Delay (minutes)')
    plt.ylabel('Predicted Delay (minutes)')
    plt.title(f'Predicted vs Actual - {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_pred_vs_actual.png', dpi=150)
    plt.close()

    # residual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.3, s=10)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Delay (minutes)')
    plt.ylabel('Residual (minutes)')
    plt.title(f'Residual vs Predicted - {model_name}')
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_residual_vs_pred.png', dpi=150)
    plt.close()

    logger.info("Saved diagnostic plots for %s", model_name)


def bootstrap_confidence_interval(y_true, y_pred, metric_func,
                                  n_bootstrap=1000, confidence=0.95,
                                  block_size=None):
    """Block bootstrap CI for time-series data.

    Standard i.i.d. bootstrap underestimates uncertainty when residuals are
    autocorrelated (as delay series typically are).  Sampling contiguous blocks
    preserves local autocorrelation structure.

    block_size: number of consecutive observations per block.
                None → auto: int(sqrt(n)).
    """
    n = len(y_true)
    if block_size is None:
        block_size = max(1, int(np.sqrt(n)))

    rng = np.random.RandomState(RANDOM_SEED)
    bootstrap_metrics = []

    for _ in range(n_bootstrap):
        # sample enough block start positions to cover n observations
        n_blocks = int(np.ceil(n / block_size))
        max_start = max(1, n - block_size + 1)
        starts = rng.randint(0, max_start, size=n_blocks)

        indices = []
        for start in starts:
            indices.extend(range(start, min(start + block_size, n)))

        indices = indices[:n]  # trim to original length

        y_true_boot = y_true.iloc[indices]
        y_pred_boot = y_pred[indices]
        bootstrap_metrics.append(metric_func(y_true_boot, y_pred_boot))

    alpha = 1 - confidence
    point_estimate = metric_func(y_true, y_pred)
    lower_bound = np.percentile(bootstrap_metrics, (alpha / 2) * 100)
    upper_bound = np.percentile(bootstrap_metrics, (1 - alpha / 2) * 100)

    return point_estimate, lower_bound, upper_bound


def main():
    config = get_config()
    config.paths.ensure_dirs()
    run_id = generate_run_id()
    config.run_id = run_id

    logger = setup_logging(config, run_id)
    logger.info("=" * 80)
    logger.info("London Underground Delay Prediction - Training Pipeline")
    logger.info("=" * 80)

    set_random_seeds(RANDOM_SEED)
    logger.info("Random seed: %d", RANDOM_SEED)

    artifact_dir = config.get_artifact_dir()
    save_config(config, artifact_dir / 'config.yaml')

    start_time = time.time()

    try:
        # ---- LOAD DATA ----
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Loading Data")
        logger.info("=" * 80)

        df, data_mode = load_data(config)
        logger.info("Data mode: %s", data_mode)

        with open(artifact_dir / 'data_info.txt', 'w') as f:
            f.write(f"Data mode: {data_mode}\n")
            f.write(f"Shape: {df.shape}\n")
            f.write(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
            f.write(f"\nColumns:\n{df.columns.tolist()}\n")
            f.write(f"\nData types:\n{df.dtypes}\n")
            f.write(f"\nMissing values:\n{df.isna().sum()}\n")

        # ---- FEATURE ENGINEERING ----
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Feature Engineering")
        logger.info("=" * 80)

        df = engineer_features(df, config, is_training=True)
        train_df, test_df = get_train_test_split(df, config)

        numeric_features, categorical_features, all_features = get_feature_columns(train_df, config)

        # placeholder — gets overwritten with real quantiles after training
        save_feature_metadata(
            numeric_features, categorical_features, artifact_dir,
            residual_quantiles={},
        )

        X_train, y_train = prepare_features_for_model(
            train_df, all_features, config.features.target_column
        )
        X_test, y_test = prepare_features_for_model(
            test_df, all_features, config.features.target_column
        )

        logger.info("Training set: %s", X_train.shape)
        logger.info("Test set: %s", X_test.shape)

        # ---- MODEL TRAINING ----
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Model Training")
        logger.info("=" * 80)

        all_metrics = {}
        models = {}

        logger.info("\n--- Naive Baseline ---")
        naive_model, naive_metrics = train_naive_baseline(X_train, y_train, X_test, y_test)
        models['naive'] = naive_model
        all_metrics['naive'] = naive_metrics

        logger.info("\n--- Ridge Regression ---")
        ridge_model, ridge_metrics = train_ridge_baseline(
            X_train, y_train, X_test, y_test,
            numeric_features, categorical_features, config
        )
        models['ridge'] = ridge_model
        all_metrics['ridge'] = ridge_metrics

        logger.info("\n--- Advanced Model ---")
        if LIGHTGBM_AVAILABLE:
            best_model, best_metrics = train_lightgbm(
                X_train, y_train, X_test, y_test,
                numeric_features, categorical_features, config
            )
            best_model_name = 'lightgbm'
        else:
            best_model, best_metrics, best_model_name = train_fallback_model(
                X_train, y_train, X_test, y_test,
                numeric_features, categorical_features, config
            )

        models['best'] = best_model
        all_metrics['best'] = best_metrics

        logger.info("\n--- XGBoost (primary comparison model) ---")
        if XGBOOST_AVAILABLE:
            xgb_model, xgb_metrics = train_xgboost(
                X_train, y_train, X_test, y_test,
                numeric_features, categorical_features, config
            )
            if xgb_model is not None:
                models['xgboost'] = xgb_model
                all_metrics['xgboost'] = xgb_metrics
        else:
            logger.info("XGBoost not available — skipping")

        # ---- MODEL COMPARISON ----
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Model Comparison")
        logger.info("=" * 80)

        _model_order = [
            ('naive',    'Naive'),
            ('ridge',    'Ridge'),
            ('xgboost',  'XGBoost'),
            ('best',     best_model_name.title()),
        ]
        _rows = [(key, label) for key, label in _model_order if key in all_metrics]
        comparison_df = pd.DataFrame({
            'Model':      [label for _, label in _rows],
            'Train MAE':  [all_metrics[k]['train_mae']  for k, _ in _rows],
            'Test MAE':   [all_metrics[k]['test_mae']   for k, _ in _rows],
            'Train RMSE': [all_metrics[k]['train_rmse'] for k, _ in _rows],
            'Test RMSE':  [all_metrics[k]['test_rmse']  for k, _ in _rows],
            'Test R²':    [all_metrics[k]['test_r2']    for k, _ in _rows],
            'Test MAPE':  [all_metrics[k].get('test_mape', float('nan')) for k, _ in _rows],
        })

        logger.info("\n" + comparison_df.to_string(index=False))
        comparison_df.to_csv(artifact_dir / 'model_comparison.csv', index=False)

        # ---- STATISTICAL SIGNIFICANCE TESTS ----
        logger.info("\n--- Pairwise Wilcoxon Significance Tests ---")
        _compare_models_statistically(models, X_test, y_test, artifact_dir)

        # ---- PER-LINE RESIDUAL QUANTILES ----
        # used by FutureDelayPredictor for proper 95% CIs
        logger.info("\n--- Computing Per-Line Residual Quantiles ---")
        y_pred_best = models['best'].predict(X_test)
        residuals_all = np.array(y_test) - y_pred_best
        residual_quantiles = {
            '__global__': {
                'q025': float(np.percentile(residuals_all, 2.5)),
                'q975': float(np.percentile(residuals_all, 97.5)),
            }
        }
        for line_name in X_test['line'].unique():
            mask = X_test['line'] == line_name
            res_line = np.array(y_test[mask]) - y_pred_best[mask]
            if len(res_line) >= 10:
                residual_quantiles[line_name] = {
                    'q025': float(np.percentile(res_line, 2.5)),
                    'q975': float(np.percentile(res_line, 97.5)),
                }
            else:
                logger.warning(
                    "Only %d test obs for %s, using global quantiles",
                    len(res_line), line_name,
                )

        # compute training-set means for network features so inference predictor
        # can use in-distribution defaults instead of hard-coded zeros
        _network_cols = [
            'network_avg_delay', 'network_delay_volatility',
            'lines_disrupted_ratio', 'is_network_wide_disruption',
        ]
        network_feature_means = {
            col: float(train_df[col].mean())
            for col in _network_cols
            if col in train_df.columns
        }

        # ---- CONFORMAL CALIBRATION SCORES ----
        # Split the test set: first half → calibration, second half → evaluation.
        # This is the standard split-conformal protocol (Vovk et al., 2005;
        # Angelopoulos & Bates, 2022).  Non-conformity score: |y_true - y_pred|.
        # At inference time, FutureDelayPredictor uses the empirical quantile of
        # these scores to build a PI with a guaranteed marginal coverage rate of
        # ≥ 1-α under exchangeability — no distributional assumptions required.
        n_cal = len(y_test) // 2
        y_pred_cal_half = models['best'].predict(X_test.iloc[:n_cal])
        conformal_cal_scores = np.abs(
            np.array(y_test.iloc[:n_cal]) - y_pred_cal_half
        )
        logger.info(
            "Conformal calibration (n=%d): median=%.3f  q95=%.3f",
            n_cal,
            float(np.median(conformal_cal_scores)),
            float(np.percentile(conformal_cal_scores, 95)),
        )

        # re-save with actual quantiles + network means + conformal scores
        save_feature_metadata(
            numeric_features, categorical_features, artifact_dir,
            residual_quantiles=residual_quantiles,
            network_feature_means=network_feature_means,
            conformal_cal_scores=conformal_cal_scores,
        )
        logger.info(
            "Per-line residual quantiles computed for %d lines",
            len(residual_quantiles) - 1,
        )

        # ---- BOOTSTRAP CIs ----
        logger.info("\n--- Computing Bootstrap Confidence Intervals ---")

        block_size = config.models.bootstrap_block_size
        mae_point, mae_lower, mae_upper = bootstrap_confidence_interval(
            y_test, y_pred_best, mean_absolute_error, block_size=block_size
        )
        _rmse = lambda y_t, y_p: float(np.sqrt(mean_squared_error(y_t, y_p)))
        rmse_point, rmse_lower, rmse_upper = bootstrap_confidence_interval(
            y_test, y_pred_best, _rmse, block_size=block_size
        )

        logger.info("Best model Test MAE:  %.3f (95%% CI: [%.3f, %.3f])",
                    mae_point, mae_lower, mae_upper)
        logger.info("Best model Test RMSE: %.3f (95%% CI: [%.3f, %.3f])",
                    rmse_point, rmse_lower, rmse_upper)

        all_metrics['best']['test_mae_ci'] = [mae_lower, mae_upper]
        all_metrics['best']['test_rmse_ci'] = [rmse_lower, rmse_upper]

        # ---- DIAGNOSTIC PLOTS ----
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Creating Diagnostic Plots")
        logger.info("=" * 80)

        for model_name in ['naive', 'ridge', 'best']:
            y_pred = models[model_name].predict(X_test)
            create_diagnostic_plots(y_test, y_pred, model_name, artifact_dir)

        # ---- SAVE EVERYTHING ----
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Saving Models and Metrics")
        logger.info("=" * 80)

        for name, model in models.items():
            joblib.dump(model, artifact_dir / f'{name}_model.pkl')
            logger.info("Saved %s model", name)

        save_metrics(all_metrics, artifact_dir / 'all_metrics.json')

        # Save CV fold scores so the dashboard can load them cheaply instead
        # of re-running cross_val_score on first visit.
        logger.info("\n--- Saving CV fold scores ---")
        _tscv_save = TimeSeriesSplit(n_splits=config.models.cv_splits)
        _cv_raw = cross_val_score(
            models['best'], X_train, y_train,
            cv=_tscv_save, scoring='neg_mean_absolute_error', n_jobs=1,
        )
        cv_fold_scores = {
            'mean': float(np.mean(-_cv_raw)),
            'std':  float(np.std(-_cv_raw)),
            'folds': [
                {
                    'fold': i + 1,
                    'train_n': int(len(tr)),
                    'val_n':   int(len(va)),
                    'mae':     float(-s),
                }
                for i, ((tr, va), s) in enumerate(
                    zip(_tscv_save.split(X_train), _cv_raw)
                )
            ],
        }
        with open(artifact_dir / 'cv_fold_scores.json', 'w') as f:
            json.dump(cv_fold_scores, f, indent=2)
        logger.info("Saved CV fold scores (mean MAE=%.4f ± %.4f)",
                    cv_fold_scores['mean'], cv_fold_scores['std'])

        test_predictions = pd.DataFrame({
            'timestamp': test_df['timestamp'].values,
            'line': test_df['line'].values,
            'actual': y_test.values,
            'pred_naive': models['naive'].predict(X_test),
            'pred_ridge': models['ridge'].predict(X_test),
            'pred_best': models['best'].predict(X_test),
            **({'pred_xgboost': models['xgboost'].predict(X_test)}
               if 'xgboost' in models else {}),
        })
        test_predictions.to_csv(artifact_dir / 'test_predictions.csv', index=False)

        X_train.to_parquet(artifact_dir / 'X_train.parquet')
        X_test.to_parquet(artifact_dir / 'X_test.parquet')
        y_test.rename('delay_severity').to_frame().to_parquet(
            artifact_dir / 'y_test.parquet'
        )
        logger.info("Saved featured train/test data for SHAP analysis")

        # Save regression targets so classify.py standalone can load them.
        # prepare_features_for_model excludes the target from X_train, so we
        # persist y_train separately rather than re-adding it to the parquet.
        y_train.reset_index(drop=True).rename('delay_minutes').to_csv(
            artifact_dir / 'y_train.csv', index=False
        )
        logger.info("Saved y_train.csv for classification pipeline")

        # Save raw TfL status labels aligned to X_train/X_test.
        # Using the original status column (not labels re-derived from delay_minutes)
        # eliminates the circularity in classify.py's ordinal task: for real data,
        # status is the true TfL label and delay_minutes is a proxy derived FROM it,
        # so re-mapping delay → status would just invert a known function.
        train_df.loc[X_train.index, 'status'].reset_index(drop=True).rename('status').to_csv(
            artifact_dir / 'y_train_status.csv', index=False
        )
        test_df.loc[X_test.index, 'status'].reset_index(drop=True).rename('status').to_csv(
            artifact_dir / 'y_test_status.csv', index=False
        )
        logger.info("Saved y_train_status.csv and y_test_status.csv for classification pipeline")

        with open(artifact_dir / 'best_model_name.txt', 'w') as f:
            f.write(best_model_name)

        _save_model_info(artifact_dir, best_model_name, data_mode, all_metrics['best'])

        # ---- CLASSIFICATION EVALUATION ----
        # Called after ALL primary artifacts are saved so a classify failure
        # never prevents the regression artifacts from being persisted.
        logger.info("\n--- Running Classification Evaluation ---")
        try:
            from classify import run_classification_evaluation
            y_train_status_arr = pd.read_csv(artifact_dir / 'y_train_status.csv')['status'].values
            y_test_status_arr  = pd.read_csv(artifact_dir / 'y_test_status.csv')['status'].values
            run_classification_evaluation(
                artifact_dir=artifact_dir,
                config=config,
                X_train=X_train,
                y_train_reg=np.array(y_train),
                X_test=X_test,
                y_test_reg=np.array(y_test),
                test_timestamps=test_df.loc[X_test.index, config.features.time_column].reset_index(drop=True),
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                y_train_status=y_train_status_arr,
                y_test_status=y_test_status_arr,
            )
            logger.info("Classification evaluation complete")
        except Exception as exc:
            logger.warning(
                "Classification evaluation failed (non-fatal): %s", exc, exc_info=True
            )

        # ---- DONE ----
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info("Total time: %s", format_duration(elapsed))
        logger.info("Artifact directory: %s", artifact_dir)
        logger.info("Best model: %s", best_model_name)
        logger.info("Best model Test MAE: %.3f severity units", all_metrics['best']['test_mae'])
        logger.info("Improvement over naive: %.1f%%",
                    (1 - all_metrics['best']['test_mae'] / all_metrics['naive']['test_mae']) * 100)
        logger.info("=" * 80)

    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        raise


if __name__ == '__main__':
    main()
