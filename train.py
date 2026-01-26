"""
Model training pipeline — naive baseline, Ridge, and LightGBM.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import time
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config, get_config, RANDOM_SEED
from utils import (
    setup_logging, generate_run_id, save_config, save_metrics,
    set_random_seeds, format_duration
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
        predictions = []
        for _, row in X.iterrows():
            pred = self.last_delays_.get(row['line'], self.global_mean_)
            predictions.append(pred)
        return np.array(predictions)


def train_naive_baseline(X_train, y_train, X_test, y_test):
    logger.info("Training naive baseline model...")

    model = NaiveBaselineModel()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_r2': r2_score(y_test, y_pred_test)
    }

    logger.info(f"Naive baseline - Test MAE: {metrics['test_mae']:.3f}, Test RMSE: {metrics['test_rmse']:.3f}")
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
        f'regressor__alpha': config.models.ridge_params['alpha']
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
    logger.info(f"Best Ridge alpha: {search.best_params_['regressor__alpha']:.4f}")

    best_model = search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    metrics = {
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_r2': r2_score(y_test, y_pred_test),
        'best_params': search.best_params_
    }

    logger.info(f"Ridge - Test MAE: {metrics['test_mae']:.3f}, Test RMSE: {metrics['test_rmse']:.3f}")
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

    param_grid = {
        f'regressor__{k}': v for k, v in config.models.lightgbm_params.items()
    }

    tscv = TimeSeriesSplit(n_splits=config.models.cv_splits)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=config.models.n_iter_search,
        cv=tscv,
        scoring=config.models.scoring,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    logger.info(f"Best LightGBM params: {search.best_params_}")

    best_model = search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    metrics = {
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_r2': r2_score(y_test, y_pred_test),
        'best_params': search.best_params_
    }

    logger.info(f"LightGBM - Test MAE: {metrics['test_mae']:.3f}, Test RMSE: {metrics['test_rmse']:.3f}")
    return best_model, metrics


def train_fallback_model(X_train, y_train, X_test, y_test,
                         numeric_features, categorical_features, config):
    """XGBoost or RandomForest fallback when LightGBM isn't installed."""
    if XGBOOST_AVAILABLE:
        logger.info("Training XGBoost (LightGBM fallback)...")
        model_name = "xgboost"

        preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(random_state=RANDOM_SEED, verbosity=0))
        ])

        param_grid = {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [3, 5, 7],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__subsample': [0.7, 0.8, 0.9, 1.0]
        }
    else:
        logger.info("Training RandomForest (LightGBM/XGBoost fallback)...")
        model_name = "randomforest"

        preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1))
        ])

        param_grid = {
            f'regressor__{k}': v for k, v in config.models.rf_params.items()
        }

    tscv = TimeSeriesSplit(n_splits=config.models.cv_splits)

    search = RandomizedSearchCV(
        pipeline, param_distributions=param_grid,
        n_iter=config.models.n_iter_search,
        cv=tscv, scoring=config.models.scoring,
        random_state=RANDOM_SEED, n_jobs=-1, verbose=1
    )

    search.fit(X_train, y_train)
    logger.info(f"Best {model_name} params: {search.best_params_}")

    best_model = search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    metrics = {
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_r2': r2_score(y_test, y_pred_test),
        'best_params': search.best_params_
    }

    logger.info(f"{model_name} - Test MAE: {metrics['test_mae']:.3f}, Test RMSE: {metrics['test_rmse']:.3f}")
    return best_model, metrics, model_name


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

    logger.info(f"Saved diagnostic plots for {model_name}")


def bootstrap_confidence_interval(y_true, y_pred, metric_func,
                                  n_bootstrap=1000, confidence=0.95):
    n = len(y_true)
    bootstrap_metrics = []
    rng = np.random.RandomState(RANDOM_SEED)

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
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
    logger.info(f"Random seed: {RANDOM_SEED}")

    artifact_dir = config.get_artifact_dir()
    save_config(config, artifact_dir / 'config.yaml')

    start_time = time.time()

    try:
        # ---- LOAD DATA ----
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Loading Data")
        logger.info("=" * 80)

        df, data_mode = load_data(config)
        logger.info(f"Data mode: {data_mode}")

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

        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")

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

        # ---- MODEL COMPARISON ----
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Model Comparison")
        logger.info("=" * 80)

        comparison_df = pd.DataFrame({
            'Model': ['Naive', 'Ridge', best_model_name.title()],
            'Train MAE': [all_metrics[m]['train_mae'] for m in ['naive', 'ridge', 'best']],
            'Test MAE': [all_metrics[m]['test_mae'] for m in ['naive', 'ridge', 'best']],
            'Train RMSE': [all_metrics[m]['train_rmse'] for m in ['naive', 'ridge', 'best']],
            'Test RMSE': [all_metrics[m]['test_rmse'] for m in ['naive', 'ridge', 'best']],
            'Test R²': [all_metrics[m]['test_r2'] for m in ['naive', 'ridge', 'best']],
        })

        logger.info("\n" + comparison_df.to_string(index=False))
        comparison_df.to_csv(artifact_dir / 'model_comparison.csv', index=False)

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
                    f"Only {len(res_line)} test obs for {line_name}, using global quantiles"
                )

        # re-save with actual quantiles
        save_feature_metadata(
            numeric_features, categorical_features, artifact_dir,
            residual_quantiles=residual_quantiles,
        )
        logger.info(
            "Per-line residual quantiles computed for %d lines",
            len(residual_quantiles) - 1,
        )

        # ---- BOOTSTRAP CIs ----
        logger.info("\n--- Computing Bootstrap Confidence Intervals ---")

        mae_point, mae_lower, mae_upper = bootstrap_confidence_interval(
            y_test, y_pred_best, mean_absolute_error
        )
        rmse_func = lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))
        rmse_point, rmse_lower, rmse_upper = bootstrap_confidence_interval(
            y_test, y_pred_best, rmse_func
        )

        logger.info(f"Best model Test MAE: {mae_point:.3f} (95% CI: [{mae_lower:.3f}, {mae_upper:.3f}])")
        logger.info(f"Best model Test RMSE: {rmse_point:.3f} (95% CI: [{rmse_lower:.3f}, {rmse_upper:.3f}])")

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
            logger.info(f"Saved {name} model")

        save_metrics(all_metrics, artifact_dir / 'all_metrics.json')

        test_predictions = pd.DataFrame({
            'timestamp': test_df['timestamp'].values,
            'line': test_df['line'].values,
            'actual': y_test.values,
            'pred_naive': models['naive'].predict(X_test),
            'pred_ridge': models['ridge'].predict(X_test),
            'pred_best': models['best'].predict(X_test)
        })
        test_predictions.to_csv(artifact_dir / 'test_predictions.csv', index=False)

        X_train.to_parquet(artifact_dir / 'X_train.parquet')
        X_test.to_parquet(artifact_dir / 'X_test.parquet')
        logger.info("Saved featured train/test data for SHAP analysis")

        with open(artifact_dir / 'best_model_name.txt', 'w') as f:
            f.write(best_model_name)

        # ---- DONE ----
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {format_duration(elapsed)}")
        logger.info(f"Artifact directory: {artifact_dir}")
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best model Test MAE: {all_metrics['best']['test_mae']:.3f}")
        logger.info(f"Improvement over naive: {(1 - all_metrics['best']['test_mae']/all_metrics['naive']['test_mae'])*100:.1f}%")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
