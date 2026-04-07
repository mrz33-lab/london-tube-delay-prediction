"""Training pipeline tests."""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch
from sklearn.metrics import mean_absolute_error

from train import (
    NaiveBaselineModel,
    bootstrap_confidence_interval,
    train_naive_baseline,
)


def _make_xy(n=50, n_lines=2):
    lines = [f'Line{i % n_lines}' for i in range(n)]
    delays = [float(i % 10) for i in range(n)]
    X = pd.DataFrame({'line': lines, 'hour': [8] * n})
    y = pd.Series(delays, name='delay_minutes')
    return X, y


class TestNaiveBaselineModel:

    def test_fit_stores_last_delay_per_line(self):
        X, y = _make_xy(n=10, n_lines=2)
        model = NaiveBaselineModel()
        model.fit(X, y)

        for line in X['line'].unique():
            idx = X[X['line'] == line].index[-1]
            assert model.last_delays_[line] == pytest.approx(y.iloc[idx])

    def test_fit_stores_global_mean(self):
        X, y = _make_xy(n=20, n_lines=3)
        model = NaiveBaselineModel()
        model.fit(X, y)
        assert model.global_mean_ == pytest.approx(y.mean())

    def test_fit_requires_line_column(self):
        X = pd.DataFrame({'hour': [8, 9, 10]})
        y = pd.Series([1.0, 2.0, 3.0])
        model = NaiveBaselineModel()
        with pytest.raises(ValueError, match="line"):
            model.fit(X, y)

    def test_predict_returns_correct_shape(self):
        X, y = _make_xy(n=30, n_lines=2)
        model = NaiveBaselineModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_predict_uses_global_mean_for_unseen_line(self):
        X_train = pd.DataFrame({'line': ['Central'] * 5, 'hour': [8] * 5})
        y_train = pd.Series([3.0, 4.0, 5.0, 6.0, 7.0])
        model = NaiveBaselineModel()
        model.fit(X_train, y_train)

        X_unseen = pd.DataFrame({'line': ['Unknown Line'], 'hour': [8]})
        pred = model.predict(X_unseen)
        assert pred[0] == pytest.approx(y_train.mean())

    def test_predict_returns_last_seen_delay(self):
        X_train = pd.DataFrame({'line': ['Central', 'Central', 'Jubilee'], 'hour': [8, 9, 10]})
        y_train = pd.Series([2.0, 8.0, 5.0])
        model = NaiveBaselineModel()
        model.fit(X_train, y_train)

        X_test = pd.DataFrame({'line': ['Central', 'Jubilee'], 'hour': [8, 8]})
        preds = model.predict(X_test)
        assert preds[0] == pytest.approx(8.0)  # last Central delay
        assert preds[1] == pytest.approx(5.0)  # last Jubilee delay


class TestBootstrapConfidenceInterval:

    def test_returns_three_values(self):
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1] * 10)
        result = bootstrap_confidence_interval(y_true, y_pred, mean_absolute_error)
        assert len(result) == 3

    def test_lower_le_point_le_upper(self):
        y_true = pd.Series(range(50), dtype=float)
        y_pred = np.arange(50, dtype=float) + np.random.RandomState(0).randn(50) * 2
        point, lower, upper = bootstrap_confidence_interval(
            y_true, y_pred, mean_absolute_error, n_bootstrap=200
        )
        assert lower <= point
        assert point <= upper

    def test_point_estimate_matches_metric(self):
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        point, _, _ = bootstrap_confidence_interval(
            y_true, y_pred, mean_absolute_error, n_bootstrap=100
        )
        assert point == pytest.approx(mean_absolute_error(y_true, y_pred))

    def test_wider_ci_at_lower_confidence(self):
        y_true = pd.Series(np.arange(100, dtype=float))
        y_pred = np.arange(100, dtype=float) + np.random.RandomState(1).randn(100) * 3
        _, lo_95, hi_95 = bootstrap_confidence_interval(
            y_true, y_pred, mean_absolute_error, n_bootstrap=500, confidence=0.95
        )
        _, lo_50, hi_50 = bootstrap_confidence_interval(
            y_true, y_pred, mean_absolute_error, n_bootstrap=500, confidence=0.50
        )
        assert (hi_95 - lo_95) >= (hi_50 - lo_50)


class TestTrainNaiveBaseline:

    def test_returns_model_and_metrics(self):
        X, y = _make_xy(n=100, n_lines=3)
        split = 80
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model, metrics = train_naive_baseline(X_train, y_train, X_test, y_test)

        assert isinstance(model, NaiveBaselineModel)
        for key in ('train_mae', 'train_rmse', 'train_r2', 'test_mae', 'test_rmse', 'test_r2'):
            assert key in metrics, f"Missing metric: {key}"

    def test_metrics_are_finite(self):
        X, y = _make_xy(n=100, n_lines=2)
        split = 80
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        _, metrics = train_naive_baseline(X_train, y_train, X_test, y_test)

        for key, val in metrics.items():
            assert np.isfinite(val), f"Metric {key} is not finite: {val}"

    def test_test_mae_non_negative(self):
        X, y = _make_xy(n=100, n_lines=2)
        split = 80
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        _, metrics = train_naive_baseline(X_train, y_train, X_test, y_test)
        assert metrics['test_mae'] >= 0.0


class TestTrainHelpers:

    def test_get_git_hash_returns_string(self):
        from train import _get_git_hash
        result = _get_git_hash()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_git_hash_returns_unknown_on_failure(self):
        from train import _get_git_hash
        with patch('train.subprocess.check_output', side_effect=OSError("git not found")):
            result = _get_git_hash()
        assert result == 'unknown'

    def test_save_model_info_creates_valid_json(self, tmp_path):
        from train import _save_model_info
        metrics = {'test_mae': 1.5, 'test_rmse': 2.3}
        _save_model_info(tmp_path, 'lightgbm', 'REAL', metrics)
        info_path = tmp_path / 'model_info.json'
        assert info_path.exists()
        info = json.loads(info_path.read_text())
        assert 'git_commit' in info
        assert 'trained_at' in info
        assert 'python_version' in info
        assert 'sklearn_version' in info
        assert info['best_model'] == 'lightgbm'
        assert info['data_mode'] == 'REAL'
        assert info['test_mae'] == pytest.approx(1.5, abs=0.001)

    def test_save_model_info_lightgbm_not_installed(self, tmp_path):
        """Covers the ImportError branch for lightgbm version."""
        import sys
        from train import _save_model_info
        with patch.dict(sys.modules, {'lightgbm': None}):
            _save_model_info(tmp_path, 'ridge', 'PROTOTYPE', {'test_mae': 2.0, 'test_rmse': 3.0})
        info = json.loads((tmp_path / 'model_info.json').read_text())
        assert info['lightgbm_version'] == 'not installed'
        assert info['best_model'] == 'ridge'

    def test_create_diagnostic_plots_saves_three_files(self, tmp_path):
        from train import create_diagnostic_plots
        y_test = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0])
        create_diagnostic_plots(y_test, y_pred, 'mymodel', tmp_path)
        assert (tmp_path / 'mymodel_residual_hist.png').exists()
        assert (tmp_path / 'mymodel_pred_vs_actual.png').exists()
        assert (tmp_path / 'mymodel_residual_vs_pred.png').exists()


class TestTrainLightGBM:
    """Tests for train_lightgbm — uses 1 Optuna trial to stay fast."""

    def test_returns_model_and_expected_metrics(self, tiny_multiline_df):
        from train import train_lightgbm
        from config import get_config
        from features import engineer_features, get_feature_columns, prepare_features_for_model

        config = get_config()
        config.models.optuna_n_trials = 1
        config.models.cv_splits = 2

        df = tiny_multiline_df
        df_featured = engineer_features(df, config, is_training=True)
        split_idx = int(len(df_featured) * 0.8)
        train_df = df_featured.iloc[:split_idx]
        test_df = df_featured.iloc[split_idx:]

        numeric_feats, cat_feats, all_feats = get_feature_columns(train_df, config)
        X_train, y_train = prepare_features_for_model(train_df, all_feats, config.features.target_column)
        X_test, y_test = prepare_features_for_model(test_df, all_feats, config.features.target_column)

        model, metrics = train_lightgbm(
            X_train, y_train, X_test, y_test,
            numeric_feats, cat_feats, config
        )

        for key in ('train_mae', 'test_mae', 'test_rmse', 'test_r2', 'best_params'):
            assert key in metrics, f"Missing metric: {key}"
        assert metrics['test_mae'] >= 0.0
        assert hasattr(model, 'predict')


class TestTrainFallbackModel:
    """Tests for train_fallback_model — covers XGBoost and RandomForest paths."""

    def test_xgboost_path_returns_model_and_metrics(self, tiny_multiline_df):
        from train import train_fallback_model, XGBOOST_AVAILABLE
        from config import get_config
        from features import engineer_features, get_feature_columns, prepare_features_for_model

        if not XGBOOST_AVAILABLE:
            pytest.skip("XGBoost not installed")

        config = get_config()
        config.models.cv_splits = 2
        config.models.n_iter_search = 1

        df = tiny_multiline_df
        df_featured = engineer_features(df, config, is_training=True)
        split_idx = int(len(df_featured) * 0.8)
        train_df = df_featured.iloc[:split_idx]
        test_df = df_featured.iloc[split_idx:]

        numeric_feats, cat_feats, all_feats = get_feature_columns(train_df, config)
        X_train, y_train = prepare_features_for_model(train_df, all_feats, config.features.target_column)
        X_test, y_test = prepare_features_for_model(test_df, all_feats, config.features.target_column)

        # XGBOOST_AVAILABLE is True; call without patching → uses XGBoost path
        model, metrics, model_name = train_fallback_model(
            X_train, y_train, X_test, y_test,
            numeric_feats, cat_feats, config
        )

        assert model_name == 'xgboost'
        for key in ('train_mae', 'test_mae', 'test_rmse', 'test_r2'):
            assert key in metrics
        assert metrics['test_mae'] >= 0.0

    def test_randomforest_path_returns_model_and_metrics(self, tiny_multiline_df):
        from train import train_fallback_model
        from config import get_config
        from features import engineer_features, get_feature_columns, prepare_features_for_model

        config = get_config()
        config.models.cv_splits = 2
        config.models.n_iter_search = 1
        config.models.rf_params = {
            'n_estimators': [10],
            'max_depth': [3],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
        }

        df = tiny_multiline_df
        df_featured = engineer_features(df, config, is_training=True)
        split_idx = int(len(df_featured) * 0.8)
        train_df = df_featured.iloc[:split_idx]
        test_df = df_featured.iloc[split_idx:]

        numeric_feats, cat_feats, all_feats = get_feature_columns(train_df, config)
        X_train, y_train = prepare_features_for_model(train_df, all_feats, config.features.target_column)
        X_test, y_test = prepare_features_for_model(test_df, all_feats, config.features.target_column)

        with patch('train.XGBOOST_AVAILABLE', False):
            model, metrics, model_name = train_fallback_model(
                X_train, y_train, X_test, y_test,
                numeric_feats, cat_feats, config
            )

        assert model_name == 'randomforest'
        for key in ('train_mae', 'test_mae', 'test_rmse', 'test_r2'):
            assert key in metrics, f"Missing metric: {key}"
        assert metrics['test_mae'] >= 0.0


class TestTrainMain:
    """Integration test covering train.main() via mocked IO and fast settings."""

    def test_main_full_pipeline_completes(self, tmp_path, tiny_multiline_df):
        import train
        from config import get_config

        config = get_config()
        artifacts_dir = tmp_path / 'artifacts'
        artifacts_dir.mkdir()
        config.paths.artifacts_dir = artifacts_dir
        config.models.cv_splits = 2
        config.models.n_iter_search = 1
        config.models.optuna_n_trials = 1
        config.models.bootstrap_block_size = 5

        tiny_df = tiny_multiline_df

        with patch('train.get_config', return_value=config), \
             patch('train.load_data', return_value=(tiny_df, 'PROTOTYPE')):
            train.main()

        run_dirs = list(artifacts_dir.glob('run_*'))
        assert len(run_dirs) == 1, "Expected exactly one run_* directory"
        artifact_dir = run_dirs[0]
        assert (artifact_dir / 'best_model.pkl').exists()
        assert (artifact_dir / 'naive_model.pkl').exists()
        assert (artifact_dir / 'ridge_model.pkl').exists()
        assert (artifact_dir / 'model_comparison.csv').exists()
        assert (artifact_dir / 'feature_metadata.pkl').exists()
        assert (artifact_dir / 'model_info.json').exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
