"""Explainability pipeline tests (mocked SHAP + I/O)."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_feature_metadata(n_features=5):
    numeric = [f'feat_{i}' for i in range(n_features)]
    return {
        'numeric_features':    numeric,
        'categorical_features': ['line'],
        'all_features':        numeric + ['line'],
    }


def _make_X(n_rows=20, n_features=5):
    data = {f'feat_{i}': np.random.randn(n_rows) for i in range(n_features)}
    data['line'] = ['Central'] * n_rows
    return pd.DataFrame(data)


import joblib


class TestLoadArtifacts:

    def test_loads_all_three_models(self, tmp_path):
        for name in ['naive_model.pkl', 'ridge_model.pkl', 'best_model.pkl',
                     'feature_metadata.pkl']:
            (tmp_path / name).touch()

        fake_meta = _make_feature_metadata()
        fake_model = MagicMock()

        with patch('explain.joblib.load') as mock_load:
            def _side_effect(path):
                if 'feature_metadata' in str(path):
                    return fake_meta
                return fake_model
            mock_load.side_effect = _side_effect

            from explain import load_artifacts
            artifacts = load_artifacts(tmp_path)

        assert 'naive_model'      in artifacts
        assert 'ridge_model'      in artifacts
        assert 'best_model'       in artifacts
        assert 'feature_metadata' in artifacts

    def test_missing_model_files_are_skipped(self, tmp_path):
        # Only the feature_metadata file is present; no model pickles.
        (tmp_path / 'feature_metadata.pkl').touch()

        fake_meta = _make_feature_metadata()

        with patch('explain.joblib.load', return_value=fake_meta):
            from explain import load_artifacts
            artifacts = load_artifacts(tmp_path)

        assert 'naive_model'      not in artifacts
        assert 'ridge_model'      not in artifacts
        assert 'best_model'       not in artifacts
        assert 'feature_metadata' in artifacts


class TestGenerateTextExplanations:

    def _make_inputs(self, n_rows=3, n_features=5):
        from config import get_config
        shap_array    = np.random.randn(n_rows, n_features)
        feature_names = [f'feat_{i}' for i in range(n_features)]
        X             = pd.DataFrame(shap_array, columns=feature_names)
        y_pred        = np.array([4.2, 7.1, 1.5])[:n_rows]
        config        = get_config()
        return shap_array, feature_names, X, y_pred, config

    def test_returns_list_of_strings(self):
        from explain import generate_text_explanations

        shap_array, feature_names, X, y_pred, config = self._make_inputs()
        results = generate_text_explanations(shap_array, feature_names, X, y_pred, config)

        assert isinstance(results, list)
        assert len(results) > 0
        for item in results:
            assert isinstance(item, str)

    def test_returns_fallback_when_shap_none(self):
        from explain import generate_text_explanations
        from config import get_config

        results = generate_text_explanations(
            None, ['feat_0'], pd.DataFrame([{'feat_0': 1.0}]),
            np.array([3.0]), get_config()
        )
        assert isinstance(results, list)
        assert len(results) > 0

    def test_n_local_examples_respected(self):
        from explain import generate_text_explanations
        from config import get_config

        config = get_config()
        n = config.explainability.n_local_examples
        shap_array    = np.random.randn(n + 5, 4)
        feature_names = [f'feat_{i}' for i in range(4)]
        X             = pd.DataFrame(shap_array, columns=feature_names)
        y_pred        = np.ones(n + 5) * 3.0

        results = generate_text_explanations(shap_array, feature_names, X, y_pred, config)
        assert len(results) == n


class TestLoadArtifactsExtended:

    def test_loads_test_predictions_csv(self, tmp_path):
        import pandas as pd
        from explain import load_artifacts

        (tmp_path / 'feature_metadata.pkl').touch()
        fake_meta = _make_feature_metadata()

        # create test_predictions.csv
        pd.DataFrame({'actual': [1.0], 'pred': [1.1]}).to_csv(
            tmp_path / 'test_predictions.csv', index=False
        )

        with patch('explain.joblib.load', return_value=fake_meta):
            artifacts = load_artifacts(tmp_path)

        assert 'test_predictions' in artifacts
        assert len(artifacts['test_predictions']) == 1

    def test_loads_parquet_files(self, tmp_path):
        import pandas as pd
        from explain import load_artifacts

        (tmp_path / 'feature_metadata.pkl').touch()
        fake_meta = _make_feature_metadata()
        df = pd.DataFrame({'a': [1.0, 2.0]})
        df.to_parquet(tmp_path / 'X_train.parquet')
        df.to_parquet(tmp_path / 'X_test.parquet')

        with patch('explain.joblib.load', return_value=fake_meta):
            artifacts = load_artifacts(tmp_path)

        assert 'X_train' in artifacts
        assert 'X_test' in artifacts

    def test_loads_best_model_name_txt(self, tmp_path):
        from explain import load_artifacts

        (tmp_path / 'feature_metadata.pkl').touch()
        (tmp_path / 'best_model_name.txt').write_text('lightgbm')
        fake_meta = _make_feature_metadata()

        with patch('explain.joblib.load', return_value=fake_meta):
            artifacts = load_artifacts(tmp_path)

        assert artifacts['best_model_name'] == 'lightgbm'


class TestShapCacheEdgeCases:

    def test_load_corrupt_cache_returns_none(self, tmp_path):
        """Covers the except branch when cache is unreadable."""
        from explain import load_shap_cache

        # Write a corrupt pickle file
        (tmp_path / 'shap_cache.pkl').write_bytes(b'not a valid pickle')
        (tmp_path / 'best_model.pkl').write_bytes(b'model')
        X = pd.DataFrame({'a': [1.0]})

        sv, fn = load_shap_cache(tmp_path, X)
        assert sv is None
        assert fn is None


class TestExplainMain:

    def test_main_returns_early_when_no_artifacts(self):
        """Covers the 'no run_* dirs' early-return branch in explain.main()."""
        import explain

        with patch('explain.get_latest_run_id', return_value=None):
            # Should not raise, just log and return
            explain.main()

    def _make_run_config(self, tmp_path):
        """Return a Config with artifacts_dir pointing to tmp_path."""
        from config import get_config
        config = get_config()
        config.paths.artifacts_dir = tmp_path
        return config

    def test_main_skips_when_no_best_model(self, tmp_path):
        """Covers setup + 'best_model not in artifacts' early-return."""
        import explain
        import logging

        run_dir = tmp_path / 'run_test'
        run_dir.mkdir()
        config = self._make_run_config(tmp_path)

        with patch('explain.get_config', return_value=config), \
             patch('explain.get_latest_run_id', return_value='run_test'), \
             patch('explain.setup_logging', return_value=logging.getLogger('t')), \
             patch('explain.load_artifacts', return_value={'feature_metadata': {}}):
            explain.main()

    def test_main_skips_when_no_x_data(self, tmp_path):
        """Covers the 'X_train/X_test not found' early-return."""
        import explain
        import logging

        run_dir = tmp_path / 'run_test'
        run_dir.mkdir()
        config = self._make_run_config(tmp_path)

        with patch('explain.get_config', return_value=config), \
             patch('explain.get_latest_run_id', return_value='run_test'), \
             patch('explain.setup_logging', return_value=logging.getLogger('t')), \
             patch('explain.load_artifacts', return_value={
                 'best_model': MagicMock(), 'feature_metadata': {}
             }):
            explain.main()

    def test_main_handles_shap_not_available(self, tmp_path):
        """Covers SHAP-not-installed branch and writes explanations.txt."""
        import explain
        import logging

        run_dir = tmp_path / 'run_test'
        run_dir.mkdir()
        config = self._make_run_config(tmp_path)
        X = pd.DataFrame({'a': [1.0, 2.0]})

        with patch('explain.get_config', return_value=config), \
             patch('explain.get_latest_run_id', return_value='run_test'), \
             patch('explain.setup_logging', return_value=logging.getLogger('t')), \
             patch('explain.SHAP_AVAILABLE', False), \
             patch('explain.load_artifacts', return_value={
                 'best_model': MagicMock(), 'feature_metadata': {}, 'X_train': X, 'X_test': X
             }):
            explain.main()

        assert (run_dir / 'explanations.txt').exists()


def _make_ridge_pipeline(n=40):
    """Minimal fitted Ridge Pipeline for explain tests."""
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer

    rng = np.random.RandomState(42)
    X = pd.DataFrame({'a': rng.randn(n), 'b': rng.randn(n)})
    y = 2.0 * X['a'] - 1.5 * X['b'] + rng.randn(n) * 0.5

    preprocessor = ColumnTransformer([('num', StandardScaler(), ['a', 'b'])])
    pipe = Pipeline([('preprocessor', preprocessor), ('regressor', Ridge())])
    pipe.fit(X, y)
    return pipe, X


class TestShapCache:

    def test_cache_key_returns_md5_string(self, tmp_path):
        from explain import _shap_cache_key
        (tmp_path / 'best_model.pkl').write_bytes(b'dummy')
        X = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        key = _shap_cache_key(tmp_path, X)
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hex digest

    def test_cache_key_differs_for_different_data(self, tmp_path):
        from explain import _shap_cache_key
        (tmp_path / 'best_model.pkl').write_bytes(b'dummy')
        X1 = pd.DataFrame({'a': [1.0, 2.0]})
        X2 = pd.DataFrame({'a': [9.0, 8.0]})
        assert _shap_cache_key(tmp_path, X1) != _shap_cache_key(tmp_path, X2)

    def test_load_cache_miss_when_no_file(self, tmp_path):
        from explain import load_shap_cache
        X = pd.DataFrame({'a': [1.0]})
        sv, fn = load_shap_cache(tmp_path, X)
        assert sv is None
        assert fn is None

    def test_save_and_load_roundtrip(self, tmp_path):
        from explain import save_shap_cache, load_shap_cache
        (tmp_path / 'best_model.pkl').write_bytes(b'model')
        X = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        fake_values = np.array([[0.1, 0.2], [0.3, 0.4]])
        feature_names = ['a', 'b']

        save_shap_cache(tmp_path, X, fake_values, feature_names)
        sv, fn = load_shap_cache(tmp_path, X)

        np.testing.assert_array_equal(sv, fake_values)
        assert fn == feature_names

    def test_load_cache_miss_on_key_mismatch(self, tmp_path):
        from explain import save_shap_cache, load_shap_cache
        (tmp_path / 'best_model.pkl').write_bytes(b'model_v1')
        X_old = pd.DataFrame({'a': [1.0, 2.0]})
        save_shap_cache(tmp_path, X_old, np.array([[0.1]]), ['a'])

        # simulate model file changing (different content → different mtime or hash)
        X_new = pd.DataFrame({'a': [9.0, 8.0]})
        sv, fn = load_shap_cache(tmp_path, X_new)
        assert sv is None
        assert fn is None


class TestCreateShapExplainer:

    def test_returns_none_when_shap_unavailable(self):
        from explain import create_shap_explainer
        from config import get_config
        with patch('explain.SHAP_AVAILABLE', False):
            result = create_shap_explainer(MagicMock(), pd.DataFrame(), get_config())
        assert result is None

    def test_returns_explainer_for_pipeline_model(self):
        from explain import create_shap_explainer
        from config import get_config
        pipe, X = _make_ridge_pipeline(n=40)
        config = get_config()
        config.explainability.shap_background_size = 100  # larger than n → no subsampling

        explainer = create_shap_explainer(pipe, X, config)
        assert explainer is not None

    def test_subsamples_large_background(self):
        from explain import create_shap_explainer
        from config import get_config
        pipe, X = _make_ridge_pipeline(n=50)
        config = get_config()
        config.explainability.shap_background_size = 5  # tiny → forces subsampling

        explainer = create_shap_explainer(pipe, X, config)
        assert explainer is not None

    def test_returns_none_on_broken_model(self):
        from explain import create_shap_explainer
        from config import get_config
        bad_model = MagicMock()
        bad_model.named_steps = {'preprocessor': None, 'regressor': None}
        config = get_config()
        # Will raise inside try/except → returns None
        result = create_shap_explainer(bad_model, pd.DataFrame({'a': [1.0]}), config)
        assert result is None


class TestComputeShapValues:

    def test_returns_none_when_explainer_none(self):
        from explain import compute_shap_values
        from config import get_config
        result = compute_shap_values(None, MagicMock(), pd.DataFrame(), get_config())
        assert result is None

    def test_returns_shap_values_for_real_model(self):
        from explain import create_shap_explainer, compute_shap_values
        from config import get_config
        pipe, X = _make_ridge_pipeline(n=40)
        config = get_config()
        config.explainability.shap_background_size = 100

        explainer = create_shap_explainer(pipe, X, config)
        shap_values = compute_shap_values(explainer, pipe, X, config)

        assert shap_values is not None
        assert hasattr(shap_values, 'values')
        assert shap_values.values.shape == (len(X), X.shape[1])

    def test_subsamples_when_max_samples_set(self):
        from explain import create_shap_explainer, compute_shap_values
        from config import get_config
        pipe, X = _make_ridge_pipeline(n=40)
        config = get_config()
        config.explainability.shap_background_size = 100

        explainer = create_shap_explainer(pipe, X, config)
        shap_values = compute_shap_values(explainer, pipe, X, config, max_samples=10)

        assert shap_values is not None
        assert shap_values.values.shape[0] == 10


class TestCreateFeatureImportancePlot:

    def test_noop_when_shap_none(self, tmp_path):
        from explain import create_feature_importance_plot
        from config import get_config
        create_feature_importance_plot(None, [], tmp_path, get_config())
        assert not (tmp_path / 'feature_importance.png').exists()

    def test_saves_plot_and_csv_with_numpy_array(self, tmp_path):
        from explain import create_feature_importance_plot
        from config import get_config
        rng = np.random.RandomState(0)
        # Pass a plain numpy array (no .values attr) — covers the else branch
        shap_array = rng.randn(20, 4)
        feature_names = ['feat_0', 'feat_1', 'feat_2', 'feat_3']
        config = get_config()
        config.explainability.top_n_features = 4

        create_feature_importance_plot(shap_array, feature_names, tmp_path, config)

        assert (tmp_path / 'feature_importance.png').exists()
        assert (tmp_path / 'feature_importance.csv').exists()


class TestCreateShapPlots:

    def test_noop_when_shap_none(self, tmp_path):
        from explain import create_shap_plots
        from config import get_config
        create_shap_plots(None, [], tmp_path, get_config())
        assert not list(tmp_path.glob('*.png'))

    def test_creates_plots_with_real_shap_values(self, tmp_path):
        from explain import create_shap_explainer, compute_shap_values, create_shap_plots
        from config import get_config
        pipe, X = _make_ridge_pipeline(n=40)
        config = get_config()
        config.explainability.shap_background_size = 100
        config.explainability.top_n_features = 2
        config.explainability.n_local_examples = 1

        explainer = create_shap_explainer(pipe, X, config)
        shap_values = compute_shap_values(explainer, pipe, X, config)
        feature_names = ['a', 'b']

        create_shap_plots(shap_values, feature_names, tmp_path, config)

        assert (tmp_path / 'shap_beeswarm.png').exists()
        assert (tmp_path / 'shap_bar.png').exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
