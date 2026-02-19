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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
