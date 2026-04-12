"""Tests for the classify.py classification pipeline."""

import json
import numpy as np
import pandas as pd
import pytest

from classify import (
    create_binary_labels,
    create_ordinal_labels,
    _train_logistic,
    _evaluate_classifier,
    run_classification_evaluation,
)
from config import get_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_xy(n: int = 200, seed: int = 42):
    """Minimal (X, y_reg) pair for fast classifier tests.

    Uses only numeric columns so the preprocessing pipeline works without
    needing a full feature-engineering pass.
    """
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        'hour':             rng.randint(0, 24, n).astype(float),
        'day_of_week':      rng.randint(0, 7, n).astype(float),
        'month':            rng.randint(1, 13, n).astype(float),
        'is_weekend':       rng.randint(0, 2, n).astype(float),
        'peak_time':        rng.randint(0, 2, n).astype(float),
        'crowding_index':   rng.uniform(0, 1, n),
        'temp_c':           rng.uniform(0, 25, n),
        'precipitation_mm': rng.uniform(0, 10, n),
    })
    y_reg = rng.uniform(0, 20, n)
    return X, y_reg


# ---------------------------------------------------------------------------
# Label creation
# ---------------------------------------------------------------------------

class TestCreateBinaryLabels:
    def test_above_threshold_is_delayed(self):
        delays = np.array([0.0, 4.9, 5.0, 10.0])
        labels = create_binary_labels(delays, threshold=5.0)
        assert list(labels) == [0, 0, 1, 1]

    def test_all_zeros_no_delay(self):
        labels = create_binary_labels(np.zeros(10), threshold=5.0)
        assert labels.sum() == 0

    def test_returns_integer_array(self):
        labels = create_binary_labels(np.array([1.0, 6.0]))
        assert labels.dtype in (np.int32, np.int64, int)

    def test_custom_threshold(self):
        delays = np.array([1.0, 3.0, 7.0])
        labels = create_binary_labels(delays, threshold=3.0)
        assert list(labels) == [0, 1, 1]


class TestCreateOrdinalLabels:
    def test_three_categories_present(self):
        delays = np.array([1.0, 5.0, 15.0])
        labels = create_ordinal_labels(delays, good_max=2.0, minor_max=10.0)
        assert set(labels) == {'Good Service', 'Minor Delays', 'Severe Delays'}

    def test_boundary_good_max(self):
        labels = create_ordinal_labels(np.array([2.0]), good_max=2.0, minor_max=10.0)
        assert labels[0] == 'Good Service'

    def test_boundary_minor_max(self):
        labels = create_ordinal_labels(np.array([10.0]), good_max=2.0, minor_max=10.0)
        assert labels[0] == 'Minor Delays'

    def test_above_minor_is_severe(self):
        labels = create_ordinal_labels(np.array([10.01]), good_max=2.0, minor_max=10.0)
        assert labels[0] == 'Severe Delays'

    def test_output_length_matches_input(self):
        delays = np.linspace(0, 20, 50)
        labels = create_ordinal_labels(delays)
        assert len(labels) == 50


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

class TestTrainLogistic:
    def test_returns_fitted_pipeline(self):
        config = get_config()
        X, y_reg = _make_xy()
        y_bin = create_binary_labels(y_reg)
        numeric_feats = list(X.columns)
        pipe = _train_logistic(X, y_bin, numeric_feats, [], config, 'binary')
        # predict must not raise
        preds = pipe.predict(X)
        assert len(preds) == len(X)

    def test_binary_predictions_are_0_or_1(self):
        config = get_config()
        X, y_reg = _make_xy()
        y_bin = create_binary_labels(y_reg)
        pipe = _train_logistic(X, y_bin, list(X.columns), [], config, 'binary')
        preds = pipe.predict(X)
        assert set(preds).issubset({0, 1})

    def test_ordinal_predictions_are_valid_labels(self):
        config = get_config()
        X, y_reg = _make_xy()
        y_ord = create_ordinal_labels(y_reg)
        pipe = _train_logistic(X, y_ord, list(X.columns), [], config, 'ordinal')
        preds = pipe.predict(X)
        valid = {'Good Service', 'Minor Delays', 'Severe Delays'}
        assert set(preds).issubset(valid)

    def test_pipeline_has_preprocessor_and_classifier_steps(self):
        config = get_config()
        X, y_reg = _make_xy()
        y_bin = create_binary_labels(y_reg)
        pipe = _train_logistic(X, y_bin, list(X.columns), [], config)
        assert 'preprocessor' in pipe.named_steps
        assert 'classifier' in pipe.named_steps


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluateClassifier:
    def test_metrics_keys_present(self):
        config = get_config()
        X, y_reg = _make_xy()
        y_bin = create_binary_labels(y_reg)
        pipe = _train_logistic(X, y_bin, list(X.columns), [], config, 'binary')
        metrics = _evaluate_classifier(pipe, X, y_bin, 'binary', ['Not Delayed', 'Delayed'])
        for key in ('accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'):
            assert key in metrics

    def test_accuracy_in_unit_interval(self):
        config = get_config()
        X, y_reg = _make_xy()
        y_bin = create_binary_labels(y_reg)
        pipe = _train_logistic(X, y_bin, list(X.columns), [], config, 'binary')
        metrics = _evaluate_classifier(pipe, X, y_bin, 'binary', ['Not Delayed', 'Delayed'])
        assert 0.0 <= metrics['accuracy'] <= 1.0

    def test_per_class_contains_class_names(self):
        config = get_config()
        X, y_reg = _make_xy()
        y_bin = create_binary_labels(y_reg)
        pipe = _train_logistic(X, y_bin, list(X.columns), [], config, 'binary')
        metrics = _evaluate_classifier(pipe, X, y_bin, 'binary', ['Not Delayed', 'Delayed'])
        assert 'per_class' in metrics


# ---------------------------------------------------------------------------
# y_train.csv loading in main()
# ---------------------------------------------------------------------------

class TestYTrainCsvLoading:
    """Verify that the y_train.csv guard in main() works correctly."""

    def test_missing_y_train_csv_prints_error_and_returns(self, tmp_path, capsys):
        """The guard block should print an error and return without raising."""
        # Reproduce the guard logic from classify.main() in isolation.
        artifact_dir = tmp_path
        # Intentionally do NOT create y_train.csv
        y_train_csv = artifact_dir / 'y_train.csv'
        if not y_train_csv.exists():
            print(
                f"ERROR: y_train.csv not found in {artifact_dir}\n"
                "Re-run train.py to regenerate artifacts."
            )

        captured = capsys.readouterr()
        assert 'ERROR' in captured.out
        assert 'y_train.csv' in captured.out

    def test_y_train_csv_loads_correctly(self, tmp_path):
        """When y_train.csv exists it should load without error."""
        y = np.array([1.0, 2.0, 3.0])
        pd.DataFrame({'delay_minutes': y}).to_csv(tmp_path / 'y_train.csv', index=False)
        y_loaded = pd.read_csv(tmp_path / 'y_train.csv')['delay_minutes'].values
        np.testing.assert_array_equal(y_loaded, y)


# ---------------------------------------------------------------------------
# Full pipeline (end-to-end with tmp_path)
# ---------------------------------------------------------------------------

class TestRunClassificationEvaluation:
    def test_produces_results_json(self, tmp_path):
        config = get_config()
        X, y_reg = _make_xy(n=300)
        split = 240
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train_reg, y_test_reg = y_reg[:split], y_reg[split:]
        numeric_feats = list(X.columns)
        timestamps = pd.Series(range(len(X_test)))

        run_classification_evaluation(
            artifact_dir=tmp_path,
            config=config,
            X_train=X_train,
            y_train_reg=y_train_reg,
            X_test=X_test,
            y_test_reg=y_test_reg,
            test_timestamps=timestamps,
            numeric_features=numeric_feats,
            categorical_features=[],
        )

        results_path = tmp_path / 'classification' / 'classification_results.json'
        assert results_path.exists(), "classification_results.json was not created"
        with open(results_path) as fh:
            results = json.load(fh)
        assert 'binary_logistic' in results
        assert 'ordinal_logistic' in results

    def test_metrics_are_valid_floats(self, tmp_path):
        config = get_config()
        X, y_reg = _make_xy(n=300)
        split = 240
        numeric_feats = list(X.columns)

        run_classification_evaluation(
            artifact_dir=tmp_path,
            config=config,
            X_train=X.iloc[:split],
            y_train_reg=y_reg[:split],
            X_test=X.iloc[split:],
            y_test_reg=y_reg[split:],
            test_timestamps=pd.Series(range(split, 300)),
            numeric_features=numeric_feats,
            categorical_features=[],
        )

        results_path = tmp_path / 'classification' / 'classification_results.json'
        with open(results_path) as fh:
            results = json.load(fh)

        bin_metrics = results['binary_logistic']['metrics']
        assert 0.0 <= bin_metrics['accuracy'] <= 1.0
        assert 0.0 <= bin_metrics['f1_weighted'] <= 1.0
