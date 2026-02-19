"""Training pipeline tests."""

import numpy as np
import pandas as pd
import pytest
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
