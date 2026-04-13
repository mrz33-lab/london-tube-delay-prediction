"""FutureDelayPredictor tests (mocked model)."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


# full feature list
_NUMERIC_FEATURES = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday', 'peak_time',
    'temp_c', 'precipitation_mm', 'humidity', 'crowding_index',
    'lag_delay_1', 'lag_delay_3', 'rolling_mean_delay_3', 'rolling_mean_delay_12',
    'rolling_std_delay_12', 'recent_disruption_rate',
    'temp_delta_1h', 'precipitation_delta_1h',
    'crowding_x_peak', 'precipitation_x_temp',
    'network_avg_delay', 'network_delay_volatility',
    'lines_disrupted_ratio', 'is_network_wide_disruption',
    'hour_sin', 'hour_cos', 'is_late_night', 'is_early_morning',
    'line_length_km', 'n_stations', 'n_interchange_stations',
    'is_deep_tube', 'zone_coverage',
    'trains_per_hour', 'service_headway_min', 'capacity_pressure',
]

_FEATURE_METADATA = {
    'numeric_features':    _NUMERIC_FEATURES,
    'categorical_features': ['line'],
    'all_features':        _NUMERIC_FEATURES + ['line'],
}


@pytest.fixture()
def mock_model():
    m = MagicMock()
    m.predict.return_value = np.array([4.2])
    return m


@pytest.fixture()
def predictor(mock_model):
    from future_prediction import FutureDelayPredictor

    with patch('future_prediction.joblib.load') as mock_load:
        mock_load.side_effect = [mock_model, _FEATURE_METADATA]
        instance = FutureDelayPredictor(
            model_path='fake_model.pkl',
            feature_metadata_path='fake_meta.pkl',
        )

    # expose mock model so tests can override predict.return_value
    instance.model = mock_model
    return instance


def _future_dt(hours=2):
    return datetime.now() + timedelta(hours=hours)


def test_predict_delay_returns_expected_keys(predictor):
    result = predictor.predict_delay('Central', _future_dt())
    for key in ('line', 'datetime', 'predicted_delay_minutes',
                'confidence_interval_95', 'status', 'features_used'):
        assert key in result, f"Missing key: {key}"


def test_predict_delay_non_negative(predictor):
    result = predictor.predict_delay('Central', _future_dt())
    assert result['predicted_delay_minutes'] >= 0


def test_predict_delay_confidence_interval_ordered(predictor):
    result = predictor.predict_delay('Central', _future_dt())
    lo, hi = result['confidence_interval_95']
    assert lo <= hi


def test_predict_delay_rejects_past_datetime(predictor):
    past = datetime.now() - timedelta(hours=1)
    with pytest.raises(ValueError, match="future"):
        predictor.predict_delay('Central', past)


def test_predict_delay_rejects_invalid_line(predictor):
    with pytest.raises(ValueError, match="Invalid line"):
        predictor.predict_delay('Hogwarts Express', _future_dt())


def test_predict_delay_status_label(predictor):
    predictor.model.predict.return_value = np.array([1.0])
    assert predictor.predict_delay('Central', _future_dt())['status'] == 'Good Service'

    predictor.model.predict.return_value = np.array([4.0])
    assert predictor.predict_delay('Central', _future_dt())['status'] == 'Minor Delays'

    predictor.model.predict.return_value = np.array([15.0])
    assert predictor.predict_delay('Central', _future_dt())['status'] == 'Severe Delays'


def test_precipitation_x_temp_formula(predictor):
    dt = _future_dt()

    features_warm = predictor._engineer_features(
        line='Central',
        target_datetime=dt,
        weather_forecast={'temperature': 20.0, 'precipitation': 5.0, 'humidity': 70.0},
        recent_delays=None,
    )
    features_cold = predictor._engineer_features(
        line='Central',
        target_datetime=dt,
        weather_forecast={'temperature': -5.0, 'precipitation': 5.0, 'humidity': 70.0},
        recent_delays=None,
    )

    warm_val = features_warm['precipitation_x_temp'].iloc[0]
    cold_val = features_cold['precipitation_x_temp'].iloc[0]

    # warm: 5 / (20 + 1) ≈ 0.238
    # cold: 5 / ( 5 + 1) ≈ 0.833  — cold should be higher
    assert cold_val > warm_val, (
        f"Expected cold precipitation_x_temp ({cold_val:.3f}) > "
        f"warm ({warm_val:.3f}). Check formula in _engineer_features."
    )


def test_validate_features_logs_warning_on_mismatch(predictor, caplog):
    import logging
    df_missing = pd.DataFrame([{'hour': 9, 'line': 'Central'}])

    with caplog.at_level(logging.WARNING, logger='future_prediction'):
        predictor._validate_features(df_missing)

    assert any('missing' in r.message.lower() for r in caplog.records)


def test_validate_features_no_warning_when_all_present(predictor, caplog):
    import logging
    all_cols = predictor.feature_metadata['all_features']
    df_full  = pd.DataFrame([{col: 0 for col in all_cols}])

    with caplog.at_level(logging.WARNING, logger='future_prediction'):
        predictor._validate_features(df_full)

    assert not any('missing' in r.message.lower() for r in caplog.records)


def test_predict_next_24_hours_shape(predictor):
    df = predictor.predict_next_24_hours('Central', interval_minutes=60)
    assert len(df) == 24
    assert 'datetime'        in df.columns
    assert 'predicted_delay' in df.columns
    assert 'status'          in df.columns


def test_predict_next_24_hours_custom_interval(predictor):
    df = predictor.predict_next_24_hours('Central', interval_minutes=30)
    assert len(df) == 48


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
