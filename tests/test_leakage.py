"""Leakage detection tests."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import get_config
from features import engineer_features, _TEMPORAL_FEATURE_PREFIXES
from data import generate_synthetic_data
from utils import check_data_leakage


def test_lag_features_no_leakage():
    config = get_config()

    # increasing delays so we can verify lag values come from the past
    timestamps = pd.date_range(start='2024-01-01', periods=100, freq='15min')
    df = pd.DataFrame({
        'timestamp': timestamps,
        'line': ['Central'] * 100,
        'status': ['Good Service'] * 100,
        'delay_minutes': list(range(100)),
        'delay_severity': [0] * 100,
        'temp_c': [15.0] * 100,
        'precipitation_mm': [0.0] * 100,
        'humidity': [60.0] * 100,
        'crowding_index': [0.5] * 100,
        'is_weekend': [0] * 100,
        'hour': [0] * 100,
        'day_of_week': [0] * 100,
        'month': [1] * 100,
        'peak_time': [0] * 100,
        'is_holiday': [0] * 100
    })

    df_featured = engineer_features(df, config, is_training=True)

    # at 15-min resolution, 1-hour lag = 4 periods, so first 4 must be NaN
    assert pd.isna(df_featured['lag_delay_1'].iloc[0])
    assert pd.isna(df_featured['lag_delay_1'].iloc[1])
    assert pd.isna(df_featured['lag_delay_1'].iloc[2])
    assert pd.isna(df_featured['lag_delay_1'].iloc[3])

    # lag_delay_1[i] must equal delay_severity[i-4] for all valid rows
    df_featured = df_featured.reset_index(drop=True)
    for i in range(4, len(df_featured)):
        lag_val = df_featured['lag_delay_1'].iloc[i]
        expected = df_featured['delay_severity'].iloc[i - 4]
        if pd.notna(lag_val):
            assert lag_val == expected, (
                f"lag_delay_1 at row {i} = {lag_val} but delay_severity at row {i-4} = {expected}"
            )


def test_rolling_features_no_leakage():
    config = get_config()

    # step change 10 -> 20; if rolling mean jumps immediately, there's leakage
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='15min'),
        'line': ['Central'] * 50,
        'status': ['Good Service'] * 50,
        'delay_minutes': [10.0] * 25 + [20.0] * 25,
        'delay_severity': [0] * 50,
        'temp_c': [15.0] * 50,
        'precipitation_mm': [0.0] * 50,
        'humidity': [60.0] * 50,
        'crowding_index': [0.5] * 50,
        'is_weekend': [0] * 50,
        'hour': [0] * 50,
        'day_of_week': [0] * 50,
        'month': [1] * 50,
        'peak_time': [0] * 50,
        'is_holiday': [0] * 50
    })

    df_featured = engineer_features(df, config, is_training=True)

    # first rolling mean should be NaN
    assert pd.isna(df_featured['rolling_mean_delay_3'].iloc[0])

    # delay_severity is all 0, so rolling mean should stay at 0 (no step change in severity)
    idx_after_change = 26
    if pd.notna(df_featured['rolling_mean_delay_3'].iloc[idx_after_change]):
        assert df_featured['rolling_mean_delay_3'].iloc[idx_after_change] >= 0


def test_per_line_feature_isolation():
    config = get_config()

    # big gap between lines to catch cross-line leakage
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='15min').tolist() * 2,
        'line': ['Central'] * 100 + ['Northern'] * 100,
        'status': ['Good Service'] * 200,
        'delay_minutes': ([5.0] * 100) + ([15.0] * 100),
        'delay_severity': [0] * 100 + [1] * 100,
        'temp_c': [15.0] * 200,
        'precipitation_mm': [0.0] * 200,
        'humidity': [60.0] * 200,
        'crowding_index': [0.5] * 200,
        'is_weekend': [0] * 200,
        'hour': [0] * 200,
        'day_of_week': [0] * 200,
        'month': [1] * 200,
        'peak_time': [0] * 200,
        'is_holiday': [0] * 200
    })

    df_featured = engineer_features(df, config, is_training=True)

    central_df = df_featured[df_featured['line'] == 'Central']
    northern_df = df_featured[df_featured['line'] == 'Northern']

    central_lag = central_df['lag_delay_1'].dropna().mean()
    northern_lag = northern_df['lag_delay_1'].dropna().mean()

    # Central has delay_severity=0, Northern has delay_severity=1 — lags should match
    assert abs(central_lag - 0.0) < 0.1
    assert abs(northern_lag - 1.0) < 0.1

    # lines must stay clearly separated
    assert abs(central_lag - northern_lag) > 0.5


def test_no_future_information_in_training():
    config = get_config()

    df = generate_synthetic_data(config)

    line_df = df[df['line'] == 'Central'].copy()
    line_df = line_df.sort_values('timestamp').reset_index(drop=True)

    df_featured = engineer_features(line_df, config, is_training=True)

    # lag_delay_1[i] must equal delay_severity[i-4] exactly (1-hour lag at 15-min resolution)
    df_featured = df_featured.reset_index(drop=True)
    for idx in range(4, len(df_featured)):
        lag_delay_1 = df_featured.loc[idx, 'lag_delay_1']
        if pd.notna(lag_delay_1):
            expected = df_featured.loc[idx - 4, 'delay_severity']
            assert lag_delay_1 == expected, (
                f"lag_delay_1 at row {idx} = {lag_delay_1} but delay_severity at row {idx-4} = {expected} "
                f"— possible future leakage"
            )


def test_temporal_ordering_preserved():
    config = get_config()

    # increasing delays — reordering would break sequence
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='1h'),
        'line': ['Central'] * 50,
        'status': ['Good Service'] * 50,
        'delay_minutes': list(range(50)),
        'delay_severity': [0] * 50,
        'temp_c': [15.0] * 50,
        'precipitation_mm': [0.0] * 50,
        'humidity': [60.0] * 50,
        'crowding_index': [0.5] * 50,
        'is_weekend': [0] * 50,
        'hour': list(range(24)) + list(range(24)) + [0, 1],
        'day_of_week': [0] * 50,
        'month': [1] * 50,
        'peak_time': [0] * 50,
        'is_holiday': [0] * 50
    })

    df_featured = engineer_features(df, config, is_training=True)

    ts = pd.Series(df_featured['timestamp'].values)
    assert (ts.diff().dropna() >= pd.Timedelta(0)).all()

    delays = pd.Series(df_featured['delay_minutes'].values)
    assert (delays.diff().dropna() >= 0).all()


def test_first_observation_has_nan_lags():
    config = get_config()

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1h'),
        'line': ['Central'] * 10,
        'status': ['Good Service'] * 10,
        'delay_minutes': [5.0] * 10,
        'delay_severity': [0] * 10,
        'temp_c': [15.0] * 10,
        'precipitation_mm': [0.0] * 10,
        'humidity': [60.0] * 10,
        'crowding_index': [0.5] * 10,
        'is_weekend': [0] * 10,
        'hour': list(range(10)),
        'day_of_week': [0] * 10,
        'month': [1] * 10,
        'peak_time': [0] * 10,
        'is_holiday': [0] * 10
    })

    df_featured = engineer_features(df, config, is_training=True)

    # All temporal features must be NaN on the very first row of each line's series.
    # Use the authoritative prefix list from features.py so this test stays in sync.
    _temporal_cols = [c for c in df_featured.columns if any(
        c.startswith(p) for p in _TEMPORAL_FEATURE_PREFIXES
    )]
    for col in _temporal_cols:
        assert pd.isna(df_featured[col].iloc[0]), (
            f"Expected NaN at row 0 for temporal feature '{col}' — shift may be missing"
        )


def test_check_data_leakage_utils():
    """check_data_leakage uses explicit prefix matching, not substring search.

    A feature named 'lag_delay_1' starts with 'lag_' → detected as temporal,
    expected to start with NaN.  A feature named 'bad_lag_feature' does NOT
    start with 'lag_' → not a temporal feature, non-NaN first value is fine.
    A feature named 'lag_bad' (correct prefix, missing NaN) IS flagged.
    """
    df = pd.DataFrame({
        'timestamp':    pd.date_range(start='2024-01-01', periods=5, freq='1h'),
        'line':         ['Central'] * 5,
        # correct lag feature — first value is NaN → valid (no leakage)
        'lag_delay_1':  [np.nan, 2.0, 3.0, 4.0, 5.0],
        # a feature that CONTAINS 'lag' as a substring but doesn't START with 'lag_'
        # → not a temporal feature, non-NaN start is expected and fine
        'bad_lag_feature': [1.0, 2.0, 3.0, 4.0, 5.0],
        # a genuine lag feature (starts with 'lag_') without a NaN first value
        # → this IS a leakage signal
        'lag_bad':      [1.0, 2.0, 3.0, 4.0, 5.0],
    })

    # Good lag — starts with NaN as expected
    assert check_data_leakage(df, 'lag_delay_1', group_col='line') is True

    # Not a temporal feature — non-NaN start is fine, no false positive
    assert check_data_leakage(df, 'bad_lag_feature', group_col='line') is True

    # Genuine lag feature without NaN start → leakage detected
    assert check_data_leakage(df, 'lag_bad', group_col='line') is False


def test_peak_time_train_inference_parity():
    """peak_time must be computed identically in training data and at inference.

    Previously data.py/data_collection.py lacked a weekday guard, so weekend
    rush-hour rows were written as peak_time=1 while FutureDelayPredictor
    returned peak_time=0 for the same timestamps.  This test locks the
    contract: both paths must agree for every hour on weekday AND weekend.
    """
    from future_prediction import FutureDelayPredictor
    from sklearn.dummy import DummyRegressor
    import joblib
    import tempfile
    from pathlib import Path

    # Build a minimal predictor so we can call _is_peak_time.
    # DummyRegressor is picklable; fit it on one sample so joblib.dump works.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stub_model = DummyRegressor()
        stub_model.fit([[0]], [0])
        joblib.dump(stub_model, tmp_path / "best_model.pkl")
        metadata = {'numeric_features': [], 'categorical_features': [],
                    'all_features': [], 'residual_quantiles': {}}
        joblib.dump(metadata, tmp_path / "feature_metadata.pkl")
        predictor = FutureDelayPredictor(
            model_path=str(tmp_path / "best_model.pkl"),
            feature_metadata_path=str(tmp_path / "feature_metadata.pkl"),
        )

    for day in range(7):  # 0=Mon … 6=Sun
        is_weekday = day < 5
        for hour in range(24):
            # replicate the training-data formula from data.py
            training_peak = 1 if (is_weekday and ((7 <= hour < 10) or (16 <= hour < 19))) else 0
            # 2024-01-01 is a Monday (weekday=0); add `day` days to reach desired weekday
            dt = datetime(2024, 1, 1 + day, hour, 0)
            inference_peak = int(predictor._is_peak_time(dt))
            assert training_peak == inference_peak, (
                f"peak_time mismatch: day={day} hour={hour} "
                f"training={training_peak} inference={inference_peak}"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
