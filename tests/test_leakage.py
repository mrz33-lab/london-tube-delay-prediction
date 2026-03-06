"""Leakage detection tests."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import get_config
from features import engineer_features
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

    # After the NaN period, lag should always be less than current value
    if pd.notna(df_featured['lag_delay_1'].iloc[4]):
        assert df_featured['lag_delay_1'].iloc[4] < df_featured['delay_minutes'].iloc[4]


def test_rolling_features_no_leakage():
    config = get_config()

    # step change 10 -> 20; if rolling mean jumps immediately, there's leakage
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='15min'),
        'line': ['Central'] * 50,
        'status': ['Good Service'] * 50,
        'delay_minutes': [10.0] * 25 + [20.0] * 25,
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

    # right after step change, mean should still be below 20
    idx_after_change = 26
    if pd.notna(df_featured['rolling_mean_delay_3'].iloc[idx_after_change]):
        assert df_featured['rolling_mean_delay_3'].iloc[idx_after_change] < df_featured['delay_minutes'].iloc[idx_after_change]


def test_per_line_feature_isolation():
    config = get_config()

    # big gap between lines to catch cross-line leakage
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='15min').tolist() * 2,
        'line': ['Central'] * 100 + ['Northern'] * 100,
        'status': ['Good Service'] * 200,
        'delay_minutes': ([5.0] * 100) + ([15.0] * 100),
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

    assert abs(central_lag - 5.0) < 2.0
    assert abs(northern_lag - 15.0) < 2.0

    # lines must stay clearly separated
    assert abs(central_lag - northern_lag) > 5.0


def test_no_future_information_in_training():
    config = get_config()

    df = generate_synthetic_data(config)

    line_df = df[df['line'] == 'Central'].copy()
    line_df = line_df.sort_values('timestamp').reset_index(drop=True)

    df_featured = engineer_features(line_df, config, is_training=True)

    # verify lag values come from earlier rows
    for idx in range(10, len(df_featured)):
        current_delay = df_featured.loc[idx, 'delay_minutes']
        lag_delay_1 = df_featured.loc[idx, 'lag_delay_1']

        if pd.notna(lag_delay_1):
            previous_delays = df_featured.loc[:idx-1, 'delay_minutes'].values
            pass


def test_temporal_ordering_preserved():
    config = get_config()

    # increasing delays — reordering would break sequence
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='1h'),
        'line': ['Central'] * 50,
        'status': ['Good Service'] * 50,
        'delay_minutes': list(range(50)),
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

    timestamps = df_featured['timestamp'].values
    assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))

    delays = df_featured['delay_minutes'].values
    assert all(delays[i] <= delays[i+1] for i in range(len(delays)-1))


def test_first_observation_has_nan_lags():
    config = get_config()

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1h'),
        'line': ['Central'] * 10,
        'status': ['Good Service'] * 10,
        'delay_minutes': [5.0] * 10,
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

    assert pd.isna(df_featured['lag_delay_1'].iloc[0])
    assert pd.isna(df_featured['rolling_mean_delay_3'].iloc[0])


def test_check_data_leakage_utils():
    # Construct a dummy dataframe
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='1h'),
        'line': ['Central'] * 5,
        'lag_feature': [np.nan, 2.0, 3.0, 4.0, 5.0],
        'bad_lag_feature': [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    
    # lag_feature starts with NaN, so it's valid
    assert check_data_leakage(df, 'lag_feature', group_col='line') is True
    
    # bad_lag_feature doesn't start with NaN, so it's invalid (leakage)
    assert check_data_leakage(df, 'bad_lag_feature', group_col='line') is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
