"""Feature engineering tests."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from config import get_config
from features import (
    engineer_features,
    get_feature_columns,
    prepare_features_for_model
)


def test_feature_engineering_output_shape():
    config = get_config()

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='1h'),
        'line': ['Central'] * 50,
        'status': ['Good Service'] * 50,
        'delay_minutes': [5.0] * 50,
        'temp_c': [15.0] * 50,
        'precipitation_mm': [0.0] * 50,
        'humidity': [60.0] * 50,
        'crowding_index': [0.5] * 50,
        'is_weekend': [0] * 50,
        'hour': [i % 24 for i in range(50)],
        'day_of_week': [0] * 50,
        'month': [1] * 50,
        'peak_time': [0] * 50,
        'is_holiday': [0] * 50
    })

    df_featured = engineer_features(df, config, is_training=True)

    assert df_featured.shape[1] > df.shape[1]

    assert 'lag_delay_1' in df_featured.columns
    assert 'lag_delay_3' in df_featured.columns

    assert 'rolling_mean_delay_3' in df_featured.columns
    assert 'rolling_mean_delay_12' in df_featured.columns

    assert 'recent_disruption_rate' in df_featured.columns

    assert 'temp_delta_1h' in df_featured.columns
    assert 'precipitation_delta_1h' in df_featured.columns

    assert 'crowding_x_peak' in df_featured.columns


def test_get_feature_columns():
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
        'is_holiday': [0] * 10,
        'lag_delay_1': [4.0] * 10,
        'rolling_mean_delay_3': [5.0] * 10
    })

    numeric_features, categorical_features, all_features = get_feature_columns(df, config)

    assert 'temp_c' in numeric_features
    assert 'delay_minutes' not in numeric_features
    assert 'timestamp' not in numeric_features

    assert 'line' in categorical_features
    assert 'status' not in categorical_features

    assert len(all_features) == len(numeric_features) + len(categorical_features)


def test_prepare_features_for_model():
    config = get_config()

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1h'),
        'line': ['Central'] * 10,
        'status': ['Good Service'] * 10,
        'delay_minutes': [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        'temp_c': [15.0] * 10,
        'lag_delay_1': [np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
    })

    feature_columns = ['temp_c', 'lag_delay_1', 'line']

    X, y = prepare_features_for_model(df, feature_columns, 'delay_minutes')

    assert len(X) == len(y)
    assert len(X) <= len(df)

    # numeric NaNs should be filled with 0
    assert not X['lag_delay_1'].isna().any()

    assert X.shape[1] == len(feature_columns)


def test_interaction_features():
    config = get_config()

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1h'),
        'line': ['Central'] * 10,
        'status': ['Good Service'] * 10,
        'delay_minutes': [5.0] * 10,
        'temp_c': [15.0] * 10,
        'precipitation_mm': [10.0] * 10,
        'humidity': [60.0] * 10,
        'crowding_index': [0.8] * 10,
        'is_weekend': [0] * 10,
        'hour': list(range(10)),
        'day_of_week': [0] * 10,
        'month': [1] * 10,
        'peak_time': [1] * 5 + [0] * 5,
        'is_holiday': [0] * 10
    })

    df_featured = engineer_features(df, config, is_training=True)

    expected = df['crowding_index'] * df['peak_time']
    pd.testing.assert_series_equal(
        df_featured['crowding_x_peak'],
        expected,
        check_names=False
    )


def test_rolling_features_calculation():
    config = get_config()

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=20, freq='1h'),
        'line': ['Central'] * 20,
        'status': ['Good Service'] * 20,
        'delay_minutes': [10.0] * 20,
        'temp_c': [15.0] * 20,
        'precipitation_mm': [0.0] * 20,
        'humidity': [60.0] * 20,
        'crowding_index': [0.5] * 20,
        'is_weekend': [0] * 20,
        'hour': [i % 24 for i in range(20)],
        'day_of_week': [0] * 20,
        'month': [1] * 20,
        'peak_time': [0] * 20,
        'is_holiday': [0] * 20
    })

    df_featured = engineer_features(df, config, is_training=True)

    rolling_mean = df_featured['rolling_mean_delay_3'].dropna()

    if len(rolling_mean) > 0:
        assert rolling_mean.mean() > 9.0
        assert rolling_mean.mean() < 11.0


def test_disruption_rate_calculation():
    config = get_config()

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=30, freq='1h'),
        'line': ['Central'] * 30,
        'status': ['Good Service'] * 15 + ['Minor Delays'] * 15,
        'delay_minutes': [5.0] * 30,
        'temp_c': [15.0] * 30,
        'precipitation_mm': [0.0] * 30,
        'humidity': [60.0] * 30,
        'crowding_index': [0.5] * 30,
        'is_weekend': [0] * 30,
        'hour': [0] * 30,
        'day_of_week': [0] * 30,
        'month': [1] * 30,
        'peak_time': [0] * 30,
        'is_holiday': [0] * 30
    })

    df_featured = engineer_features(df, config, is_training=True)

    assert df_featured['recent_disruption_rate'].min() >= 0
    assert df_featured['recent_disruption_rate'].max() <= 1


def test_weather_delta_features():
    config = get_config()

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1h'),
        'line': ['Central'] * 10,
        'status': ['Good Service'] * 10,
        'delay_minutes': [5.0] * 10,
        'temp_c': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        'precipitation_mm': [0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0],
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

    # positive mean delta for rising temps
    temp_delta = df_featured['temp_delta_1h'].dropna()

    if len(temp_delta) > 0:
        assert temp_delta.mean() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
