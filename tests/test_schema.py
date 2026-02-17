"""Schema validation tests."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import get_config
from data import validate_schema, generate_synthetic_data


def test_schema_validation_success():
    config = get_config()

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1h'),
        'line': ['Central'] * 10,
        'status': ['Good Service'] * 10,
        'delay_minutes': [2.5] * 10,
        'temp_c': [15.0] * 10,
        'precipitation_mm': [0.0] * 10,
        'humidity': [60.0] * 10,
        'crowding_index': [0.5] * 10,
        'is_weekend': [0] * 10,
        'hour': [i for i in range(10)],
        'day_of_week': [0] * 10,
        'month': [1] * 10,
        'peak_time': [0] * 10,
        'is_holiday': [0] * 10
    })

    validate_schema(df, config)


def test_schema_validation_missing_column():
    config = get_config()

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1h'),
        'line': ['Central'] * 10,
        'status': ['Good Service'] * 10,
        # 'delay_minutes' is deliberately omitted.
        'temp_c': [15.0] * 10,
        'precipitation_mm': [0.0] * 10,
        'humidity': [60.0] * 10,
        'crowding_index': [0.5] * 10,
        'is_weekend': [0] * 10,
        'hour': [i for i in range(10)],
        'day_of_week': [0] * 10,
        'month': [1] * 10,
        'peak_time': [0] * 10,
        'is_holiday': [0] * 10
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df, config)


def test_schema_validation_invalid_line():
    config = get_config()

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1h'),
        'line': ['InvalidLine'] * 10,
        'status': ['Good Service'] * 10,
        'delay_minutes': [2.5] * 10,
        'temp_c': [15.0] * 10,
        'precipitation_mm': [0.0] * 10,
        'humidity': [60.0] * 10,
        'crowding_index': [0.5] * 10,
        'is_weekend': [0] * 10,
        'hour': [i for i in range(10)],
        'day_of_week': [0] * 10,
        'month': [1] * 10,
        'peak_time': [0] * 10,
        'is_holiday': [0] * 10
    })

    with pytest.raises(ValueError, match="Invalid tube lines"):
        validate_schema(df, config)


def test_schema_validation_invalid_crowding():
    config = get_config()

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1h'),
        'line': ['Central'] * 10,
        'status': ['Good Service'] * 10,
        'delay_minutes': [2.5] * 10,
        'temp_c': [15.0] * 10,
        'precipitation_mm': [0.0] * 10,
        'humidity': [60.0] * 10,
        'crowding_index': [1.5] * 10,
        'is_weekend': [0] * 10,
        'hour': [i for i in range(10)],
        'day_of_week': [0] * 10,
        'month': [1] * 10,
        'peak_time': [0] * 10,
        'is_holiday': [0] * 10
    })

    with pytest.raises(ValueError, match="crowding_index"):
        validate_schema(df, config)


def test_schema_validation_negative_delay():
    config = get_config()

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1h'),
        'line': ['Central'] * 10,
        'status': ['Good Service'] * 10,
        'delay_minutes': [-1.0] * 10,
        'temp_c': [15.0] * 10,
        'precipitation_mm': [0.0] * 10,
        'humidity': [60.0] * 10,
        'crowding_index': [0.5] * 10,
        'is_weekend': [0] * 10,
        'hour': [i for i in range(10)],
        'day_of_week': [0] * 10,
        'month': [1] * 10,
        'peak_time': [0] * 10,
        'is_holiday': [0] * 10
    })

    with pytest.raises(ValueError, match="delay_minutes has values below minimum"):
        validate_schema(df, config)


def test_synthetic_data_generation():
    config = get_config()

    df = generate_synthetic_data(config)

    validate_schema(df, config)

    assert len(df) > 0
    assert df['line'].nunique() == len(config.data.tube_lines)
    assert df['delay_minutes'].min() >= 0
    assert df['crowding_index'].min() >= 0
    assert df['crowding_index'].max() <= 1

    assert df['hour'].min() >= 0
    assert df['hour'].max() <= 23
    assert df['day_of_week'].min() >= 0
    assert df['day_of_week'].max() <= 6
    assert df['month'].min() >= 1
    assert df['month'].max() <= 12


def test_timestamp_consistency():
    config = get_config()

    df = generate_synthetic_data(config)

    for line in df['line'].unique():
        line_df = df[df['line'] == line]
        timestamps = line_df['timestamp'].values

        assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
