"""Shared test fixtures."""

import pytest
import pandas as pd
import numpy as np

from config import get_config


@pytest.fixture()
def tiny_multiline_df() -> pd.DataFrame:
    """Minimal multi-line DataFrame (20 timestamps × 11 lines) for fast tests."""
    lines = [
        'Bakerloo', 'Central', 'Circle', 'District',
        'Hammersmith & City', 'Jubilee', 'Metropolitan',
        'Northern', 'Piccadilly', 'Victoria', 'Waterloo & City',
    ]
    rng = np.random.RandomState(0)
    _sev_map = {'Good Service': 0, 'Minor Delays': 1, 'Severe Delays': 2}
    # 2024-02-05 is a Monday with no UK bank holiday — safe for tests that
    # indirectly compute is_holiday from the timestamp.
    timestamps = pd.date_range('2024-02-05', periods=20, freq='1h')
    records = []
    for ts in timestamps:
        for line in lines:
            delay = float(rng.uniform(0, 15))
            status = 'Good Service' if delay < 3 else ('Minor Delays' if delay < 10 else 'Severe Delays')
            records.append({
                'timestamp': ts, 'line': line,
                'status': status,
                'delay_minutes': round(delay, 2),
                'delay_severity': _sev_map[status],
                'temp_c': 12.0, 'precipitation_mm': 0.0, 'humidity': 70.0,
                'crowding_index': 0.5, 'is_weekend': 0,
                'hour': ts.hour, 'day_of_week': ts.dayofweek,
                'month': ts.month, 'peak_time': 0, 'is_holiday': 0,
            })
    return pd.DataFrame(records)


@pytest.fixture(scope='session')
def config():
    return get_config()


@pytest.fixture()
def sample_df():
    """50-row single-line DataFrame with all 14 schema columns."""
    n = 50
    return pd.DataFrame({
        'timestamp':        pd.date_range(start='2024-02-05', periods=n, freq='1h'),
        'line':             ['Central'] * n,
        'status':           ['Good Service'] * n,
        'delay_minutes':    [5.0] * n,
        'delay_severity':   [0] * n,
        'temp_c':           [15.0] * n,
        'precipitation_mm': [0.0] * n,
        'humidity':         [60.0] * n,
        'crowding_index':   [0.5] * n,
        'is_weekend':       [0] * n,
        'hour':             [i % 24 for i in range(n)],
        'day_of_week':      [0] * n,
        'month':            [1] * n,
        'peak_time':        [0] * n,
        'is_holiday':       [0] * n,
    })


@pytest.fixture()
def multi_line_df():
    """All 11 lines, 20 rows each — for cross-line tests."""
    lines = [
        'Bakerloo', 'Central', 'Circle', 'District',
        'Hammersmith & City', 'Jubilee', 'Metropolitan',
        'Northern', 'Piccadilly', 'Victoria', 'Waterloo & City',
    ]
    n_per_line = 20
    dfs = []
    for line in lines:
        dfs.append(pd.DataFrame({
            'timestamp':        pd.date_range(start='2024-02-05', periods=n_per_line, freq='1h'),
            'line':             [line] * n_per_line,
            'status':           ['Good Service'] * n_per_line,
            'delay_minutes':    np.random.RandomState(42).uniform(1, 10, n_per_line).tolist(),
            'delay_severity':   [0] * n_per_line,
            'temp_c':           [15.0] * n_per_line,
            'precipitation_mm': [0.0] * n_per_line,
            'humidity':         [60.0] * n_per_line,
            'crowding_index':   [0.5] * n_per_line,
            'is_weekend':       [0] * n_per_line,
            'hour':             [i % 24 for i in range(n_per_line)],
            'day_of_week':      [0] * n_per_line,
            'month':            [1] * n_per_line,
            'peak_time':        [0] * n_per_line,
            'is_holiday':       [0] * n_per_line,
        }))
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture()
def featured_df(sample_df, config):
    """sample_df after running through engineer_features."""
    from features import engineer_features
    return engineer_features(sample_df, config, is_training=True)
