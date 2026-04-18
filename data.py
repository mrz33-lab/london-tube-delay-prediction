"""
Data loading, validation, and synthetic generation.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays

from config import Config, RANDOM_SEED
from utils import validate_datetime_column, set_random_seeds
from exceptions import SchemaValidationError
from line_metadata import LINE_BASE_DELAYS
from data_collection import LINE_CROWDING_WEIGHT

# Lines whose crowding weight places them in the top tier (weight >= 0.10).
_HIGH_CROWDING_LINES = frozenset(
    line for line, w in LINE_CROWDING_WEIGHT.items() if w >= 0.10
)


logger = logging.getLogger(__name__)


_STATUS_TO_SEVERITY = {
    'Good Service': 0,
    'Minor Delays': 1,
    'Severe Delays': 2,
}


def _add_delay_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Add delay_severity (0/1/2) from the real TfL status label.

    This is the primary regression target: a real, measured ordinal quantity
    directly from the TfL Unified API, unlike delay_minutes which is a
    synthetic proxy.
    """
    if 'delay_severity' not in df.columns:
        df = df.copy()
        df['delay_severity'] = df['status'].map(_STATUS_TO_SEVERITY).fillna(0).astype(int)
        logger.info(
            "Added delay_severity from status: %s",
            df['delay_severity'].value_counts().to_dict(),
        )
    return df


def load_data(config: Config) -> Tuple[pd.DataFrame, str]:
    """Load from real CSV if it exists, otherwise generate synthetic data."""
    real_data_path = config.paths.data_dir / config.paths.real_data_file

    if real_data_path.exists():
        logger.info(f"Found real data at {real_data_path}")
        logger.info("Running in REAL MODE")
        df = pd.read_csv(real_data_path)
        df = validate_datetime_column(df, 'timestamp')
        df = _add_delay_severity(df)
        mode = 'REAL'
    else:
        logger.warning(f"Real data not found at {real_data_path}")
        logger.info("Running in PROTOTYPE MODE - generating synthetic data")
        df = generate_synthetic_data(config)
        mode = 'PROTOTYPE'

    validate_schema(df, config)

    logger.info(f"Loaded data with shape {df.shape}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df, mode


def validate_schema(df: pd.DataFrame, config: Config):
    """Check required columns, types, and value ranges."""
    required_cols = set(config.data.required_columns)
    actual_cols = set(df.columns)

    missing_cols = required_cols - actual_cols
    if missing_cols:
        raise SchemaValidationError(f"Missing required columns: {missing_cols}")

    _nan_checked = ['delay_minutes', 'delay_severity', 'temp_c', 'precipitation_mm',
                    'humidity', 'crowding_index']
    for col in _nan_checked:
        if col in df.columns and df[col].isna().any():
            raise SchemaValidationError(
                f"Column '{col}' contains NaN values — check data pipeline before training"
            )

    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        raise SchemaValidationError("'timestamp' must be datetime type")

    if not pd.api.types.is_numeric_dtype(df['delay_minutes']):
        raise SchemaValidationError("'delay_minutes' must be numeric")

    if df['delay_minutes'].min() < config.data.delay_min:
        raise SchemaValidationError(f"delay_minutes has values below minimum {config.data.delay_min}")

    if df['delay_minutes'].max() > config.data.delay_max:
        raise SchemaValidationError(
            f"delay_minutes has values above maximum {config.data.delay_max}: "
            f"max={df['delay_minutes'].max():.1f}"
        )

    if not pd.api.types.is_numeric_dtype(df['delay_severity']):
        raise SchemaValidationError("'delay_severity' must be numeric")

    invalid_severity = set(df['delay_severity'].unique()) - {0, 1, 2}
    if invalid_severity:
        raise SchemaValidationError(f"delay_severity must be 0/1/2, got: {invalid_severity}")

    invalid_lines = set(df['line'].unique()) - set(config.data.tube_lines)
    if invalid_lines:
        raise SchemaValidationError(f"Invalid tube lines found: {invalid_lines}")

    invalid_status = set(df['status'].unique()) - set(config.data.status_categories)
    if invalid_status:
        raise SchemaValidationError(f"Invalid status values found: {invalid_status}")

    if df['temp_c'].min() < config.data.temp_min or df['temp_c'].max() > config.data.temp_max:
        raise SchemaValidationError(
            f"Temperature outside expected range [{config.data.temp_min}, {config.data.temp_max}]: "
            f"min={df['temp_c'].min():.1f}, max={df['temp_c'].max():.1f}. "
            "Check for sensor faults or unit errors before training."
        )

    if df['crowding_index'].min() < config.data.crowding_min or df['crowding_index'].max() > config.data.crowding_max:
        raise SchemaValidationError(f"crowding_index must be in [{config.data.crowding_min}, {config.data.crowding_max}]")

    logger.info("Schema validation passed")


def generate_synthetic_data(config: Config) -> pd.DataFrame:
    """Generate realistic-ish synthetic delay data for prototyping."""
    # Use an explicit RandomState so results are reproducible regardless of
    # any other code that may call np.random before this function runs.
    rng = np.random.RandomState(RANDOM_SEED)
    logger.info("Generating synthetic data...")

    # Fixed reference date ensures synthetic timestamps are reproducible
    # regardless of when generate_synthetic_data() is called.
    start_date = datetime(2024, 1, 1)
    timestamps = pd.date_range(
        start=start_date,
        periods=config.data.synthetic_n_days * config.data.synthetic_samples_per_day,
        freq='15min'
    )

    end_date = start_date + timedelta(days=config.data.synthetic_n_days)
    uk_holidays = holidays.country_holidays('GB', years=list(range(start_date.year, end_date.year + 1)))

    records = []

    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.dayofweek
        month = ts.month
        is_weekend = 1 if day_of_week >= 5 else 0
        peak_time = 1 if (day_of_week < 5 and ((7 <= hour < 10) or (16 <= hour < 19))) else 0
        is_holiday = 1 if ts.date() in uk_holidays else 0

        # seasonal temperature cycle
        base_temp = 15 + 10 * np.sin(2 * np.pi * (month - 3) / 12)
        temp_c = np.clip(base_temp + rng.normal(0, 3), config.data.temp_min, config.data.temp_max)

        rain_probability = 0.3 if month in [11, 12, 1, 2] else 0.15
        precipitation_mm = rng.exponential(5) if rng.random() < rain_probability else 0

        humidity = np.clip(60 + rng.normal(0, 15) + precipitation_mm, 30, 100)

        for line in config.data.tube_lines:
            line_base_delay = LINE_BASE_DELAYS.get(line, 3.0)

            base_crowding = 0.3
            if peak_time:
                base_crowding += 0.4
            if is_weekend:
                base_crowding -= 0.15
            if line in _HIGH_CROWDING_LINES:
                base_crowding += 0.1

            crowding_index = np.clip(base_crowding + rng.normal(0, 0.1), 0, 1)

            delay = line_base_delay
            if peak_time:
                delay += rng.uniform(2, 5)
            if precipitation_mm > 10:
                delay += rng.uniform(3, 8)
            elif precipitation_mm > 0:
                delay += rng.uniform(0.5, 2)
            if temp_c < 0 or temp_c > 30:
                delay += rng.uniform(1, 4)

            delay += crowding_index * rng.uniform(1, 3)

            if is_weekend and not is_holiday:
                delay *= 0.7
            if is_holiday:
                delay *= 0.5

            delay += rng.exponential(1.5)
            delay = max(0, delay)

            # assign status from delay thresholds
            if delay < config.data.status_good_max:
                status = 'Good Service'
            elif delay < config.data.status_minor_max:
                status = 'Minor Delays'
            else:
                status = 'Severe Delays'

            # rare spike events (~2% chance) — signal failures etc
            if rng.random() < 0.02:
                delay += rng.uniform(15, 45)
                status = 'Severe Delays'

            records.append({
                'timestamp': ts, 'line': line, 'status': status,
                'delay_minutes': round(delay, 2),
                'delay_severity': _STATUS_TO_SEVERITY[status],
                'temp_c': round(temp_c, 2),
                'precipitation_mm': round(precipitation_mm, 2),
                'humidity': round(humidity, 2),
                'crowding_index': round(crowding_index, 3),
                'is_weekend': is_weekend, 'hour': hour,
                'day_of_week': day_of_week, 'month': month,
                'peak_time': peak_time, 'is_holiday': is_holiday
            })

    df = pd.DataFrame(records)

    logger.info(f"Generated {len(df)} synthetic records")
    logger.info(f"Lines: {df['line'].nunique()}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Mean delay: {df['delay_minutes'].mean():.2f} minutes")
    logger.info(f"Status distribution:\n{df['status'].value_counts()}")

    return df


def get_train_test_split(df, config, time_col='timestamp'):
    """Chronological split — no shuffling."""
    df = df.sort_values(time_col).reset_index(drop=True)
    split_idx = int(len(df) * config.models.train_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    logger.info(f"Train set: {len(train_df)} records ({train_df[time_col].min()} to {train_df[time_col].max()})")
    logger.info(f"Test set: {len(test_df)} records ({test_df[time_col].min()} to {test_df[time_col].max()})")

    if train_df[time_col].max() >= test_df[time_col].min():
        logger.warning("Potential temporal overlap detected in train/test split")

    return train_df, test_df


def save_data(df: pd.DataFrame, output_path: Path):
    df.to_csv(output_path, index=False)
    logger.info(f"Saved data to {output_path}")
