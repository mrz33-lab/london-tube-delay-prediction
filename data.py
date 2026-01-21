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


logger = logging.getLogger(__name__)

# baseline delays per line — roughly matches TfL published averages
LINE_BASE_DELAYS = {
    'Central': 3.0, 'Jubilee': 2.5, 'Northern': 3.5,
    'Victoria': 2.0, 'Piccadilly': 2.8, 'Bakerloo': 3.2,
    'District': 3.8, 'Circle': 4.0, 'Metropolitan': 2.5,
    'Hammersmith & City': 3.5, 'Waterloo & City': 1.5,
}


def load_data(config: Config) -> Tuple[pd.DataFrame, str]:
    """Load from real CSV if it exists, otherwise generate synthetic data."""
    real_data_path = config.paths.data_dir / config.paths.real_data_file

    if real_data_path.exists():
        logger.info(f"Found real data at {real_data_path}")
        logger.info("Running in REAL MODE")
        df = pd.read_csv(real_data_path)
        df = validate_datetime_column(df, 'timestamp')
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
        raise ValueError(f"Missing required columns: {missing_cols}")

    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        raise ValueError("'timestamp' must be datetime type")

    if not pd.api.types.is_numeric_dtype(df['delay_minutes']):
        raise ValueError("'delay_minutes' must be numeric")

    if df['delay_minutes'].min() < config.data.delay_min:
        raise ValueError(f"delay_minutes has values below minimum {config.data.delay_min}")

    invalid_lines = set(df['line'].unique()) - set(config.data.tube_lines)
    if invalid_lines:
        raise ValueError(f"Invalid tube lines found: {invalid_lines}")

    invalid_status = set(df['status'].unique()) - set(config.data.status_categories)
    if invalid_status:
        raise ValueError(f"Invalid status values found: {invalid_status}")

    # warn rather than error for temperature — extreme values do happen
    if df['temp_c'].min() < config.data.temp_min or df['temp_c'].max() > config.data.temp_max:
        logger.warning(f"Temperature outside expected range [{config.data.temp_min}, {config.data.temp_max}]")

    if df['crowding_index'].min() < config.data.crowding_min or df['crowding_index'].max() > config.data.crowding_max:
        raise ValueError(f"crowding_index must be in [{config.data.crowding_min}, {config.data.crowding_max}]")

    logger.info("Schema validation passed")


def generate_synthetic_data(config: Config) -> pd.DataFrame:
    """Generate realistic-ish synthetic delay data for prototyping."""
    set_random_seeds(RANDOM_SEED)
    logger.info("Generating synthetic data...")

    start_date = datetime.now() - timedelta(days=config.data.synthetic_n_days)
    timestamps = pd.date_range(
        start=start_date,
        periods=config.data.synthetic_n_days * config.data.synthetic_samples_per_day,
        freq='15min'
    )

    uk_holidays = holidays.UK(years=list(range(start_date.year, datetime.now().year + 1)))

    records = []

    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.dayofweek
        month = ts.month
        is_weekend = 1 if day_of_week >= 5 else 0
        peak_time = 1 if (7 <= hour < 10) or (16 <= hour < 19) else 0
        is_holiday = 1 if ts.date() in uk_holidays else 0

        # seasonal temperature cycle
        base_temp = 15 + 10 * np.sin(2 * np.pi * (month - 3) / 12)
        temp_c = base_temp + np.random.normal(0, 3)

        rain_probability = 0.3 if month in [11, 12, 1, 2] else 0.15
        precipitation_mm = np.random.exponential(5) if np.random.random() < rain_probability else 0

        humidity = np.clip(60 + np.random.normal(0, 15) + precipitation_mm, 30, 100)

        for line in config.data.tube_lines:
            line_base_delay = LINE_BASE_DELAYS.get(line, 3.0)

            base_crowding = 0.3
            if peak_time:
                base_crowding += 0.4
            if is_weekend:
                base_crowding -= 0.15
            if line in ['Central', 'Northern', 'Victoria', 'Jubilee']:
                base_crowding += 0.1  # busier lines

            crowding_index = np.clip(base_crowding + np.random.normal(0, 0.1), 0, 1)

            delay = line_base_delay
            if peak_time:
                delay += np.random.uniform(2, 5)
            if precipitation_mm > 10:
                delay += np.random.uniform(3, 8)
            elif precipitation_mm > 0:
                delay += np.random.uniform(0.5, 2)
            if temp_c < 0 or temp_c > 30:
                delay += np.random.uniform(1, 4)

            delay += crowding_index * np.random.uniform(1, 3)

            if is_weekend and not is_holiday:
                delay *= 0.7
            if is_holiday:
                delay *= 0.5

            delay += np.random.exponential(1.5)
            delay = max(0, delay)

            # assign status from delay thresholds
            if delay < config.data.status_good_max:
                status = 'Good Service'
            elif delay < config.data.status_minor_max:
                status = 'Minor Delays'
            else:
                status = 'Severe Delays'

            # rare spike events (~2% chance) — signal failures etc
            if np.random.random() < 0.02:
                delay += np.random.uniform(15, 45)
                status = 'Severe Delays'

            records.append({
                'timestamp': ts, 'line': line, 'status': status,
                'delay_minutes': round(delay, 2),
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
