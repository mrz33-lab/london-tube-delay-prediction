"""
Feature engineering pipeline with leakage protection.

Computes lag, rolling, network effect, temporal, topology, and frequency
features. Uses shift() throughout to avoid data leakage.
"""

import logging
from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from pathlib import Path

from config import Config
from line_metadata import (
    LINE_LENGTH_KM,
    LINE_N_STATIONS,
    LINE_N_INTERCHANGES,
    LINE_IS_DEEP_TUBE,
    LINE_ZONE_COVERAGE,
    LINE_PEAK_TPH,
    LINE_OFFPEAK_TPH,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    config: Config,
    is_training: bool = True
) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    All lag/rolling features are per-line and use only past data.
    """
    logger.info("Engineering features with leakage protection...")

    df = df.copy()

    df = df.sort_values(
        [config.features.group_column, config.features.time_column]
    ).reset_index(drop=True)

    target_col = config.features.target_column
    group_col  = config.features.group_column

    # figure out data frequency so lag/rolling windows are correct
    time_diffs = df.groupby(group_col)[config.features.time_column].diff()
    raw_median = time_diffs.median()
    if pd.notna(raw_median) and raw_median.total_seconds() > 0:
        periods_per_hour: float = pd.Timedelta(hours=1) / raw_median
    else:
        logger.warning(
            "Could not detect data frequency (median_diff=%s), "
            "defaulting to 1 period/hour", raw_median
        )
        periods_per_hour = 1.0

    # === LAG FEATURES ===
    for lag_hours in config.features.lag_features:
        feature_name  = f'lag_delay_{lag_hours}'
        shift_periods = max(1, int(lag_hours * periods_per_hour))
        df[feature_name] = df.groupby(group_col)[target_col].shift(shift_periods)
        logger.info(f"Created {feature_name} (shift {shift_periods} periods)")

    # === ROLLING FEATURES ===
    for window_hours in config.features.rolling_windows:
        window_periods = max(1, int(window_hours * periods_per_hour))

        # shift(1) so current row is excluded
        feature_name = f'rolling_mean_delay_{window_hours}'
        df[feature_name] = df.groupby(group_col)[target_col].transform(
            lambda x: x.shift(1).rolling(window=window_periods, min_periods=1).mean()
        )
        logger.info(f"Created {feature_name}")

        if window_hours >= 12:
            feature_name = f'rolling_std_delay_{window_hours}'
            df[feature_name] = df.groupby(group_col)[target_col].transform(
                lambda x: x.shift(1).rolling(window=window_periods, min_periods=min(2, window_periods)).std()
            )
            logger.info(f"Created {feature_name}")

    # === DISRUPTION RATE ===
    df['_is_disrupted'] = (df['status'] != 'Good Service').astype(int)
    df['recent_disruption_rate'] = df.groupby(group_col)['_is_disrupted'].transform(
        lambda x: x.shift(1).rolling(window=12, min_periods=1).mean()
    )
    df = df.drop(columns=['_is_disrupted'])
    logger.info("Created recent_disruption_rate")

    # === WEATHER DELTA ===
    _weather_shift = max(1, int(periods_per_hour))
    df['temp_delta_1h']          = df.groupby(group_col)['temp_c'].diff(periods=_weather_shift)
    df['precipitation_delta_1h'] = df.groupby(group_col)['precipitation_mm'].diff(periods=_weather_shift)
    logger.info("Created weather delta features")

    # === INTERACTION FEATURES ===
    df['crowding_x_peak']      = df['crowding_index'] * df['peak_time']
    df['precipitation_x_temp'] = df['precipitation_mm'] * (1 / (df['temp_c'].abs() + 1))
    logger.info("Created interaction features")

    # === EXTRA FEATURE GROUPS ===
    df = add_network_effect_features(df, config)
    df = add_special_event_features(df)
    df = add_topology_features(df)
    df = add_train_frequency_features(df)

    # === LEAKAGE CHECK ===
    _verify_no_leakage(df, config)

    nan_counts = df.isna().sum()
    features_with_nan = nan_counts[nan_counts > 0]
    if len(features_with_nan) > 0:
        logger.info("Features with NaN values (expected for lag/rolling):")
        for feat, count in features_with_nan.items():
            logger.info(f"  {feat}: {count} ({100*count/len(df):.1f}%)")

    return df


# ---------------------------------------------------------------------------
# network effects (leave-one-out)
# ---------------------------------------------------------------------------

def add_network_effect_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add network-wide disruption signals.

    Uses leave-one-out so each line's features come from OTHER lines only.
    Everything is shifted by 1 period to avoid seeing current-time data.
    """
    time_col   = config.features.time_column
    group_col  = config.features.group_column
    target_col = config.features.target_column

    # aggregate per timestamp in one pass
    ts_agg = (
        df.groupby(time_col)[target_col]
        .agg(net_sum='sum', net_sum_sq=lambda x: (x ** 2).sum(), net_count='count')
        .reset_index()
    )

    df = df.merge(ts_agg, on=time_col, how='left')

    # leave-one-out mean
    loo_count = (df['net_count'] - 1).clip(lower=1)
    loo_sum   = df['net_sum'] - df[target_col]
    df['network_avg_delay'] = loo_sum / loo_count

    # leave-one-out variance: E[X²] - (E[X])²
    loo_sum_sq  = df['net_sum_sq'] - df[target_col] ** 2
    loo_mean_sq = loo_sum_sq / loo_count
    loo_mean    = loo_sum / loo_count
    loo_var     = (loo_mean_sq - loo_mean ** 2).clip(lower=0.0)
    df['network_delay_volatility'] = np.sqrt(loo_var)

    df = df.drop(columns=['net_sum', 'net_sum_sq', 'net_count'])

    if 'status' in df.columns:
        df['_disrupted'] = (df['status'] != 'Good Service').astype(int)

        ts_dis = (
            df.groupby(time_col)['_disrupted']
            .agg(dis_sum='sum', dis_count='count')
            .reset_index()
        )
        df = df.merge(ts_dis, on=time_col, how='left')

        loo_dis_count = (df['dis_count'] - 1).clip(lower=1)
        loo_dis_sum   = df['dis_sum'] - df['_disrupted']
        df['lines_disrupted_ratio'] = loo_dis_sum / loo_dis_count
        df['is_network_wide_disruption'] = (
            df['lines_disrupted_ratio'] >= 0.5
        ).astype(int)

        df = df.drop(columns=['_disrupted', 'dis_sum', 'dis_count'])
    else:
        df['lines_disrupted_ratio']      = 0.0
        df['is_network_wide_disruption'] = 0

    # shift everything by 1 so we only see T-1 network state
    for col in ['network_avg_delay', 'network_delay_volatility',
                'lines_disrupted_ratio', 'is_network_wide_disruption']:
        df[col] = df.groupby(group_col)[col].shift(1)

    logger.info("Created network effect features (lagged T-1)")
    return df


# ---------------------------------------------------------------------------
# temporal / event features
# ---------------------------------------------------------------------------

def add_special_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical hour encoding + service window flags."""
    hour = df['hour']
    df['hour_sin']         = np.sin(2 * np.pi * hour / 24)
    df['hour_cos']         = np.cos(2 * np.pi * hour / 24)
    df['is_late_night']    = (hour >= 22).astype(int)
    df['is_early_morning'] = (hour < 6).astype(int)

    logger.info("Created hour_sin, hour_cos, is_late_night, is_early_morning")
    return df


# ---------------------------------------------------------------------------
# topology features
# ---------------------------------------------------------------------------

def add_topology_features(df: pd.DataFrame) -> pd.DataFrame:
    """Static line infrastructure metadata — no temporal leakage possible."""
    df['line_length_km']         = df['line'].map(LINE_LENGTH_KM).fillna(0.0)
    df['n_stations']             = df['line'].map(LINE_N_STATIONS).fillna(0).astype(float)
    df['n_interchange_stations'] = df['line'].map(LINE_N_INTERCHANGES).fillna(0).astype(float)
    df['is_deep_tube']           = df['line'].map(LINE_IS_DEEP_TUBE).fillna(0).astype(float)
    df['zone_coverage']          = df['line'].map(LINE_ZONE_COVERAGE).fillna(0).astype(float)

    logger.info("Created topology features")
    return df


# ---------------------------------------------------------------------------
# train frequency features
# ---------------------------------------------------------------------------

def add_train_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """TPH-based features — low frequency lines are more fragile."""
    max_peak_tph = max(LINE_PEAK_TPH.values())  # 36 for Victoria/Jubilee

    peak_tph    = df['line'].map(LINE_PEAK_TPH).fillna(12)
    offpeak_tph = df['line'].map(LINE_OFFPEAK_TPH).fillna(8)

    df['trains_per_hour']     = np.where(df['peak_time'] == 1, peak_tph, offpeak_tph)
    df['service_headway_min'] = 60.0 / df['trains_per_hour']
    df['capacity_pressure']   = (
        df['crowding_index'] * df['trains_per_hour'] / max_peak_tph
    )

    logger.info("Created train frequency features")
    return df


# ---------------------------------------------------------------------------
# leakage verification
# ---------------------------------------------------------------------------

def _verify_no_leakage(df: pd.DataFrame, config: Config):
    """Check lag features are NaN at series start (no future data leaked)."""
    group_col = config.features.group_column
    time_col  = config.features.time_column

    lag_features = [f'lag_delay_{lag}' for lag in config.features.lag_features]
    rolling_features = (
        [f'rolling_mean_delay_{w}' for w in config.features.rolling_windows]
        + [f'rolling_std_delay_{w}' for w in config.features.rolling_windows if w >= 12]
    )
    all_temporal = lag_features + rolling_features + ['recent_disruption_rate']

    for feature in all_temporal:
        if feature not in df.columns:
            continue
        for line in df[group_col].unique():
            line_df   = df[df[group_col] == line].sort_values(time_col)
            first_val = line_df[feature].iloc[0]
            if pd.notna(first_val):
                logger.warning(
                    f"Potential leakage in {feature} for {line}: "
                    "first value is not NaN"
                )

    logger.info("Leakage verification completed")


# ---------------------------------------------------------------------------
# feature helpers
# ---------------------------------------------------------------------------

def get_feature_columns(df, config):
    """Return (numeric_features, categorical_features, all_features)."""
    exclude_cols = set(
        config.features.exclude_columns + [config.features.target_column]
    )
    all_cols = [col for col in df.columns if col not in exclude_cols]

    numeric_features = []
    categorical_features = []

    for col in all_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    logger.info(f"Identified {len(numeric_features)} numeric features")
    logger.info(f"Identified {len(categorical_features)} categorical features")

    return numeric_features, categorical_features, numeric_features + categorical_features


def create_preprocessing_pipeline(numeric_features, categorical_features):
    """StandardScaler for numeric, OneHotEncoder for categorical."""
    transformers = []
    if numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append((
            'cat',
            OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            categorical_features
        ))

    preprocessor = ColumnTransformer(transformers=transformers)
    logger.info("Created preprocessing pipeline")
    return preprocessor


def save_feature_metadata(
    numeric_features, categorical_features, output_dir,
    residual_quantiles=None,
):
    metadata = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'all_features': numeric_features + categorical_features,
    }
    if residual_quantiles is not None:
        metadata['residual_quantiles'] = residual_quantiles
    joblib.dump(metadata, output_dir / 'feature_metadata.pkl')
    logger.info(f"Saved feature metadata to {output_dir}")


def load_feature_metadata(artifact_dir):
    return joblib.load(artifact_dir / 'feature_metadata.pkl')


def prepare_features_for_model(df, feature_columns, target_column='delay_minutes'):
    """Prepare X/y matrices, filling NaNs with safe defaults."""
    df_clean = df.dropna(subset=[target_column]).copy()

    X = df_clean[feature_columns].copy()
    y = df_clean[target_column].copy()

    # 0 for numeric = "no historical data available"
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(0)

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    X[categorical_cols] = X[categorical_cols].fillna('Unknown')

    logger.info(f"Prepared features: X shape {X.shape}, y shape {y.shape}")
    return X, y
