"""
Shared utility functions used across the London Tube delay prediction pipeline.

Covers four areas:
  1. Logging        — per-run file + console loggers
  2. Artifact I/O   — saving/loading configs, metrics, and run IDs
  3. Data helpers   — datetime validation, data-leakage checks, safe arithmetic
  4. Evaluation     — a single evaluate_model() used by every trainer so metrics
                      are always computed the same way (Objective 3 & 4)
"""

import logging
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from dataclasses import asdict

from config import Config, RANDOM_SEED


# ---------------------------------------------------------------------------
# 1. Logging
# ---------------------------------------------------------------------------

def setup_logging(config: Config, run_id: str) -> logging.Logger:
    """Create a logger that writes to both a file and the console for one training run.

    Each run gets its own log file at:
        artifacts/<run_id>/run.log

    We use a *named* logger ('tube_delay.<run_id>') rather than the global root
    logger for two reasons:
      - Multiple runs (e.g. during hyperparameter search) can coexist without
        their log messages mixing together.
      - Test frameworks like pytest capture root-logger output; using a named
        logger avoids accidentally interfering with that.

    Args:
        config:  Project configuration (provides log level, format, artifact dir).
        run_id:  Unique identifier for this run (e.g. 'run_20240415_143022').

    Returns:
        A configured Logger instance ready to use throughout the run.
    """
    # Create the per-run artifact folder if it does not already exist.
    artifact_dir = config.paths.artifacts_dir / run_id
    artifact_dir.mkdir(exist_ok=True, parents=True)

    log_file = artifact_dir / "run.log"
    log_level = config.logging.get_log_level()
    formatter = logging.Formatter(
        config.logging.log_format, datefmt=config.logging.log_date_format
    )

    # Get (or create) a logger scoped to this run — clearing any leftover handlers
    # prevents duplicate output if this function is called more than once for the
    # same run_id (e.g. during testing or interactive sessions).
    logger = logging.getLogger(f"tube_delay.{run_id}")
    logger.setLevel(log_level)
    logger.handlers.clear()
    # Stop messages bubbling up to the root logger, which would print them twice.
    logger.propagate = False

    # Write every log message to the run's log file for reproducibility.
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Also print messages to the terminal so progress is visible while training.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# ---------------------------------------------------------------------------
# 2. Artifact I/O
# ---------------------------------------------------------------------------

def generate_run_id() -> str:
    """Return a timestamped run identifier such as 'run_20240415_143022'.

    Using a timestamp makes run folders self-documenting and sorts them
    chronologically in any file explorer or directory listing.
    """
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def save_config(config: Config, output_path: Path):
    """Serialise the full project config to a YAML file for reproducibility.

    Saving the config alongside every training artefact means that any run can
    be reproduced exactly by reloading its config — important for the
    dissertation's experimental methodology (Objective 3).

    Path objects are converted to plain strings before serialisation because
    the standard YAML library cannot dump pathlib.Path instances.

    Args:
        config:       The Config dataclass instance used for this run.
        output_path:  Destination .yaml file path.
    """
    config_dict = asdict(config)

    def convert_paths(obj):
        """Walk the config dict tree and convert every Path to a string."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        return obj

    config_dict = convert_paths(config_dict)

    with open(output_path, 'w') as f:
        # sort_keys=False preserves the field order defined in the dataclass,
        # making the saved YAML easier to read.
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def load_config_from_yaml(config_path: Path) -> Dict[str, Any]:
    """Load a previously saved config YAML back into a plain Python dict.

    Returns a dict (not a Config dataclass) so callers can inspect individual
    fields without needing to reconstruct the full object hierarchy.

    Args:
        config_path:  Path to the .yaml config file written by save_config().

    Returns:
        A nested dict mirroring the structure of the Config dataclass.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _coerce_numpy(obj):
    """Recursively convert numpy scalar types to plain Python floats/ints.

    NumPy types (np.float32, np.int64, etc.) are not JSON-serialisable by
    default. This helper walks any nested combination of dicts, lists, and
    scalars and replaces numpy values so the result can be passed to
    json.dump() without error.

    Args:
        obj:  Anything — scalar, dict, list, or tuple.

    Returns:
        The same structure with all numpy scalars replaced by Python floats.
    """
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _coerce_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_numpy(i) for i in obj]
    return obj


def save_metrics(metrics: Dict[str, float], output_path: Path):
    """Write a metrics dictionary to a JSON file.

    Handles nested dicts and numpy scalar types automatically so callers do
    not need to pre-process model outputs before saving.

    Args:
        metrics:      Dict of metric names to values (e.g. {'test_mae': 1.23}).
                      May be nested (e.g. per-line or per-class breakdowns).
        output_path:  Destination .json file path.
    """
    with open(output_path, 'w') as f:
        json.dump(_coerce_numpy(metrics), f, indent=2)


def load_metrics(metrics_path: Path) -> Dict[str, float]:
    """Load a metrics JSON file back into a Python dict.

    Args:
        metrics_path:  Path to a JSON file previously written by save_metrics().

    Returns:
        Dict of metric names to float values.
    """
    with open(metrics_path, 'r') as f:
        return json.load(f)


def set_random_seeds(seed: int = RANDOM_SEED):
    """Fix all random-number generator seeds to make experiments reproducible.

    Called at the start of every training run so that model training,
    train/test splits, and any stochastic operations produce the same result
    each time — a requirement for a scientifically valid comparison between
    models (Objective 3 and 4).

    Covers Python's built-in random, NumPy, and PyTorch (if installed).

    Args:
        seed:  Integer seed value; defaults to RANDOM_SEED from config.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch is optional — only present if the user has installed it.
    try:
        import torch  # type: ignore[import-untyped]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            # Seed all GPU devices, not just the primary one.
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_latest_run_id(artifacts_dir: Path) -> Optional[str]:
    """Return the name of the most recently created run directory, or None.

    Looks for sub-folders matching the pattern 'run_YYYYMMDD_HHMMSS' inside
    artifacts_dir and returns the lexicographically largest name, which is
    also the chronologically newest because the timestamp format sorts
    correctly as a string.

    Using name-based sorting (rather than file modification time) is reliable
    even on network drives or when multiple runs start within the same second.

    Args:
        artifacts_dir:  Directory that contains run_* sub-folders.

    Returns:
        The run ID string (e.g. 'run_20240415_143022'), or None if no runs
        exist yet.
    """
    if not artifacts_dir.exists():
        return None
    run_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda x: x.name, reverse=True)
    return run_dirs[0].name


# ---------------------------------------------------------------------------
# 3. Data helpers
# ---------------------------------------------------------------------------

def validate_datetime_column(df: pd.DataFrame, column: str = 'timestamp') -> pd.DataFrame:
    """Ensure a DataFrame column contains proper timezone-naive datetime values.

    Two problems are fixed automatically:
      - String timestamps are parsed into pandas datetime objects.
      - Timezone-aware datetimes are stripped of their timezone info so that
        all timestamps in the dataset use the same naive UTC-equivalent format.

    This is called during data loading (Objective 2) to guard against silent
    issues where string comparisons replace proper time arithmetic.

    Args:
        df:      The DataFrame to validate.
        column:  Name of the datetime column to check (default 'timestamp').

    Returns:
        The same DataFrame with the column guaranteed to be datetime64[ns].

    Raises:
        ValueError:  If the column is missing or cannot be parsed as datetime.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        try:
            df[column] = pd.to_datetime(df[column])
        except Exception as e:
            raise ValueError(f"Failed to convert '{column}' to datetime: {e}")

    # Drop timezone info so all timestamps are comparable without conversion.
    if df[column].dt.tz is not None:
        df[column] = df[column].dt.tz_localize(None)

    return df


def check_data_leakage(df, feature_col, time_col='timestamp', group_col=None):
    """Check whether a feature column appears to contain future information (data leakage).

    Data leakage occurs when a feature used for training was computed using
    information that would not have been available at prediction time — a
    common problem with lag and rolling features when the dataset is not
    sorted correctly before feature engineering (Objective 4).

    The check is simple: after sorting by time, a lag/rolling feature should
    *not* have a valid (non-NaN) value in the very first row, because there is
    no prior row to compute it from.  If it does, the data was likely not
    sorted before rolling-window calculations were applied.

    If group_col is provided the check is applied per group (e.g. per Tube
    line), because leakage within one line does not imply leakage in another.

    Args:
        df:          DataFrame sorted (or to be sorted) by time.
        feature_col: Name of the feature column to inspect.
        time_col:    Name of the timestamp column (default 'timestamp').
        group_col:   Optional column to group by before checking (e.g. 'line').

    Returns:
        True if no leakage is detected, False otherwise.
    """
    df = df.sort_values(time_col)

    if group_col:
        for group in df[group_col].unique():
            group_df = df[df[group_col] == group].copy()
            if not _check_temporal_order(group_df, feature_col, time_col):
                return False
    else:
        return _check_temporal_order(df, feature_col, time_col)

    return True


# Features that are computed via shift() or rolling() naturally produce NaN
# in the first row(s) of each group — that is expected and correct behaviour.
# We use startswith() matching against this explicit list rather than a
# substring search (e.g. 'lag' in name) to avoid false positives:
#   - 'flag_route' contains 'lag' but is NOT a temporal lag feature.
#   - 'catalogue_id' contains 'lag' but is NOT a temporal lag feature.
_TEMPORAL_FEATURE_PREFIXES: tuple = (
    'lag_',                      # e.g. lag_delay_1h — delay from N hours ago
    'rolling_',                  # e.g. rolling_mean_delay_3h — window averages
    'recent_disruption_rate',    # fraction of recent windows with disruptions
    'temp_delta_1h',             # change in temperature vs. 1 hour prior
    'precipitation_delta_1h',    # change in rainfall vs. 1 hour prior
    'network_avg_delay',         # average delay across all lines at that time
    'network_delay_volatility',  # standard deviation of delays across lines
    'lines_disrupted_ratio',     # proportion of lines currently disrupted
    'is_network_wide_disruption',# binary flag: majority of lines disrupted
)


def _check_temporal_order(df, feature_col, time_col):
    """Return False if a temporal feature has a non-NaN value in the first row.

    This is the inner check used by check_data_leakage(). It works on a single
    group (or the full DataFrame when no grouping is needed).

    A lag/rolling feature should always start with NaN because the first row
    has no preceding rows to compute from.  A non-NaN value in position 0
    suggests the feature was calculated on an unsorted dataset, meaning future
    information leaked into earlier rows.

    Args:
        df:          A time-sorted (sub-)DataFrame.
        feature_col: Column name to inspect.
        time_col:    Timestamp column (currently unused here but kept for
                     symmetry with the public-facing function signature).

    Returns:
        True if the column looks clean, False if leakage is suspected.
    """
    non_null_mask = df[feature_col].notna()

    # A fully-NaN column cannot have leakage — skip it.
    if non_null_mask.sum() == 0:
        return True

    first_non_null_idx = non_null_mask.idxmax()
    first_idx = df.index[0]

    # Check whether this column is expected to start with NaN.
    is_lagged = any(
        feature_col.startswith(p) or feature_col == p
        for p in _TEMPORAL_FEATURE_PREFIXES
    )

    # If the feature is lagged AND the first non-null value is in the very
    # first row, there is likely a leakage problem.
    if is_lagged and first_non_null_idx == first_idx:
        return False

    return True


def format_duration(seconds):
    """Convert a duration in seconds to a human-readable string.

    Used in log messages and the Streamlit dashboard to display how long
    training, back-testing, or data-fetching steps took.

    Examples:
        45.3  -> '45.3s'
        130   -> '2m 10s'
        3665  -> '1h 1m'

    Args:
        seconds:  Duration as a float or int number of seconds.

    Returns:
        A formatted string such as '3m 42s'.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def safe_divide(numerator, denominator, default=0.0):
    """Divide two numbers, returning a safe default instead of raising on zero.

    Prevents ZeroDivisionError and NaN propagation in metric calculations
    (e.g. when computing per-line delay rates for a line that had zero
    records in a time window).

    Args:
        numerator:    The top value of the division.
        denominator:  The bottom value; if zero or NaN, default is returned.
        default:      Value to return when division is unsafe (default 0.0).

    Returns:
        numerator / denominator, or default if the denominator is zero/NaN.
    """
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except Exception:
        return default


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error (MAPE), ignoring near-zero actuals.

    Rows where the actual delay is less than 0.1 minutes are excluded because
    dividing by a near-zero value would produce unrealistically large percentage
    errors that skew the metric.  This situation arises for 'on-time' records
    where the true delay is essentially zero.

    Args:
        y_true:  Array of actual delay values (minutes).
        y_pred:  Array of model predictions (minutes), same length as y_true.

    Returns:
        MAPE as a percentage (e.g. 15.3 means 15.3 %), or NaN if all actual
        values are near zero.
    """
    mask = np.abs(y_true) > 0.1
    if not mask.any():
        return float('nan')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Compute a standard set of regression metrics on both the train and test splits.

    Centralising metric computation here (rather than duplicating it in each
    trainer) guarantees that LightGBM, XGBoost, and any future model are
    always evaluated identically — a prerequisite for a fair comparison in
    the dissertation (Objective 3 and 4).

    Metrics computed:
        MAE   — Mean Absolute Error (minutes); easy to interpret as average
                error magnitude.
        RMSE  — Root Mean Squared Error; penalises large errors more than MAE,
                useful for catching severe delay mis-predictions.
        R²    — Coefficient of determination; proportion of delay variance
                explained by the model (1.0 is perfect, 0.0 is no better than
                always predicting the mean).
        MAPE  — Mean Absolute Percentage Error; scale-free error useful for
                comparing across lines with different baseline delay levels.

    Args:
        model:    A fitted scikit-learn compatible model with a predict() method.
        X_train:  Training feature matrix.
        y_train:  Training target (delay severity).
        X_test:   Test feature matrix (held-out, unseen during training).
        y_test:   Test target.

    Returns:
        Dict with eight keys:
            train_mae, train_rmse, train_r2, train_mape,
            test_mae,  test_rmse,  test_r2,  test_mape
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    return {
        'train_mae':  mean_absolute_error(y_train, y_pred_train),
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        'train_r2':   r2_score(y_train, y_pred_train),
        'train_mape': _mape(np.array(y_train), y_pred_train),
        'test_mae':   mean_absolute_error(y_test, y_pred_test),
        'test_rmse':  float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        'test_r2':    r2_score(y_test, y_pred_test),
        'test_mape':  _mape(np.array(y_test), y_pred_test),
    }
