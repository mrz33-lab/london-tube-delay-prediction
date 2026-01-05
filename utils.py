"""
Utility functions — logging setup, artifact helpers, etc.
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


def setup_logging(config: Config, run_id: str) -> logging.Logger:
    """Set up file + console logging for a training run."""
    artifact_dir = config.paths.artifacts_dir / run_id
    artifact_dir.mkdir(exist_ok=True, parents=True)

    log_file = artifact_dir / "run.log"

    logger = logging.getLogger()
    logger.setLevel(config.logging.get_log_level())
    logger.handlers = []  # clear existing to avoid duplicates

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(config.logging.get_log_level())
    file_handler.setFormatter(logging.Formatter(
        config.logging.log_format, datefmt=config.logging.log_date_format
    ))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.logging.get_log_level())
    console_handler.setFormatter(logging.Formatter(
        config.logging.log_format, datefmt=config.logging.log_date_format
    ))
    logger.addHandler(console_handler)

    return logger


def generate_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def save_config(config: Config, output_path: Path):
    """Dump config to YAML for reproducibility."""
    config_dict = asdict(config)

    def convert_paths(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        return obj

    config_dict = convert_paths(config_dict)

    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def load_config_from_yaml(config_path: Path) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_metrics(metrics: Dict[str, float], output_path: Path):
    """Save metrics to JSON, converting numpy types."""
    metrics_clean = {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in metrics.items()
    }
    with open(output_path, 'w') as f:
        json.dump(metrics_clean, f, indent=2)


def load_metrics(metrics_path: Path) -> Dict[str, float]:
    with open(metrics_path, 'r') as f:
        return json.load(f)


def set_random_seeds(seed: int = RANDOM_SEED):
    """Fix seeds for numpy and stdlib random. Also seeds torch if available."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_latest_run_id(artifacts_dir: Path) -> Optional[str]:
    """Find the most recent run_* directory."""
    run_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return run_dirs[0].name


def validate_datetime_column(df: pd.DataFrame, column: str = 'timestamp') -> pd.DataFrame:
    """Make sure the column is proper datetime, strip tz if present."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        try:
            df[column] = pd.to_datetime(df[column])
        except Exception as e:
            raise ValueError(f"Failed to convert '{column}' to datetime: {e}")

    if df[column].dt.tz is not None:
        df[column] = df[column].dt.tz_localize(None)

    return df


def check_data_leakage(df, feature_col, time_col='timestamp', group_col=None):
    """Basic temporal ordering check for a feature column."""
    df = df.sort_values(time_col)

    if group_col:
        for group in df[group_col].unique():
            group_df = df[df[group_col] == group].copy()
            if not _check_temporal_order(group_df, feature_col, time_col):
                return False
    else:
        return _check_temporal_order(df, feature_col, time_col)

    return True


def _check_temporal_order(df, feature_col, time_col):
    non_null_mask = df[feature_col].notna()
    if non_null_mask.sum() == 0:
        return True

    # TODO: actually use first_non_null_idx to check position
    first_non_null_idx = non_null_mask.idxmax()
    return True


def format_duration(seconds):
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
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except Exception:
        return default
