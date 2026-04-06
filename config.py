"""
Central config — all settings in one place.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import logging

from line_metadata import LINE_LENGTH_KM


RANDOM_SEED = 42


@dataclass
class PathConfig:
    """File system paths."""

    project_root: Path = field(default_factory=lambda: Path(__file__).parent)
    data_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    real_data_file: str = "tfl_merged.csv"

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.artifacts_dir = self.project_root / "artifacts"

    def ensure_dirs(self):
        """Create data/ and artifacts/ dirs.

        Kept separate from __post_init__ so importing config
        doesn't create directories as a side effect.
        """
        self.data_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)


@dataclass
class DataConfig:
    """Schema and synthetic data settings."""

    required_columns: List[str] = field(default_factory=lambda: [
        'timestamp', 'line', 'status', 'delay_minutes',
        'temp_c', 'precipitation_mm', 'humidity', 'crowding_index',
        'is_weekend', 'hour', 'day_of_week', 'month', 'peak_time', 'is_holiday'
    ])

    # Single source of truth: derived from line_metadata.LINE_LENGTH_KM keys.
    # Adding a line to line_metadata automatically makes it valid here too.
    tube_lines: List[str] = field(
        default_factory=lambda: list(LINE_LENGTH_KM.keys())
    )

    status_categories: List[str] = field(default_factory=lambda: [
        'Good Service', 'Minor Delays', 'Severe Delays'
    ])

    # thresholds for mapping predicted delay -> status label
    status_good_max: float = 3.0
    status_minor_max: float = 10.0

    synthetic_n_days: int = 365
    synthetic_samples_per_day: int = 96  # every 15 min

    # value ranges for schema validation
    delay_min: float = 0.0
    delay_max: float = 120.0
    temp_min: float = -5.0
    temp_max: float = 35.0
    precipitation_min: float = 0.0
    precipitation_max: float = 50.0
    humidity_min: float = 0.0
    humidity_max: float = 100.0
    crowding_min: float = 0.0
    crowding_max: float = 1.0


@dataclass
class FeatureConfig:
    """Feature engineering settings."""

    lag_features: List[int] = field(default_factory=lambda: [1, 3])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 12])
    target_column: str = 'delay_minutes'
    group_column: str = 'line'
    time_column: str = 'timestamp'
    # Number of past periods to use when computing the recent disruption rate.
    # Defaults to 12 (equivalent to 3 hours at 15-min data frequency).
    disruption_rate_window: int = 12
    exclude_columns: List[str] = field(default_factory=lambda: [
        'timestamp', 'status'
    ])



@dataclass
class ModelConfig:
    """Training and evaluation settings."""

    train_ratio: float = 0.8
    cv_splits: int = 5
    n_iter_search: int = 20
    # Block size for block bootstrap CI (None = auto: int(sqrt(n)))
    bootstrap_block_size: Optional[int] = None

    models_to_train: List[str] = field(default_factory=lambda: [
        'naive', 'ridge', 'lightgbm'
    ])

    optuna_n_trials: int = 50

    lightgbm_optuna_space: dict = field(default_factory=lambda: {
        'num_leaves': {'low': 15, 'high': 127},
        'max_depth': {'low': 3, 'high': 10},
        'learning_rate': {'low': 0.01, 'high': 0.2},
        'n_estimators': {'low': 50, 'high': 500},
        'min_child_samples': {'low': 10, 'high': 100},
        'subsample': {'low': 0.5, 'high': 1.0},
        'colsample_bytree': {'low': 0.5, 'high': 1.0},
        'reg_alpha': {'low': 1e-8, 'high': 10.0},
        'reg_lambda': {'low': 1e-8, 'high': 10.0}
    })

    ridge_params: dict = field(default_factory=lambda: {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    })

    rf_params: dict = field(default_factory=lambda: {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    })

    scoring: str = 'neg_mean_absolute_error'

    # __post_init__ placed after all field declarations so references to
    # sibling fields are unambiguous and the class is easy to read top-to-bottom.
    def __post_init__(self):
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError(f"train_ratio must be in (0, 1), got {self.train_ratio}")
        if self.cv_splits < 2:
            raise ValueError(f"cv_splits must be >= 2, got {self.cv_splits}")
        if self.n_iter_search < 1:
            raise ValueError(f"n_iter_search must be >= 1, got {self.n_iter_search}")
        if self.optuna_n_trials < 1:
            raise ValueError(f"optuna_n_trials must be >= 1, got {self.optuna_n_trials}")


@dataclass
class ExplainabilityConfig:
    """SHAP settings."""

    shap_sample_size: int = 500
    shap_background_size: int = 100
    top_n_features: int = 10
    n_local_examples: int = 3
    # Fallback prediction std (minutes) used only when a model was trained
    # before per-line residual quantiles were introduced.
    ci_fallback_std: float = 1.5


@dataclass
class LoggingConfig:
    log_level: str = 'INFO'
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_date_format: str = '%Y-%m-%d %H:%M:%S'

    def get_log_level(self) -> int:
        return getattr(logging, self.log_level.upper(), logging.INFO)


@dataclass
class Config:
    """Top-level config, aggregates everything."""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    random_seed: int = RANDOM_SEED
    run_id: Optional[str] = None

    def get_artifact_dir(self) -> Path:
        if self.run_id is None:
            raise ValueError("run_id must be set before getting artifact directory")
        artifact_dir = self.paths.artifacts_dir / self.run_id
        artifact_dir.mkdir(exist_ok=True, parents=True)
        return artifact_dir


def get_config() -> Config:
    return Config()
