import pytest
import pandas as pd
import numpy as np
import logging
import yaml
import json
from pathlib import Path
from datetime import datetime

from utils import (
    setup_logging, generate_run_id, save_config, load_config_from_yaml,
    save_metrics, load_metrics, set_random_seeds, get_latest_run_id,
    validate_datetime_column, format_duration, safe_divide
)

# A simple mock Config for testing
class MockLogging:
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_date_format = '%Y-%m-%d %H:%M:%S'
    def get_log_level(self):
        return logging.INFO

class MockPaths:
    def __init__(self, tmp_path):
        self.artifacts_dir = tmp_path / "artifacts"

class MockConfig:
    def __init__(self, tmp_path):
        self.logging = MockLogging()
        self.paths = MockPaths(tmp_path)


def test_generate_run_id():
    run_id = generate_run_id()
    assert run_id.startswith("run_")
    assert len(run_id) > 10


def test_setup_logging(tmp_path):
    config = MockConfig(tmp_path)
    run_id = "test_run"
    logger = setup_logging(config, run_id)
    
    assert logger.level == logging.INFO
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    
    log_file = config.paths.artifacts_dir / run_id / "run.log"
    assert log_file.exists()


def test_save_load_config(tmp_path):
    from dataclasses import dataclass
    
    @dataclass
    class DummyConfig:
        name: str
        value: int
        paths: list

    config = DummyConfig(name="test", value=42, paths=[tmp_path / "test.txt"])
    
    out_file = tmp_path / "config.yaml"
    save_config(config, out_file)
    assert out_file.exists()
    
    loaded = load_config_from_yaml(out_file)
    assert loaded["name"] == "test"
    assert loaded["value"] == 42
    assert loaded["paths"][0] == str(tmp_path / "test.txt")


def test_save_load_metrics(tmp_path):
    metrics = {
        "test_mae": 1.23,
        "test_rmse": np.float64(2.34), # ensure numpy types are serialized
        "count": np.int64(42)
    }
    out_file = tmp_path / "metrics.json"
    save_metrics(metrics, out_file)
    assert out_file.exists()
    
    loaded = load_metrics(out_file)
    assert loaded["test_mae"] == 1.23
    assert loaded["test_rmse"] == 2.34
    assert loaded["count"] == 42


def test_set_random_seeds():
    # Smoke test just to ensure it doesn't crash
    set_random_seeds(42)


def test_get_latest_run_id(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    
    assert get_latest_run_id(artifacts_dir) is None
    
    (artifacts_dir / "run_1").mkdir()
    (artifacts_dir / "run_2").mkdir()
    
    latest = get_latest_run_id(artifacts_dir)
    assert latest in ["run_1", "run_2"] # ordering by mtime so it will pick whichever was written last


def test_validate_datetime_column():
    df = pd.DataFrame({
        "timestamp": ["2026-01-01 10:00:00", "2026-01-01 11:00:00"],
        "other": [1, 2]
    })
    
    # Needs conversion
    df = validate_datetime_column(df, "timestamp")
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    
    # Missing column
    with pytest.raises(ValueError):
        validate_datetime_column(df, "wrong_col")
        
    # Invalid conversion
    df_invalid = pd.DataFrame({"ts": ["not_a_time"]})
    with pytest.raises(ValueError):
        validate_datetime_column(df_invalid, "ts")
        
    # Timezone removal
    df_tz = pd.DataFrame({"ts": pd.date_range("2026-01-01", periods=2, tz="UTC")})
    df_tz = validate_datetime_column(df_tz, "ts")
    assert df_tz["ts"].dt.tz is None


def test_format_duration():
    assert format_duration(30) == "30.0s"
    assert format_duration(90) == "1m 30s"
    assert format_duration(3660) == "1h 1m"


def test_safe_divide():
    assert safe_divide(10, 2) == 5.0
    assert safe_divide(10, 0) == 0.0
    assert safe_divide(10, np.nan) == 0.0
    assert safe_divide(10, None) == 0.0
