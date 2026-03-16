import pytest
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import joblib

from app.data_loading import load_artifacts, load_collection_status

def test_load_artifacts_empty(tmp_path):
    # Should handle empty directory gracefully
    artifacts = load_artifacts(str(tmp_path))
    assert isinstance(artifacts, dict)
    assert not artifacts

def test_load_artifacts_with_data(tmp_path):
    # Create some dummy files
    model_path = tmp_path / "best_model.pkl"
    joblib.dump({"dummy": "model"}, model_path)
    
    metrics_path = tmp_path / "all_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"test_mae": 1.5}, f)
        
    pred_path = tmp_path / "test_predictions.csv"
    pd.DataFrame({
        "timestamp": ["2026-01-01 10:00:00"],
        "actual": [1.0]
    }).to_csv(pred_path, index=False)
    
    best_txt_path = tmp_path / "best_model_name.txt"
    best_txt_path.write_text("lightgbm\n")
    
    artifacts = load_artifacts(str(tmp_path))
    assert "best_model" in artifacts
    assert artifacts["metrics"]["test_mae"] == 1.5
    assert "test_predictions" in artifacts
    assert pd.api.types.is_datetime64_any_dtype(artifacts["test_predictions"]["timestamp"])
    assert artifacts["best_model_name"] == "lightgbm"

def test_load_collection_status_empty(tmp_path):
    status = load_collection_status(str(tmp_path))
    assert status["record_count"] == 0
    assert status["has_data"] is False
    assert status["is_active"] is False

def test_load_collection_status_with_data(tmp_path):
    csv_path = tmp_path / "tfl_merged.csv"
    now = datetime.now()
    
    # Create 3 records spanning 2 hours.
    df = pd.DataFrame({
        "timestamp": [now - timedelta(hours=2), now - timedelta(hours=1), now],
        "line": ["District", "Circle", "District"]
    })
    df.to_csv(csv_path, index=False)
    
    status = load_collection_status(str(tmp_path))
    
    assert status["record_count"] == 3
    assert status["has_data"] is True
    assert status["is_active"] is True
    assert "District" in status["lines_present"]
    assert "Circle" in status["lines_present"]
    assert status["rate_per_hour"] == 1.5 # 3 records / 2 hours

def test_load_collection_status_inactive(tmp_path):
    csv_path = tmp_path / "tfl_merged.csv"
    old_time = datetime.now() - timedelta(hours=2)
    
    # All records older than 30 mins
    df = pd.DataFrame({
        "timestamp": [old_time, old_time + timedelta(minutes=1)]
    })
    df.to_csv(csv_path, index=False)
    
    status = load_collection_status(str(tmp_path))
    assert status["is_active"] is False
