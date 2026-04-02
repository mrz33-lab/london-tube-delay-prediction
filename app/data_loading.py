"""
Cached data loading for the Streamlit dashboard.
"""

import streamlit as st
import pandas as pd
import json
import joblib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

from app.constants import DATA_COLLECTION_TARGET

# NaiveBaselineModel import is needed so joblib can unpickle the model
from train import NaiveBaselineModel  # noqa: F401


logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def load_artifacts(artifact_dir: str) -> Dict:
    """Load training artifacts from disk (cached)."""
    
    # Fix for unpickling NaiveBaselineModel which was saved in __main__
    import sys
    import train
    sys.modules["__main__"].NaiveBaselineModel = train.NaiveBaselineModel
    
    path = Path(artifact_dir)
    artifacts: Dict = {}

    try:
        for name in ("naive", "ridge", "best"):
            p = path / f"{name}_model.pkl"
            if p.exists():
                artifacts[f"{name}_model"] = joblib.load(p)

        metrics_p = path / "all_metrics.json"
        if metrics_p.exists():
            with open(metrics_p) as f:
                artifacts["metrics"] = json.load(f)

        pred_p = path / "test_predictions.csv"
        if pred_p.exists():
            df = pd.read_csv(pred_p)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            artifacts["test_predictions"] = df

        comp_p = path / "model_comparison.csv"
        if comp_p.exists():
            artifacts["model_comparison"] = pd.read_csv(comp_p)

        feat_p = path / "feature_importance.csv"
        if feat_p.exists():
            artifacts["feature_importance"] = pd.read_csv(feat_p)

        best_p = path / "best_model_name.txt"
        if best_p.exists():
            artifacts["best_model_name"] = best_p.read_text().strip()

    except Exception as exc:
        st.error(f"Error loading artifacts: {exc}")

    return artifacts


@st.cache_data(show_spinner=False, ttl=60)
def load_collection_status(data_dir: str) -> Dict:
    """Read real data CSV and compute collection progress. TTL=60s."""
    result: Dict = {
        "record_count": 0,
        "target":       DATA_COLLECTION_TARGET,
        "pct":          0.0,
        "rate_per_hour":0.0,
        "first_ts":     None,
        "last_ts":      None,
        "is_active":    False,
        "eta_hours":    None,
        "lines_present":[],
        "has_data":     False,
    }

    csv_path = Path(data_dir) / "tfl_merged.csv"
    if not csv_path.exists():
        return result

    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        n = len(df)
        result["record_count"] = n
        result["has_data"] = n > 0

        if n > 0:
            result["pct"] = min(n / DATA_COLLECTION_TARGET * 100, 100)
            result["first_ts"] = df["timestamp"].min()
            result["last_ts"]  = df["timestamp"].max()

            elapsed_h = (result["last_ts"] - result["first_ts"]).total_seconds() / 3600
            if elapsed_h > 0:
                result["rate_per_hour"] = n / elapsed_h
                remaining = DATA_COLLECTION_TARGET - n
                if result["rate_per_hour"] > 0:
                    result["eta_hours"] = remaining / result["rate_per_hour"]

            if "line" in df.columns:
                result["lines_present"] = sorted(df["line"].unique().tolist())

            # Collection is considered active if the most recent record
            # is less than 30 minutes old.
            cutoff = datetime.now() - timedelta(minutes=30)
            result["is_active"] = result["last_ts"].to_pydatetime().replace(tzinfo=None) > cutoff

    except Exception as e:
        logger.error(f"Error loading collection status: {e}")

    return result
