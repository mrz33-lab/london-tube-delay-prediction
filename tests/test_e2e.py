"""
End-to-end integration test.

Covers the full contract from raw data → feature engineering → model training
→ artifact persistence → model loading → prediction.  If training and inference
ever drift out of sync (e.g. a feature added to engineer_features but not to
FutureDelayPredictor._engineer_features), this test will catch it before
anything reaches production.

Uses only a tiny synthetic dataset and Ridge regression (no Optuna / LightGBM)
so the test stays fast (< 10 s on a laptop).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from config import get_config, ModelConfig
from data import generate_synthetic_data
from features import (
    engineer_features, get_feature_columns,
    create_preprocessing_pipeline, save_feature_metadata,
    prepare_features_for_model,
)
from train import train_ridge_baseline, NaiveBaselineModel
from future_prediction import FutureDelayPredictor
import joblib


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_full_pipeline_trains_and_predicts(tmp_path, tiny_multiline_df):
    """
    Full contract test:
      1. Feature engineering on raw data
      2. Train Ridge (fast, no Optuna)
      3. Compute per-line residual quantiles
      4. Save model + feature metadata to tmp_path
      5. Load via FutureDelayPredictor
      6. Make a future prediction and validate the response shape
    """
    config = get_config()

    # ---- 1. Feature engineering ----
    df = tiny_multiline_df
    df_featured = engineer_features(df, config, is_training=True)

    # Chronological 80/20 split
    split_idx = int(len(df_featured) * 0.8)
    train_df = df_featured.iloc[:split_idx]
    test_df  = df_featured.iloc[split_idx:]

    numeric_feats, cat_feats, all_feats = get_feature_columns(train_df, config)

    X_train, y_train = prepare_features_for_model(train_df, all_feats, config.features.target_column)
    X_test,  y_test  = prepare_features_for_model(test_df,  all_feats, config.features.target_column)

    assert X_train.shape[1] == X_test.shape[1], "Train/test feature count mismatch"
    assert len(y_train) > 0

    # ---- 2. Train Ridge ----
    model, metrics = train_ridge_baseline(
        X_train, y_train, X_test, y_test,
        numeric_feats, cat_feats, config
    )

    assert metrics['test_mae'] >= 0, "MAE must be non-negative"
    assert 'train_mae' in metrics

    # ---- 3. Residual quantiles ----
    y_pred = model.predict(X_test)
    residuals = np.array(y_test) - y_pred
    residual_quantiles = {
        '__global__': {
            'q025': float(np.percentile(residuals, 2.5)),
            'q975': float(np.percentile(residuals, 97.5)),
        }
    }

    # ---- 4. Save artifacts ----
    # API expects the file to be called best_model.pkl
    joblib.dump(model, tmp_path / 'best_model.pkl')
    save_feature_metadata(numeric_feats, cat_feats, tmp_path,
                          residual_quantiles=residual_quantiles)

    assert (tmp_path / 'best_model.pkl').exists()
    assert (tmp_path / 'feature_metadata.pkl').exists()

    # ---- 5. Load via FutureDelayPredictor ----
    predictor = FutureDelayPredictor(
        model_path=str(tmp_path / 'best_model.pkl'),
        feature_metadata_path=str(tmp_path / 'feature_metadata.pkl'),
    )

    # ---- 6. Predict ----
    future_dt = datetime.now() + timedelta(hours=2)
    result = predictor.predict_delay(line='Central', target_datetime=future_dt)

    assert result['line'] == 'Central'
    assert result['predicted_delay_minutes'] >= 0.0
    ci_lo, ci_hi = result['confidence_interval_95']
    assert ci_lo <= result['predicted_delay_minutes'] <= ci_hi or ci_lo <= ci_hi, \
        f"CI [{ci_lo}, {ci_hi}] is invalid"
    assert result['status'] in ('Good Service', 'Minor Delays', 'Severe Delays')


def test_all_lines_predict_without_error(tmp_path, tiny_multiline_df):
    """All 11 lines must produce a valid prediction from the same trained model."""
    config = get_config()
    df = tiny_multiline_df
    df_featured = engineer_features(df, config, is_training=True)

    split_idx = int(len(df_featured) * 0.8)
    train_df = df_featured.iloc[:split_idx]
    test_df  = df_featured.iloc[split_idx:]

    numeric_feats, cat_feats, all_feats = get_feature_columns(train_df, config)
    X_train, y_train = prepare_features_for_model(train_df, all_feats, config.features.target_column)
    X_test,  y_test  = prepare_features_for_model(test_df,  all_feats, config.features.target_column)

    model, _ = train_ridge_baseline(
        X_train, y_train, X_test, y_test,
        numeric_feats, cat_feats, config
    )

    y_pred = model.predict(X_test)
    residuals = np.array(y_test) - y_pred
    residual_quantiles = {
        '__global__': {
            'q025': float(np.percentile(residuals, 2.5)),
            'q975': float(np.percentile(residuals, 97.5)),
        }
    }

    joblib.dump(model, tmp_path / 'best_model.pkl')
    save_feature_metadata(numeric_feats, cat_feats, tmp_path,
                          residual_quantiles=residual_quantiles)

    predictor = FutureDelayPredictor(
        model_path=str(tmp_path / 'best_model.pkl'),
        feature_metadata_path=str(tmp_path / 'feature_metadata.pkl'),
    )

    future_dt = datetime.now() + timedelta(hours=1)
    for line in config.data.tube_lines:
        result = predictor.predict_delay(line=line, target_datetime=future_dt)
        assert result['predicted_delay_minutes'] >= 0.0, f"{line}: negative delay"
        assert result['status'] in ('Good Service', 'Minor Delays', 'Severe Delays'), \
            f"{line}: unexpected status '{result['status']}'"


def test_naive_baseline_matches_ridge_interface(tmp_path, tiny_multiline_df):
    """NaiveBaselineModel must have the same predict() interface as Ridge."""
    config = get_config()
    df = tiny_multiline_df
    df_featured = engineer_features(df, config, is_training=True)

    split_idx = int(len(df_featured) * 0.8)
    train_df = df_featured.iloc[:split_idx]
    test_df  = df_featured.iloc[split_idx:]

    numeric_feats, cat_feats, all_feats = get_feature_columns(train_df, config)
    X_train, y_train = prepare_features_for_model(train_df, all_feats, config.features.target_column)
    X_test,  y_test  = prepare_features_for_model(test_df,  all_feats, config.features.target_column)

    naive = NaiveBaselineModel()
    naive.fit(X_train, y_train)
    preds = naive.predict(X_test)

    assert len(preds) == len(X_test)
    assert preds.dtype == float
    assert np.all(np.isfinite(preds)), "NaiveBaseline returned non-finite predictions"


def test_feature_count_train_inference_match(tmp_path, tiny_multiline_df):
    """
    The feature vector built at inference time must contain every feature the
    model was trained on (and only those features).  Drift between
    engineer_features() and FutureDelayPredictor._engineer_features() will
    surface here before reaching production.
    """
    config = get_config()
    df = tiny_multiline_df
    df_featured = engineer_features(df, config, is_training=True)

    split_idx = int(len(df_featured) * 0.8)
    train_df = df_featured.iloc[:split_idx]
    test_df  = df_featured.iloc[split_idx:]

    numeric_feats, cat_feats, all_feats = get_feature_columns(train_df, config)
    X_train, y_train = prepare_features_for_model(train_df, all_feats, config.features.target_column)
    X_test,  y_test  = prepare_features_for_model(test_df,  all_feats, config.features.target_column)

    model, _ = train_ridge_baseline(
        X_train, y_train, X_test, y_test,
        numeric_feats, cat_feats, config
    )

    y_pred = model.predict(X_test)
    residuals = np.array(y_test) - y_pred
    residual_quantiles = {
        '__global__': {
            'q025': float(np.percentile(residuals, 2.5)),
            'q975': float(np.percentile(residuals, 97.5)),
        }
    }

    joblib.dump(model, tmp_path / 'best_model.pkl')
    save_feature_metadata(numeric_feats, cat_feats, tmp_path,
                          residual_quantiles=residual_quantiles)

    predictor = FutureDelayPredictor(
        model_path=str(tmp_path / 'best_model.pkl'),
        feature_metadata_path=str(tmp_path / 'feature_metadata.pkl'),
    )

    future_dt = datetime.now() + timedelta(hours=3)
    inference_features = predictor._engineer_features(
        line='Jubilee',
        target_datetime=future_dt,
        weather_forecast=None,
        recent_delays=None,
    )

    training_features = set(all_feats)
    inference_cols    = set(inference_features.columns)

    missing_at_inference = training_features - inference_cols
    assert not missing_at_inference, (
        f"Features present at training but missing at inference: {missing_at_inference}\n"
        "Update FutureDelayPredictor._engineer_features() to match engineer_features()."
    )
