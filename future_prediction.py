"""
Inference-time delay predictor.

Loads a trained model and mirrors the feature engineering from features.py
so training and inference stay consistent.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import joblib
import logging
import holidays

from config import get_config
from features import (
    add_temporal_encoding_features,
    add_topology_features,
    add_train_frequency_features,
)
from events import get_event_features
from line_metadata import LINE_BASE_DELAYS
from data_collection import LINE_CROWDING_WEIGHT


logger = logging.getLogger(__name__)


class FutureDelayPredictor:
    """Wraps trained model with feature engineering for real-time predictions."""

    def __init__(self, model_path: str, feature_metadata_path: str):
        logger.info(f"Loading model from: {model_path}")
        try:
            self.model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        except Exception as exc:
            logger.error(f"Failed to load model: {exc}")
            raise

        try:
            self.feature_metadata = joblib.load(feature_metadata_path)
            logger.info("Feature metadata loaded")
        except Exception as exc:
            logger.error(f"Failed to load feature metadata: {exc}")
            raise

        self.uk_holidays = holidays.UK()
        self._config = get_config()
        logger.info("FutureDelayPredictor ready")

    def predict_delay(self, line, target_datetime, weather_forecast=None,
                      recent_delays=None) -> Dict:
        """Predict delay for a single line at a future datetime."""
        # Normalise to naive datetime before comparing so that tz-aware inputs
        # don't raise TypeError against datetime.now() (which is always naive).
        target_naive = (
            target_datetime.replace(tzinfo=None)
            if target_datetime.tzinfo is not None
            else target_datetime
        )
        if target_naive < datetime.now():
            raise ValueError("target_datetime must be in the future (or present)")

        if line not in self._get_valid_lines():
            raise ValueError(f"Invalid line: {line}")

        features = self._engineer_features(
            line=line, target_datetime=target_datetime,
            weather_forecast=weather_forecast, recent_delays=recent_delays,
        )

        self._validate_features(features)

        try:
            prediction = float(self.model.predict(features)[0])
        except Exception as exc:
            logger.error(f"Prediction error: {exc}")
            raise

        lower_bound, upper_bound = self._get_confidence_interval(prediction, line)

        return {
            'line': line,
            'datetime': target_datetime,
            'predicted_delay_minutes': max(0.0, prediction),
            'confidence_interval_95': (lower_bound, upper_bound),
            'status': self._get_status_label(prediction),
            'features_used': features.values.tolist()[0],
        }

    def predict_next_24_hours(self, line, interval_minutes=60) -> pd.DataFrame:
        """Generate predictions for the next 24h at regular intervals."""
        num_predictions = (24 * 60) // interval_minutes
        base_time = datetime.now() + timedelta(minutes=interval_minutes)

        predictions = []
        for i in range(num_predictions):
            target_time = base_time + timedelta(minutes=i * interval_minutes)
            try:
                pred = self.predict_delay(line, target_time)
                predictions.append({
                    'datetime': target_time,
                    'predicted_delay': pred['predicted_delay_minutes'],
                    'status': pred['status'],
                })
            except Exception as exc:
                logger.warning(f"Could not predict for {target_time}: {exc}")

        return pd.DataFrame(predictions)

    def predict_from_features(self, features: pd.DataFrame, line: str) -> Dict:
        """Run model.predict + CI for a pre-built feature row.

        Accepts a DataFrame already prepared by _engineer_features (or any
        caller that assembles the correct columns).  Does NOT validate that
        target_datetime is in the future, so this is safe to call for
        historical/scenario datetimes from the dashboard.

        Returns:
            dict with keys: prediction, ci_lo, ci_hi
        """
        self._validate_features(features)
        pred = float(self.model.predict(features)[0])
        lo, hi = self._get_confidence_interval(pred, line)
        return {
            "prediction": max(0.0, pred),
            "ci_lo":      max(0.0, float(lo)),
            "ci_hi":      float(hi),
        }

    # ------------------------------------------------------------------

    def _engineer_features(self, line, target_datetime, weather_forecast,
                           recent_delays):
        f: Dict = {}

        # temporal
        hour = target_datetime.hour
        f['hour'] = hour
        f['day_of_week'] = target_datetime.weekday()
        f['month'] = target_datetime.month
        f['is_weekend'] = int(target_datetime.weekday() >= 5)
        f['is_holiday'] = int(target_datetime.date() in self.uk_holidays)
        f['peak_time'] = int(self._is_peak_time(target_datetime))

        # weather
        if weather_forecast:
            f['temp_c'] = weather_forecast.get('temperature', 12.0)
            f['precipitation_mm'] = weather_forecast.get('precipitation', 0.0)
            f['humidity'] = weather_forecast.get('humidity', 70.0)
        else:
            seasonal = self._get_typical_weather(target_datetime.month)
            f['temp_c'] = seasonal['temp_c']
            f['precipitation_mm'] = seasonal['precipitation_mm']
            f['humidity'] = seasonal['humidity']

        f['crowding_index'] = self._estimate_crowding(target_datetime, line)

        # lag / rolling — use recent history if available, otherwise defaults
        if recent_delays is not None and len(recent_delays) > 0:
            delays = recent_delays['delay_minutes']
            f['lag_delay_1'] = delays.iloc[-1]
            f['lag_delay_3'] = delays.iloc[-3] if len(delays) >= 3 else delays.iloc[-1]
            f['rolling_mean_delay_3'] = delays.tail(3).mean()
            f['rolling_mean_delay_12'] = delays.tail(12).mean()
            f['rolling_std_delay_12'] = delays.tail(12).std() if len(delays) >= 2 else 0.0
            f['recent_disruption_rate'] = (
                (recent_delays['status'].tail(12) != 'Good Service').mean()
            )
        else:
            # Use per-line baseline delays from line_metadata rather than
            # a single magic number — lines like Waterloo & City (1.5 min
            # baseline) and Bakerloo (3.2 min) differ meaningfully.
            base = LINE_BASE_DELAYS.get(line, 3.0)
            f['lag_delay_1'] = base
            f['lag_delay_3'] = base
            f['rolling_mean_delay_3'] = base
            f['rolling_mean_delay_12'] = base
            f['rolling_std_delay_12'] = base * 0.4  # ~40% coefficient of variation is typical for London tube delay series
            f['recent_disruption_rate'] = 0.2

        # weather deltas default to 0 (no prior-hour reading at inference time)
        f['temp_delta_1h'] = 0.0
        f['precipitation_delta_1h'] = 0.0

        # interactions
        f['crowding_x_peak'] = f['crowding_index'] * f['peak_time']
        f['precipitation_x_temp'] = f['precipitation_mm'] * (1.0 / (abs(f['temp_c']) + 1))

        # network effects — use training-set means stored in feature_metadata so
        # inference inputs are in-distribution; is_network_wide_disruption stays 0
        # (conservative: assume no widespread disruption when state is unknown)
        net_defaults = self.feature_metadata.get('network_feature_means', {})
        f['network_avg_delay'] = net_defaults.get('network_avg_delay', 2.0)
        f['network_delay_volatility'] = net_defaults.get('network_delay_volatility', 1.0)
        f['lines_disrupted_ratio'] = net_defaults.get('lines_disrupted_ratio', 0.2)
        f['is_network_wide_disruption'] = 0

        f['line'] = line
        df_row = pd.DataFrame([f])

        # reuse the same functions as features.py to avoid drift
        df_row = add_temporal_encoding_features(df_row)
        df_row = add_topology_features(df_row)
        df_row = add_train_frequency_features(df_row)

        # event features — procedural calendar, no external state required
        event_feats = get_event_features(
            timestamps=pd.Series([target_datetime], index=df_row.index),
            lines=df_row['line'],
        )
        df_row = df_row.join(event_feats, how='left')
        df_row[['is_major_event', 'event_crowd_boost', 'seasonal_demand_factor']] = (
            df_row[['is_major_event', 'event_crowd_boost', 'seasonal_demand_factor']].fillna(0)
        )

        return df_row

    def _get_confidence_interval(
        self, prediction: float, line: str, alpha: float = 0.05
    ) -> tuple:
        """Build a 95% prediction interval using the best available method.

        Priority order:
          1. Split-conformal PI (Vovk et al. 2005; Angelopoulos & Bates 2022)
             — distribution-free coverage guarantee of ≥ 1-α under exchangeability.
             Requires conformal_cal_scores stored in feature_metadata at training time.
          2. Per-line empirical residual quantiles — sensible default when conformal
             scores are unavailable (models trained before this feature was added).
          3. Gaussian fallback (±1.96σ) — last resort, requires no calibration data.

        The conformal half-width is the ⌈(n+1)(1-α)⌉/n empirical quantile of
        |y_true - y_pred| from the held-out calibration split, which guarantees
        that P(y_true ∈ [ŷ - q̂, ŷ + q̂]) ≥ 1-α in finite samples.
        """
        # --- 1. Split-conformal PI ---
        cal_scores = self.feature_metadata.get('conformal_cal_scores')
        if cal_scores is not None and len(cal_scores) >= 10:
            n = len(cal_scores)
            q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
            q_hat = float(np.quantile(cal_scores, q_level))
            return max(0.0, prediction - q_hat), prediction + q_hat

        # --- 2. Per-line empirical residual quantiles ---
        rq_store = self.feature_metadata.get('residual_quantiles', {})
        line_q = rq_store.get(line, rq_store.get('__global__'))
        if line_q is not None:
            return (
                max(0.0, prediction + line_q['q025']),
                prediction + line_q['q975'],
            )

        # --- 3. Gaussian fallback ---
        prediction_std = self._config.explainability.ci_fallback_std
        return (
            max(0.0, prediction - 1.96 * prediction_std),
            prediction + 1.96 * prediction_std,
        )

    def _validate_features(self, features):
        if not hasattr(self, 'feature_metadata') or self.feature_metadata is None:
            return

        expected = set(self.feature_metadata.get('all_features', []))
        actual = set(features.columns)

        missing = expected - actual
        if missing:
            logger.warning(f"Missing features vs training: {missing}")

    def _get_valid_lines(self):
        return self._config.data.tube_lines

    def _get_status_label(self, delay_minutes):
        if delay_minutes < self._config.data.status_good_max:
            return 'Good Service'
        elif delay_minutes < self._config.data.status_minor_max:
            return 'Minor Delays'
        return 'Severe Delays'

    def _is_peak_time(self, dt):
        if dt.weekday() < 5:
            return (7 <= dt.hour < 10) or (16 <= dt.hour < 19)
        return False

    def _estimate_crowding(self, dt, line):
        base = LINE_CROWDING_WEIGHT.get(line, 0.05) * 5.0
        if self._is_peak_time(dt):
            base += 0.35
        elif 10 <= dt.hour < 16:
            base += 0.15
        if dt.weekday() >= 5:
            base *= 0.6
        return round(max(0.0, min(1.0, base)), 3)

    def _get_typical_weather(self, month):
        # Monthly averages for central London.
        # Source: Met Office UK climate averages 1991–2020
        # (https://www.metoffice.gov.uk/research/climate/maps-and-data/uk-climate-averages).
        # Temperature = mean daily temp (°C); precipitation = mean daily total (mm);
        # humidity = mean relative humidity at 0900 UTC (%).
        seasonal = {
            1:  {'temp_c': 7.0,  'precipitation_mm': 2.2, 'humidity': 80.0},
            2:  {'temp_c': 7.0,  'precipitation_mm': 1.6, 'humidity': 77.0},
            3:  {'temp_c': 9.0,  'precipitation_mm': 1.7, 'humidity': 72.0},
            4:  {'temp_c': 11.0, 'precipitation_mm': 1.8, 'humidity': 68.0},
            5:  {'temp_c': 15.0, 'precipitation_mm': 2.0, 'humidity': 66.0},
            6:  {'temp_c': 18.0, 'precipitation_mm': 1.8, 'humidity': 65.0},
            7:  {'temp_c': 20.0, 'precipitation_mm': 1.8, 'humidity': 64.0},
            8:  {'temp_c': 20.0, 'precipitation_mm': 2.0, 'humidity': 66.0},
            9:  {'temp_c': 17.0, 'precipitation_mm': 2.0, 'humidity': 70.0},
            10: {'temp_c': 14.0, 'precipitation_mm': 2.7, 'humidity': 75.0},
            11: {'temp_c': 10.0, 'precipitation_mm': 2.4, 'humidity': 80.0},
            12: {'temp_c': 7.0,  'precipitation_mm': 2.2, 'humidity': 82.0},
        }
        return seasonal.get(month, seasonal[1])
