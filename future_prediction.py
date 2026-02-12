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

from config import Config, get_config
from features import (
    add_special_event_features,
    add_topology_features,
    add_train_frequency_features,
)


logger = logging.getLogger(__name__)


class FutureDelayPredictor:
    """Wraps trained model with feature engineering for real-time predictions."""

    def __init__(self, model_path: str, feature_metadata_path: str):
        logger.info("Loading model from: %s", model_path)
        try:
            self.model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise

        try:
            self.feature_metadata = joblib.load(feature_metadata_path)
            logger.info("Feature metadata loaded")
        except Exception as exc:
            logger.error("Failed to load feature metadata: %s", exc)
            raise

        self.uk_holidays = holidays.UK()
        self._config = get_config()
        logger.info("FutureDelayPredictor ready")

    def predict_delay(self, line, target_datetime, weather_forecast=None,
                      recent_delays=None) -> Dict:
        """Predict delay for a single line at a future datetime."""
        if target_datetime <= datetime.now():
            raise ValueError("target_datetime must be in the future")

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
            logger.error("Prediction error: %s", exc)
            raise

        # build 95% CI from empirical residual quantiles (per-line if available)
        rq_store = self.feature_metadata.get('residual_quantiles', {})
        line_q = rq_store.get(line, rq_store.get('__global__'))
        if line_q is not None:
            lower_bound = max(0.0, prediction + line_q['q025'])
            upper_bound = prediction + line_q['q975']
        else:
            # fallback for models trained before residual quantiles were added
            prediction_std = 1.5
            lower_bound = max(0.0, prediction - 1.96 * prediction_std)
            upper_bound = prediction + 1.96 * prediction_std

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
                logger.warning("Could not predict for %s: %s", target_time, exc)

        return pd.DataFrame(predictions)

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

        f['crowding_index'] = self._estimate_crowding(target_datetime)

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
            # reasonable defaults — zero would be unrealistically optimistic
            f['lag_delay_1'] = 2.0
            f['lag_delay_3'] = 2.0
            f['rolling_mean_delay_3'] = 2.0
            f['rolling_mean_delay_12'] = 2.0
            f['rolling_std_delay_12'] = 1.0
            f['recent_disruption_rate'] = 0.2

        # weather deltas default to 0 (no prior-hour reading at inference time)
        f['temp_delta_1h'] = 0.0
        f['precipitation_delta_1h'] = 0.0

        # interactions
        f['crowding_x_peak'] = f['crowding_index'] * f['peak_time']
        f['precipitation_x_temp'] = f['precipitation_mm'] * (1.0 / (abs(f['temp_c']) + 1))

        # network effects — neutral defaults
        # TODO: could call TfL API for all lines and compute real leave-one-out
        f['network_avg_delay'] = 0.0
        f['network_delay_volatility'] = 0.0
        f['lines_disrupted_ratio'] = 0.0
        f['is_network_wide_disruption'] = 0

        f['line'] = line
        df_row = pd.DataFrame([f])

        # reuse the same functions as features.py to avoid drift
        df_row = add_special_event_features(df_row)
        df_row = add_topology_features(df_row)
        df_row = add_train_frequency_features(df_row)

        return df_row

    def _validate_features(self, features):
        if not hasattr(self, 'feature_metadata') or self.feature_metadata is None:
            return

        expected = set(self.feature_metadata.get('all_features', []))
        actual = set(features.columns)

        missing = expected - actual
        if missing:
            logger.warning("Missing features vs training: %s", missing)

    def _get_valid_lines(self):
        return Config().data.tube_lines

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

    def _estimate_crowding(self, dt):
        if self._is_peak_time(dt):
            return 0.8
        elif 10 <= dt.hour < 16:
            return 0.5
        elif 19 <= dt.hour < 22:
            return 0.6
        return 0.2

    def _get_typical_weather(self, month):
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
