"""
Write feature_metadata.pkl with the 21-feature schema expected by the trained
model, using 'line_encoded' as the final positional feature.
"""

import joblib
from pathlib import Path

meta = {
    'feature_names': [
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday', 'peak_time',
        'temp_c', 'precipitation_mm', 'humidity',
        'crowding_index',
        'lag_delay_1', 'lag_delay_3',
        'rolling_mean_delay_3', 'rolling_mean_delay_12', 'rolling_std_delay_12',
        'recent_disruption_rate',
        'temp_delta_1h', 'precipitation_delta_1h',
        'crowding_x_peak', 'precipitation_x_temp',
        'line_encoded'
    ],
    'all_features': [
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday', 'peak_time',
        'temp_c', 'precipitation_mm', 'humidity',
        'crowding_index',
        'lag_delay_1', 'lag_delay_3',
        'rolling_mean_delay_3', 'rolling_mean_delay_12', 'rolling_std_delay_12',
        'recent_disruption_rate',
        'temp_delta_1h', 'precipitation_delta_1h',
        'crowding_x_peak', 'precipitation_x_temp',
        'line_encoded'
    ]
}

output_path = Path('artifacts/run_20260210_153030/feature_metadata.pkl')
joblib.dump(meta, output_path)

print(f'Updated metadata: {len(meta["feature_names"])} features')
print('Features:', meta['feature_names'][:5], '...')
