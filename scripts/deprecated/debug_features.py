from future_prediction import FutureDelayPredictor
from datetime import datetime, timedelta
import pandas as pd

predictor = FutureDelayPredictor(
    'artifacts/run_20260210_153030/best_model.pkl',
    'artifacts/run_20260210_153030/feature_metadata.pkl'
)

meta = predictor.feature_metadata
print("Feature names in metadata:", len(meta.get('feature_names', [])))
print(meta.get('feature_names', []))

target_dt = datetime.now() + timedelta(hours=2)
features_df = predictor._engineer_features('Central', target_dt, None, None)

print("\nDataFrame created:")
print(f"Shape: {features_df.shape}")
print(f"Columns: {list(features_df.columns)}")
print(f"Number of columns: {len(features_df.columns)}")
