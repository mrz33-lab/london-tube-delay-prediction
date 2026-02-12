"""
Test suite for the London Underground Delay Prediction System.

Modules:
    test_schema        -- Validates raw data meets column/type/range contracts.
    test_leakage       -- Verifies that no future information leaks into features.
    test_features      -- Checks the 37-feature engineering pipeline end-to-end.
    test_api           -- Exercises FastAPI endpoints (predict, batch, health, lines).
    test_explain       -- Validates SHAP explainability artefacts.
    test_future_prediction -- Tests inference-time feature engineering and CI generation.
"""

__all__ = [
    "test_schema",
    "test_leakage",
    "test_features",
    "test_api",
    "test_explain",
    "test_future_prediction",
]
