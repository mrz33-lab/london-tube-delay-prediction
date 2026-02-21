"""API integration tests (mocked predictor, no model files needed)."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import datetime as dt
import pandas as pd


def _make_mock_predictor():
    mock = MagicMock()
    mock.predict_delay.return_value = {
        'line': 'Central',
        'datetime': dt.datetime(2027, 1, 1, 9, 0),
        'predicted_delay_minutes': 3.5,
        'confidence_interval_95': (0.57, 6.43),
        'status': 'Minor Delays',
        'features_used': [],
    }
    mock.predict_next_24_hours.return_value = pd.DataFrame([
        {'datetime': dt.datetime(2027, 1, 1, h), 'predicted_delay': 2.0, 'status': 'Good Service'}
        for h in range(24)
    ])
    return mock


@pytest.fixture()
def client():
    import api
    mock_predictor = _make_mock_predictor()

    async def _fake_load(app):
        app.state.predictor     = mock_predictor
        app.state.active_run_id = 'run_test'

    with patch.object(api, '_load_model', _fake_load):
        with TestClient(api.app) as test_client:
            yield test_client


@pytest.fixture()
def client_no_model():
    import api

    async def _no_model(app):
        app.state.predictor     = None
        app.state.active_run_id = None

    with patch.object(api, '_load_model', _no_model):
        with TestClient(api.app) as test_client:
            yield test_client


def test_root_returns_info(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "endpoints" in data


def test_health_check_healthy(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "run_test"


def test_health_check_unhealthy(client_no_model):
    response = client_no_model.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["model_loaded"] is False


def test_predict_valid_request(client):
    response = client.post("/predict", json={
        "line": "Central",
        "datetime": "2027-06-01T09:00:00"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["line"] == "Central"
    assert "predicted_delay_minutes" in data
    assert "confidence_interval_95" in data
    assert "status" in data
    assert data["model_version"] == "run_test"


def test_predict_invalid_line(client):
    response = client.post("/predict", json={
        "line": "Hogwarts Express",
        "datetime": "2027-06-01T09:00:00"
    })
    assert response.status_code == 422


def test_predict_past_datetime(client):
    response = client.post("/predict", json={
        "line": "Central",
        "datetime": "2000-01-01T09:00:00"
    })
    assert response.status_code == 422


def test_predict_no_model(client_no_model):
    response = client_no_model.post("/predict", json={
        "line": "Central",
        "datetime": "2027-06-01T09:00:00"
    })
    assert response.status_code == 503


def test_get_lines(client):
    response = client.get("/lines")
    assert response.status_code == 200
    lines = response.json()["lines"]
    assert len(lines) == 11
    assert "Central" in lines
    assert "Waterloo & City" in lines


def test_batch_predict_valid(client):
    response = client.post("/predict/batch", json={
        "predictions": [
            {"line": "Central", "datetime": "2027-06-01T09:00:00"},
            {"line": "Jubilee", "datetime": "2027-06-01T10:00:00"},
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["total_requested"] == 2
    assert data["successful"] == 2
    assert data["failed"] == 0


def test_batch_predict_exceeds_limit(client):
    predictions = [
        {"line": "Central", "datetime": "2027-06-01T09:00:00"}
        for _ in range(101)
    ]
    response = client.post("/predict/batch", json={"predictions": predictions})
    assert response.status_code == 400


def test_resolve_latest_run_id_picks_most_recent(tmp_path):
    from api import _resolve_latest_run_id

    (tmp_path / "run_20260101_000000").mkdir()
    (tmp_path / "run_20260210_153030").mkdir()
    (tmp_path / "run_20260301_120000").mkdir()
    (tmp_path / "some_other_dir").mkdir()

    result = _resolve_latest_run_id(tmp_path)
    assert result == "run_20260301_120000"


def test_resolve_latest_run_id_empty(tmp_path):
    from api import _resolve_latest_run_id
    assert _resolve_latest_run_id(tmp_path) is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
