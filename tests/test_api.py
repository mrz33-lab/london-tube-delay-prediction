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


def test_forecast_endpoint_returns_forecast(client):
    response = client.post("/predict/forecast?line=Central&hours_ahead=4&interval_minutes=60")
    assert response.status_code == 200
    data = response.json()
    assert data["line"] == "Central"
    assert "forecast" in data
    assert len(data["forecast"]) == 4
    assert data["interval_minutes"] == 60


def test_forecast_endpoint_no_model(client_no_model):
    response = client_no_model.post("/predict/forecast?line=Central")
    assert response.status_code == 503


def test_forecast_endpoint_exceeds_horizon(client):
    response = client.post("/predict/forecast?line=Central&hours_ahead=200")
    assert response.status_code == 400


def test_predict_batch_no_model(client_no_model):
    response = client_no_model.post("/predict/batch", json={
        "predictions": [{"line": "Central", "datetime": "2027-06-01T09:00:00"}]
    })
    assert response.status_code == 503


def test_predict_with_weather_forecast(client):
    response = client.post("/predict", json={
        "line": "Central",
        "datetime": "2027-06-01T09:00:00",
        "weather_forecast": {"temperature": 20.0, "precipitation": 5.0, "humidity": 65.0},
    })
    assert response.status_code == 200


def test_predict_value_error_returns_400(client):
    import api
    from fastapi.testclient import TestClient as _TC
    mock_predictor = _make_mock_predictor()
    mock_predictor.predict_delay.side_effect = ValueError("bad input")

    async def _load_with_error(app):
        app.state.predictor = mock_predictor
        app.state.active_run_id = 'run_test'

    with patch.object(api, '_load_model', _load_with_error):
        with _TC(api.app) as c:
            response = c.post("/predict", json={
                "line": "Central",
                "datetime": "2027-06-01T09:00:00",
            })
    assert response.status_code == 400


def test_rate_limiter_allows_and_blocks():
    from api import _RateLimiter
    limiter = _RateLimiter(max_calls=3, window_seconds=60)
    assert limiter.is_allowed("test_ip") is True
    assert limiter.is_allowed("test_ip") is True
    assert limiter.is_allowed("test_ip") is True
    # 4th call within window should be blocked
    assert limiter.is_allowed("test_ip") is False


def test_rate_limiter_different_ips_are_independent():
    from api import _RateLimiter
    limiter = _RateLimiter(max_calls=1, window_seconds=60)
    assert limiter.is_allowed("ip_a") is True
    assert limiter.is_allowed("ip_b") is True
    assert limiter.is_allowed("ip_a") is False


def test_rate_limit_middleware_returns_429(client):
    import api
    with patch.object(api._limiter, 'is_allowed', return_value=False):
        response = client.get("/health")
    assert response.status_code == 429


def test_predict_general_exception_returns_500(client):
    import api
    from fastapi.testclient import TestClient as _TC
    mock_predictor = _make_mock_predictor()
    mock_predictor.predict_delay.side_effect = RuntimeError("unexpected crash")

    async def _setup(app):
        app.state.predictor = mock_predictor
        app.state.active_run_id = 'run_test'

    with patch.object(api, '_load_model', _setup):
        with _TC(api.app) as c:
            response = c.post("/predict", json={
                "line": "Central",
                "datetime": "2027-06-01T09:00:00",
            })
    assert response.status_code == 500


def test_batch_predict_partial_error(client):
    import api
    mock_predictor = _make_mock_predictor()
    call_count = [0]

    def _side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("second prediction failed")
        return _make_mock_predictor().predict_delay.return_value

    mock_predictor.predict_delay.side_effect = _side_effect

    async def _setup(app):
        app.state.predictor = mock_predictor
        app.state.active_run_id = 'run_test'

    from fastapi.testclient import TestClient as _TC
    with patch.object(api, '_load_model', _setup):
        with _TC(api.app) as c:
            response = c.post("/predict/batch", json={
                "predictions": [
                    {"line": "Central", "datetime": "2027-06-01T09:00:00"},
                    {"line": "Jubilee", "datetime": "2027-06-01T09:00:00"},
                ]
            })
    assert response.status_code == 200
    data = response.json()
    assert data["successful"] == 1
    assert data["failed"] == 1


def test_forecast_exception_returns_500(client):
    import api
    from fastapi.testclient import TestClient as _TC
    mock_predictor = _make_mock_predictor()
    mock_predictor.predict_next_24_hours.side_effect = RuntimeError("forecast failed")

    async def _setup(app):
        app.state.predictor = mock_predictor
        app.state.active_run_id = 'run_test'

    with patch.object(api, '_load_model', _setup):
        with _TC(api.app) as c:
            response = c.post("/predict/forecast?line=Central")
    assert response.status_code == 500


def test_load_model_no_run_dirs():
    """Covers _load_model() early-return when no artifacts exist."""
    import asyncio
    import api
    from fastapi import FastAPI
    app_obj = FastAPI()
    app_obj.state.predictor = None

    with patch.object(api, '_resolve_latest_run_id', return_value=None):
        asyncio.run(api._load_model(app_obj))

    assert app_obj.state.predictor is None


def test_load_model_raises_on_failed_predictor():
    """Covers _load_model() RuntimeError path when model file is missing."""
    import asyncio
    import api
    from fastapi import FastAPI
    app_obj = FastAPI()
    app_obj.state.predictor = None
    app_obj.state.active_run_id = None

    with patch.object(api, '_resolve_latest_run_id', return_value='run_test'):
        with patch('api.FutureDelayPredictor', side_effect=FileNotFoundError("no model")):
            with pytest.raises(RuntimeError, match="Model loading failed"):
                asyncio.run(api._load_model(app_obj))


def test_load_model_success():
    """Covers the happy path of _load_model()."""
    import asyncio
    import api
    from fastapi import FastAPI
    app_obj = FastAPI()
    app_obj.state.predictor = None
    app_obj.state.active_run_id = None

    mock_predictor = _make_mock_predictor()

    with patch.object(api, '_resolve_latest_run_id', return_value='run_20260101_000000'):
        with patch('api.FutureDelayPredictor', return_value=mock_predictor):
            asyncio.run(api._load_model(app_obj))

    assert app_obj.state.predictor is mock_predictor
    assert app_obj.state.active_run_id == 'run_20260101_000000'


def test_lifespan_degraded_on_runtime_error():
    """Covers the graceful degraded-mode startup in lifespan.

    When model artifacts exist but fail to load, the API should start up
    without crashing (no exception propagated) and serve 503 on /predict.
    The /health endpoint should report 'unhealthy'.

    This replaces the previous test that expected a raise, because
    Improvement #5 intentionally changed the behaviour: the app now logs
    a CRITICAL message and stays alive rather than crash-looping.
    """
    import api
    from fastapi.testclient import TestClient as _TC

    async def _fails(app):
        raise RuntimeError("hard failure")

    # The app should start without raising despite the load failure
    with patch.object(api, '_load_model', _fails):
        with _TC(api.app) as c:
            # predictor is None, so /predict must return 503
            response = c.post("/predict", json={
                "line": "Central",
                "datetime": "2027-06-01T09:00:00",
            })
            assert response.status_code == 503

            # health endpoint must report 'unhealthy'
            health = c.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "unhealthy"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
