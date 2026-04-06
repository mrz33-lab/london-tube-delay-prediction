"""
FastAPI service for delay predictions.

    uvicorn api:app --reload --port 8000

Environment variables:
    ALLOWED_ORIGINS  Comma-separated list of allowed CORS origins.
                     Defaults to "" (wildcard, no credentials).
                     Set to e.g. "http://localhost:3000,https://yourdomain.com"
                     to restrict access and enable credentials.
"""

import os
import time
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, List
from pathlib import Path
import logging

from config import DataConfig
from future_prediction import FutureDelayPredictor
from exceptions import ModelNotLoadedError

_DATA_CONFIG = DataConfig()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# simple in-memory rate limiter (no extra dependencies)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Token-bucket rate limiter keyed by client IP."""

    def __init__(self, max_calls: int, window_seconds: int):
        self._max = max_calls
        self._window = window_seconds
        self._buckets: dict = {}  # plain dict — only populated on first request per key
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            # evict timestamps older than the window; delete key when bucket empties
            calls = [t for t in self._buckets.get(key, []) if now - t < self._window]
            if len(calls) >= self._max:
                self._buckets[key] = calls
                return False
            calls.append(now)
            self._buckets[key] = calls
            return True


_limiter = _RateLimiter(max_calls=60, window_seconds=60)  # 60 req/min per IP


# ---------------------------------------------------------------------------
# CORS — read allowed origins from environment so the default (wildcard) is
# safe for demos while production can lock it down without code changes.
# ---------------------------------------------------------------------------

_cors_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
if _cors_origins_env:
    _cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    _allow_credentials = True   # safe because origins are explicit
else:
    _cors_origins = ["*"]
    _allow_credentials = False  # credentials + wildcard is forbidden by spec


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup.

    If no artifacts exist (pre-training), starts without a model — the health
    endpoint will report 'unhealthy' and /predict will return 503.

    If artifacts exist but fail to load, logs a critical error and starts in a
    degraded state rather than crashing.  This avoids crash-loop backoffs in
    container orchestrators (e.g. Kubernetes) while giving ops teams time to
    inspect the corrupted artifact.  /health will still report 'unhealthy'.
    """
    app.state.predictor = None
    app.state.active_run_id = None
    try:
        await _load_model(app)
    except RuntimeError as exc:
        # Degraded start — log loudly but don't crash the process.
        logger.critical(
            f"Model failed to load at startup — serving in degraded mode: {exc}"
        )
    yield


app = FastAPI(
    title="London Underground Delay Prediction API",
    description="ML-powered API for predicting tube delays",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if not _limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests — limit is 60/minute per IP"},
        )
    return await call_next(request)


def _resolve_latest_run_id(artifacts_dir: Path) -> Optional[str]:
    if not artifacts_dir.exists():
        return None
    run_dirs = sorted(
        [d.name for d in artifacts_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    )
    return run_dirs[-1] if run_dirs else None


# ---------------------------------------------------------------------------
# request / response models
# ---------------------------------------------------------------------------

class WeatherForecast(BaseModel):
    """Typed weather payload — prevents silent key-name mismatches."""
    temperature: float = Field(12.0, ge=-30.0, le=50.0, description="Temperature in °C")
    precipitation: float = Field(0.0, ge=0.0, le=200.0, description="Precipitation in mm")
    humidity: float = Field(70.0, ge=0.0, le=100.0, description="Relative humidity 0–100")


class PredictionRequest(BaseModel):
    line: str = Field(..., description="Tube line name")
    datetime: str = Field(..., description="Target datetime (ISO 8601)")
    weather_forecast: Optional[WeatherForecast] = Field(None, description="Optional weather data")

    @field_validator('line')
    @classmethod
    def validate_line(cls, v):
        valid_lines = _DATA_CONFIG.tube_lines
        if v not in valid_lines:
            raise ValueError(f"Invalid line. Must be one of: {', '.join(valid_lines)}")
        return v

    @field_validator('datetime')
    @classmethod
    def validate_datetime(cls, v):
        try:
            dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
            dt_naive = dt.replace(tzinfo=None)
            if dt_naive < datetime.now():
                raise ValueError("Datetime must be in the future (or present)")
            return v
        except ValueError as exc:
            raise ValueError(f"Invalid datetime format. Use ISO 8601: {exc}") from exc


class PredictionResponse(BaseModel):
    line: str
    datetime: str
    predicted_delay_minutes: float
    confidence_interval_95: tuple
    status: str
    status_emoji: str
    model_version: str


class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

async def _load_model(app: FastAPI):
    """Load model from the latest run directory.

    If no artifacts exist yet (pre-training), logs a warning and leaves
    predictor as None — the health endpoint will report 'unhealthy'.

    If artifacts exist but the model fails to load, raises RuntimeError so
    the application fails fast rather than silently serving 503 forever.
    """
    artifacts_root = Path("artifacts")
    run_id = _resolve_latest_run_id(artifacts_root)

    if run_id is None:
        logger.warning("No run_* directories under artifacts/ — starting without model")
        return  # Acceptable: training hasn't run yet

    # Artifacts exist — any failure from here is a hard error
    artifacts_dir = artifacts_root / run_id
    try:
        app.state.predictor = FutureDelayPredictor(
            model_path=str(artifacts_dir / "best_model.pkl"),
            feature_metadata_path=str(artifacts_dir / "feature_metadata.pkl")
        )
        app.state.active_run_id = run_id
        logger.info(f"Model loaded from {artifacts_dir}")
    except Exception as exc:
        logger.critical(f"Failed to load model from {artifacts_dir}: {exc}", exc_info=True)
        raise RuntimeError(f"Model loading failed: {exc}") from exc


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "London Underground Delay Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Single prediction",
            "/predict/batch": "POST - Batch predictions",
            "/predict/forecast": "POST - 24h forecast",
            "/health": "GET - Health check",
            "/docs": "GET - Swagger docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(http_request: Request):
    predictor = http_request.app.state.predictor
    active_run_id = http_request.app.state.active_run_id
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        model_version=active_run_id if predictor else None,
        timestamp=datetime.now().isoformat()
    )


def _make_prediction_response(predictor, active_run_id, body):
    """Shared logic for single and batch predictions."""
    target_dt = datetime.fromisoformat(
        body.datetime.replace('Z', '+00:00')
    ).replace(tzinfo=None)

    prediction = predictor.predict_delay(
        line=body.line,
        target_datetime=target_dt,
        # Convert Pydantic model → plain dict so future_prediction.py
        # can use .get() without caring about the API model type.
        weather_forecast=body.weather_forecast.model_dump() if body.weather_forecast else None,
    )

    emoji_map = {
        'Good Service': '🟢', 'Minor Delays': '🟡', 'Severe Delays': '🔴',
    }

    return PredictionResponse(
        line=prediction['line'],
        datetime=prediction['datetime'].isoformat(),
        predicted_delay_minutes=round(prediction['predicted_delay_minutes'], 2),
        confidence_interval_95=tuple(
            round(x, 2) for x in prediction['confidence_interval_95']
        ),
        status=prediction['status'],
        status_emoji=emoji_map.get(prediction['status'], '⚪'),
        model_version=active_run_id or "unknown",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_delay(body: PredictionRequest, http_request: Request):
    predictor = http_request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return _make_prediction_response(
            predictor, http_request.app.state.active_run_id, body
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict/batch", tags=["Predictions"])
async def predict_batch(body: BatchPredictionRequest, http_request: Request):
    predictor = http_request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(body.predictions) > 100:
        raise HTTPException(status_code=400, detail="Max 100 predictions per batch")

    active_run_id = http_request.app.state.active_run_id
    results, errors = [], []

    for idx, pred_body in enumerate(body.predictions):
        try:
            result = _make_prediction_response(predictor, active_run_id, pred_body)
            results.append(result.model_dump())
        except Exception as e:
            errors.append({"index": idx, "line": pred_body.line, "error": str(e)})

    return {
        "predictions": results, "errors": errors,
        "total_requested": len(body.predictions),
        "successful": len(results), "failed": len(errors),
    }


@app.post("/predict/forecast", tags=["Predictions"])
async def predict_forecast(http_request: Request, line: str,
                           hours_ahead: int = 24, interval_minutes: int = 60):
    predictor = http_request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if hours_ahead > 168:
        raise HTTPException(status_code=400, detail="Max forecast horizon is 168 hours (1 week)")

    try:
        forecast_df = predictor.predict_next_24_hours(
            line=line, interval_minutes=interval_minutes,
        )

        max_predictions = hours_ahead * 60 // interval_minutes
        forecast_df = forecast_df.head(max_predictions)

        forecast = [
            {
                "datetime": row['datetime'].isoformat(),
                "predicted_delay_minutes": round(row['predicted_delay'], 2),
                "status": row['status'],
            }
            for _, row in forecast_df.iterrows()
        ]

        return {
            "line": line, "forecast": forecast,
            "interval_minutes": interval_minutes,
            "model_version": http_request.app.state.active_run_id or "unknown",
        }
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/lines", tags=["Info"])
async def get_lines():
    return {"lines": _DATA_CONFIG.tube_lines}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
