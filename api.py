"""
FastAPI service for delay predictions.

    uvicorn api:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
import logging

from future_prediction import FutureDelayPredictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup."""
    app.state.predictor = None
    app.state.active_run_id = None
    await _load_model(app)
    yield


app = FastAPI(
    title="London Underground Delay Prediction API",
    description="ML-powered API for predicting tube delays",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # wide open for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_latest_run_id(artifacts_dir: Path) -> Optional[str]:
    if not artifacts_dir.exists():
        return None
    run_dirs = sorted(
        [d.name for d in artifacts_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    )
    return run_dirs[-1] if run_dirs else None


# ---- request / response models ----

class PredictionRequest(BaseModel):
    line: str = Field(..., description="Tube line name")
    datetime: str = Field(..., description="Target datetime (ISO 8601)")
    weather_forecast: Optional[Dict] = Field(None, description="Optional weather data")

    @field_validator('line')
    @classmethod
    def validate_line(cls, v):
        from config import DataConfig
        valid_lines = DataConfig().tube_lines
        if v not in valid_lines:
            raise ValueError(f"Invalid line. Must be one of: {', '.join(valid_lines)}")
        return v

    @field_validator('datetime')
    @classmethod
    def validate_datetime(cls, v):
        try:
            dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
            dt_naive = dt.replace(tzinfo=None)
            if dt_naive <= datetime.now():
                raise ValueError("Datetime must be in the future")
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


# ---- model loading ----

async def _load_model(app: FastAPI):
    try:
        artifacts_root = Path("artifacts")
        run_id = _resolve_latest_run_id(artifacts_root)

        if run_id is None:
            logger.error("No run_* directories under artifacts/ — can't load model")
            return

        artifacts_dir = artifacts_root / run_id

        app.state.predictor = FutureDelayPredictor(
            model_path=str(artifacts_dir / "best_model.pkl"),
            feature_metadata_path=str(artifacts_dir / "feature_metadata.pkl")
        )
        app.state.active_run_id = run_id

        logger.info("Model loaded from %s", artifacts_dir)

    except Exception as e:
        logger.error("Failed to load model: %s", e)
        import traceback
        traceback.print_exc()
        app.state.predictor = None


# ---- endpoints ----

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
        weather_forecast=body.weather_forecast,
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
        logger.error("Prediction error: %s", e)
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
        logger.error("Forecast error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/lines", tags=["Info"])
async def get_lines():
    from config import DataConfig
    return {"lines": DataConfig().tube_lines}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
