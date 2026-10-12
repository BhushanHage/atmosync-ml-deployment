from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from forecasting.tcn_tft_inference import AtmosyncTCNTFTPredictor

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("atmosync-api")

BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = Path(
    os.getenv(
        "ATMOSYNC_MODEL_EXPORT_DIR",
        str(BASE_DIR / "models" / "TCN-TFT" / "export"),
    )
).resolve()

predictor: AtmosyncTCNTFTPredictor | None = None
startup_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, startup_error
    try:
        predictor = AtmosyncTCNTFTPredictor(export_dir=EXPORT_DIR, device="cpu")
        startup_error = None
        logger.info(
            "Model loaded: %s; lookback=%s; horizon=%s; export_dir=%s",
            predictor.model_name,
            predictor.lookback,
            predictor.horizon,
            EXPORT_DIR,
        )
    except Exception as exc:
        predictor = None
        startup_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Model failed to load")
    yield


app = FastAPI(
    title="ATMOSYNC Forecast API",
    description=(
        "Inference API for the existing exported ATMOSYNC model. "
        "Supply 30 consecutive daily rows with the configured model features."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class HistoryRow(BaseModel):
    observation_date: str
    # The model's exact feature columns are validated dynamically against its config.
    # Extra fields are allowed so callers may include source metadata.
    model_config = {"extra": "allow"}


class PredictionRequest(BaseModel):
    history: list[dict[str, Any]] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Exactly 30 consecutive daily rows, oldest first.",
    )


@app.get("/")
def root():
    return {
        "service": "ATMOSYNC Forecast API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "POST /predict",
    }


@app.get("/health")
def health():
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "model_not_ready",
                "message": startup_error or "Model has not loaded.",
            },
        )
    return {
        "status": "ok",
        "model": predictor.model_name,
        "device": str(predictor.device),
        "lookback_days": predictor.lookback,
        "forecast_horizon_days": predictor.horizon,
        "target_names": predictor.targets,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "model_not_ready",
                "message": startup_error or "Model has not loaded.",
            },
        )

    if len(request.history) != predictor.lookback:
        raise HTTPException(
            status_code=422,
            detail=f"Exactly {predictor.lookback} history rows are required.",
        )

    history = pd.DataFrame(request.history)
    required = ["observation_date", *predictor.features]
    missing = [column for column in required if column not in history.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "missing_required_columns",
                "missing_columns": missing,
                "message": (
                    "Provide every configured model feature for each of the 30 days. "
                    "Missing values must be handled upstream; this endpoint does not "
                    "invent or silently impute weather values."
                ),
            },
        )

    try:
        predictions = predictor.predict(history)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs for diagnostic details.",
        ) from exc

    # Convert pandas timestamps and NumPy scalars into JSON-compatible values.
    predictions = predictions.astype(object).where(pd.notna(predictions), None)
    records = predictions.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if hasattr(value, "isoformat"):
                record[key] = value.isoformat()
            elif hasattr(value, "item"):
                record[key] = value.item()

    return {
        "model": predictor.model_name,
        "lookback_days": predictor.lookback,
        "forecast_horizon_days": predictor.horizon,
        "units": {
            "rainfall_total": "mm",
            "snowfall_total": "cm",
            "temperature_targets": "°C",
        },
        "forecasts": records,
    }
