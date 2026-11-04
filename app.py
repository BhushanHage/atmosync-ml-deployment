
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests

from fastapi import FastAPI, HTTPException, Header, Query
from pydantic import BaseModel, Field

from forecasting.lightgbm_inference import LightGBMForecaster, STATIONS


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("atmosync-api")

# Model instances
tcn_predictor = None
tcn_startup_error = None
lightgbm_predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tcn_predictor, tcn_startup_error, lightgbm_predictor

    # Load legacy TCN model
    try:
        from forecasting.tcn_missing_robust_inference import (
            AtmosyncMissingRobustTCNPredictor,
        )
        from pathlib import Path

        export_dir = Path(
            os.getenv(
                "ATMOSYNC_MODEL_EXPORT_DIR",
                str(
                    Path(__file__).resolve().parent
                    / "models"
                    / "atmosync_tcn_missing_robust"
                    / "export"
                ),
            )
        )

        tcn_predictor = AtmosyncMissingRobustTCNPredictor(
            export_dir=export_dir,
            device=os.getenv("ATMOSYNC_DEVICE", "cpu"),
        )

        tcn_startup_error = None

    except Exception as exc:
        tcn_predictor = None
        tcn_startup_error = f"{type(exc).__name__}: {exc}"
        log.warning("Legacy TCN unavailable: %s", tcn_startup_error)

    # Load LightGBM model
    try:
        lightgbm_predictor = LightGBMForecaster()

        log.info(
            "LightGBM loaded: lookback=%s horizon=%s",
            lightgbm_predictor.lookback,
            lightgbm_predictor.horizon,
        )

    except Exception as exc:
        lightgbm_predictor = None
        log.exception("LightGBM failed to load: %s", exc)

    yield


app = FastAPI(
    title="ATMOSYNC Forecast API",
    description="ATMOSYNC LightGBM saved-forecast API with legacy TCN routes retained.",
    version="2.1.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    history: list[dict[str, Any]] = Field(
        ...,
        min_length=30,
        max_length=30,
    )


@app.get("/")
def root():
    return {
        "service": "ATMOSYNC Forecast API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "legacy_prediction_endpoint": "POST /predict",
        "forecasts_endpoint": "GET /forecasts",
        "refresh_endpoint": "POST /forecasts/refresh",
    }


@app.get("/health")
def health():
    return {
        "status": (
            "ok"
            if (
                lightgbm_predictor is not None
                or tcn_predictor is not None
            )
            else "degraded"
        ),
        "models": {
            "lightgbm": {
                "ready": lightgbm_predictor is not None,
                "lookback_days": 90,
                "forecast_horizon_days": 7,
            },
            "legacy_tcn": {
                "ready": tcn_predictor is not None,
                "error": tcn_startup_error,
            },
        },
        "forecast_storage_configured": bool(
            os.getenv("SUPABASE_URL")
            and (
                os.getenv("SUPABASE_SERVICE_ROLE_KEY")
                or os.getenv("SUPABASE_KEY")
            )
        ),
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    if tcn_predictor is None:
        raise HTTPException(
            503,
            detail={
                "status": "legacy_model_not_ready",
                "message": tcn_startup_error or "TCN export not loaded",
            },
        )

    if len(request.history) != tcn_predictor.lookback:
        raise HTTPException(
            422,
            detail=f"Exactly {tcn_predictor.lookback} history rows are required.",
        )

    history = pd.DataFrame(request.history)

    if "observation_date" not in history.columns:
        raise HTTPException(
            422,
            detail="The observation_date column is required.",
        )

    try:
        forecasts = tcn_predictor.predict(history)

    except (ValueError, TypeError) as exc:
        raise HTTPException(422, detail=str(exc)) from exc

    except Exception as exc:
        log.exception("TCN prediction failed")
        raise HTTPException(
            500,
            detail="Prediction failed",
        ) from exc

    forecasts = (
        forecasts.replace([np.inf, -np.inf], np.nan)
        .astype(object)
        .where(pd.notna(forecasts), None)
    )

    return {
        "model": tcn_predictor.model_name,
        "lookback_days": tcn_predictor.lookback,
        "forecast_horizon_days": tcn_predictor.horizon,
        "forecasts": forecasts.to_dict(orient="records"),
    }


# --------------------------------------------------
# SUPABASE CONFIGURATION
# --------------------------------------------------

def supabase_config():

    base = os.getenv("SUPABASE_URL", "").rstrip("/")

    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
    )

    table = os.getenv(
        "SUPABASE_FORECASTS_TABLE",
        "weather_forecasts",
    )

    if not base or not key:
        raise HTTPException(
            503,
            detail=(
                "Forecast storage is not configured. "
                "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
                "(or SUPABASE_KEY)."
            ),
        )

    return f"{base}/rest/v1/{table}", key


def sb_headers(key, prefer=None):

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    if prefer:
        headers["Prefer"] = prefer

    return headers


# --------------------------------------------------
# READ SAVED FORECASTS
# --------------------------------------------------

@app.get("/forecasts")
def get_forecasts(
    station: str | None = Query(default=None),
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = Query(default=500, ge=1, le=2000),
):

    url, key = supabase_config()

    params = {
        "select": "*",
        "order": "forecast_date.asc",
        "limit": str(limit),
    }

    if station:
        slug = station.strip().lower().replace(" ", "_")

        valid_slugs = {
            station_name.replace(" ", "_")
            for station_name in STATIONS
        }

        if slug not in valid_slugs:
            raise HTTPException(
                422,
                detail=f"Unknown station. Choose: {', '.join(sorted(valid_slugs))}",
            )

        params["station"] = f"eq.{slug}"

    if start_date and end_date:
        params["and"] = (
            f"(forecast_date.gte.{start_date.isoformat()},"
            f"forecast_date.lte.{end_date.isoformat()})"
        )

    elif start_date:
        params["forecast_date"] = f"gte.{start_date.isoformat()}"

    elif end_date:
        params["forecast_date"] = f"lte.{end_date.isoformat()}"

    try:
        r = requests.get(
            url,
            headers=sb_headers(key),
            params=params,
            timeout=30,
        )

        r.raise_for_status()
        rows = r.json()

    except requests.RequestException as exc:
        log.exception("Supabase forecast read failed")

        raise HTTPException(
            502,
            detail="Could not read saved forecasts",
        ) from exc

    return {
        "count": len(rows),
        "forecasts": rows,
    }


# --------------------------------------------------
# GENERATE AND SAVE LIGHTGBM FORECASTS
# --------------------------------------------------

@app.post("/forecasts/refresh")
def refresh_forecasts(
    end_date: date | None = None,
    x_atmosync_refresh_token: str | None = Header(default=None),
):

    # Validate refresh token
    expected = os.getenv("ATMOSYNC_REFRESH_TOKEN")

    if not expected:
        raise HTTPException(
            503,
            detail=(
                "Forecast refresh is disabled until "
                "ATMOSYNC_REFRESH_TOKEN is configured."
            ),
        )

    if x_atmosync_refresh_token != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid refresh token",
        )

    # Validate model
    if lightgbm_predictor is None:
        raise HTTPException(
            503,
            detail=(
                "LightGBM model is not ready; "
                "check model artifact paths and server logs."
            ),
        )

    url, key = supabase_config()

    # Use yesterday as the historical data cutoff.
    as_of = end_date or (date.today() - timedelta(days=1))

    generated = []
    errors = []

    # Generate predictions for all stations.
    for station in STATIONS:

        try:
            station_forecasts = lightgbm_predictor.predict(
                station,
                as_of,
            )

            generated.extend(station_forecasts)

        except Exception as exc:
            log.exception(
                "Forecast generation failed for %s",
                station,
            )

            errors.append({
                "station": station,
                "error": str(exc),
            })

    if not generated:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "No forecasts generated",
                "errors": errors,
            },
        )

    # --------------------------------------------------
    # MAP MODEL OUTPUTS TO SUPABASE SCHEMA
    # --------------------------------------------------

    generated_at = pd.Timestamp.now(tz="UTC").isoformat()

    payload = []

    for row in generated:

        station = (
            row["station"]
            .strip()
            .replace(" ", "_")
        )

        station_key = station.replace("_", " ")

        # Prefer the station ID returned by inference.
        # Fall back to the official station mapping.
        station_id = (
            row.get("station_id")
            or STATIONS.get(station_key)
        )

        if not station_id:
            raise HTTPException(
                status_code=500,
                detail=f"Missing station_id for station: {station}",
            )

        payload.append({
            "station": station,
            "station_id": station_id,
            "forecast_date": row["forecast_date"],
            "generated_at": generated_at,
            "model_name": "LightGBM",
            "model_version": "lightgbm-v1",
            "max_temp_c": row.get("predicted_max_temp_c"),
            "min_temp_c": row.get("predicted_min_temp_c"),
            "mean_temp_c": row.get("predicted_mean_temp_c"),
            "precip_mm": row.get("predicted_precip_mm"),
            "updated_at": generated_at,
        })

    # --------------------------------------------------
    # SAVE PREDICTIONS TO SUPABASE
    # --------------------------------------------------

    try:

        r = requests.post(
            url,
            headers=sb_headers(
                key,
                "resolution=merge-duplicates,return=representation",
            ),
            params={
                "on_conflict": "station,forecast_date",
            },
            json=payload,
            timeout=90,
        )

        r.raise_for_status()
        saved = r.json()

    except requests.RequestException as exc:

        log.exception(
            "Supabase forecast write failed: %s",
            exc,
        )

        if (
            isinstance(exc, requests.HTTPError)
            and exc.response is not None
        ):
            log.error(
                "Supabase HTTP status: %s | Response: %s",
                exc.response.status_code,
                exc.response.text,
            )

        raise HTTPException(
            status_code=502,
            detail=(
                "Supabase forecast upsert failed. "
                "Check Render logs."
            ),
        ) from exc

    # --------------------------------------------------
    # RESPONSE
    # --------------------------------------------------

    return {
        "generated_count": len(generated),
        "saved_count": len(saved) if isinstance(saved, list) else len(generated),
        "history_end_date": as_of.isoformat(),
        "forecast_start_date": (
            as_of + timedelta(days=1)
        ).isoformat(),
        "forecast_end_date": (
            as_of + timedelta(days=7)
        ).isoformat(),
        "errors": errors,
        "forecasts": saved if isinstance(saved, list) else payload,
    }
