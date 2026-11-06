
from __future__ import annotations

import logging
import os
import time
import math
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Any
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests

from fastapi import FastAPI, HTTPException, Header, Query
from pydantic import BaseModel, Field

from forecasting.lightgbm_inference import (
    LightGBMForecaster,
    STATIONS,
)


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
    description="ATMOSYNC forecast API with read-only observation access and field-level forecast fallback.",
    version="2.2.0",
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
        "weather_endpoint": "GET /weather/{city}",
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
        "observation_storage_configured": bool(
            os.getenv("OBSERVATION_DATABASE_URL")
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

    # Default window: today + next 6 days (7 days total).
    # Use the same timezone as the forecast generation pipeline.
    if start_date is None and end_date is None:
        today = pd.Timestamp.now(
            tz=ZoneInfo("Asia/Kolkata")
        ).date()

        start_date = today
        end_date = today + timedelta(days=6)

    # Validate the requested date range.
    if (
        start_date is not None
        and end_date is not None
        and start_date > end_date
    ):
        raise HTTPException(
            status_code=400,
            detail="start_date must be on or before end_date.",
        )

    params = {
        "select": "*",
        "order": "forecast_date.asc",
        "limit": str(limit),
    }

    # Validate and filter station.
    if station:
        slug = station.strip().lower().replace(" ", "_")

        valid_slugs = {
            station_name.replace(" ", "_")
            for station_name in STATIONS
        }

        if slug not in valid_slugs:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unknown station. Choose: "
                    f"{', '.join(sorted(valid_slugs))}"
                ),
            )

        params["station"] = f"eq.{slug}"

    # Apply inclusive date filters.
    if start_date is not None and end_date is not None:
        params["and"] = (
            f"(forecast_date.gte.{start_date.isoformat()},"
            f"forecast_date.lte.{end_date.isoformat()})"
        )

    elif start_date is not None:
        params["forecast_date"] = (
            f"gte.{start_date.isoformat()}"
        )

    elif end_date is not None:
        params["forecast_date"] = (
            f"lte.{end_date.isoformat()}"
        )

    try:
        r = requests.get(
            url,
            headers=sb_headers(key),
            params=params,
            timeout=30,
        )

        r.raise_for_status()
        rows = r.json()

        if not isinstance(rows, list):
            raise ValueError(
                "Forecast database returned an invalid response."
            )

    except requests.RequestException as exc:
        log.exception("Supabase forecast read failed")

        raise HTTPException(
            status_code=502,
            detail="Could not read saved forecasts",
        ) from exc

    except ValueError as exc:
        log.exception("Invalid forecast response")

        raise HTTPException(
            status_code=502,
            detail="Invalid response from forecast storage",
        ) from exc

    return {
        "count": len(rows),
        "forecasts": rows,
    }




# --------------------------------------------------
# WEATHER OBSERVATIONS WITH SAVED-FORECAST FALLBACK
# --------------------------------------------------

OBSERVATION_COLUMNS = (
    "mean_temperature", "mean_temperature_flag",
    "min_temperature", "min_temperature_flag",
    "max_temperature", "max_temperature_flag",
    "total_precipitation", "total_precipitation_flag",
    "total_rain", "total_rain_flag",
    "total_snow", "total_snow_flag",
    "snow_on_ground", "snow_on_ground_flag",
    "direction_max_gust", "direction_max_gust_flag",
    "speed_max_gust", "speed_max_gust_flag",
    "cooling_degree_days", "cooling_degree_days_flag",
    "heating_degree_days", "heating_degree_days_flag",
    "min_relative_humidity", "min_relative_humidity_flag",
)

FORECAST_FALLBACK_FIELDS = {
    "min_temperature": "min_temp_c",
    "mean_temperature": "mean_temp_c",
    "max_temperature": "max_temp_c",
    "total_precipitation": "precip_mm",
}

def normalize_station(city: str) -> tuple[str, str]:
    """Return canonical inference key and public climate station ID."""
    key = city.strip().lower().replace("_", " ")
    if key not in STATIONS:
        valid = ", ".join(sorted(STATIONS))
        raise HTTPException(404, detail=f"Unknown station. Choose: {valid}")
    return key, STATIONS[key]


def read_observations_read_only(station_id: str, start_date: date, end_date: date):
    database_url = os.getenv("OBSERVATION_DATABASE_URL")
    if not database_url:
        raise HTTPException(
            503,
            detail="Observation database is not configured. Set OBSERVATION_DATABASE_URL.",
        )

    columns = ", ".join(("climate_identifier", "observation_date", *OBSERVATION_COLUMNS))
    query = f"""
        SELECT {columns}
        FROM weather_observations
        WHERE climate_identifier = %s
          AND observation_date BETWEEN %s AND %s
        ORDER BY observation_date
    """
    try:
        import psycopg
        from psycopg.rows import dict_row

        # Enforce read-only at the PostgreSQL session level as a second safety layer.
        with psycopg.connect(
            database_url,
            options="-c default_transaction_read_only=on",
            row_factory=dict_row,
            connect_timeout=15,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SET TRANSACTION READ ONLY")
                cursor.execute(query, (station_id, start_date, end_date))
                return cursor.fetchall()
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Read-only observation query failed")
        raise HTTPException(
            502,
            detail="Could not read actual weather observations.",
        ) from exc


def read_saved_forecasts(station_id: str, start_date: date, end_date: date):
    url, key = supabase_config()
    try:
        response = requests.get(
            url,
            headers=sb_headers(key),
            params={
                "select": "station,station_id,forecast_date,min_temp_c,mean_temp_c,max_temp_c,precip_mm,model_name,model_version,generated_at",
                "station_id": f"eq.{station_id}",
                "and": f"(forecast_date.gte.{start_date.isoformat()},forecast_date.lte.{end_date.isoformat()})",
                "order": "forecast_date.asc",
                "limit": "1000",
            },
            timeout=30,
        )
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            raise ValueError("Forecast store returned a non-list response")
        return rows
    except Exception as exc:
        log.exception("Saved forecast fallback read failed")
        raise HTTPException(
            502,
            detail="Could not read saved forecast fallback data.",
        ) from exc


@app.get("/weather/{city}")
def get_weather_with_forecast_fallback(
    city: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
):
    """Compatible weather route; actual observations take priority field by field."""
    if start_date > end_date:
        raise HTTPException(400, detail="start_date must be on or before end_date")

    _, station_id = normalize_station(city)
    actual_rows = read_observations_read_only(station_id, start_date, end_date)
    forecast_rows = read_saved_forecasts(station_id, start_date, end_date)

    actual_by_date = {
        row["observation_date"]: dict(row)
        for row in actual_rows
    }
    forecast_by_date = {
        date.fromisoformat(row["forecast_date"]): row
        for row in forecast_rows
        if row.get("forecast_date")
    }

    merged = []
    for day in sorted(set(actual_by_date) | set(forecast_by_date)):
        actual = actual_by_date.get(day)
        forecast = forecast_by_date.get(day)
        if actual is None and forecast is None:
            continue

        record = actual or {
            "climate_identifier": station_id,
            "observation_date": day,
            **{column: None for column in OBSERVATION_COLUMNS},
        }
        field_sources = {}
        for observation_field, forecast_field in FORECAST_FALLBACK_FIELDS.items():
            actual_value = record.get(observation_field)
            forecast_value = forecast.get(forecast_field) if forecast else None
            if actual_value is not None:
                field_sources[observation_field] = "actual"
            elif forecast_value is not None:
                record[observation_field] = forecast_value
                field_sources[observation_field] = "forecast"
            else:
                field_sources[observation_field] = None

        record["observation_date"] = day.isoformat()
        record["field_sources"] = field_sources
        merged.append(record)

    return merged


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
    india_today = pd.Timestamp.now(
    tz=ZoneInfo("Asia/Kolkata")
    ).date()
    as_of = end_date or (india_today - timedelta(days=1))

    generated = []
    errors = []
    expected_dates = {
        (as_of + timedelta(days=lead)).isoformat()
        for lead in range(1, 8)
    }

    # Three total attempts per station, with bounded exponential backoff.
    for station, station_id in STATIONS.items():
        station_error = None
        for attempt in range(1, 4):
            try:
                station_forecasts = lightgbm_predictor.predict(station, as_of)
                if not isinstance(station_forecasts, list) or len(station_forecasts) != 7:
                    raise ValueError("Expected exactly seven forecast rows")

                dates = [str(row.get("forecast_date")) for row in station_forecasts]
                if len(set(dates)) != 7 or set(dates) != expected_dates:
                    raise ValueError("Forecast dates must be the next seven unique calendar days")

                for row in station_forecasts:
                    if row.get("station") != station:
                        raise ValueError("Forecast row station does not match requested station")
                    if str(row.get("station_id")) != station_id:
                        raise ValueError("Forecast row station_id does not match station mapping")
                    for field in (
                        "predicted_min_temp_c",
                        "predicted_mean_temp_c",
                        "predicted_max_temp_c",
                        "predicted_precip_mm",
                    ):
                        value = row.get(field)
                        if value is None or not math.isfinite(float(value)):
                            raise ValueError(f"Invalid or non-finite prediction: {field}")

                generated.extend(station_forecasts)
                station_error = None
                break
            except Exception as exc:
                station_error = exc
                log.exception(
                    "Forecast generation attempt %s/3 failed for %s",
                    attempt,
                    station,
                )
                if attempt < 3:
                    time.sleep(2 ** (attempt - 1))

        if station_error is not None:
            errors.append({
                "station": station,
                "station_id": station_id,
                "attempts": 3,
                "error": str(station_error),
            })

    if not generated:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "No station forecasts generated after three attempts each",
                "errors": errors,
            },
        )

    # --------------------------------------------------
    # MAP MODEL OUTPUTS TO SUPABASE SCHEMA
    # --------------------------------------------------

    generated_at = pd.Timestamp.now(tz="Asia/Kolkata").isoformat()

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
         
        # Intentionally omit created_at: Supabase sets it on first insert;
        # upserts must preserve the original creation timestamp.
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

    saved_count = len(saved) if isinstance(saved, list) else 0
    expected_count = len(STATIONS) * 7
    complete = (
        not errors
        and len(generated) == expected_count
        and len(payload) == expected_count
        and saved_count == expected_count
    )

    result = {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "expected_count": expected_count,
        "generated_count": len(generated),
        "saved_count": saved_count,
        "history_end_date": as_of.isoformat(),
        "forecast_start_date": (as_of + timedelta(days=1)).isoformat(),
        "forecast_end_date": (as_of + timedelta(days=7)).isoformat(),
        "errors": errors,
        "forecasts": saved if isinstance(saved, list) else [],
    }

    if not complete:
        raise HTTPException(
            status_code=503,
            detail={
                **result,
                "message": (
                    "Forecast refresh incomplete. Successful station forecasts "
                    "were saved; failed stations will be retried on the next run."
                ),
            },
        )

    return result
