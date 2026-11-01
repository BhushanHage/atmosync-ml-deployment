
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

EXPORT_DIR = ROOT / "models" / "atmosync_tcn_missing_robust" / "export"
HISTORY_JSON = ROOT / "tests_" / "atmosync_predict.json"
OUTPUT_DIR = ROOT / "tests_"

API_URL = "https://atmosyncv2-0.onrender.com/weather/toronto"

HISTORY_START = "2026-09-13"
HISTORY_END = "2026-10-12"

FORECAST_START = "2026-10-13"
FORECAST_END = "2026-10-19"

FEATURES = [
    "air_temperature_mean",
    "air_temperature_max",
    "air_temperature_min",
    "dew_point_temperature",
    "relative_humidity",
    "wind_speed",
    "surface_pressure",
    "solar_radiation",
    "month",
    "day_of_year",
    "seasonal_cycle_sin",
    "seasonal_cycle_cos",
    "temperature_imputed_flag",
]

# Actual API observation field for each temperature target.
ACTUAL_FIELDS = {
    "air_temperature_min": "min_temperature",
    "air_temperature_mean": "mean_temperature",
    "air_temperature_max": "max_temperature",
}

# ---------------------------------------------------------
# IMPORT THE EXISTING TRAINED MODEL INFERENCE CLASS
# ---------------------------------------------------------

sys.path.insert(0, str(ROOT))

from forecasting.tcn_missing_robust_inference import (
    AtmosyncMissingRobustTCNPredictor,
)


def fetch_records(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch records from the ATMOSYNC weather endpoint."""
    response = requests.get(
        API_URL,
        params={
            "start_date": start_date,
            "end_date": end_date,
        },
        timeout=90,
    )
    response.raise_for_status()

    records = response.json()
    if not isinstance(records, list):
        raise ValueError(
            f"Expected a list from the API; got {type(records).__name__}"
        )

    if not records:
        raise ValueError(
            f"No API records returned for {start_date} to {end_date}"
        )

    df = pd.DataFrame(records)

    if "observation_date" not in df.columns:
        raise ValueError("API response has no observation_date field.")

    df["observation_date"] = pd.to_datetime(
        df["observation_date"], errors="coerce"
    )

    df = df.dropna(subset=["observation_date"])
    df = df.sort_values("observation_date").reset_index(drop=True)

    df["observation_date"] = (
        df["observation_date"].dt.normalize()
    )

    if df["observation_date"].duplicated().any():
        raise ValueError("The API returned duplicate observation dates.")

    return df


def get_saved_seasonal_features() -> dict:
    """
    Reuse the existing history JSON's calendar features when available.
    This avoids silently changing the seasonal formula used by that file.
    """
    if not HISTORY_JSON.exists():
        return {}

    try:
        payload = json.loads(
            HISTORY_JSON.read_text(encoding="utf-8")
        )
        records = payload.get("history", [])
        lookup = {}

        for record in records:
            date_value = record.get("observation_date")
            if not date_value:
                continue

            if (
                "seasonal_cycle_sin" in record
                and "seasonal_cycle_cos" in record
            ):
                lookup[pd.Timestamp(date_value).strftime("%Y-%m-%d")] = (
                    record["seasonal_cycle_sin"],
                    record["seasonal_cycle_cos"],
                )

        return lookup

    except (OSError, ValueError, TypeError) as exc:
        print("Warning: Could not reuse saved seasonal features:", exc)
        return {}


def map_api_history(api: pd.DataFrame) -> pd.DataFrame:
    """Map API observations into the model's expected input schema."""
    frame = pd.DataFrame()
    frame["observation_date"] = api["observation_date"]

    # Direct temperature mappings.
    for target, source in [
        ("air_temperature_mean", "mean_temperature"),
        ("air_temperature_max", "max_temperature"),
        ("air_temperature_min", "min_temperature"),
    ]:
        frame[target] = pd.to_numeric(
            api[source] if source in api.columns else np.nan,
            errors="coerce",
        )

    # These are not available as equivalent measurements from this API.
    # Do not substitute minimum RH for mean RH or gust speed for mean wind.
    for column in [
        "dew_point_temperature",
        "relative_humidity",
        "wind_speed",
        "surface_pressure",
        "solar_radiation",
    ]:
        frame[column] = np.nan

    dates = pd.to_datetime(frame["observation_date"])
    frame["month"] = dates.dt.month
    frame["day_of_year"] = dates.dt.dayofyear

    # Use seasonal features from the existing history file where possible.
    # For dates without a saved value, use the documented fallback below.
    # Verify this fallback against the original Kaggle training pipeline.
    seasonal_lookup = get_saved_seasonal_features()

    sin_values = []
    cos_values = []

    for timestamp in dates:
        key = timestamp.strftime("%Y-%m-%d")

        if key in seasonal_lookup:
            sin_value, cos_value = seasonal_lookup[key]
        else:
            angle = 2 * np.pi * timestamp.dayofyear / 365.0
            sin_value = np.sin(angle)
            cos_value = np.cos(angle)

        sin_values.append(sin_value)
        cos_values.append(cos_value)

    frame["seasonal_cycle_sin"] = sin_values
    frame["seasonal_cycle_cos"] = cos_values

    temperature_columns = [
        "air_temperature_mean",
        "air_temperature_max",
        "air_temperature_min",
    ]

    # Flag records where any temperature measurement was absent
    # BEFORE the model's imputer is applied.
    frame["temperature_imputed_flag"] = (
        frame[temperature_columns].isna().any(axis=1).astype(float)
    )

    # Keep the exact feature order from model_config.json.
    config_path = EXPORT_DIR / "model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_features = config["features"]

    for feature in model_features:
        if feature not in frame.columns:
            frame[feature] = np.nan

    return frame[["observation_date"] + model_features]


def calculate_metrics(
    comparison: pd.DataFrame,
    predicted_column: str,
    actual_column: str,
    target: str,
) -> dict | None:
    """Calculate metrics only on paired, valid predictions and observations."""
    predicted = pd.to_numeric(
        comparison[predicted_column], errors="coerce"
    )
    actual = pd.to_numeric(
        comparison[actual_column], errors="coerce"
    )

    valid = predicted.notna() & actual.notna()

    if not valid.any():
        print(f"{target}: no matching valid actual observations.")
        return None

    errors = predicted[valid] - actual[valid]

    return {
        "target": target,
        "valid_days": int(valid.sum()),
        "MAE": float(errors.abs().mean()),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "mean_error_predicted_minus_actual": float(errors.mean()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ATMOSYNC MISSING-AWARE TCN: API FORECAST TEST")
    print("=" * 65)

    print("\n1. Loading exported model...")

    predictor = AtmosyncMissingRobustTCNPredictor(
        export_dir=EXPORT_DIR,
        device="cpu",
        max_missing_fraction=0.60,
    )

    print("Model loaded:", predictor.model_name)
    print("Lookback days:", predictor.lookback)
    print("Forecast horizon:", predictor.horizon)
    print("Model features:", predictor.features)
    print("Model targets:", predictor.targets)

    if predictor.lookback != 30:
        raise ValueError(
            f"This test expects 30 history days; model expects "
            f"{predictor.lookback}."
        )

    if predictor.horizon != 7:
        raise ValueError(
            f"This test expects a 7-day forecast; model expects "
            f"{predictor.horizon}."
        )

    print("\n2. Fetching historical API data...")

    api_history = fetch_records(HISTORY_START, HISTORY_END)

    expected_dates = pd.date_range(
        HISTORY_START, HISTORY_END, freq="D"
    )
    actual_dates = pd.DatetimeIndex(api_history["observation_date"])

    if len(api_history) != predictor.lookback:
        raise ValueError(
            f"API returned {len(api_history)} rows; "
            f"model requires exactly {predictor.lookback}."
        )

    if not actual_dates.equals(expected_dates):
        missing = expected_dates.difference(actual_dates)
        extra = actual_dates.difference(expected_dates)
        raise ValueError(
            "Historical dates do not match the expected consecutive "
            f"30-day period. Missing dates: {list(missing)}; "
            f"unexpected dates: {list(extra)}. "
            "No dates will be synthesized."
        )

    print("Historical API rows:", len(api_history))
    print(
        "History range:",
        api_history["observation_date"].min().date(),
        "to",
        api_history["observation_date"].max().date(),
    )

    history = map_api_history(api_history)

    # Save the actual inputs that will be passed to the predictor.
    input_file = OUTPUT_DIR / "api_model_input_used.csv"
    history.to_csv(input_file, index=False)
    print("Mapped model input saved:", input_file)

    print("\n3. Running the trained model...")

    forecast = predictor.predict(history.copy())

    forecast["forecast_date"] = pd.to_datetime(
        forecast["forecast_date"]
    ).dt.strftime("%Y-%m-%d")

    forecast_file = OUTPUT_DIR / "api_model_forecast_2026-10-13_to_19.csv"
    forecast.to_csv(forecast_file, index=False)

    print("\nFORECAST GENERATED")
    print(forecast.to_string(index=False))
    print("\nForecast saved:", forecast_file)

    if len(forecast) != 7:
        raise RuntimeError(
            f"Expected 7 forecast rows, received {len(forecast)}."
        )

    expected_forecast_dates = pd.date_range(
        FORECAST_START, FORECAST_END, freq="D"
    ).strftime("%Y-%m-%d").tolist()

    if forecast["forecast_date"].tolist() != expected_forecast_dates:
        raise ValueError(
            "The model's forecast dates do not match the requested "
            f"period: {forecast['forecast_date'].tolist()}"
        )

    print("\n4. Fetching actual observations for comparison...")

    actuals = fetch_records(FORECAST_START, FORECAST_END)

    actual_fields = [
        "observation_date",
        "min_temperature",
        "mean_temperature",
        "max_temperature",
        "total_precipitation",
        "total_rain",
        "total_snow",
    ]

    for field in actual_fields:
        if field not in actuals.columns:
            actuals[field] = np.nan

    actuals = actuals[actual_fields].copy()
    actuals["observation_date"] = (
        pd.to_datetime(actuals["observation_date"])
        .dt.strftime("%Y-%m-%d")
    )

    actuals = actuals.rename(
        columns={"observation_date": "forecast_date"}
    )

    comparison = forecast.merge(
        actuals,
        on="forecast_date",
        how="left",
        validate="one_to_one",
    )

    print("\n5. Predicted vs actual temperatures...")

    metric_rows = []

    for target, actual_field in ACTUAL_FIELDS.items():
        if target not in predictor.targets:
            print(f"Target not present in model: {target}")
            continue

        predicted_field = target

        if predicted_field not in comparison.columns:
            print(f"Prediction column missing: {predicted_field}")
            continue

        actual_column = f"actual_{target}"
        comparison[actual_column] = pd.to_numeric(
            comparison[actual_field], errors="coerce"
        )

        comparison[f"error_{target}"] = (
            pd.to_numeric(comparison[predicted_field], errors="coerce")
            - comparison[actual_column]
        )

        comparison[f"absolute_error_{target}"] = (
            comparison[f"error_{target}"].abs()
        )

        result = calculate_metrics(
            comparison,
            predicted_field,
            actual_column,
            target,
        )

        if result is not None:
            metric_rows.append(result)

    display_columns = ["forecast_date"]

    for target in ACTUAL_FIELDS:
        for column in [
            target,
            f"actual_{target}",
            f"error_{target}",
        ]:
            if column in comparison.columns:
                display_columns.append(column)

    for column in [
        "total_precipitation",
        "total_rain",
        "total_snow",
        "input_missing_fraction",
    ]:
        if column in comparison.columns:
            display_columns.append(column)

    print("\nDATE-BY-DATE COMPARISON")
    print(
        comparison[
            list(dict.fromkeys(display_columns))
        ].to_string(index=False)
    )

    if metric_rows:
        metrics = pd.DataFrame(metric_rows)
        metrics_file = OUTPUT_DIR / "api_forecast_temperature_metrics.csv"
        metrics.to_csv(metrics_file, index=False)

        print("\nTEMPERATURE ACCURACY METRICS")
        print(metrics.to_string(index=False))
        print("Metrics saved:", metrics_file)
    else:
        print(
            "\nNo temperature accuracy metrics could be calculated. "
            "Check whether the API returned valid observations."
        )

    comparison_file = OUTPUT_DIR / "api_forecast_actual_comparison.csv"
    comparison.to_csv(comparison_file, index=False)

    print("\nPRECIPITATION VALIDATION")
    print(
        "API total_precipitation is not automatically equivalent to "
        "the model's rainfall_total target. It cannot establish "
        "separate rain and snow amounts."
    )
    print(
        "Valid total_rain observations:",
        pd.to_numeric(actuals["total_rain"], errors="coerce").notna().sum(),
        "/",
        len(actuals),
    )
    print(
        "Valid total_snow observations:",
        pd.to_numeric(actuals["total_snow"], errors="coerce").notna().sum(),
        "/",
        len(actuals),
    )

    print("\nCOMPARISON SAVED:", comparison_file)
    print("\nTest finished.")


if __name__ == "__main__":
    main()
