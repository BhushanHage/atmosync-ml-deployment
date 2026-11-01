"""
ATMOSYNC v2.0 — local LightGBM inference matching the uploaded training notebook.

Fetches 90 days of daily API observations and predicts the next 7 days using
the saved list of 28 LightGBM estimators (7 forecast leads x 4 targets).

Run from the deployment project root:
    python atmosync_local_7day_forecast.py --station toronto --end-date 2026-10-31

Requires:
    pip install pandas numpy requests joblib lightgbm
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "Light_GBM"
MODEL_PATH = MODEL_DIR / "lightgbm_models.joblib"
METADATA_PATH = MODEL_DIR / "inference_metadata.json"
DEFAULT_API_TEMPLATE = (
    "https://atmosyncv2-0.onrender.com/weather/{station}"
    "?start_date={start_date}&end_date={end_date}"
)

# API station slugs -> training station IDs from inference_metadata.json.
STATION_IDS = {
    "vancouver": "1108446",
    "edmonton": "3012209",
    "calgary": "3031094",
    "winnipeg": "502S001",
    "ottawa": "6105978",
    "toronto": "6158355",
    "montreal": "7014160",
    "quebec_city": "701S001",
    "halifax": "8202251",
}

# Training feature names -> possible historical API field names.
API_ALIASES = {
    "Max Temp (°C)": ["max_temperature", "max_temp_c", "maximum_temperature"],
    "Min Temp (°C)": ["min_temperature", "min_temp_c", "minimum_temperature"],
    "Mean Temp (°C)": ["mean_temperature", "mean_temp_c", "average_temperature"],
    "Heat Deg Days (°C)": [
        "heating_degree_days", "heat_degree_days", "heat_deg_days",
        "heating_degree_days_c", "degree_days_heat",
    ],
    "Cool Deg Days (°C)": [
        "cooling_degree_days", "cool_degree_days", "cool_deg_days",
        "cooling_degree_days_c", "degree_days_cool",
    ],
    "precip_input_mm": [
        "total_precipitation", "precipitation", "precip_mm",
        "precipitation_mm", "PRECTOTCORR",
    ],
}

def parse_records(payload):
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in ("data", "results", "records", "weather", "observations"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        if records is None:
            raise ValueError(f"Cannot find daily records in API response keys: {list(payload)}")
    else:
        raise ValueError(f"Unexpected API response type: {type(payload).__name__}")
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("API returned no records.")
    return df

def fetch_history(station, station_id, end_date, lookback, api_template):
    start_date = end_date - timedelta(days=lookback - 1)
    url = api_template.format(
        station=station,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )
    print(f"GET {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    df = parse_records(response.json())

    date_col = next((c for c in ("observation_date", "Date/Time", "date", "forecast_date")
                     if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"No date column in API response. Columns: {list(df.columns)}")
    df["Date/Time"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date/Time"])
    df = df[(df["Date/Time"] >= pd.Timestamp(start_date)) &
            (df["Date/Time"] <= pd.Timestamp(end_date))]
    df = df.sort_values("Date/Time").drop_duplicates("Date/Time", keep="last")
    if df.empty:
        raise ValueError("No valid daily observations in the requested date range.")

    # Reindex to a complete daily timeline, exactly as training did.
    df = df.set_index("Date/Time").asfreq("D")
    df["atm_station_id"] = station_id

    for feature, aliases in API_ALIASES.items():
        source = next((c for c in aliases if c in df.columns), None)
        if source is None:
            df[feature] = np.nan
            print(f"WARNING: API did not provide {feature}; training median imputation will be used.")
        else:
            df[feature] = pd.to_numeric(df[source], errors="coerce")

    df.loc[df["precip_input_mm"] < 0, "precip_input_mm"] = np.nan

    # Match training calendar feature construction.
    doy = df.index.dayofyear.to_numpy(dtype=float)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    year_fraction = doy / np.where(df.index.is_leap_year, 366.0, 365.0)
    df["year_sin"] = np.sin(2 * np.pi * year_fraction)
    df["year_cos"] = np.cos(2 * np.pi * year_fraction)

    if len(df) != lookback:
        raise ValueError(f"Expected {lookback} daily rows, got {len(df)}.")
    return df

def load_artifacts():
    model_path = Path(MODEL_PATH)
    metadata_path = Path(METADATA_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    models = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(models, (list, tuple)):
        raise TypeError(f"Expected training notebook's model list, got {type(models).__name__}")
    if len(models) != 28:
        raise ValueError(f"Expected 28 models (7 leads x 4 targets), found {len(models)}.")
    return models, metadata

def build_window_features(history, metadata, station_id):
    feature_names = metadata["features"]
    medians = np.asarray(metadata["feature_medians"], dtype=np.float32)
    x_mean = np.asarray(metadata["feature_mean"], dtype=np.float32)
    x_std = np.asarray(metadata["feature_std"], dtype=np.float32)
    station_map = {str(k): int(v) for k, v in metadata["station_to_index"].items()}

    if len(feature_names) != 10 or len(medians) != 10 or len(x_mean) != 10 or len(x_std) != 10:
        raise ValueError("Metadata does not match expected 10-feature training schema.")
    if station_id not in station_map:
        raise ValueError(
            f"Station ID {station_id!r} is absent from training station_to_index mapping."
        )

    # Shape is [90 days, 10 features], matching make_windows() in the notebook.
    x = history[feature_names].to_numpy(dtype=np.float32)
    for j in range(x.shape[1]):
        bad = ~np.isfinite(x[:, j])
        x[bad, j] = medians[j]
    x = (x - x_mean) / np.where(x_std == 0, 1.0, x_std)

    # Training flattened [lookback, features] row-major, then appended station index.
    flattened = x.reshape(1, -1)
    station_column = np.asarray([[station_map[station_id]]], dtype=np.float32)
    return np.concatenate([flattened, station_column], axis=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", default="toronto", help="Station slug, e.g. toronto")
    parser.add_argument("--end-date", default=None, help="Last historical date YYYY-MM-DD; defaults to yesterday")
    parser.add_argument("--api-url-template", default=DEFAULT_API_TEMPLATE)
    parser.add_argument("--output-dir", default=str(ROOT / "forecasting" / "local_runs"))
    args = parser.parse_args()

    station = args.station.strip().lower().replace(" ", "_")
    station_id = STATION_IDS.get(station)
    if station_id is None:
        parser.error(f"Unknown station '{args.station}'. Choose: {', '.join(STATION_IDS)}")

    end_date = date.fromisoformat(args.end_date) if args.end_date else date.today() - timedelta(days=1)

    try:
        models, metadata = load_artifacts()
        lookback = int(metadata["lookback_days"])
        horizon = int(metadata["forecast_horizon_days"])
        target_names = metadata["target_names"]
        target_mean = np.asarray(metadata["target_mean"], dtype=np.float32)
        target_std = np.asarray(metadata["target_std"], dtype=np.float32)

        if horizon != 7 or len(target_names) != 4:
            raise ValueError(f"Unexpected metadata horizon/targets: {horizon}, {target_names}")

        history = fetch_history(
            station, station_id, end_date, lookback, args.api_url_template
        )
        x = build_window_features(history, metadata, station_id)

        # Notebook training order: model index = horizon_index * 4 + target_index.
        # Each model predicts a standardized target; invert target scaling.
        predictions = np.full((horizon, len(target_names)), np.nan, dtype=np.float32)
        for lead in range(horizon):
            for target_index, target_name in enumerate(target_names):
                model_index = lead * len(target_names) + target_index
                model = models[model_index]
                if model is None:
                    continue
                if not hasattr(model, "predict"):
                    raise TypeError(f"Artifact item {model_index} is not a predictor: {type(model).__name__}")
                expected = getattr(model, "n_features_in_", None)
                if expected is not None and int(expected) != x.shape[1]:
                    raise ValueError(
                        f"Model {model_index} expects {expected} features; constructed {x.shape[1]}."
                    )
                standardized = float(np.asarray(model.predict(x)).reshape(-1)[0])
                predictions[lead, target_index] = (
                    standardized * target_std[target_index] + target_mean[target_index]
                )

        # Physical output constraints: preserve min <= mean <= max and nonnegative precipitation.
        output_rows = []
        target_index_map = {name: i for i, name in enumerate(target_names)}
        for lead in range(horizon):
            row = {
                "station": station,
                "station_id": station_id,
                "history_start": history.index.min().date().isoformat(),
                "history_end": end_date.isoformat(),
                "forecast_date": (end_date + timedelta(days=lead + 1)).isoformat(),
                "forecast_horizon_day": lead + 1,
                "selected_model": metadata.get("selected_model", "LightGBM"),
            }
            for i, name in enumerate(target_names):
                value = predictions[lead, i]
                row[f"predicted_{name}"] = float(value) if np.isfinite(value) else None

            keys = ["predicted_min_temp_c", "predicted_mean_temp_c", "predicted_max_temp_c"]
            if all(row.get(k) is not None for k in keys):
                vals = sorted(float(row[k]) for k in keys)
                row["predicted_min_temp_c"], row["predicted_mean_temp_c"], row["predicted_max_temp_c"] = vals
            if row.get("predicted_precip_mm") is not None:
                row["predicted_precip_mm"] = max(0.0, float(row["predicted_precip_mm"]))
            output_rows.append(row)

        result = pd.DataFrame(output_rows)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = date.today().isoformat()
        csv_path = out_dir / f"forecast_{station}_{stamp}.csv"
        json_path = out_dir / f"forecast_{station}_{stamp}.json"
        result.to_csv(csv_path, index=False)
        json_path.write_text(result.to_json(orient="records", indent=2), encoding="utf-8")
        print(f"\nSUCCESS: generated {len(result)} forecast days.")
        print(result.to_string(index=False))
        print(f"\nCSV:  {csv_path}")
        print(f"JSON: {json_path}")
    except Exception as exc:
        print(f"FORECAST FAILED: {exc}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
