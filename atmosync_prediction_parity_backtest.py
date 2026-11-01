"""
ATMOSYNC v2.0 — local prediction-parity / historical backtest test.

This script checks the downloaded local LightGBM artifact against known
historical actuals over one or more seven-day forecast windows. It uses the
same 90-day lookback and feature-building logic as atmosync_local_7day_forecast.py.

IMPORTANT:
- This is a parity/backtest utility, not a guarantee of forecast quality.
- It assumes the model list order is lead-major then target-major:
  model_index = lead * 4 + target_index.
- For strict notebook parity, the notebook must use the same input preprocessing,
  station mapping, and target scaling. Any differences are reported as limitations.
- Actual values are fetched from the historical API after each forecast window.
"""

from __future__ import annotations

import argparse
import json
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
API_TEMPLATE = (
    "https://atmosyncv2-0.onrender.com/weather/{station}"
    "?start_date={start_date}&end_date={end_date}"
)

STATION_IDS = {
    "vancouver": "1108446", "edmonton": "3012209", "calgary": "3031094",
    "winnipeg": "502S001", "ottawa": "6105978", "toronto": "6158355",
    "montreal": "7014160", "quebec_city": "701S001", "halifax": "8202251",
}
API_ALIASES = {
    "Max Temp (°C)": ["max_temperature", "max_temp_c", "maximum_temperature"],
    "Min Temp (°C)": ["min_temperature", "min_temp_c", "minimum_temperature"],
    "Mean Temp (°C)": ["mean_temperature", "mean_temp_c", "average_temperature"],
    "Heat Deg Days (°C)": ["heating_degree_days", "heat_degree_days", "heat_deg_days", "heating_degree_days_c"],
    "Cool Deg Days (°C)": ["cooling_degree_days", "cool_degree_days", "cool_deg_days", "cooling_degree_days_c"],
    "precip_input_mm": ["total_precipitation", "precipitation", "precip_mm", "precipitation_mm", "PRECTOTCORR"],
}
TARGET_API_ALIASES = {
    "max_temp_c": ["max_temperature", "max_temp_c", "maximum_temperature"],
    "min_temp_c": ["min_temperature", "min_temp_c", "minimum_temperature"],
    "mean_temp_c": ["mean_temperature", "mean_temp_c", "average_temperature"],
    "precip_mm": ["total_precipitation", "precipitation", "precip_mm", "precipitation_mm"],
}

def get_records(payload):
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = next((payload[k] for k in ("data", "results", "records", "weather", "observations")
                        if isinstance(payload.get(k), list)), None)
        if records is None:
            raise ValueError(f"No records list in API response. Keys={list(payload)}")
    else:
        raise ValueError(f"Unexpected API response: {type(payload).__name__}")
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("API returned no records")
    return frame

def get_api_frame(station, start, end):
    url = API_TEMPLATE.format(station=station, start_date=start.isoformat(), end_date=end.isoformat())
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    frame = get_records(response.json())
    date_col = next((c for c in ("observation_date", "Date/Time", "date", "forecast_date") if c in frame.columns), None)
    if not date_col:
        raise ValueError(f"No date column; API columns={list(frame.columns)}")
    frame["_date"] = pd.to_datetime(frame[date_col], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["_date"])
    frame = frame[(frame["_date"] >= pd.Timestamp(start)) & (frame["_date"] <= pd.Timestamp(end))]
    frame = frame.sort_values("_date").drop_duplicates("_date", keep="last")
    return frame.set_index("_date").asfreq("D")

def prepare_history(station, station_id, end, lookback, metadata):
    start = end - timedelta(days=lookback - 1)
    raw = get_api_frame(station, start, end)
    if raw.empty:
        raise ValueError(f"No historical data from {start} to {end}")
    raw["atm_station_id"] = station_id
    for feature, aliases in API_ALIASES.items():
        source = next((c for c in aliases if c in raw.columns), None)
        raw[feature] = pd.to_numeric(raw[source], errors="coerce") if source else np.nan
    raw.loc[raw["precip_input_mm"] < 0, "precip_input_mm"] = np.nan
    doy = raw.index.dayofyear.to_numpy(dtype=float)
    raw["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    raw["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    year_fraction = doy / np.where(raw.index.is_leap_year, 366.0, 365.0)
    raw["year_sin"] = np.sin(2 * np.pi * year_fraction)
    raw["year_cos"] = np.cos(2 * np.pi * year_fraction)
    if len(raw) != lookback:
        raise ValueError(f"Expected {lookback} history rows, received {len(raw)}")
    features = metadata["features"]
    medians = np.asarray(metadata["feature_medians"], dtype=np.float32)
    means = np.asarray(metadata["feature_mean"], dtype=np.float32)
    stds = np.asarray(metadata["feature_std"], dtype=np.float32)
    x = raw[features].to_numpy(dtype=np.float32)
    for j in range(x.shape[1]):
        invalid = ~np.isfinite(x[:, j])
        x[invalid, j] = medians[j]
    x = (x - means) / np.where(stds == 0, 1.0, stds)
    station_index = {str(k): int(v) for k, v in metadata["station_to_index"].items()}[station_id]
    flattened = x.reshape(1, -1)
    model_input = np.concatenate([flattened, np.array([[station_index]], dtype=np.float32)], axis=1)
    return raw, model_input

def predict_seven(models, metadata, model_input):
    targets = metadata["target_names"]
    target_means = np.asarray(metadata["target_mean"], dtype=np.float32)
    target_stds = np.asarray(metadata["target_std"], dtype=np.float32)
    output = []
    for lead in range(7):
        row = {}
        for ti, target in enumerate(targets):
            model = models[lead * len(targets) + ti]
            y_scaled = float(np.asarray(model.predict(model_input)).reshape(-1)[0])
            row[target] = y_scaled * target_stds[ti] + target_means[ti]
        row["max_temp_c"], row["mean_temp_c"], row["min_temp_c"] = sorted(
            [row["max_temp_c"], row["mean_temp_c"], row["min_temp_c"]]
        )
        row["precip_mm"] = max(0.0, row["precip_mm"])
        output.append(row)
    return output

def main():
    parser = argparse.ArgumentParser(description="Backtest ATMOSYNC local LightGBM on historical API data.")
    parser.add_argument("--station", default="toronto")
    parser.add_argument("--end-date", default="2025-12-24",
                        help="Forecast origin date; following 7 days must have actuals in API.")
    parser.add_argument("--windows", type=int, default=4,
                        help="Number of weekly forecast windows, walking backward.")
    parser.add_argument("--output-dir", default=str(ROOT / "forecasting" / "parity_test"))
    args = parser.parse_args()

    station = args.station.lower().replace(" ", "_")
    if station not in STATION_IDS:
        parser.error(f"Unknown station {station}. Options: {', '.join(STATION_IDS)}")
    station_id = STATION_IDS[station]
    origin0 = date.fromisoformat(args.end_date)

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        print(f"Missing model/metadata: {MODEL_PATH} or {METADATA_PATH}", file=sys.stderr)
        return 1
    models = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(models, (list, tuple)) or len(models) != 28:
        print(f"Expected 28-model list, got {type(models).__name__}, length={len(models) if isinstance(models, (list, tuple)) else 'n/a'}", file=sys.stderr)
        return 1

    rows = []
    for window in range(args.windows):
        origin = origin0 - timedelta(days=7 * window)
        hist_start = origin - timedelta(days=int(metadata["lookback_days"]) - 1)
        try:
            history, x = prepare_history(station, station_id, origin, int(metadata["lookback_days"]), metadata)
            preds = predict_seven(models, metadata, x)
            actual_start, actual_end = origin + timedelta(days=1), origin + timedelta(days=7)
            actuals = get_api_frame(station, actual_start, actual_end)
            for lead, pred in enumerate(preds, start=1):
                target_date = origin + timedelta(days=lead)
                if pd.Timestamp(target_date) not in actuals.index:
                    continue
                actual_row = actuals.loc[pd.Timestamp(target_date)]
                row = {
                    "station": station,
                    "forecast_origin": origin.isoformat(),
                    "forecast_date": target_date.isoformat(),
                    "forecast_horizon_day": lead,
                }
                for target, aliases in TARGET_API_ALIASES.items():
                    col = next((c for c in aliases if c in actuals.columns), None)
                    if col is None:
                        row[f"actual_{target}"] = np.nan
                    else:
                        value = pd.to_numeric(pd.Series([actual_row[col]]), errors="coerce").iloc[0]
                        row[f"actual_{target}"] = float(value) if pd.notna(value) else np.nan
                    row[f"predicted_{target}"] = float(pred[target])
                rows.append(row)
            print(f"Window {window + 1}/{args.windows}: origin={origin}, actual period={actual_start}..{actual_end}")
        except Exception as exc:
            print(f"Window failed (origin {origin}): {exc}", file=sys.stderr)

    if not rows:
        print("No backtest rows produced. Check API coverage and station field names.", file=sys.stderr)
        return 2
    result = pd.DataFrame(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"backtest_{station}_{origin0.isoformat()}.csv"
    result.to_csv(detail_path, index=False)

    metric_rows = []
    for target in ("max_temp_c", "min_temp_c", "mean_temp_c", "precip_mm"):
        actual = pd.to_numeric(result[f"actual_{target}"], errors="coerce")
        predicted = pd.to_numeric(result[f"predicted_{target}"], errors="coerce")
        valid = actual.notna() & predicted.notna()
        if valid.sum() == 0:
            metric_rows.append({"target": target, "valid_rows": 0, "MAE": np.nan, "RMSE": np.nan})
            continue
        err = predicted[valid] - actual[valid]
        metric_rows.append({
            "target": target,
            "valid_rows": int(valid.sum()),
            "MAE": float(np.abs(err).mean()),
            "RMSE": float(np.sqrt(np.square(err).mean())),
        })
    metrics = pd.DataFrame(metric_rows)
    metrics_path = out_dir / f"metrics_{station}_{origin0.isoformat()}.csv"
    metrics.to_csv(metrics_path, index=False)
    print("\nBacktest metrics (actuals available from API):")
    print(metrics.to_string(index=False))
    print(f"\nDetailed rows: {detail_path}")
    print(f"Metrics:       {metrics_path}")
    print("\nNote: this evaluates API-history backtesting, not exact notebook prediction parity unless preprocessing and target handling are confirmed identical.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
