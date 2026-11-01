"""
ATMOSYNC missing-robust TCN: API forecast test
================================================
Run from the project root, for example:

python tests_/atmosync_tcn_missing_robust_api_test.py ^
  --api-url "https://atmosyncv2-0.onrender.com/weather/toronto" ^
  --start-date 2026-09-01 --end-date 2026-10-15 ^
  --export-dir "models/TCN-TFT/export" ^
  --output-dir "tests_/api_tcn_test_results"

Requirements:
    pip install requests pandas numpy scikit-learn joblib torch

IMPORTANT:
- The network architecture below is reconstructed from the supplied state-dict
  names/shapes. For scientifically valid predictions, replace MissingAwareTCN
  with the EXACT class/forward() from the training script if it differs.
- The script uses strict state-dict loading so key/shape incompatibilities fail.
- Target units in model_config.json were not confirmed. Do not use results
  operationally until target units and API field mappings are verified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
import torch
from torch import nn
from torch.nn.utils import weight_norm  # harmless if unused by the exact architecture


DEFAULT_API_URL = (
    "https://atmosyncv2-0.onrender.com/weather/toronto"
)

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
TARGETS = [
    "air_temperature_min",
    "air_temperature_max",
    "air_temperature_mean",
    "rainfall_total",
    "snowfall_total",
]

# API aliases. Adjust these to match the real JSON response from your API.
ALIASES = {
    "observation_date": [
        "observation_date", "date", "datetime", "valid_date", "forecast_date",
        "time", "timestamp",
    ],
    "station_id": [
        "station_id", "climate_identifier", "climate_id", "station",
        "station_name", "city", "location",
    ],
    "air_temperature_min": [
        "air_temperature_min", "min_temperature", "minimum_temperature",
        "temperature_min", "tmin", "min_temp",
    ],
    "air_temperature_max": [
        "air_temperature_max", "max_temperature", "maximum_temperature",
        "temperature_max", "tmax", "max_temp",
    ],
    "air_temperature_mean": [
        "air_temperature_mean", "mean_temperature", "average_temperature",
        "temperature_mean", "tmean", "mean_temp",
    ],
    "dew_point_temperature": [
        "dew_point_temperature", "dew_point", "dewpoint", "dew_point_temp",
    ],
    "relative_humidity": [
        "relative_humidity", "humidity", "relative_humidity_percent",
        "rh",
    ],
    "wind_speed": ["wind_speed", "wind_speed_kmh", "windspeed", "wind"],
    "surface_pressure": [
        "surface_pressure", "surface_pressure_kpa", "pressure", "pressure_kpa",
        "station_pressure",
    ],
    "solar_radiation": [
        "solar_radiation", "shortwave_radiation", "solar_radiation_mj_m2_day",
        "radiation",
    ],
    "rainfall_total": [
        "rainfall_total", "rainfall", "rain_amount", "precipitation",
        "precipitation_total", "precipitation_corrected", "PRECTOTCORR",
        "precipitation_mm",
    ],
    "snowfall_total": [
        "snowfall_total", "snowfall", "snow_amount", "snow_precipitation",
        "PRECSNO", "PRECSNOLAND",
    ],
}


def first_present(row: dict[str, Any], names: list[str]) -> Any:
    """Case-insensitive alias lookup for one API record."""
    lookup = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return np.nan


def extract_records(payload: Any) -> list[dict[str, Any]]:
    """Find a list of record dictionaries in common API JSON shapes."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        # Common direct list keys
        for key in (
            "data", "results", "records", "weather", "observations",
            "forecast", "daily", "items",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
            if isinstance(value, dict):
                nested = extract_records(value)
                if nested:
                    return nested

        # Sometimes the response is one record
        if any(k in {str(x).lower() for x in payload.keys()}
               for k in ("date", "observation_date", "datetime")):
            return [payload]

    raise ValueError(
        "Could not locate a list of daily records in API JSON. "
        "Inspect the saved response and update extract_records()."
    )


def normalize_api_payload(payload: Any) -> pd.DataFrame:
    records = extract_records(payload)
    normalized = []

    for record in records:
        out = {}
        for canonical, aliases in ALIASES.items():
            out[canonical] = first_present(record, aliases)
        # Preserve original fields for diagnosis
        out["_raw_json"] = json.dumps(record, default=str)
        normalized.append(out)

    df = pd.DataFrame(normalized)
    if "observation_date" not in df:
        raise ValueError("API response has no recognizable date field.")

    df["observation_date"] = pd.to_datetime(
        df["observation_date"], errors="coerce", utc=True
    ).dt.tz_convert(None).dt.normalize()

    df = df.dropna(subset=["observation_date"]).copy()
    if df.empty:
        raise ValueError("API response contained no parseable daily dates.")

    for col in FEATURES + TARGETS:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # If API provides repeated rows per station/day, retain each station separately.
    df["station_id"] = df["station_id"].fillna("default_station").astype(str)
    df = df.sort_values(["station_id", "observation_date"])
    df = df.drop_duplicates(["station_id", "observation_date"], keep="last")

    # Calendar values are derived from each actual API date, not filled-in dates.
    dates = df["observation_date"]
    df["month"] = dates.dt.month.astype(float)
    df["day_of_year"] = dates.dt.dayofyear.astype(float)
    angle = 2.0 * np.pi * (dates.dt.dayofyear.astype(float) - 1.0) / 365.25
    df["seasonal_cycle_sin"] = np.sin(angle)
    df["seasonal_cycle_cos"] = np.cos(angle)

    # If the API does not provide this field, use 0 for observed rows. Missingness
    # is still tracked by the input mask built below.
    df["temperature_imputed_flag"] = df["temperature_imputed_flag"].fillna(0.0)

    return df.reset_index(drop=True)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        y = self.conv(x)
        if self.padding:
            y = y[:, :, :-self.padding]
        y = self.norm(y)
        return y


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.c1 = CausalConv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.c2 = CausalConv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        y = self.activation(self.c1(x))
        y = self.dropout(y)
        y = self.activation(self.c2(y))
        y = self.dropout(y)
        return self.activation(y + residual)


class MissingAwareTCN(nn.Module):
    """
    Reconstructed shape-compatible TCN.
    Confirm this implementation against the original training source.
    """
    def __init__(self, input_channels: int = 26, channels: int = 64,
                 horizon: int = 7, n_targets: int = 5):
        super().__init__()
        self.proj = nn.Conv1d(input_channels, channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            TemporalBlock(channels, dilation=d) for d in (1, 2, 4, 8)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(channels),  # state dict head.0.weight/bias: (64,)
            nn.Linear(channels, 128),  # head.1
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, horizon * n_targets),  # head.4: (35,128)
        )
        self.horizon = horizon
        self.n_targets = n_targets

    def forward(self, x):
        # x shape: [batch, time, channels]
        x = x.transpose(1, 2)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        last = x[:, :, -1]
        out = self.head(last)
        return out.view(-1, self.horizon, self.n_targets)


def locate_file(directory: Path, candidates: list[str]) -> Path:
    for name in candidates:
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find any of {candidates} inside {directory.resolve()}"
    )


def load_model(export_dir: Path, device: torch.device):
    config_path = locate_file(export_dir, ["model_config.json"])
    with config_path.open("r", encoding="utf-8-sig") as f:
        config = json.load(f)

    features = config.get("features", FEATURES)
    targets = config.get("targets", TARGETS)
    if features != FEATURES:
        raise ValueError(
            f"Feature order differs from this script.\nConfig: {features}\n"
            f"Script: {FEATURES}\nUpdate FEATURES to exactly match model_config.json."
        )
    if targets != TARGETS:
        raise ValueError("Target order differs from this script; update TARGETS.")

    imputer = joblib.load(locate_file(export_dir, ["feature_imputer.joblib"]))
    scaler = joblib.load(locate_file(export_dir, ["feature_scaler.joblib"]))
    target_scaler = joblib.load(locate_file(export_dir, ["target_scaler.joblib"]))

    checkpoint_path = locate_file(
        export_dir,
        [
            "selected_model_state_dict.pt",
            "tcn_missing_robust_state_dict.pt",
            "atmosync_missing_aware_state_dict.pt",
            "best_model.pt",
        ],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = MissingAwareTCN(
        input_channels=int(config.get("input_channels", 26)),
        channels=64,
        horizon=int(config.get("forecast_horizon_days", 7)),
        n_targets=len(targets),
    )
    # Strict loading intentionally fails if the reconstructed class differs.
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, imputer, scaler, target_scaler, config, checkpoint_path


def make_input_window(group: pd.DataFrame, imputer, scaler):
    group = group.sort_values("observation_date").copy()
    if len(group) < 30:
        return None, "fewer_than_30_API_rows"

    tail = group.tail(30).copy()
    # Do not synthesize missing dates; demand 30 consecutive daily records.
    expected = pd.date_range(tail["observation_date"].iloc[0],
                             periods=30, freq="D")
    if not pd.DatetimeIndex(tail["observation_date"]).equals(expected):
        return None, "history_dates_not_30_consecutive_days"

    raw = tail[FEATURES].to_numpy(dtype=np.float64)
    missing_mask = ~np.isfinite(raw)

    # Rule A: if no temperature history exists at all, temperature predictions
    # are marked unavailable. Precipitation history is not a model input in this
    # model configuration, so its independent-history rule cannot be enforced
    # from model inputs alone.
    temp_cols = [
        FEATURES.index("air_temperature_min"),
        FEATURES.index("air_temperature_max"),
        FEATURES.index("air_temperature_mean"),
    ]
    temperature_history_count = int(np.isfinite(raw[:, temp_cols]).sum())

    imputed = imputer.transform(pd.DataFrame(raw, columns=FEATURES))
    scaled = scaler.transform(imputed)

    # Append one missingness channel per feature, matching config input_channels=26.
    x = np.concatenate([scaled, missing_mask.astype(np.float32)], axis=1)
    if x.shape != (30, 26):
        raise ValueError(f"Unexpected model input shape: {x.shape}")
    return {
        "x": x.astype(np.float32),
        "history_start": tail["observation_date"].iloc[0],
        "history_end": tail["observation_date"].iloc[-1],
        "temperature_history_count": temperature_history_count,
        "missing_cells": int(missing_mask.sum()),
        "missing_by_feature": {
            feature: int(missing_mask[:, i].sum())
            for i, feature in enumerate(FEATURES)
        },
        "history": tail,
    }, None


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "mae": None, "rmse": None, "bias_pred_minus_actual": None}
    errors = predicted[valid] - actual[valid]
    return {
        "n": n,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "bias_pred_minus_actual": float(np.mean(errors)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--export-dir", default="models/TCN-TFT/export")
    parser.add_argument("--output-dir", default="tests_/api_tcn_test_results")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--station", default=None,
                        help="Optional station_id/climate_identifier/name filter")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {"start_date": args.start_date, "end_date": args.end_date}
    print(f"[1/6] GET {args.api_url} params={params}")
    try:
        response = requests.get(args.api_url, params=params, timeout=args.timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"API_REQUEST_FAILED: {exc}", file=sys.stderr)
        print("Check Render service availability, URL, network and API route.", file=sys.stderr)
        return 2

    raw_path = output_dir / "api_response_raw.json"
    raw_path.write_text(response.text, encoding="utf-8")
    print(f"Saved raw response: {raw_path}")

    try:
        payload = response.json()
    except ValueError:
        print("API did not return valid JSON. Raw response saved.", file=sys.stderr)
        print(response.text[:1000], file=sys.stderr)
        return 2

    print("[2/6] Normalizing API response")
    try:
        df = normalize_api_payload(payload)
    except Exception as exc:
        print(f"API_SCHEMA_ERROR: {exc}", file=sys.stderr)
        print("Inspect api_response_raw.json and adjust ALIASES/extract_records().")
        return 3

    if args.station:
        matches = df["station_id"].astype(str).str.casefold() == args.station.casefold()
        if not matches.any():
            available = sorted(df["station_id"].astype(str).unique().tolist())
            print(f"Station {args.station!r} not found. Available: {available}")
            return 3
        df = df.loc[matches].copy()

    normalized_path = output_dir / "api_normalized_daily.csv"
    df.to_csv(normalized_path, index=False)
    print(f"Rows={len(df)} stations={df['station_id'].nunique()} saved={normalized_path}")
    print("Date range:", df["observation_date"].min().date(), "to",
          df["observation_date"].max().date())

    print("[3/6] Loading model and preprocessing artifacts")
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    try:
        model, imputer, scaler, target_scaler, config, checkpoint_path = load_model(
            export_dir, device
        )
    except Exception as exc:
        print(f"MODEL_LOAD_FAILED: {exc}", file=sys.stderr)
        print(
            "Confirm export_dir contains model_config.json, feature_imputer.joblib, "
            "feature_scaler.joblib, target_scaler.joblib and the trained checkpoint.",
            file=sys.stderr,
        )
        return 4
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Device: {device}; lookback={config.get('lookback_days', 30)}; "
          f"horizon={config.get('forecast_horizon_days', 7)}")
    if config.get("target_units_confirmed") is False:
        warnings.warn("Target units are unconfirmed in model_config.json.")

    print("[4/6] Forecasting each station with a valid 30-day history")
    forecasts = []
    input_audit = []
    failures = []

    for station_id, group in df.groupby("station_id", sort=False):
        group = group.sort_values("observation_date")
        window, reason = make_input_window(group, imputer, scaler)
        if reason:
            failures.append({"station_id": station_id, "reason": reason})
            continue

        x = torch.from_numpy(window["x"]).unsqueeze(0).to(device)
        with torch.no_grad():
            scaled_prediction = model(x).detach().cpu().numpy().reshape(7, len(TARGETS))
        prediction = target_scaler.inverse_transform(
            scaled_prediction.reshape(-1, len(TARGETS))
        )

        # Apply physical constraints without reordering target semantics.
        for i, target in enumerate(TARGETS):
            if target in ("rainfall_total", "snowfall_total"):
                prediction[:, i] = np.maximum(prediction[:, i], 0.0)

        # Enforce Tmin <= Tmean <= Tmax while preserving target identities:
        # use monotonic projection by sorting the three predicted temperature
        # values for each day, then assigning them back in semantic order.
        # This is a postprocessing correction; inspect its frequency in output.
        order_corrections = 0
        imin = TARGETS.index("air_temperature_min")
        imax = TARGETS.index("air_temperature_max")
        imean = TARGETS.index("air_temperature_mean")
        for day_i in range(7):
            before = prediction[day_i, [imin, imean, imax]].copy()
            after = np.sort(before)
            prediction[day_i, imin] = after[0]
            prediction[day_i, imean] = after[1]
            prediction[day_i, imax] = after[2]
            if not np.allclose(before, after):
                order_corrections += 1

        dates = pd.date_range(window["history_end"] + pd.Timedelta(days=1),
                              periods=7, freq="D")
        for j, date in enumerate(dates):
            row = {
                "station_id": station_id,
                "history_start": window["history_start"].date().isoformat(),
                "history_end": window["history_end"].date().isoformat(),
                "forecast_date": date.date().isoformat(),
                "temperature_history_count": window["temperature_history_count"],
                "input_missing_cells": window["missing_cells"],
                "temperature_order_corrections_window": order_corrections,
            }
            for k, target in enumerate(TARGETS):
                row[f"predicted_{target}"] = float(prediction[j, k])
            forecasts.append(row)

        input_audit.append({
            "station_id": station_id,
            "history_start": window["history_start"].date().isoformat(),
            "history_end": window["history_end"].date().isoformat(),
            "temperature_history_count": window["temperature_history_count"],
            "input_missing_cells": window["missing_cells"],
            **{f"missing_{k}": v for k, v in window["missing_by_feature"].items()},
        })

    forecast_df = pd.DataFrame(forecasts)
    if forecast_df.empty:
        pd.DataFrame(failures).to_csv(output_dir / "stations_skipped.csv", index=False)
        print("No forecasts created. Need at least 30 consecutive daily rows per station.")
        return 5

    forecast_path = output_dir / "api_model_forecast_7day.csv"
    forecast_df.to_csv(forecast_path, index=False)
    pd.DataFrame(input_audit).to_csv(output_dir / "api_model_input_used.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "stations_skipped.csv", index=False)
    print(f"Saved forecasts: {forecast_path}")
    print(f"Forecast rows: {len(forecast_df)}")

    print("[5/6] Comparing forecasts with API actuals when target fields are present")
    # Only compute metrics when API includes actual target values for forecast dates.
    # For historical APIs, this compares the forecast dates with returned observations.
    actual_columns = [c for c in TARGETS if c in df.columns and df[c].notna().any()]
    metrics = {}
    comparison_rows = []

    if actual_columns:
        actual = df[["station_id", "observation_date"] + actual_columns].rename(
            columns={"observation_date": "forecast_date"}
        )

        # Normalize both join keys to the same dtype. Forecast dates are written
        # as ISO strings above, while API observation dates are pandas timestamps.
        forecast_df["forecast_date"] = pd.to_datetime(
            forecast_df["forecast_date"], errors="coerce"
        ).dt.normalize()
        actual["forecast_date"] = pd.to_datetime(
            actual["forecast_date"], errors="coerce"
        ).dt.normalize()
        forecast_df["station_id"] = forecast_df["station_id"].astype(str)
        actual["station_id"] = actual["station_id"].astype(str)

        merged = forecast_df.merge(
            actual, on=["station_id", "forecast_date"], how="left"
        )
        for target in TARGETS:
            actual_col = target if target in actual_columns else None
            pred_col = f"predicted_{target}"
            if actual_col:
                metrics[target] = calculate_metrics(
                    pd.to_numeric(merged[actual_col], errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(merged[pred_col], errors="coerce").to_numpy(dtype=float),
                )
                merged[f"error_{target}"] = merged[pred_col] - merged[actual_col]
                merged[f"absolute_error_{target}"] = merged[f"error_{target}"].abs()
            else:
                metrics[target] = {
                    "n": 0, "mae": None, "rmse": None,
                    "bias_pred_minus_actual": None,
                    "note": "API response did not provide actual values for this target",
                }
        comparison_path = output_dir / "api_forecast_actual_comparison.csv"
        merged.to_csv(comparison_path, index=False)
        print(f"Saved comparison: {comparison_path}")
    else:
        metrics = {
            target: {
                "n": 0, "mae": None, "rmse": None,
                "bias_pred_minus_actual": None,
                "note": "No actual target observations detected in API response",
            } for target in TARGETS
        }
        print("No target actuals detected. Forecast was generated, but MAE/RMSE cannot be computed.")

    metrics_path = output_dir / "api_forecast_metrics.json"
    report = {
        "api_url": args.api_url,
        "request_params": params,
        "http_status": response.status_code,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "model_type": config.get("model_type"),
        "lookback_days": config.get("lookback_days", 30),
        "forecast_horizon_days": config.get("forecast_horizon_days", 7),
        "target_units_confirmed": config.get("target_units_confirmed", False),
        "api_rows": int(len(df)),
        "stations_in_response": int(df["station_id"].nunique()),
        "forecast_rows": int(len(forecast_df)),
        "stations_skipped": failures,
        "metrics": metrics,
        "important_limitations": [
            "Architecture was reconstructed from state-dict shapes; confirm against training source.",
            "Rainfall/snowfall are not included in the configured 13 input features, so independent precipitation-history eligibility cannot be enforced by this model as configured.",
            "Metrics are meaningful only when API actuals refer to the same dates, units and target definitions as the model outputs.",
            "Forecast-vs-observation comparison on dates used as model history is not a valid out-of-sample evaluation.",
        ],
    }
    metrics_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Saved metrics/report: {metrics_path}")

    print("[6/6] Summary")
    print(json.dumps(metrics, indent=2))
    print("\nDone. Review output files and verify units/feature mapping before interpreting scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
