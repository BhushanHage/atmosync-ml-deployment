"""
ATMOSYNC all-station retrospective API evaluation.

Default evaluation:
  Forecast origin: 2026-10-01
  Input history:   2026-09-02 through 2026-10-01 (30 days)
  Verification:    2026-10-02 through 2026-10-08 (7 days)

Run from project root:
  python tests_/atmosync_tcn_all_station_api_test.py `
    --export-dir "models/atmosync_tcn_missing_robust/export" `
    --output-dir "tests_/all_station_api_test_results"

Requirements:
  pip install requests pandas numpy scikit-learn joblib torch

Important:
- Imports the existing single-station test script for model/preprocessing functions.
- The model architecture in that script is reconstructed; confirm it matches the
  training source before treating predictions as verified.
- API total_precipitation is compared only as a proxy/context column, NOT as
  verified rainfall target accuracy. total_rain/total_snow are null in the
  example records, so separate rain/snow accuracy may be unavailable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests


STATIONS = [
    {"city": "Toronto", "climate_identifier": "6158355", "slug": "toronto"},
    {"city": "Montreal", "climate_identifier": "7014160", "slug": "montreal"},
    {"city": "Vancouver", "climate_identifier": "1108446", "slug": "vancouver"},
    {"city": "Calgary", "climate_identifier": "3031094", "slug": "calgary"},
    {"city": "Edmonton", "climate_identifier": "3012209", "slug": "edmonton"},
    {"city": "Ottawa", "climate_identifier": "6105978", "slug": "ottawa"},
    {"city": "Winnipeg", "climate_identifier": "502S001", "slug": "winnipeg"},
    {"city": "Quebec City", "climate_identifier": "701S001", "slug": "quebec city"},
    {"city": "Halifax", "climate_identifier": "8202251", "slug": "halifax"},
]
BASE_URL = "https://atmosyncv2-0.onrender.com/weather"


def load_test_module(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(
            f"Required helper script not found: {script_path}. "
            "Keep atmosync_tcn_missing_robust_api_test.py in tests_."
        )
    spec = importlib.util.spec_from_file_location("atmosync_single_station_test", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import helper script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_api_data(city_slug: str, start_date: str, end_date: str, timeout: int):
    url = f"{BASE_URL}/{city_slug}"
    response = requests.get(
        url,
        params={"start_date": start_date, "end_date": end_date},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload, response.status_code, url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", default="models/atmosync_tcn_missing_robust/export")
    parser.add_argument("--output-dir", default="tests_/all_station_api_test_results")
    parser.add_argument("--helper-script", default="tests_/atmosync_tcn_missing_robust_api_test.py")
    parser.add_argument("--api-base-url", default=BASE_URL)
    parser.add_argument("--history-start", default="2026-09-02")
    parser.add_argument("--origin-date", default="2026-10-01")
    parser.add_argument("--verification-end", default="2026-10-08")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    history_start = pd.Timestamp(args.history_start)
    origin = pd.Timestamp(args.origin_date)
    verification_end = pd.Timestamp(args.verification_end)
    if (origin - history_start).days != 29:
        parser.error("history-start must be exactly 29 days before origin-date for 30 daily rows.")
    if (verification_end - origin).days != 7:
        parser.error("verification-end must be exactly 7 days after origin-date for a 7-day horizon.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    helper_path = Path(args.helper_script)
    helper = load_test_module(helper_path)

    device = helper.torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and helper.torch.cuda.is_available()) else "cpu"
    )
    print("[1/5] Loading model and preprocessing artifacts...")
    model, imputer, scaler, target_scaler, config, checkpoint_path = helper.load_model(
        Path(args.export_dir), device
    )
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    if config.get("target_units_confirmed") is False:
        print("WARNING: target units are marked unconfirmed in model_config.json.")

    api_end = verification_end.strftime("%Y-%m-%d")
    api_start = history_start.strftime("%Y-%m-%d")
    all_forecasts = []
    all_comparisons = []
    station_status = []
    raw_dir = out_dir / "raw_api"
    raw_dir.mkdir(exist_ok=True)

    print(f"[2/5] Fetching all 9 cities: {api_start} through {api_end}")
    for station in STATIONS:
        city = station["city"]
        slug = station["slug"]
        sid_expected = station["climate_identifier"]
        print(f"  - {city} ({sid_expected}) ...", end=" ", flush=True)

        try:
            url = f"{args.api_base_url.rstrip('/')}/{slug}"
            response = requests.get(
                url,
                params={"start_date": api_start, "end_date": api_end},
                timeout=args.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            (raw_dir / f"{slug}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            df = helper.normalize_api_payload(payload)
        except Exception as exc:
            print(f"FAILED: {exc}")
            station_status.append({
                "city": city, "climate_identifier": sid_expected,
                "status": "api_or_schema_failed", "detail": str(exc),
            })
            continue

        # Use the mapped ID where API records have it; otherwise retain city key.
        df["station_id"] = df["station_id"].astype(str)
        matching = df[df["station_id"] == sid_expected].copy()
        if matching.empty:
            # Some endpoints may not return climate_identifier; accept their rows
            # only if all returned records belong to this city endpoint.
            matching = df.copy()
            matching["station_id"] = sid_expected
        df = matching.sort_values("observation_date")
        history = df[
            (df["observation_date"] >= history_start) &
            (df["observation_date"] <= origin)
        ].copy()
        actuals = df[
            (df["observation_date"] > origin) &
            (df["observation_date"] <= verification_end)
        ].copy()

        if len(history) != 30:
            print(f"SKIPPED: expected 30 history rows, got {len(history)}")
            station_status.append({
                "city": city, "climate_identifier": sid_expected,
                "status": "insufficient_history_rows",
                "history_rows": int(len(history)),
                "actual_rows": int(len(actuals)),
            })
            continue

        window, reason = helper.make_input_window(history, imputer, scaler)
        if reason:
            print(f"SKIPPED: {reason}")
            station_status.append({
                "city": city, "climate_identifier": sid_expected,
                "status": "invalid_history", "detail": reason,
                "history_rows": int(len(history)), "actual_rows": int(len(actuals)),
            })
            continue

        x = helper.torch.from_numpy(window["x"]).unsqueeze(0).to(device)
        with helper.torch.no_grad():
            pred_scaled = model(x).detach().cpu().numpy().reshape(7, len(helper.TARGETS))
        pred = target_scaler.inverse_transform(pred_scaled)

        # Enforce nonnegative precipitation and preserve target identities.
        for idx, target in enumerate(helper.TARGETS):
            if target in ("rainfall_total", "snowfall_total"):
                pred[:, idx] = np.maximum(pred[:, idx], 0.0)

        imin = helper.TARGETS.index("air_temperature_min")
        imean = helper.TARGETS.index("air_temperature_mean")
        imax = helper.TARGETS.index("air_temperature_max")
        corrections = 0
        for day_idx in range(7):
            before = pred[day_idx, [imin, imean, imax]].copy()
            after = np.sort(before)
            pred[day_idx, imin], pred[day_idx, imean], pred[day_idx, imax] = after
            corrections += int(not np.allclose(before, after))

        forecast_dates = pd.date_range(origin + pd.Timedelta(days=1), periods=7, freq="D")
        city_forecasts = []
        for i, date in enumerate(forecast_dates):
            row = {
                "city": city,
                "station_id": sid_expected,
                "history_start": history_start.date().isoformat(),
                "history_end": origin.date().isoformat(),
                "forecast_date": date.date().isoformat(),
                "input_missing_cells": window["missing_cells"],
                "temperature_order_corrections_window": corrections,
            }
            for j, target in enumerate(helper.TARGETS):
                row[f"predicted_{target}"] = float(pred[i, j])
            city_forecasts.append(row)
        forecast_df = pd.DataFrame(city_forecasts)
        all_forecasts.append(forecast_df)

        actuals = actuals.copy()
        actuals["forecast_date"] = pd.to_datetime(actuals["observation_date"]).dt.normalize()
        forecast_df["forecast_date"] = pd.to_datetime(forecast_df["forecast_date"]).dt.normalize()
        actuals["station_id"] = sid_expected
        actual_cols = [
            "air_temperature_min", "air_temperature_max", "air_temperature_mean"
        ]
        # Normalize API precipitation aliases separately. It is a proxy only;
        # do not relabel total_precipitation as actual rainfall.
        raw_records = helper.extract_records(payload)
        raw_lookup = {}
        for record in raw_records:
            low = {str(k).lower(): v for k, v in record.items()}
            date_value = low.get("observation_date", low.get("date"))
            if date_value is None:
                continue
            date_key = pd.to_datetime(date_value, errors="coerce")
            if pd.isna(date_key):
                continue
            date_key = date_key.normalize()
            rec_sid = str(low.get("climate_identifier", sid_expected))
            if rec_sid == sid_expected:
                raw_lookup[date_key] = low

        actuals["actual_total_precipitation_proxy"] = actuals["observation_date"].map(
            lambda d: pd.to_numeric(
                raw_lookup.get(pd.Timestamp(d).normalize(), {}).get("total_precipitation"),
                errors="coerce",
            )
        )
        for target in actual_cols:
            actuals[f"actual_{target}"] = actuals[target]

        merged = forecast_df.merge(
            actuals[["station_id", "forecast_date"] +
                    [f"actual_{t}" for t in actual_cols] +
                    ["actual_total_precipitation_proxy"]],
            on=["station_id", "forecast_date"], how="left"
        )
        for target in actual_cols:
            pred_col = f"predicted_{target}"
            actual_col = f"actual_{target}"
            merged[f"error_{target}"] = merged[pred_col] - merged[actual_col]
            merged[f"absolute_error_{target}"] = merged[f"error_{target}"].abs()
        all_comparisons.append(merged)
        print(f"OK: history=30, actual verification rows={int(merged['actual_air_temperature_mean'].notna().sum())}")
        station_status.append({
            "city": city, "climate_identifier": sid_expected,
            "status": "evaluated", "history_rows": int(len(history)),
            "actual_rows": int(len(actuals)),
        })

    print("[3/5] Writing forecast and comparison CSVs...")
    forecasts_all = pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()
    comparisons_all = pd.concat(all_comparisons, ignore_index=True) if all_comparisons else pd.DataFrame()
    forecasts_all.to_csv(out_dir / "all_station_forecasts_7day.csv", index=False)
    comparisons_all.to_csv(out_dir / "all_station_forecast_actual_comparison.csv", index=False)
    pd.DataFrame(station_status).to_csv(out_dir / "station_status.csv", index=False)

    print("[4/5] Calculating per-station and overall metrics...")
    metric_rows = []
    if not comparisons_all.empty:
        targets = [
            ("air_temperature_min", "actual_air_temperature_min"),
            ("air_temperature_mean", "actual_air_temperature_mean"),
            ("air_temperature_max", "actual_air_temperature_max"),
        ]
        groups = list(comparisons_all.groupby(["city", "station_id"], dropna=False))
        groups.append(("OVERALL", "ALL"))
        for key, group in groups:
            city, sid = key if isinstance(key, tuple) else (key, "ALL")
            for target, actual_col in targets:
                if actual_col not in group:
                    continue
                actual = pd.to_numeric(group[actual_col], errors="coerce").to_numpy(float)
                predicted = pd.to_numeric(group[f"predicted_{target}"], errors="coerce").to_numpy(float)
                valid = np.isfinite(actual) & np.isfinite(predicted)
                if valid.sum() == 0:
                    mae = rmse = bias = np.nan
                else:
                    error = predicted[valid] - actual[valid]
                    mae = float(np.mean(np.abs(error)))
                    rmse = float(np.sqrt(np.mean(error ** 2)))
                    bias = float(np.mean(error))
                metric_rows.append({
                    "city": city, "station_id": sid, "target": target,
                    "n": int(valid.sum()), "mae": mae, "rmse": rmse, "bias_pred_minus_actual": bias,
                })
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "metrics_by_station_and_overall.csv", index=False)

    # A descriptive precipitation proxy comparison, not rainfall accuracy.
    if not comparisons_all.empty and "actual_total_precipitation_proxy" in comparisons_all:
        proxy_rows = []
        for city, group in comparisons_all.groupby("city"):
            actual = pd.to_numeric(group["actual_total_precipitation_proxy"], errors="coerce").to_numpy(float)
            predicted = pd.to_numeric(group["predicted_rainfall_total"], errors="coerce").to_numpy(float)
            valid = np.isfinite(actual) & np.isfinite(predicted)
            error = predicted[valid] - actual[valid]
            proxy_rows.append({
                "city": city, "n": int(valid.sum()),
                "mae_vs_total_precipitation_proxy": float(np.mean(np.abs(error))) if valid.any() else np.nan,
                "rmse_vs_total_precipitation_proxy": float(np.sqrt(np.mean(error**2))) if valid.any() else np.nan,
                "warning": "Proxy only: total precipitation is not confirmed equivalent to rainfall_total",
            })
        pd.DataFrame(proxy_rows).to_csv(out_dir / "precipitation_proxy_metrics_NOT_rain_accuracy.csv", index=False)

    print("[5/5] Summary")
    print(pd.DataFrame(station_status).to_string(index=False))
    if not metrics_df.empty:
        print("\nOverall temperature metrics:")
        print(metrics_df[metrics_df["city"] == "OVERALL"].to_string(index=False))
    print(f"\nOutputs saved under: {out_dir.resolve()}")
    print("Separate rainfall/snowfall accuracy requires non-null observed total_rain/total_snow or equivalent validated targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
