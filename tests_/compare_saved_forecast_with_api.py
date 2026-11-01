
import json
import math
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]

PREDICTION_FILE = (
    ROOT / "tests_" / "atmosync_predict.json"
)

API_URL = "https://atmosyncv2-0.onrender.com/weather/toronto"

START_DATE = "2026-10-13"
END_DATE = "2026-10-19"

TARGETS = {
    "air_temperature_min": "min_temperature",
    "air_temperature_mean": "mean_temperature",
    "air_temperature_max": "max_temperature",
}


def find_prediction_records(obj):
    """Find date-indexed prediction records in a JSON structure."""
    found = []

    if isinstance(obj, list):
        for item in obj:
            found.extend(find_prediction_records(item))

    elif isinstance(obj, dict):
        date_keys = (
            "date",
            "forecast_date",
            "target_date",
            "observation_date",
        )

        date_value = next(
            (obj[k] for k in date_keys if k in obj),
            None,
        )

        if date_value is not None and any(
            key in obj for key in TARGETS
        ):
            found.append(obj)

        else:
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    found.extend(find_prediction_records(value))

    return found


def load_predictions():
    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Forecast JSON not found: {PREDICTION_FILE}"
        )

    with open(PREDICTION_FILE, "r", encoding="utf-8") as f:
        payload = json.load(f)

    records = find_prediction_records(payload)

    if not records:
        print("Could not automatically identify prediction rows.")
        print("JSON top-level keys:")
        if isinstance(payload, dict):
            print(list(payload.keys()))
        print("\nJSON preview:")
        print(json.dumps(payload, indent=2)[:5000])
        raise ValueError(
            "Please check the JSON structure and its prediction keys."
        )

    df = pd.DataFrame(records)

    date_column = next(
        (
            c for c in (
                "date",
                "forecast_date",
                "target_date",
                "observation_date",
            )
            if c in df.columns
        ),
        None,
    )

    if date_column is None:
        raise ValueError("No forecast date field found.")

    df["date"] = pd.to_datetime(
        df[date_column], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    for target in TARGETS:
        if target in df.columns:
            df[target] = pd.to_numeric(
                df[target], errors="coerce"
            )

    return df


def fetch_actuals():
    response = requests.get(
        API_URL,
        params={
            "start_date": START_DATE,
            "end_date": END_DATE,
        },
        timeout=90,
    )
    response.raise_for_status()

    records = response.json()
    if not isinstance(records, list):
        raise ValueError("Expected API to return a list.")

    df = pd.DataFrame(records)

    if df.empty:
        return pd.DataFrame(columns=["date"])

    df["date"] = pd.to_datetime(
        df["observation_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    actual_fields = [
        "min_temperature",
        "mean_temperature",
        "max_temperature",
        "total_precipitation",
        "total_rain",
        "total_snow",
    ]

    for field in actual_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(
                df[field], errors="coerce"
            )

    return df


def main():
    print("Loading saved model predictions...")
    predictions = load_predictions()

    predictions = predictions[
        predictions["date"].between(START_DATE, END_DATE)
    ].copy()

    print("Prediction rows found:", len(predictions))

    if predictions.empty:
        raise ValueError(
            "No predictions found for the requested dates."
        )

    print("\nFetching actual API observations...")
    actuals = fetch_actuals()
    print("Actual API records:", len(actuals))

    comparison = predictions.merge(
        actuals,
        on="date",
        how="left",
        suffixes=("_predicted", "_actual"),
    )

    metrics = []

    for predicted_field, actual_field in TARGETS.items():
        if predicted_field not in comparison.columns:
            print(f"\nPrediction target missing: {predicted_field}")
            continue

        if actual_field not in comparison.columns:
            print(f"\nActual field missing: {actual_field}")
            continue

        pred = pd.to_numeric(
            comparison[predicted_field], errors="coerce"
        )
        actual = pd.to_numeric(
            comparison[actual_field], errors="coerce"
        )

        valid = pred.notna() & actual.notna()

        comparison[predicted_field + "_actual"] = actual
        comparison[predicted_field + "_error"] = pred - actual
        comparison[predicted_field + "_absolute_error"] = (
            (pred - actual).abs()
        )

        if valid.any():
            errors = pred[valid] - actual[valid]
            mae = errors.abs().mean()
            rmse = math.sqrt((errors ** 2).mean())

            metrics.append({
                "target": predicted_field,
                "valid_days": int(valid.sum()),
                "MAE": mae,
                "RMSE": rmse,
            })

    output_file = ROOT / "tests_" / "forecast_api_comparison.csv"
    comparison.to_csv(output_file, index=False)

    print("\nDATE-BY-DATE TEMPERATURE COMPARISON")
    display_columns = ["date"]

    for target in TARGETS:
        for suffix in ("", "_actual", "_error"):
            column = target + suffix
            if column in comparison.columns:
                display_columns.append(column)

    print(
        comparison[
            list(dict.fromkeys(display_columns))
        ].to_string(index=False)
    )

    print("\nACCURACY METRICS")
    if metrics:
        print(pd.DataFrame(metrics).to_string(index=False))
    else:
        print(
            "No matching predicted and actual temperatures "
            "were available for scoring."
        )

    print("\nPRECIPITATION VALIDATION")
    print(
        "The API total_precipitation field is not automatically "
        "equivalent to the model rainfall_total target."
    )
    print(
        "Rainfall and snowfall accuracy cannot be calculated "
        "when total_rain and total_snow are null."
    )

    print(f"\nSaved comparison CSV: {output_file}")


if __name__ == "__main__":
    main()
