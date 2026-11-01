
from datetime import date, timedelta

import pandas as pd
import requests

BASE_URL = "https://atmosyncv2-0.onrender.com"

STATIONS = [
    "vancouver",
    "edmonton",
    "calgary",
    "winnipeg",
    "ottawa",
    "toronto",
    "montreal",
    "quebec city",
    "halifax",
]

WEATHER_FIELDS = [
    "max_temperature",
    "min_temperature",
    "mean_temperature",
    "heating_degree_days",
    "cooling_degree_days",
    "total_precipitation",
]


def extract_records(payload):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in (
            "data", "results", "records", "weather", "observations"
        ):
            if isinstance(payload.get(key), list):
                return payload[key]

    raise ValueError("Could not locate records in API response")


def inspect_station(station, start_date, end_date):
    print(f"\n{'=' * 60}")
    print(f"STATION: {station}")

    url = f"{BASE_URL}/weather/{station}"

    try:
        response = requests.get(
            url,
            params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            timeout=60,
        )

        print(f"HTTP status: {response.status_code}")
        response.raise_for_status()

        df = pd.DataFrame(extract_records(response.json()))

        if df.empty:
            print("ERROR: No records returned")
            return

        if "observation_date" not in df.columns:
            print("ERROR: observation_date is missing")
            return

        df["observation_date"] = pd.to_datetime(
            df["observation_date"], errors="coerce"
        ).dt.normalize()

        invalid_dates = int(df["observation_date"].isna().sum())
        df = df.dropna(subset=["observation_date"])

        df = df[
            (df["observation_date"] >= pd.Timestamp(start_date))
            & (df["observation_date"] <= pd.Timestamp(end_date))
        ]

        duplicates = int(
            df["observation_date"].duplicated().sum()
        )

        df = df.drop_duplicates(
            subset=["observation_date"], keep="last"
        )

        expected_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq="D",
        )

        observed_dates = pd.DatetimeIndex(df["observation_date"])
        missing_dates = expected_dates.difference(observed_dates)

        print(f"Expected dates: {len(expected_dates)}")
        print(f"Unique dates returned: {len(observed_dates)}")
        print(f"Invalid date records: {invalid_dates}")
        print(f"Duplicate dates: {duplicates}")

        if len(missing_dates):
            print("MISSING DATES:")
            for missing_date in missing_dates:
                print(f"  {missing_date.date()}")
        else:
            print("MISSING DATES: None")

        print("\nMISSING WEATHER VALUES:")

        found_missing_values = False

        for field in WEATHER_FIELDS:
            if field not in df.columns:
                print(f"  {field}: COLUMN MISSING")
                found_missing_values = True
                continue

            numeric = pd.to_numeric(df[field], errors="coerce")
            bad_dates = df.loc[
                numeric.isna(), "observation_date"
            ]

            if not bad_dates.empty:
                found_missing_values = True
                dates = [
                    value.date().isoformat()
                    for value in bad_dates
                ]
                print(f"  {field}: {dates}")

        if not found_missing_values:
            print("  None")

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")


def main():
    end_date = date(2026, 10, 31)
    start_date = end_date - timedelta(days=89)

    print(f"Validation period: {start_date} to {end_date}")

    for station in STATIONS:
        inspect_station(station, start_date, end_date)


if __name__ == "__main__":
    main()
