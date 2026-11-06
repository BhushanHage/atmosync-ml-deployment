from __future__ import annotations
import json, os
from datetime import date, timedelta
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(
    os.getenv("ATMOSYNC_LIGHTGBM_MODEL_DIR", ROOT / "models" / "Light_GBM")
)
API_BASE = os.getenv(
    "ATMOSYNC_WEATHER_API_BASE_URL", "https://atmosyncv2-0.onrender.com"
).rstrip("/")
STATIONS = {
    "vancouver": "1108446",
    "edmonton": "3012209",
    "calgary": "3031094",
    "winnipeg": "502S001",
    "ottawa": "6105978",
    "toronto": "6158355",
    "montreal": "7014160",
    "quebec city": "701S001",
    "halifax": "8202251",
}
# API route slugs are not always identical to the public-facing station names.
API_STATION_SLUGS = {
    'vancouver': 'vancouver',
    'edmonton': 'edmonton',
    'calgary': 'calgary',
    'winnipeg': 'winnipeg',
    'ottawa': 'ottawa',
    'toronto': 'toronto',
    'montreal': 'montreal',
    'quebec city': 'Quebec City',
    'halifax': 'halifax',
}
ALIASES = {
    "Max Temp (°C)": ["max_temperature", "max_temp_c", "maximum_temperature"],
    "Min Temp (°C)": ["min_temperature", "min_temp_c", "minimum_temperature"],
    "Mean Temp (°C)": ["mean_temperature", "mean_temp_c", "average_temperature"],
    "Heat Deg Days (°C)": ["heating_degree_days", "heat_degree_days", "heat_deg_days"],
    "Cool Deg Days (°C)": ["cooling_degree_days", "cool_degree_days", "cool_deg_days"],
    "precip_input_mm": [
        "total_precipitation",
        "precipitation",
        "precip_mm",
        "precipitation_mm",
        "PRECTOTCORR",
    ],
}


class LightGBMForecaster:
    def __init__(self):
        model_path = MODEL_DIR / "lightgbm_models.joblib"
        meta_path = MODEL_DIR / "inference_metadata.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"LightGBM artifacts missing in {MODEL_DIR}")
        self.models = joblib.load(model_path)
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.lookback = int(self.meta["lookback_days"])
        self.horizon = int(self.meta["forecast_horizon_days"])
        if len(self.models) != self.horizon * len(self.meta["target_names"]):
            raise ValueError("Model count does not match metadata horizon and targets")
        if self.lookback != 90 or self.horizon != 7:
            raise ValueError("Unexpected model lookback/horizon")
        self.target_names = self.meta["target_names"]

    def fetch_history(self, station: str, end_date: date) -> pd.DataFrame:
        if station not in STATIONS:
            raise ValueError(
                f'Unknown station {station!r}. Choose from {", ".join(STATIONS)}'
            )
        start = end_date - timedelta(days=self.lookback - 1)
        api_station = API_STATION_SLUGS[station]
        r = requests.get(
            f"{API_BASE}/weather/{api_station}",
            params={"start_date": start.isoformat(), "end_date": end_date.isoformat()},
            timeout=60,
        )
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = next(
                (
                    payload[k]
                    for k in ("data", "results", "records", "weather", "observations")
                    if isinstance(payload.get(k), list)
                ),
                None,
            )
            if records is None:
                raise ValueError(
                    f"Unexpected weather API response keys: {list(payload)}"
                )
        else:
            raise ValueError(
                "Weather API response must be a list or object containing records"
            )
        df = pd.DataFrame(records)
        if df.empty:
            raise ValueError("Weather API returned no history records")
        date_col = next(
            (
                c
                for c in ("observation_date", "Date/Time", "date", "forecast_date")
                if c in df.columns
            ),
            None,
        )
        if not date_col:
            raise ValueError("Weather API response has no observation date column")
        df["__date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
        df = df.dropna(subset=["__date"])
        df = (
            df[
                (df["__date"] >= pd.Timestamp(start))
                & (df["__date"] <= pd.Timestamp(end_date))
            ]
            .sort_values("__date")
            .drop_duplicates("__date", keep="last")
            .set_index("__date")
        )

        # Reindex to the requested calendar, not the first/last returned observation.
        # This preserves missing leading, trailing, and internal dates as NaN rows.
        calendar = pd.date_range(start=start, end=end_date, freq="D", name="__date")
        df = df.reindex(calendar)
        for feature, aliases in ALIASES.items():
            col = next((c for c in aliases if c in df.columns), None)
            df[feature] = pd.to_numeric(df[col], errors="coerce") if col else np.nan
        df.loc[df["precip_input_mm"] < 0, "precip_input_mm"] = np.nan
        doy = df.index.dayofyear.to_numpy(float)
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        yf = doy / np.where(df.index.is_leap_year, 366.0, 365.0)
        df["year_sin"] = np.sin(2 * np.pi * yf)
        df["year_cos"] = np.cos(2 * np.pi * yf)
        if len(df) != self.lookback:
            raise ValueError(f"Expected {self.lookback} calendar days; got {len(df)}")
        if (
            df[list(self.meta["features"][:5])].isna().all(axis=1).sum()
            > self.lookback // 3
        ):
            raise ValueError(
                "Too many days have all temperature/degree-day features missing; refusing unreliable forecast"
            )
        return df

    def predict(
        self, station: str, end_date: date | None = None
    ) -> list[dict[str, Any]]:
        end_date = end_date or (date.today() - timedelta(days=1))
        sid = STATIONS[station]
        hist = self.fetch_history(station, end_date)
        names = self.meta["features"]
        med = np.asarray(self.meta["feature_medians"], np.float32)
        mean = np.asarray(self.meta["feature_mean"], np.float32)
        std = np.asarray(self.meta["feature_std"], np.float32)
        x = hist[names].to_numpy(dtype=np.float32)
        for j in range(x.shape[1]):
            x[~np.isfinite(x[:, j]), j] = med[j]
        x = (x - mean) / np.where(std == 0, 1.0, std)
        station_idx = {str(k): int(v) for k, v in self.meta["station_to_index"].items()}
        if sid not in station_idx:
            raise ValueError(f"Station ID {sid} missing from model metadata")
        features = np.concatenate(
            [x.reshape(1, -1), np.asarray([[station_idx[sid]]], np.float32)], axis=1
        )
        tm = np.asarray(self.meta["target_mean"], np.float32)
        ts = np.asarray(self.meta["target_std"], np.float32)
        rows = []
        for lead in range(self.horizon):
            vals = []
            for j in range(len(self.target_names)):
                model = self.models[lead * len(self.target_names) + j]
                if model is None:
                    raise ValueError(
                        f"Model estimator missing at index {lead*len(self.target_names)+j}"
                    )
                expected = getattr(model, "n_features_in_", None)
                if expected is not None and int(expected) != features.shape[1]:
                    raise ValueError(
                        f"Estimator expects {expected} features; got {features.shape[1]}"
                    )
                vals.append(
                    float(np.asarray(model.predict(features)).reshape(-1)[0])
                    * float(ts[j])
                    + float(tm[j])
                )
            row = {
                "station": station,
                "station_id": sid,
                "history_start": hist.index.min().date().isoformat(),
                "history_end": end_date.isoformat(),
                "forecast_date": (end_date + timedelta(days=lead + 1)).isoformat(),
                "forecast_horizon_day": lead + 1,
                "model": "LightGBM",
            }
            for name, val in zip(self.meta["target_names"], vals):
                row[
                    "predicted_"
                    + name.replace("precip_target_observed_mm", "precip_mm")
                ] = (val if np.isfinite(val) else None)
            temp_keys = [
                "predicted_min_temp_c",
                "predicted_mean_temp_c",
                "predicted_max_temp_c",
            ]
            if all(row.get(k) is not None for k in temp_keys):
                ordered = sorted(row[k] for k in temp_keys)
                for k, v in zip(temp_keys, ordered):
                    row[k] = v
            if row.get("predicted_precip_mm") is not None:
                row["predicted_precip_mm"] = max(0.0, row["predicted_precip_mm"])
            rows.append(row)
        return rows
