from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

import forecasting.lightgbm_inference as inference

def make_forecaster():
    obj = inference.LightGBMForecaster.__new__(inference.LightGBMForecaster)
    obj.lookback = 90
    obj.meta = {
        "features": [
            "Max Temp (°C)", "Min Temp (°C)", "Mean Temp (°C)",
            "Heat Deg Days (°C)", "Cool Deg Days (°C)",
            "precip_input_mm", "doy_sin", "doy_cos", "year_sin", "year_cos",
        ]
    }
    return obj


def test_fetch_history_reindexes_missing_leading_trailing_and_internal_dates(monkeypatch):
    end = date(2026, 11, 1)
    start = end - timedelta(days=89)
    missing_dates = {start, start + timedelta(days=10), end}
    records = []
    for i in range(90):
        day = start + timedelta(days=i)
        if day in missing_dates:
            continue
        records.append({
            "observation_date": day.isoformat(),
            "max_temperature": 20,
            "min_temperature": 10,
            "mean_temperature": 15,
            "heating_degree_days": 0,
            "cooling_degree_days": 0,
            "total_precipitation": 2,
        })

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return records

    captured = {}

    def fake_get(url, params, timeout):
        captured.update(url=url, params=params, timeout=timeout)
        return Response()

    monkeypatch.setattr(inference.requests, "get", fake_get)
    hist = make_forecaster().fetch_history("quebec city", end)

    assert len(hist) == 90
    assert hist.index.min().date() == start
    assert hist.index.max().date() == end
    for day in missing_dates:
        assert hist.loc[pd.Timestamp(day), "Max Temp (°C)"] is np.nan or pd.isna(
            hist.loc[pd.Timestamp(day), "Max Temp (°C)"]
        )
    assert captured["url"].endswith("/weather/quebec_city")
    assert captured["params"]["start_date"] == start.isoformat()
    assert captured["params"]["end_date"] == end.isoformat()


def test_missing_precipitation_stays_missing_in_history(monkeypatch):
    end = date(2026, 11, 1)
    start = end - timedelta(days=89)
    records = []
    for i in range(90):
        day = start + timedelta(days=i)
        records.append({
            "observation_date": day.isoformat(),
            "max_temperature": 20,
            "min_temperature": 10,
            "mean_temperature": 15,
            "heating_degree_days": 0,
            "cooling_degree_days": 0,
            "total_precipitation": None if i == 20 else 0,
        })

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return records

    monkeypatch.setattr(inference.requests, "get", lambda *args, **kwargs: Response())
    hist = make_forecaster().fetch_history("toronto", end)
    assert pd.isna(hist.iloc[20]["precip_input_mm"])
