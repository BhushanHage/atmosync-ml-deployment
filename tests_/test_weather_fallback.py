from datetime import date

import pytest

import app as api


def test_actual_zero_is_preserved(monkeypatch):
    monkeypatch.setattr(
        api,
        "read_observations_read_only",
        lambda *_: [{
            "climate_identifier": "6158355",
            "observation_date": date(2026, 1, 1),
            "min_temperature": 0,
            "mean_temperature": None,
            "max_temperature": 2,
            "total_precipitation": None,
            "total_rain": 0,
            "total_snow": None,
        }],
    )
    monkeypatch.setattr(
        api,
        "read_saved_forecasts",
        lambda *_: [{
            "forecast_date": "2026-01-01",
            "min_temp_c": -5,
            "mean_temp_c": 1,
            "max_temp_c": 3,
            "precip_mm": 4,
        }],
    )

    rows = api.get_weather_with_forecast_fallback(
        "toronto", date(2026, 1, 1), date(2026, 1, 1)
    )
    row = rows[0]
    assert row["min_temperature"] == 0
    assert row["mean_temperature"] == 1
    assert row["max_temperature"] == 2
    assert row["total_precipitation"] == 4
    assert row["total_rain"] == 0
    assert row["field_sources"] == {
        "min_temperature": "actual",
        "mean_temperature": "forecast",
        "max_temperature": "actual",
        "total_precipitation": "forecast",
    }


def test_missing_actual_row_is_synthesized_only_with_forecast(monkeypatch):
    monkeypatch.setattr(api, "read_observations_read_only", lambda *_: [])
    monkeypatch.setattr(
        api,
        "read_saved_forecasts",
        lambda *_: [{
            "forecast_date": "2026-01-02",
            "min_temp_c": -4,
            "mean_temp_c": -1,
            "max_temp_c": 2,
            "precip_mm": 0,
        }],
    )

    rows = api.get_weather_with_forecast_fallback(
        "toronto", date(2026, 1, 1), date(2026, 1, 3)
    )
    assert len(rows) == 1
    assert rows[0]["observation_date"] == "2026-01-02"
    assert rows[0]["min_temperature"] == -4
    assert rows[0]["total_precipitation"] == 0
    assert rows[0]["total_rain"] is None
    assert set(rows[0]["field_sources"].values()) == {"forecast"}


def test_missing_actual_and_forecast_produces_no_synthetic_row(monkeypatch):
    monkeypatch.setattr(api, "read_observations_read_only", lambda *_: [])
    monkeypatch.setattr(api, "read_saved_forecasts", lambda *_: [])
    rows = api.get_weather_with_forecast_fallback(
        "toronto", date(2026, 1, 1), date(2026, 1, 3)
    )
    assert rows == []


def test_invalid_date_range_rejected():
    with pytest.raises(api.HTTPException) as exc:
        api.get_weather_with_forecast_fallback(
            "toronto", date(2026, 1, 3), date(2026, 1, 1)
        )
    assert exc.value.status_code == 400
