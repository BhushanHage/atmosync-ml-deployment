# ATMOSYNC Render API starter

These files are intended to be copied into the existing `ml_model` directory.
Do not replace or remove the exported model files.

## Files
- `app.py`: FastAPI wrapper around `forecasting/tcn_tft_inference.py`
- `requirements.txt`: runtime dependencies
- `Procfile`: Render web start command

## Local test (from the `ml_model` directory)
```powershell
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/health` and `http://127.0.0.1:8000/docs`.

## Render settings
- Runtime: Python
- Root Directory: `ml_model` if the Git repository contains this directory as a subdirectory; otherwise use the repository root where these files and the model directories live.
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

Keep these paths in the deployed source:
- `forecasting/tcn_tft_inference.py`
- `models/TCN-TFT/export/model_config.json`
- `models/TCN-TFT/export/selected_model_state_dict.pt`
- `models/TCN-TFT/export/feature_scaler.joblib`
- `models/TCN-TFT/export/target_scaler.joblib`

Do not deploy `.venv` folders.

## Prediction request
POST `/predict` with JSON:
```json
{
  "history": [
    {
      "observation_date": "2026-09-01",
      "air_temperature_mean": 12.3,
      "air_temperature_max": 16.0,
      "air_temperature_min": 8.1,
      "dew_point_temperature": 6.0,
      "relative_humidity": 70.0,
      "wind_speed": 3.2,
      "surface_pressure": 100.8,
      "solar_radiation": 15.0,
      "month": 9,
      "day_of_year": 244,
      "seasonal_cycle_sin": -0.99,
      "seasonal_cycle_cos": -0.10,
      "temperature_imputed_flag": 0
    }
  ]
}
```
The `history` array must contain exactly 30 consecutive daily rows. The example above illustrates the row shape only; send 30 actual rows and all feature names from `model_config.json`. Do not use the example values as real weather data.

The endpoint validates required features and relies on the predictor to reject missing/non-finite values or nonconsecutive dates. It does not fetch city weather yet; that integration remains a separate step because the existing weather API response needs missing inputs handled first.

Rainfall output unit: mm. Snowfall output unit: cm. Temperature output unit: °C.
