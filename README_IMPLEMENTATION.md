# ATMOSYNC LightGBM forecast API implementation package

## Implemented in this package
- Keeps the legacy `GET /`, `GET /health`, and `POST /predict` route shapes; `/predict` remains the existing 30-row TCN route when the original TCN export is deployed.
- Adds LightGBM inference from `models/Light_GBM/lightgbm_models.joblib` plus `inference_metadata.json`.
- Adds `GET /forecasts` to read saved forecast rows from Supabase.
- Adds token-protected `POST /forecasts/refresh` to generate seven days for all nine stations and upsert them into the new table.
- Adds SQL for a new `weather_forecasts` table only. It does not modify other tables.

## Required Render environment variables
- `SUPABASE_URL`: project URL.
- `SUPABASE_SERVICE_ROLE_KEY`: server-side service-role key. Never expose it to the frontend.
- `SUPABASE_FORECASTS_TABLE`: optional; defaults to `weather_forecasts`.
- `ATMOSYNC_REFRESH_TOKEN`: secret required by `POST /forecasts/refresh`.
- `ATMOSYNC_WEATHER_API_BASE_URL`: optional; defaults to `https://atmosyncv2-0.onrender.com`.
- `ATMOSYNC_LIGHTGBM_MODEL_DIR`: optional; defaults to `models/Light_GBM`.
- Existing `ATMOSYNC_MODEL_EXPORT_DIR` and `ATMOSYNC_DEVICE` continue to control legacy TCN loading.

## Deployment checklist
1. Copy the package files into the ML API repository, retaining the `forecasting/` package and original TCN export if `/predict` must remain operational.
2. Run `supabase_new_table.sql` in Supabase SQL editor. It creates a new table only; do not run if the table already exists without reviewing its schema.
3. Set the Render environment variables above. Do not commit secrets.
4. Deploy the ML API service. This package does not deploy itself or apply SQL remotely because no Render/Supabase credentials are available in this workspace.
5. Test `/health`, `/docs`, `GET /forecasts`, and authorized `POST /forecasts/refresh`.

## Important model limitations
- LightGBM predicts temperature and **total precipitation** only. It does not separately predict rainfall and snowfall. Do not rename `predicted_precip_mm` as rain or snow.
- Inputs are 90 calendar days and forecast horizon is 7 days. The legacy TCN `/predict` contract remains 30 rows.
- Missing input values use the training metadata's feature medians, matching the supplied local inference implementation. Missing precipitation is not silently treated as measured zero; it may be replaced by the model's training median and should be understood as imputed input.
- Output temperatures are sorted to preserve min <= mean <= max, and precipitation is constrained to nonnegative values.
- A forecast refresh requires working access to the historical API and Supabase.
