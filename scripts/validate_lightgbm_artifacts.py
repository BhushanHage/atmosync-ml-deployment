import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Light_GBM"

MODEL_PATH = MODEL_DIR / "lightgbm_models.joblib"
METADATA_PATH = MODEL_DIR / "inference_metadata.json"


def main():
    for path in (MODEL_PATH, METADATA_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing artifact: {path}")

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )
    models = joblib.load(MODEL_PATH)

    print("=== ARTIFACT VALIDATION ===")
    print(f"Model container: {type(models).__name__}")
    print(f"Estimator count: {len(models)}")
    print(f"Selected model: {metadata.get('selected_model')}")
    print(f"Lookback days: {metadata.get('lookback_days')}")
    print(f"Forecast horizon: {metadata.get('forecast_horizon_days')}")
    print(f"Features: {metadata.get('features')}")
    print(f"Targets: {metadata.get('target_names')}")

    if not isinstance(models, (list, tuple)):
        raise TypeError("Expected a list or tuple of estimators.")

    if len(models) != 28:
        raise ValueError(
            f"Expected 28 estimators, found {len(models)}."
        )

    if metadata.get("lookback_days") != 90:
        raise ValueError("Expected a 90-day lookback.")

    if metadata.get("forecast_horizon_days") != 7:
        raise ValueError("Expected a seven-day forecast horizon.")

    features = metadata.get("features", [])
    targets = metadata.get("target_names", [])

    if len(features) != 10:
        raise ValueError(
            f"Expected 10 daily features, found {len(features)}."
        )

    if len(targets) != 4:
        raise ValueError(
            f"Expected four targets, found {len(targets)}."
        )

    expected_input_size = 90 * 10 + 1

    print(f"\nExpected estimator input size: {expected_input_size}")
    print("\n=== INDIVIDUAL ESTIMATORS ===")

    for index, model in enumerate(models):
        if model is None:
            print(f"{index}: MISSING")
            continue

        if not hasattr(model, "predict"):
            raise TypeError(
                f"Estimator {index} has no predict() method."
            )

        actual_size = getattr(model, "n_features_in_", None)

        if actual_size is not None and actual_size != expected_input_size:
            raise ValueError(
                f"Estimator {index}: expected {expected_input_size} "
                f"inputs, found {actual_size}."
            )

        print(
            f"{index}: {type(model).__name__}, "
            f"input_size={actual_size}"
        )

    for key in (
        "feature_medians",
        "feature_mean",
        "feature_std",
    ):
        values = np.asarray(metadata.get(key, []), dtype=float)

        if values.shape != (10,):
            raise ValueError(
                f"{key} must contain exactly 10 values."
            )

        if not np.isfinite(values).all():
            raise ValueError(f"{key} contains invalid values.")

    for key in ("target_mean", "target_std"):
        values = np.asarray(metadata.get(key, []), dtype=float)

        if values.shape != (4,):
            raise ValueError(
                f"{key} must contain exactly four values."
            )

        if not np.isfinite(values).all():
            raise ValueError(f"{key} contains invalid values.")

    target_std = np.asarray(metadata["target_std"], dtype=float)

    if np.any(target_std <= 0):
        raise ValueError("Target standard deviations must be positive.")

    station_map = metadata.get("station_to_index", {})

    if not station_map:
        raise ValueError("Station mapping is missing.")

    print(f"\nStations configured: {len(station_map)}")
    print("\nPASS: Artifact structure and metadata checks succeeded.")


if __name__ == "__main__":
    main()