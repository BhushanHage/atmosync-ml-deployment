from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

logger = logging.getLogger(__name__)


class CausalConv1d(nn.Module):
    """Causal convolution matching the ATMOSYNC training notebook."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.15):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=self.pad, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.pad:
            y = y[:, :, :-self.pad]
        return self.dropout(torch.nn.functional.gelu(self.norm(y)))


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.15):
        super().__init__()
        self.c1 = CausalConv1d(channels, 3, dilation, dropout)
        self.c2 = CausalConv1d(channels, 3, dilation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.c2(self.c1(x))


class MissingAwareTCN(nn.Module):
    """Architecture matching MissingAwareTCN in the Kaggle training notebook."""

    def __init__(self, n_input_channels: int, n_targets: int = 5, horizon: int = 7,
                 channels: int = 64, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Conv1d(n_input_channels, channels, kernel_size=1)
        self.blocks = nn.Sequential(*[ResidualTCNBlock(channels, d, dropout) for d in [1, 2, 4, 8]])
        self.head = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, horizon * n_targets)
        )
        self.horizon = horizon
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x.transpose(1, 2))
        z = self.blocks(z)
        return self.head(z[:, :, -1]).view(-1, self.horizon, self.n_targets)


class AtmosyncMissingRobustTCNPredictor:
    """Loads the exported ATMOSYNC missing-aware TCN and forecasts seven days."""

    def __init__(self, export_dir: str | Path, device: str = "cpu", max_missing_fraction: float = 0.60):
        self.export_dir = Path(export_dir).resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        config_path = self.export_dir / "model_config.json"
        weights_path = self.export_dir / "tcn_missing_robust_state_dict.pt"
        imputer_path = self.export_dir / "feature_imputer.joblib"
        feature_scaler_path = self.export_dir / "feature_scaler.joblib"
        target_scaler_path = self.export_dir / "target_scaler.joblib"
        for path in [config_path, weights_path, imputer_path, feature_scaler_path, target_scaler_path]:
            if not path.is_file():
                raise FileNotFoundError(f"Required model artifact not found: {path}")

        self.config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        self.model_name = str(self.config.get("model_type", "missing_aware_tcn"))
        self.features = list(self.config["features"])
        self.targets = list(self.config["targets"])
        self.lookback = int(self.config["lookback_days"])
        self.horizon = int(self.config["forecast_horizon_days"])
        self.value_feature_count = int(self.config["value_feature_count"])
        self.mask_feature_count = int(self.config["mask_feature_count"])
        self.input_channels = int(self.config["input_channels"])
        self.max_missing_fraction = float(max_missing_fraction)

        if self.value_feature_count != len(self.features) or self.mask_feature_count != len(self.features):
            raise ValueError("Model config feature/mask counts do not match the configured feature list.")
        if self.input_channels != len(self.features) * 2:
            raise ValueError(f"Expected {len(self.features) * 2} input channels, got {self.input_channels}.")
        if not 0.0 < self.max_missing_fraction <= 1.0:
            raise ValueError("max_missing_fraction must be in the interval (0, 1].")

        self.imputer = joblib.load(imputer_path)
        self.feature_scaler = joblib.load(feature_scaler_path)
        self.target_scaler = joblib.load(target_scaler_path)

        self.model = MissingAwareTCN(self.input_channels, len(self.targets), self.horizon).to(self.device)
        try:
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
        except TypeError:  # Compatibility with older PyTorch versions.
            checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        logger.info("Loaded %s from %s", self.model_name, self.export_dir)

    def _prepare_history(self, history: pd.DataFrame) -> tuple[pd.Timestamp, np.ndarray, float]:
        if not isinstance(history, pd.DataFrame):
            history = pd.DataFrame(history)
        if "observation_date" not in history.columns:
            raise ValueError("The observation_date column is required.")
        if len(history) != self.lookback:
            raise ValueError(f"Exactly {self.lookback} history rows are required.")

        frame = history.copy()
        try:
            dates = pd.to_datetime(frame["observation_date"], errors="raise", utc=True).dt.tz_convert(None).dt.normalize()
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid observation_date value.") from exc
        if dates.isna().any():
            raise ValueError("observation_date cannot contain missing values.")
        frame["observation_date"] = dates
        frame = frame.sort_values("observation_date").reset_index(drop=True)
        dates = pd.DatetimeIndex(frame["observation_date"])
        if dates.has_duplicates:
            raise ValueError("History contains duplicate dates.")
        expected = pd.date_range(end=dates[-1], periods=self.lookback, freq="D")
        if not dates.equals(expected):
            raise ValueError(f"History must contain {self.lookback} consecutive calendar days; missing dates are not synthesized.")

        # Record missingness BEFORE imputation. Missing feature columns are accepted as missing measurements.
        for feature in self.features:
            if feature not in frame.columns:
                frame[feature] = np.nan
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
        raw = frame[self.features].to_numpy(dtype=np.float64)
        raw[~np.isfinite(raw)] = np.nan
        missing_mask = ~np.isfinite(raw)
        missing_fraction = float(missing_mask.mean()) if missing_mask.size else 1.0
        if missing_fraction > self.max_missing_fraction:
            raise ValueError(
                f"Insufficient measurement coverage: {missing_fraction:.1%} of feature values are missing; "
                f"the configured maximum is {self.max_missing_fraction:.0%}."
            )

        # The notebook fits the imputer on a DataFrame and the scaler on a NumPy array.
        flat = pd.DataFrame(raw, columns=self.features)
        imputed = self.imputer.transform(flat)
        scaled = self.feature_scaler.transform(imputed)
        x = np.concatenate([scaled.astype(np.float32), missing_mask.astype(np.float32)], axis=-1)
        if x.shape != (self.lookback, self.input_channels):
            raise ValueError(f"Prepared input shape {x.shape} is invalid; expected {(self.lookback, self.input_channels)}.")
        if not np.isfinite(x).all():
            raise ValueError("Preprocessing produced non-finite model inputs.")
        return dates[-1], x[None, :, :], missing_fraction

    @torch.inference_mode()
    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        last_date, x, missing_fraction = self._prepare_history(history)
        tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self.model.eval()
        scaled_predictions = self.model(tensor).detach().cpu().numpy()
        if scaled_predictions.shape != (1, self.horizon, len(self.targets)):
            raise RuntimeError(f"Unexpected model output shape: {scaled_predictions.shape}")
        predictions = self.target_scaler.inverse_transform(
            scaled_predictions.reshape(-1, len(self.targets))
        ).reshape(self.horizon, len(self.targets))
        if not np.isfinite(predictions).all():
            raise RuntimeError("The model produced non-finite predictions.")

        result = pd.DataFrame(predictions, columns=self.targets)
        result.insert(0, "forecast_date", pd.date_range(last_date + pd.Timedelta(days=1), periods=self.horizon, freq="D"))
        result["input_missing_fraction"] = missing_fraction
        return result
