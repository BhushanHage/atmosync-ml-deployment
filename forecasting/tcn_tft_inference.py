"""
ATMOSYNC TCN-TFT inference loader
---------------------------------
Loads the exported model, configuration, and scalers produced by
5-parameters (1).ipynb. Input must contain exactly 30 consecutive daily
rows and the configured feature columns. Output is seven days x five targets.

Place this file at:
    ml_model/forecasting/tcn_tft_inference.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.15):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3,
            padding=2 * dilation, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=3,
            padding=2 * dilation, dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        z = self.conv1(x)
        z = z[:, :, :x.shape[-1]]
        z = self.dropout(torch.relu(self.norm1(z)))
        z = self.conv2(z)
        z = z[:, :, :x.shape[-1]]
        z = self.dropout(torch.relu(self.norm2(z)))
        return x + z


class TCNModel(nn.Module):
    def __init__(self, n_features, n_targets, horizon, channels=64, lookback=30):
        super().__init__()
        self.proj = nn.Conv1d(n_features, channels, 1)
        self.blocks = nn.Sequential(
            *[ResidualTCNBlock(channels, d) for d in [1, 2, 4, 8]]
        )
        self.head = nn.Sequential(
            nn.Linear(channels, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, horizon * n_targets)
        )
        self.horizon = horizon
        self.n_targets = n_targets

    def forward(self, x):
        z = x.transpose(1, 2)
        z = self.proj(z)
        z = self.blocks(z)
        return self.head(z[:, :, -1]).view(-1, self.horizon, self.n_targets)


class AttentionTemporalModel(nn.Module):
    """Architecture matching the notebook's TFT-inspired attention model."""
    def __init__(self, n_features, n_targets, horizon, d_model=64, nhead=4, lookback=30):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, lookback, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.15,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, horizon * n_targets),
        )
        self.horizon = horizon
        self.n_targets = n_targets

    def forward(self, x):
        z = self.input_proj(x) + self.pos[:, :x.shape[1], :]
        z = self.encoder(z)
        return self.head(z[:, -1]).view(-1, self.horizon, self.n_targets)


class HybridTCNTFT(nn.Module):
    def __init__(self, n_features, n_targets, horizon, lookback=30):
        super().__init__()
        self.tcn_proj = nn.Conv1d(n_features, 64, 1)
        self.tcn = nn.Sequential(
            *[ResidualTCNBlock(64, d) for d in [1, 2, 4]]
        )
        self.attn_proj = nn.Linear(n_features, 64)
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.15,
            batch_first=True,
            activation="gelu",
        )
        self.attn = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, horizon * n_targets),
        )
        self.horizon = horizon
        self.n_targets = n_targets

    def forward(self, x):
        a = self.tcn(self.tcn_proj(x.transpose(1, 2)))[:, :, -1]
        b = self.attn(self.attn_proj(x))[:, -1]
        return self.head(torch.cat([a, b], dim=-1)).view(
            -1, self.horizon, self.n_targets
        )


MODEL_BUILDERS = {
    "tcn": TCNModel,
    "tft_inspired": AttentionTemporalModel,
    "hybrid_tcn_tft": HybridTCNTFT,
}


class AtmosyncTCNTFTPredictor:
    def __init__(self, export_dir: str | Path, device: Optional[str] = None):
        self.export_dir = Path(export_dir).resolve()
        config_path = self.export_dir / "model_config.json"
        checkpoint_path = self.export_dir / "selected_model_state_dict.pt"

        if not config_path.is_file():
            raise FileNotFoundError(f"Missing model configuration: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing selected checkpoint: {checkpoint_path}")

        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.features = list(self.config["features"])
        self.targets = list(self.config["targets"])
        self.lookback = int(self.config["lookback_days"])
        self.horizon = int(self.config["forecast_horizon_days"])
        self.model_name = self.config.get(
            "selected_model", self.config.get("model_type", "")
        )

        if self.lookback != 30 or self.horizon != 7:
            raise ValueError(
                f"Expected lookback=30 and horizon=7; got "
                f"{self.lookback} and {self.horizon}."
            )
        if len(self.targets) != 5:
            raise ValueError(f"Expected five targets, found {len(self.targets)}.")
        if self.model_name not in MODEL_BUILDERS:
            raise ValueError(f"Unsupported model type in config: {self.model_name}")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        def scaler_path(key: str) -> Path:
            raw = Path(self.config[key])
            candidate = raw if raw.is_absolute() else self.export_dir / raw
            candidate = candidate.resolve()
            if not candidate.is_file():
                # Backward-compatible recovery for the old ../preprocessing paths:
                # use the scaler beside model_config.json only if it exists there.
                local_candidate = self.export_dir / raw.name
                if local_candidate.is_file():
                    candidate = local_candidate
                else:
                    raise FileNotFoundError(
                        f"Cannot resolve {key}={self.config[key]!r}. "
                        f"Checked {candidate} and {local_candidate}."
                    )
            return candidate

        self.feature_scaler = joblib.load(scaler_path("feature_scaler"))
        self.target_scaler = joblib.load(scaler_path("target_scaler"))

        self.model = MODEL_BUILDERS[self.model_name](
            n_features=len(self.features),
            n_targets=len(self.targets),
            horizon=self.horizon,
            lookback=self.lookback,
        ).to(self.device)

        # Checkpoints are generated by this project and contain a state_dict wrapper.
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
        except TypeError:  # Compatibility with older PyTorch releases.
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        """
        history: 30 consecutive daily rows, oldest first, including configured features.
        Returns a 7-row DataFrame with forecast dates and the five targets.
        """
        if "observation_date" not in history.columns:
            raise ValueError("history must include an observation_date column.")

        frame = history.copy()
        frame["observation_date"] = pd.to_datetime(
            frame["observation_date"], errors="raise"
        ).dt.normalize()
        frame = frame.sort_values("observation_date").reset_index(drop=True)

        if len(frame) != self.lookback:
            raise ValueError(
                f"Expected exactly {self.lookback} history rows; got {len(frame)}."
            )

        dates = pd.DatetimeIndex(frame["observation_date"])
        if dates.duplicated().any():
            raise ValueError("Duplicate dates found in history.")
        expected_dates = pd.date_range(dates[0], periods=self.lookback, freq="D")
        if not dates.equals(expected_dates):
            raise ValueError("History must contain 30 consecutive calendar days.")

        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing required model features: {missing}")

        raw = frame[self.features].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(raw).all():
            raise ValueError("History contains NaN or infinite feature values.")

        scaled = self.feature_scaler.transform(raw).astype(np.float32)
        batch = torch.from_numpy(scaled).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_scaled = self.model(batch).cpu().numpy()

        if pred_scaled.shape != (1, self.horizon, len(self.targets)):
            raise RuntimeError(f"Unexpected prediction shape: {pred_scaled.shape}")
        if not np.isfinite(pred_scaled).all():
            raise RuntimeError("Model produced NaN or infinite predictions.")

        pred = self.target_scaler.inverse_transform(
            pred_scaled.reshape(-1, len(self.targets))
        ).reshape(self.horizon, len(self.targets))

        forecast_dates = pd.date_range(
            dates[-1] + pd.Timedelta(days=1),
            periods=self.horizon,
            freq="D",
        )
        result = pd.DataFrame(pred, columns=self.targets)
        result.insert(0, "forecast_date", forecast_dates)
        result.insert(0, "last_observed_date", dates[-1])
        result.insert(0, "model", self.model_name)
        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ATMOSYNC TCN-TFT seven-day inference smoke test"
    )
    parser.add_argument(
        "--export-dir",
        default=str(Path(__file__).resolve().parents[1] / "models" / "TCN-TFT" / "export"),
        help="Directory containing model_config.json, scalers, and checkpoint.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Preprocessed daily CSV containing the model features.",
    )
    parser.add_argument("--station-id", required=True)
    parser.add_argument(
        "--as-of",
        required=True,
        help="Last historical date to use, YYYY-MM-DD.",
    )
    parser.add_argument("--output", default="atmosync_smoke_test_predictions.csv")
    args = parser.parse_args()

    predictor = AtmosyncTCNTFTPredictor(args.export_dir)
    data = pd.read_csv(args.csv, dtype={"station_id": str})
    data["observation_date"] = pd.to_datetime(data["observation_date"], errors="raise").dt.normalize()
    cutoff = pd.Timestamp(args.as_of).normalize()
    history = data[
        (data["station_id"].astype(str) == str(args.station_id))
        & (data["observation_date"] <= cutoff)
    ].sort_values("observation_date").tail(predictor.lookback)

    predictions = predictor.predict(history)
    predictions.to_csv(args.output, index=False)
    print("Device:", predictor.device)
    print("Model:", predictor.model_name)
    print("Input shape:", (1, predictor.lookback, len(predictor.features)))
    print("Output shape:", (1, predictor.horizon, len(predictor.targets)))
    print(predictions.to_string(index=False))
    print("Saved:", Path(args.output).resolve())
