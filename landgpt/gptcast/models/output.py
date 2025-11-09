"""Regression head used by the soil moisture model."""

from __future__ import annotations

import torch
from torch import nn


class OutputModule(nn.Module):
    """Regression head that maps fused features to deterministic soil moisture forecasts."""

    def __init__(self, in_channels: int, forecast_steps: int) -> None:
        super().__init__()
        self.forecast_steps = forecast_steps
        hidden = max(in_channels, 32)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, forecast_steps, kernel_size=1),
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """Use the final fused frame to produce a horizon of predictions."""

        last_state = fused_features[:, -1]
        preds = self.head(last_state)
        return preds.unsqueeze(2)


class QuantileHead(nn.Module):
    """Output head that predicts multiple quantiles per forecast step."""

    def __init__(self, in_channels: int, forecast_steps: int, quantiles: list[float]) -> None:
        super().__init__()
        if not quantiles:
            raise ValueError("QuantileHead requires a non-empty list of quantiles.")
        self.forecast_steps = forecast_steps
        self.quantiles = list(quantiles)
        hidden = max(in_channels, 32)
        out_channels = forecast_steps * len(self.quantiles)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """Produce quantile forecasts for each horizon."""

        last_state = fused_features[:, -1]
        raw = self.head(last_state)
        batch, _, height, width = raw.shape
        raw = raw.view(batch, self.forecast_steps, len(self.quantiles), height, width)
        return raw
