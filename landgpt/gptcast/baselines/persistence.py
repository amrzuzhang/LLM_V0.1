"""Persistence baseline for soil moisture forecasting."""

from __future__ import annotations

from typing import Dict

import torch


class PersistenceBaseline(torch.nn.Module):
    """Propagate the last observation across the forecast horizon."""

    def __init__(self, forecast_steps: int = 1, target_channel: int = 0) -> None:
        super().__init__()
        self.forecast_steps = forecast_steps
        self.target_channel = target_channel

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = batch["input_raw"]
        if inputs.dim() != 5:
            raise ValueError("input_raw must have shape (B, T, H, W, C)")
        last_frame = inputs[:, -1:, :, :, self.target_channel : self.target_channel + 1]
        preds = last_frame.repeat(1, self.forecast_steps, 1, 1, 1)
        return preds.permute(0, 1, 4, 2, 3)


def evaluate_persistence(batch: Dict[str, torch.Tensor], target_channel: int = 0) -> Dict[str, torch.Tensor]:
    """Compute MSE/MAE for the persistence baseline."""

    model = PersistenceBaseline(
        forecast_steps=batch["target"].shape[1],
        target_channel=target_channel,
    )
    preds = model(batch)
    target = batch["target"].permute(0, 1, 4, 2, 3)
    mse = torch.mean((preds - target) ** 2)
    mae = torch.mean(torch.abs(preds - target))
    return {"mse": mse, "mae": mae}
