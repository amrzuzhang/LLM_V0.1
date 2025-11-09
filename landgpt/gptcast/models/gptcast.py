"""Lightning module used for soil moisture forecasting."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import lightning as L
import torch
from torch import nn

from gptcast.models.feature_fusion import FusionModule
from gptcast.models.output import OutputModule, QuantileHead
from gptcast.utils.hydrology import hydro_balance_penalty


class SoilMoistureGPTCast(L.LightningModule):
    """Encoder-transformer-regressor tailored to LandBench soil moisture."""

    def __init__(
        self,
        *,
        in_channels: int,
        input_steps: int,
        forecast_steps: int,
        target_channel: int = 0,
        cnn_channels: int = 64,
        fusion_channels: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_hyd: float = 0.0,
        use_area_weighted_loss: bool = False,
        area_weights: Optional[torch.Tensor] = None,
        uncertainty: str = "none",
        quantiles: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["area_weights"])

        self.in_channels = in_channels
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.target_channel = target_channel
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_hyd = lambda_hyd
        self.use_area_weighted_loss = use_area_weighted_loss
        self.uncertainty = (uncertainty or "none").lower()
        if self.uncertainty not in {"none", "quantile"}:
            raise ValueError("uncertainty must be 'none' or 'quantile'.")

        aux_channels = max(in_channels - 1, 0)

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.fusion = FusionModule(cnn_channels, aux_channels, fusion_channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_channels,
            nhead=transformer_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        if self.uncertainty == "quantile":
            quantile_levels = quantiles or [0.05, 0.5, 0.95]
            quantile_levels = sorted(float(q) for q in quantile_levels)
            self.quantile_levels = tuple(quantile_levels)
            quant_tensor = torch.tensor(self.quantile_levels, dtype=torch.float32).view(1, 1, len(self.quantile_levels), 1, 1)
            self.register_buffer("quantiles_tensor", quant_tensor)
            if 0.5 in self.quantile_levels:
                self.median_index = self.quantile_levels.index(0.5)
            else:
                self.median_index = len(self.quantile_levels) // 2
            self.output_head = QuantileHead(fusion_channels, forecast_steps, self.quantile_levels)
        else:
            self.quantile_levels = None
            self.quantiles_tensor = None  # type: ignore[assignment]
            self.median_index = 0
            self.output_head = OutputModule(fusion_channels, forecast_steps)
            self.loss_fn = nn.MSELoss()

        if area_weights is not None:
            area_tensor = torch.as_tensor(area_weights, dtype=torch.float32)
            total = torch.sum(area_tensor)
            if total > 0:
                area_tensor = area_tensor / total
            self.register_buffer("area_weights", area_tensor)
        else:
            self.area_weights = None

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the forward pass for a batch produced by ``SoilMoistureDataset``."""

        inputs = batch["input_raw"].permute(0, 1, 4, 2, 3).contiguous()
        aux_indices = [idx for idx in range(self.in_channels) if idx != self.target_channel]

        primary = inputs[:, :, self.target_channel : self.target_channel + 1]
        aux = inputs[:, :, aux_indices] if aux_indices else None

        batch_size, seq_len, _, height, width = primary.shape
        encoded = self.frame_encoder(primary.view(batch_size * seq_len, 1, height, width))
        encoded = encoded.view(batch_size, seq_len, -1, height, width)
        fused = self.fusion(encoded, aux)

        fused_flat = fused.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, seq_len, -1)
        encoded_seq = self.temporal_encoder(fused_flat)
        encoded_last = encoded_seq[:, -1, :].view(batch_size, height, width, -1).permute(0, 3, 1, 2)
        encoded_last = encoded_last.unsqueeze(1)

        return self.output_head(encoded_last)

    def _step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        preds = self.forward(batch)
        target = batch["target"].permute(0, 1, 4, 2, 3).contiguous()
        point_preds = self._point_forecast(preds)
        diff_point = point_preds - target
        mse_unweighted = torch.mean(diff_point ** 2)
        mse_area_weighted = self._area_weighted_mse(diff_point ** 2)
        if self.uncertainty == "quantile":
            loss = self._quantile_loss(preds, target)
        else:
            loss = (
                mse_area_weighted
                if self.use_area_weighted_loss and self.area_weights is not None
                else mse_unweighted
            )
        mae = torch.mean(torch.abs(diff_point))
        hydro_penalty = preds.new_tensor(0.0)
        if self.lambda_hyd > 0.0 and "hydro" in batch:
            hydro = batch["hydro"]
            hydro_penalty = hydro_balance_penalty(
                point_preds,
                hydro["sm_prev"],
                precip=hydro["precip"],
                precip_mask=hydro["precip_mask"],
                evap=hydro["evap"],
                runoff=hydro["runoff"],
                valid_mask=hydro["valid_mask"],
            )
            loss = loss + self.lambda_hyd * hydro_penalty

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, batch_size=preds.size(0))
        self.log(f"{stage}_mae", mae, prog_bar=False, on_epoch=True, batch_size=preds.size(0))
        self.log(f"{stage}_mse_unweighted", mse_unweighted, prog_bar=False, on_epoch=True, batch_size=preds.size(0))
        self.log(
            f"{stage}_mse_area_weighted",
            mse_area_weighted,
            prog_bar=False,
            on_epoch=True,
            batch_size=preds.size(0),
        )
        if self.lambda_hyd > 0.0:
            self.log(
                f"{stage}_hydro_penalty",
                hydro_penalty,
                prog_bar=False,
                on_epoch=True,
                batch_size=preds.size(0),
            )
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def _area_weighted_mse(self, squared_errors: torch.Tensor) -> torch.Tensor:
        if self.area_weights is None:
            return torch.mean(squared_errors)
        height, width = self.area_weights.shape
        spatial_weights = self.area_weights.view(1, 1, 1, height, width)
        weighted = (squared_errors * spatial_weights).sum(dim=(-1, -2))
        return weighted.mean()

    def _point_forecast(self, preds: torch.Tensor) -> torch.Tensor:
        if self.uncertainty == "quantile":
            return preds[:, :, self.median_index : self.median_index + 1, ...]
        return preds

    def _quantile_loss(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.quantiles_tensor is None:
            raise RuntimeError("Quantile tensor is not initialized.")
        diffs = target.unsqueeze(2) - preds
        quantiles = self.quantiles_tensor
        pinball = torch.where(diffs >= 0, quantiles * diffs, (quantiles - 1.0) * diffs)
        if self.use_area_weighted_loss and self.area_weights is not None:
            height, width = self.area_weights.shape
            spatial = self.area_weights.view(1, 1, 1, height, width)
            pinball = (pinball * spatial).sum(dim=(-1, -2))
            return pinball.mean()
        return pinball.mean()
