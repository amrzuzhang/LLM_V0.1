"""Feature fusion block used by the soil moisture model."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class FusionModule(nn.Module):
    """Fuse CNN features with auxiliary predictors via lightweight projections."""

    def __init__(self, cnn_channels: int, aux_channels: int, out_channels: int) -> None:
        super().__init__()
        self.cnn_project = nn.Conv2d(cnn_channels, out_channels, kernel_size=1)
        self.aux_project: Optional[nn.Conv2d]
        if aux_channels > 0:
            self.aux_project = nn.Conv2d(aux_channels, out_channels, kernel_size=1)
            fused_channels = out_channels * 2
        else:
            self.aux_project = None
            fused_channels = out_channels
        self.post_project = nn.Conv2d(fused_channels, out_channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, cnn_features: torch.Tensor, aux_features: Optional[torch.Tensor]) -> torch.Tensor:
        """Project and optionally concatenate CNN and auxiliary channels."""

        batch_size, seq_len, _, height, width = cnn_features.shape
        cnn_flat = cnn_features.view(batch_size * seq_len, -1, height, width)
        cnn_proj = self.cnn_project(cnn_flat)

        if self.aux_project is not None and aux_features is not None:
            aux_flat = aux_features.view(batch_size * seq_len, -1, height, width)
            aux_proj = self.aux_project(aux_flat)
            fused = torch.cat([cnn_proj, aux_proj], dim=1)
        else:
            fused = cnn_proj

        fused = self.post_project(self.activation(fused))
        return fused.view(batch_size, seq_len, -1, height, width)
