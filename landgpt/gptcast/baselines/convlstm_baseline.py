"""ConvLSTM baseline implementation for soil moisture forecasting."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn


def _pair(value: int | Tuple[int, int]) -> Tuple[int, int]:
    """Normalise integer or tuple kernel sizes into a 2-tuple."""

    if isinstance(value, tuple):
        return value
    return value, value


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell adapted from the LandBench reference implementation."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int | Tuple[int, int], bias: bool = True):
        super().__init__()
        k_h, k_w = _pair(kernel_size)
        padding = k_h // 2, k_w // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=(k_h, k_w),
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur

    def init_state(self, batch_size: int, spatial_size: Tuple[int, int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = spatial_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return h, c


class ConvLSTM(nn.Module):
    """Stacked ConvLSTM encoder used by the baseline."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        kernel_size: int | Tuple[int, int] = 3,
        bias: bool = True,
        return_all_layers: bool = False,
    ) -> None:
        super().__init__()
        hidden_dims = tuple(hidden_dims)
        self.return_all_layers = return_all_layers
        cell_list = []
        for i, hidden_dim in enumerate(hidden_dims):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cell_list.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size, bias=bias))
        self.cells = nn.ModuleList(cell_list)

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Sequence[Tuple[torch.Tensor, torch.Tensor]]]:
        batch, seq_len, _, height, width = x.shape
        device = x.device

        if hidden_states is None:
            hidden_states = [cell.init_state(batch, (height, width), device) for cell in self.cells]

        cur_input = x
        new_states = []
        outputs = []

        for cell, state in zip(self.cells, hidden_states):
            h, c = state
            out = []
            for t in range(seq_len):
                h, c = cell(cur_input[:, t, ...], (h, c))
                out.append(h)
            out_stack = torch.stack(out, dim=1)
            outputs.append(out_stack)
            new_states.append((h, c))
            cur_input = out_stack

        if self.return_all_layers:
            return outputs, new_states
        return outputs[-1], new_states


class ConvLSTMBaseline(nn.Module):
    """Baseline ConvLSTM forecaster compatible with :class:`SoilMoistureDataset`."""

    def __init__(
        self,
        input_channels: int,
        forecast_steps: int,
        hidden_dims: Sequence[int] = (32, 32),
        kernel_size: int | Tuple[int, int] = 3,
        bias: bool = True,
        target_channel: int = 0,
        use_all_features: bool = True,
    ) -> None:
        super().__init__()
        self.forecast_steps = forecast_steps
        self.target_channel = target_channel
        self.use_all_features = use_all_features
        effective_channels = input_channels if use_all_features else 1
        self.encoder = ConvLSTM(
            input_dim=effective_channels,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            bias=bias,
            return_all_layers=False,
        )
        last_hidden = hidden_dims[-1]
        self.readout = nn.Conv2d(last_hidden, forecast_steps, kernel_size=1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = batch["input_raw"]
        if inputs.dim() != 5:
            raise ValueError("input_raw must have shape (B, T, H, W, C)")
        inputs = inputs.permute(0, 1, 4, 2, 3).contiguous()
        if not self.use_all_features:
            inputs = inputs[:, :, self.target_channel : self.target_channel + 1]
        outputs, _ = self.encoder(inputs)
        last_state = outputs[:, -1]
        preds = self.readout(last_state)
        return preds.unsqueeze(2)
