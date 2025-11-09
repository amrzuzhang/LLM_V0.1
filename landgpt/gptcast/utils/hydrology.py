"""Hydrology-aware regularization utilities."""

from __future__ import annotations

import torch


def _huber(residual: torch.Tensor, delta: float) -> torch.Tensor:
    abs_res = residual.abs()
    quadratic = torch.clamp(abs_res, max=delta)
    linear = abs_res - quadratic
    return 0.5 * quadratic**2 + delta * linear


def hydro_balance_penalty(
    preds: torch.Tensor,
    sm_prev: torch.Tensor,
    *,
    precip: torch.Tensor,
    precip_mask: torch.Tensor,
    evap: torch.Tensor,
    runoff: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    delta: float = 0.02,
) -> torch.Tensor:
    """Penalize violations of ΔSM ≈ P - ET - R for the first forecast step."""

    if precip_mask is None:
        return preds.new_tensor(0.0)

    delta_sm = preds[:, 0, 0] - sm_prev
    driver = precip - evap - runoff

    mask = precip_mask
    if valid_mask is not None:
        vm = valid_mask
        if vm.dim() == 2:
            vm = vm.unsqueeze(0)
        if vm.shape[0] == 1 and mask.shape[0] > 1:
            vm = vm.expand_as(mask)
        mask = mask * vm

    weight = mask
    if weight.sum() == 0:
        return preds.new_tensor(0.0)

    residual = delta_sm - driver
    loss = _huber(residual, delta) * weight
    return loss.sum() / weight.sum()


__all__ = ["hydro_balance_penalty"]
