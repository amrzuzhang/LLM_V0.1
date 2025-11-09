"""Calibration utilities for probabilistic soil moisture forecasts."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _ensure_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Quantile levels must be a 1-D sequence.")
    if not np.all(np.diff(arr) >= 0):
        raise ValueError("Quantile levels must be sorted in ascending order.")
    return arr


def _pinball_losses(observations: np.ndarray, quantile_levels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    diff = observations[..., None] - predictions
    losses = np.where(diff >= 0, quantile_levels * diff, (quantile_levels - 1.0) * diff)
    return losses


def crps_from_quantiles(
    observations: np.ndarray,
    quantile_levels: Iterable[float],
    predictions: np.ndarray,
) -> np.ndarray:
    """Approximate CRPS by integrating the pinball loss over the provided quantiles."""

    tau = _ensure_array(quantile_levels)
    obs = np.asarray(observations, dtype=np.float64)
    preds = np.asarray(predictions, dtype=np.float64)
    if preds.shape[-1] != tau.shape[0]:
        raise ValueError("Last dimension of predictions must match number of quantile levels.")

    pinball = _pinball_losses(obs, tau, preds)
    crps = np.zeros_like(obs, dtype=np.float64)
    for idx in range(len(tau) - 1):
        weight = tau[idx + 1] - tau[idx]
        segment = 0.5 * (pinball[..., idx] + pinball[..., idx + 1])
        crps += weight * segment
    return 2.0 * crps


def pit_histogram(
    observations: np.ndarray,
    quantile_levels: Iterable[float],
    predictions: np.ndarray,
    bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PIT values and histogram counts for quantile forecasts."""

    tau = _ensure_array(quantile_levels)
    obs = np.asarray(observations, dtype=np.float64).ravel()
    preds = np.asarray(predictions, dtype=np.float64).reshape(obs.shape[0], tau.shape[0])

    tau_ext = np.concatenate(([0.0], tau, [1.0]))
    pit_values = np.empty_like(obs)
    for idx, y in enumerate(obs):
        q = preds[idx]
        q_ext = np.concatenate(([q[0] - 1.0], q, [q[-1] + 1.0]))
        interval = np.searchsorted(q_ext, y, side="right") - 1
        interval = np.clip(interval, 0, len(q_ext) - 2)
        q_low, q_high = q_ext[interval], q_ext[interval + 1]
        tau_low, tau_high = tau_ext[interval], tau_ext[interval + 1]
        if q_high > q_low:
            frac = (y - q_low) / (q_high - q_low)
            pit_values[idx] = tau_low + frac * (tau_high - tau_low)
        else:
            pit_values[idx] = tau_high
    pit_values = np.clip(pit_values, 0.0, 1.0)
    hist, edges = np.histogram(pit_values, bins=bins, range=(0.0, 1.0))
    return hist, edges, pit_values


def reliability_curve(
    observations: np.ndarray,
    quantile_levels: Iterable[float],
    predictions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return nominal vs empirical coverage for each quantile."""

    tau = _ensure_array(quantile_levels)
    obs = np.asarray(observations, dtype=np.float64).reshape(-1, 1)
    preds = np.asarray(predictions, dtype=np.float64).reshape(obs.shape[0], tau.shape[0])
    empirical = (obs <= preds).mean(axis=0)
    return tau, empirical


__all__ = ["crps_from_quantiles", "pit_histogram", "reliability_curve"]
