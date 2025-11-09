"""Latitude/longitude helper utilities."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float], Iterable[float]]


def _prepare_lat_grid(height: int, width: int, lat_grid: Optional[ArrayLike]) -> np.ndarray:
    """Return a 2-D latitude grid in degrees."""

    if lat_grid is None:
        delta = 180.0 / height
        centers = np.linspace(90.0 - delta / 2.0, -90.0 + delta / 2.0, height, dtype=np.float64)
        grid = np.repeat(centers[:, None], width, axis=1)
        return grid

    lat_arr = np.asarray(lat_grid, dtype=np.float64)
    if lat_arr.ndim == 1:
        if lat_arr.shape[0] != height:
            raise ValueError(f"Latitude vector length {lat_arr.shape[0]} does not match height {height}.")
        lat_arr = np.repeat(lat_arr[:, None], width, axis=1)
    elif lat_arr.ndim == 2:
        if lat_arr.shape != (height, width):
            raise ValueError(f"Latitude grid shape {lat_arr.shape} does not match {(height, width)}.")
    else:
        raise ValueError("Latitude grid must be 1-D or 2-D.")
    return lat_arr


def build_coslat_weights(
    height: int,
    width: int,
    mask: Optional[ArrayLike] = None,
    lat_grid: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Construct normalized cos(latitude) weights for a grid.

    Args:
        height: Number of latitude rows.
        width: Number of longitude columns.
        mask: Optional boolean/float mask where True/1 marks valid land pixels.
        lat_grid: Optional latitude values in degrees (1-D or 2-D).

    Returns:
        A float32 array of shape ``(height, width)`` whose elements sum to 1
        over valid (unmasked) pixels.
    """

    latitudes = _prepare_lat_grid(height, width, lat_grid)
    weights = np.cos(np.deg2rad(latitudes))
    weights = np.clip(weights, 0.0, None)

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != (height, width):
            raise ValueError(f"Mask shape {mask_arr.shape} does not match {(height, width)}.")
        weights = np.where(mask_arr, weights, 0.0)

    total = float(np.sum(weights))
    if total <= 0:
        if mask is not None:
            weights = np.where(mask_arr, 1.0, 0.0)
            total = float(np.sum(weights))
        if total <= 0:
            weights = np.ones((height, width), dtype=np.float32)
            total = float(np.sum(weights))
    weights = weights / total
    return weights.astype(np.float32)


__all__ = ["build_coslat_weights"]
