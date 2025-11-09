from __future__ import annotations

from pathlib import Path
from typing import Union


Numeric = Union[int, float, str]


def resolution_dirname(spatial_resolution: Numeric) -> str:
    """Return a canonical folder-friendly resolution label."""
    try:
        value = float(spatial_resolution)
    except (TypeError, ValueError):
        return str(spatial_resolution)
    if value.is_integer():
        return str(int(value))
    return str(spatial_resolution)


def resolve_root(base: Union[str, Path], product: str, spatial_resolution: Numeric) -> Path:
    """Return the directory that stores artifacts for a product & resolution."""

    base_path = Path(base)
    resolution_folder = resolution_dirname(spatial_resolution)
    candidates = []
    if product:
        candidates.append(base_path / product / resolution_folder)
    candidates.append(base_path / resolution_folder)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # fallback to the first candidate even if it does not exist (legacy behaviour)
    return candidates[0]


def mask_filename(spatial_resolution: Numeric) -> str:
    return f"Mask with {resolution_dirname(spatial_resolution)} spatial resolution.npy"


def resolve_mask_path(base: Union[str, Path], spatial_resolution: Numeric) -> Path:
    base_path = Path(base)
    canonical = base_path / mask_filename(spatial_resolution)
    if canonical.exists():
        return canonical
    legacy = base_path / f"Mask with {spatial_resolution} spatial resolution.npy"
    if legacy.exists():
        return legacy
    return canonical


def lat_filename(spatial_resolution: Numeric) -> str:
    return f"lat_{resolution_dirname(spatial_resolution)}.npy"


def lon_filename(spatial_resolution: Numeric) -> str:
    return f"lon_{resolution_dirname(spatial_resolution)}.npy"
