"""Utilities to load exogenous (future) variables such as GraphCast/AIFS/FourCastNet."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


class ExogReader:
    """Best-effort loader that returns future driver fields for requested lead times."""

    def __init__(
        self,
        root: Path,
        source: str,
        variables: Sequence[str],
        lead_hours: Sequence[int],
        lat_grid: Optional[np.ndarray],
        lon_grid: Optional[np.ndarray],
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.source = source
        self.variables: List[str] = list(variables or [])
        self.lead_hours: List[int] = sorted({int(abs(lead)) for lead in lead_hours})
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self._datasets: Dict[str, np.ndarray] = {}
        self._warned_missing = False

        base = self.root / self.source if self.source else self.root
        for var in self.variables:
            dataset = self._open_dataset(base, var)
            if dataset is not None:
                self._datasets[var] = dataset

        if not self._datasets and not self._warned_missing:
            warnings.warn(
                f"[ExogReader] No external datasets found under {base}. Falling back to all-zero tensors.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_missing = True

    def load(self, t_index: int, height: int, width: int) -> np.ndarray:
        """Return array shaped (num_leads, height, width, num_vars)."""

        tensor = np.zeros(
            (len(self.lead_hours), height, width, len(self.variables)),
            dtype=np.float32,
        )
        for var_idx, var in enumerate(self.variables):
            dataset = self._datasets.get(var)
            if dataset is None:
                continue
            for lead_idx, lead in enumerate(self.lead_hours):
                slice_ = self._extract_frame(dataset, t_index, lead)
                if slice_ is None:
                    continue
                if slice_.shape != (height, width):
                    slice_ = self._resize_if_possible(slice_, height, width)
                    if slice_ is None:
                        continue
                tensor[lead_idx, :, :, var_idx] = slice_.astype(np.float32, copy=False)
        return tensor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _open_dataset(self, base: Path, variable: str) -> Optional[np.ndarray]:
        """Try several filename conventions for external drivers."""

        candidates = [
            base / f"{variable}.npy",
            base / f"{variable}.npz",
            base / variable / "data.npy",
            base / variable / "data.npz",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                if path.suffix == ".npz":
                    npz = np.load(path)
                    # Pick the first array-like entry
                    for key in npz.files:
                        data = npz[key]
                        if isinstance(data, np.ndarray):
                            return data
                else:
                    return np.load(path, mmap_mode="r")
            except Exception:
                continue
        return None

    def _extract_frame(self, dataset: np.ndarray, t_index: int, lead_hour: int) -> Optional[np.ndarray]:
        """Pick the requested time/lead slice if possible."""

        try:
            if dataset.ndim == 4:
                # (time, lead, lat, lon)
                lead_axis = 1
            elif dataset.ndim == 3:
                # (time, lat, lon) – treat as lead=0 only
                lead_axis = None
            else:
                return None

            if lead_axis is None:
                if 0 <= t_index < dataset.shape[0]:
                    return dataset[t_index]
                return None

            lead_indices = self._match_lead_indices(dataset.shape[lead_axis])
            lead_idx = lead_indices.get(lead_hour)
            if lead_idx is None:
                return None

            if 0 <= t_index < dataset.shape[0]:
                return dataset[t_index, lead_idx]
            return None
        except Exception:
            return None

    def _match_lead_indices(self, lead_dim: int) -> Dict[int, int]:
        """Map requested lead hours to dataset indices (assume evenly spaced)."""

        if lead_dim <= 0:
            return {}
        # Assume dataset provides equally spaced leads and matches sorted order.
        mapping = {}
        for idx, lead in enumerate(self.lead_hours):
            if idx < lead_dim:
                mapping[lead] = idx
        return mapping

    def _resize_if_possible(self, data: np.ndarray, height: int, width: int) -> Optional[np.ndarray]:
        """Resample data via simple nearest-neighbour if sizes mismatch."""

        try:
            return np.asarray(
                np.resize(data, (height, width)),
                dtype=np.float32,
            )
        except Exception:
            if not self._warned_missing:
                warnings.warn(
                    "[ExogReader] Unable to resize exogenous grid; returning zeros.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_missing = True
            return None


def summarize_exog(source: str, variables: Sequence[str], lead_hours: Sequence[int]) -> str:
    if not variables or not lead_hours:
        return ""
    lead_str = ",".join(str(h) for h in sorted({int(abs(h)) for h in lead_hours}))
    var_str = ",".join(variables)
    return f"{source}: vars={var_str} | lead_hours={lead_str}"


__all__ = ["ExogReader", "summarize_exog"]
