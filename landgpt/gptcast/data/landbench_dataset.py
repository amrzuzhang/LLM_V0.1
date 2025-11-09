"""Dataset utilities for reading LandBench preprocessing outputs.

LandBench's official pipeline (see ``landbench/src/data.py``) converts the raw
NetCDF archives into ``x_train_norm.npy``, ``x_test_norm.npy``, ``static_norm.npy``
and other memmap files. This dataset simply streams those files and builds the
``(time, height, width, channels)`` sliding windows expected by GPTCast.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from gptcast.config import CONFIG
from gptcast.utils.latlon_weights import build_coslat_weights


@dataclass
class SoilMoistureSample:
    """Metadata describing a single temporal sample window."""

    start_idx: int
    input_slice: slice
    target_slice: slice


class SoilMoistureDataset(Dataset):
    """Dataset backed by LandBench memmap outputs."""

    def __init__(
        self,
        root: str,
        subset: str = "train",
        input_steps: int = 6,
        forecast_steps: int = 3,
        stride: int = 1,
        normalize: bool = False,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"LandBench root directory not found: {self.root}")

        subset = subset.lower()
        if subset not in {"train", "test"}:
            raise ValueError("subset must be 'train' or 'test'")
        self.subset = subset

        self.input_steps = int(input_steps)
        self.forecast_steps = int(forecast_steps)
        self.forecast_offset = self.input_steps
        self.stride = int(stride)

        self.data_dir = self._resolve_data_dir(self.root)
        print(f"Using LandBench preprocessed directory: {self.data_dir}")

        self._dynamic_data = self._load_dynamic_array(self.data_dir, subset)
        self._target_data = self._load_target_array(self.data_dir, subset)
        self._static_data = self._load_static_array(self.data_dir)
        self._height = self._dynamic_data.shape[1]
        self._width = self._dynamic_data.shape[2]
        self._mask = self._load_mask(self.data_dir)
        self._mask_broadcast = self._compute_mask_broadcast(self._mask)
        self._lat_grid = self._load_lat_grid(self.data_dir, self._height, self._width)
        self._static_features = self._prepare_static_features(self._static_data, self._mask)
        self._area_weights = build_coslat_weights(
            self._height,
            self._width,
            mask=self._mask,
            lat_grid=self._lat_grid,
        )
        self._target_channel = 0  # soil moisture channel produced by LandBench
        self._normalization = None
        self._valid_mask = (
            self._mask.astype(np.float32) if self._mask is not None else np.ones((self._height, self._width), dtype=np.float32)
        )
        self._valid_mask_tensor = torch.from_numpy(self._valid_mask)

        self._forcing_names = self._load_variable_names("forcing", self._default_variable_list("forcing_list"))
        self._land_surface_names = self._load_variable_names("land_surface", self._default_variable_list("land_surface_list"))
        self._channel_lookup = self._build_channel_lookup()
        self._component_indices = self._resolve_component_indices()

        self._samples = self._build_index()
        print(f"Loaded {len(self)} sliding windows from preprocessed data.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self._samples[index]

        inputs = np.array(self._dynamic_data[sample.input_slice], copy=False)
        targets = np.array(self._target_data[sample.target_slice], copy=False)
        inputs = self._apply_mask(inputs)
        targets = self._apply_mask(targets)
        inputs = np.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)
        targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

        if self._static_data is not None:
            inputs = np.concatenate(
                [inputs, np.repeat(self._static_features[None, ...], inputs.shape[0], axis=0)],
                axis=-1,
            )

        inputs_tensor = torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32))
        targets_tensor = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32))

        batch_dict: Dict[str, torch.Tensor] = {
            "input_raw": inputs_tensor,
            "target": targets_tensor,
        }
        batch_dict["hydro"] = {
            key: torch.from_numpy(value.astype(np.float32))
            for key, value in self._build_hydro_features(sample).items()
        }
        batch_dict["hydro"]["valid_mask"] = self._valid_mask_tensor
        return batch_dict

    def _build_index(self) -> List[SoilMoistureSample]:
        num_timesteps = self._dynamic_data.shape[0]
        samples: List[SoilMoistureSample] = []
        for start in range(0, num_timesteps - self.input_steps - self.forecast_steps + 1, self.stride):
            input_slice = slice(start, start + self.input_steps)
            target_slice = slice(start + self.forecast_offset, start + self.forecast_offset + self.forecast_steps)
            samples.append(SoilMoistureSample(start, input_slice, target_slice))
        if not samples:
            raise ValueError("No valid samples could be generated with the given configuration.")
        print(f"Built {len(samples)} sliding windows.")
        return samples

    def _load_or_fit_normalization(self) -> Optional[Dict[str, np.ndarray]]:
        return None

    def _apply_normalization(self) -> None:
        return

    @property
    def target_channel(self) -> int:
        return self._target_channel

    @property
    def feature_dim(self) -> int:
        total = self._dynamic_data.shape[-1]
        if self._static_data is not None:
            total += self._static_data.shape[-1]
        return total

    @property
    def area_weights(self) -> np.ndarray:
        return self._area_weights

    @property
    def valid_mask(self) -> np.ndarray:
        return self._valid_mask

    def _resolve_data_dir(self, path: Path) -> Path:
        """Locate the directory that contains LandBench memmap outputs."""

        expected = path / "x_train_norm.npy"
        if expected.exists():
            return path
        candidates = list(path.glob("*/x_train_norm.npy"))
        if len(candidates) == 1:
            return candidates[0].parent
        if len(candidates) > 1:
            raise ValueError(
                f"Found multiple preprocessed directories under {path}: "
                f"{', '.join(str(p.parent) for p in candidates)}. "
                "Point --nc-root directly to the desired resolution directory."
            )
        raise FileNotFoundError(
            f"x_train_norm.npy not found under {path}. "
            "Run landbench/src/data.py to generate preprocessed files first."
        )

    def _load_dynamic_array(self, data_dir: Path, subset: str) -> np.ndarray:
        """Load memmap array for the requested subset (train/test)."""

        fname = data_dir / f"x_{subset}_norm.npy"
        if not fname.exists():
            raise FileNotFoundError(f"Missing memmap file: {fname}")

        shape_path = data_dir / f"x_{subset}_norm_shape.npy"
        if shape_path.exists():
            shape = tuple(int(dim) for dim in np.load(shape_path))
            arr = np.memmap(fname, dtype=np.float32, mode="r", shape=shape)
            print(f"{subset} dynamic memmap shape: {arr.shape}")
            return arr

        arr = np.load(fname, mmap_mode="r")
        print(f"{subset} dynamic array shape: {arr.shape}")
        return arr

    def _load_target_array(self, data_dir: Path, subset: str) -> np.ndarray:
        """Load soil-moisture targets for the requested subset."""

        fname = data_dir / f"y_{subset}_norm.npy"
        if not fname.exists():
            raise FileNotFoundError(
                f"Missing target memmap file: {fname}. Ensure LandBench preprocessing generated y_{subset}_norm.npy."
            )

        shape_path = data_dir / f"y_{subset}_norm_shape.npy"
        if shape_path.exists():
            shape = tuple(int(dim) for dim in np.load(shape_path))
            arr = np.memmap(fname, dtype=np.float32, mode="r", shape=shape)
            print(f"{subset} target memmap shape: {arr.shape}")
            return arr

        arr = np.load(fname, mmap_mode="r")
        print(f"{subset} target array shape: {arr.shape}")
        return arr

    def _load_static_array(self, data_dir: Path) -> Optional[np.ndarray]:
        """Load spatially normalized static features, if available."""

        fname = data_dir / "static_norm.npy"
        if not fname.exists():
            print("static_norm.npy not found; training will use only dynamic features.")
            return None
        static = np.load(fname, mmap_mode="r", allow_pickle=True)
        print(f"Static feature shape: {static.shape}")
        return static

    def _load_mask(self, data_dir: Path) -> Optional[np.ndarray]:
        """Load LandBench land mask if it exists next to the memmaps."""

        candidates = sorted(data_dir.glob("Mask with * spatial resolution.npy"))
        if not candidates:
            print("Mask file not found; proceeding without spatial filtering.")
            return None
        mask = np.load(candidates[0]).astype(bool)
        print(f"Loaded land mask from {candidates[0].name} with shape {mask.shape}")
        return mask

    def _load_lat_grid(self, data_dir: Path, height: int, width: int) -> Optional[np.ndarray]:
        """Load latitude grid if available for cos(lat) weighting."""

        preferred = [
            data_dir / "lat.npy",
            data_dir / "latitude.npy",
        ]
        glob_patterns = ("lat_*.npy", "latitude_*.npy")
        for pattern in glob_patterns:
            preferred.extend(sorted(data_dir.glob(pattern)))
        for candidate in preferred:
            if candidate.exists():
                lat = np.load(candidate)
                print(f"Loaded latitude grid from {candidate.name} with shape {lat.shape}")
                return lat
        print("Latitude grid not found; falling back to synthetic equal-area latitudes.")
        return None

    def _compute_mask_broadcast(self, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        return mask.astype(bool, copy=False)[None, ..., None]

    def _prepare_static_features(
        self, static: Optional[np.ndarray], mask: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if static is None:
            return None
        static_arr = np.asarray(static, dtype=np.float32)
        if mask is not None:
            static_arr = np.where(mask[..., None], static_arr, 0.0)
        return static_arr

    def _apply_mask(self, array: np.ndarray) -> np.ndarray:
        if self._mask_broadcast is None:
            return array
        return np.where(self._mask_broadcast, array, 0.0)

    def _apply_mask_grid(self, array: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return array
        return np.where(self._mask, array, 0.0)

    def _default_variable_list(self, key: str) -> List[str]:
        landbench_cfg = CONFIG.settings.get("landbench", {})
        values = landbench_cfg.get(key, [])
        if isinstance(values, str):
            return [values]
        return list(values) if values else []

    def _load_variable_names(self, kind: str, fallback: List[str]) -> List[str]:
        metadata = self.data_dir / f"{kind}_variables.json"
        if metadata.exists():
            try:
                contents = json.loads(metadata.read_text(encoding="utf-8"))
                if isinstance(contents, list):
                    return [str(item) for item in contents]
            except Exception:
                pass
        return list(fallback)

    def _build_channel_lookup(self) -> Dict[str, int]:
        total_channels = self._dynamic_data.shape[-1]
        forcing_count = self._infer_component_count("forcing_memmap.npy") or len(self._forcing_names)
        forcing_count = min(forcing_count, total_channels)
        land_surface_count = total_channels - forcing_count

        forcing_names = self._pad_or_trim(self._forcing_names, forcing_count, "forcing")
        land_surface_names = self._pad_or_trim(self._land_surface_names, land_surface_count, "land_surface")

        lookup: Dict[str, int] = {}
        channel_idx = 0
        for name in forcing_names + land_surface_names:
            normalized = self._normalize_var_name(name)
            if normalized and normalized not in lookup:
                lookup[normalized] = channel_idx
            channel_idx += 1
        self._forcing_names = forcing_names
        self._land_surface_names = land_surface_names
        return lookup

    def _pad_or_trim(self, names: List[str], count: int, prefix: str) -> List[str]:
        names = list(names)
        if len(names) >= count:
            return names[:count]
        pad_count = count - len(names)
        names.extend([f"{prefix}_{idx}" for idx in range(pad_count)])
        return names

    def _infer_component_count(self, filename: str) -> Optional[int]:
        path = self.data_dir / filename
        if not path.exists():
            return None
        try:
            arr = np.load(path, mmap_mode="r")
            count = arr.shape[-1]
            del arr
            return count
        except (ValueError, OSError):
            return None

    def _normalize_var_name(self, name: str) -> str:
        normalized = "".join(ch if ch.isalnum() or ch in ("_",) else "_" for ch in name.lower().strip())
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        return normalized.strip("_")

    def _resolve_component_indices(self) -> Dict[str, Optional[int]]:
        component_aliases = {
            "precip": ["precipitation", "total_precipitation", "tp", "rainfall", "pr"],
            "et": ["evapotranspiration", "evaporation", "total_evaporation", "evap", "surface_latent_heat_flux"],
            "runoff": ["total_runoff", "runoff", "ro"],
        }
        return {key: self._find_component_index(aliases) for key, aliases in component_aliases.items()}

    def _find_component_index(self, aliases: Sequence[str]) -> Optional[int]:
        for alias in aliases:
            normalized = self._normalize_var_name(alias)
            if normalized in self._channel_lookup:
                return self._channel_lookup[normalized]
        return None

    def _build_hydro_features(self, sample: SoilMoistureSample) -> Dict[str, np.ndarray]:
        sm_prev = np.array(self._target_data[sample.input_slice.stop - 1, ..., self._target_channel], copy=False)
        sm_prev = self._apply_mask_grid(sm_prev).astype(np.float32)

        hydro_time = sample.target_slice.start
        precip, precip_mask = self._extract_component(hydro_time, self._component_indices.get("precip"))
        et, _ = self._extract_component(hydro_time, self._component_indices.get("et"))
        runoff, _ = self._extract_component(hydro_time, self._component_indices.get("runoff"))

        return {
            "sm_prev": sm_prev,
            "precip": precip,
            "precip_mask": precip_mask,
            "evap": et,
            "runoff": runoff,
        }

    def _extract_component(self, time_index: int, channel_index: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
        zeros = np.zeros((self._height, self._width), dtype=np.float32)
        if channel_index is None:
            return zeros, zeros

        values = np.array(self._dynamic_data[time_index, ..., channel_index], copy=False)
        values = self._apply_mask_grid(values).astype(np.float32)
        mask = self._valid_mask.astype(np.float32).copy()
        finite = np.isfinite(values)
        mask = mask * finite
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return values, mask
