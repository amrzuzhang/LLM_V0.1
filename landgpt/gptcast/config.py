"""Project-level configuration helpers for LandGPT."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Stores repository paths and default runtime options."""

    repo_root: Path
    data_root: Path
    outputs_root: Path
    default_resolution: str
    default_device: str
    default_device_count: int
    default_years: Optional[Tuple[int, ...]]
    settings: Dict[str, object]

    @property
    def dataset_root(self) -> Path:
        """Return the resolution-specific dataset directory."""

        candidate = self.data_root / self.default_resolution
        if candidate.exists():
            return candidate.resolve()
        if self.data_root.name == self.default_resolution:
            return self.data_root.resolve()
        return candidate.resolve()

    @property
    def accelerator(self) -> str:
        """Return the default Lightning accelerator keyword."""

        return "gpu" if self.default_device.lower() == "gpu" else "cpu"

    @property
    def torch_device(self) -> str:
        """Return the default torch device identifier."""

        return "cuda" if self.default_device.lower() == "gpu" else "cpu"

    @property
    def year_span(self) -> Optional[Tuple[int, int]]:
        """Return the (start, end) tuple for the configured training years."""

        if not self.default_years:
            return None
        return self.default_years[0], self.default_years[-1]

    @staticmethod
    def _parse_years(spec: str) -> Tuple[int, ...]:
        """Parse an environment string into an ordered year tuple."""

        clean = spec.strip()
        if not clean:
            return tuple()
        bracketed = clean[0] in "([{" and clean[-1] in ")]}"
        if bracketed:
            clean = clean[1:-1].strip()

        normalized = clean.replace(" ", "")
        if "-" in normalized and normalized.count("-") == 1 and "," not in normalized:
            start_str, end_str = normalized.split("-", 1)
            start, end = int(start_str), int(end_str)
            if start > end:
                start, end = end, start
            return tuple(range(start, end + 1))

        parts = [part for part in clean.replace(" ", "").split(",") if part]
        if bracketed and len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            if start > end:
                start, end = end, start
            return tuple(range(start, end + 1))

        return tuple(sorted(int(part) for part in parts))

    @classmethod
    def build(
        cls,
        *,
        repo_root: Optional[Path] = None,
        data_root: Optional[Path] = None,
        outputs_root: Optional[Path] = None,
        resolution: Optional[str] = None,
        device: Optional[str] = None,
        device_count: Optional[int] = None,
        years: Optional[Sequence[int]] = None,
    ) -> "ProjectConfig":
        base = repo_root or Path(__file__).resolve().parent.parent

        settings = cls._load_settings(base)
        paths_cfg = settings.get("paths", {})
        gptcast_cfg = settings.get("gptcast", {})

        env_data_root = os.getenv("LANDGPT_DATASET_BASE")
        env_outputs_root = os.getenv("LANDGPT_OUTPUTS_ROOT")
        env_resolution = os.getenv("LANDGPT_RESOLUTION")
        env_device = os.getenv("LANDGPT_DEVICE")
        env_device_count = os.getenv("LANDGPT_DEVICE_COUNT")
        env_years = os.getenv("LANDGPT_TRAIN_YEARS")

        resolved_data_root = Path(env_data_root) if env_data_root else data_root
        if resolved_data_root is None:
            home = Path.home()
            candidates = [
                Path(paths_cfg["preprocessed"]) if paths_cfg.get("preprocessed") else None,
                home / "data" / "LandBench",
                home / "data" / "LandBench" / "LandBench1" / "0.1",
                base / "landbench" / "adhub",
                base / "landbench" / "adhub" / "LandBench1" / "0.1",
                base / "adhub",
                base / "adhub" / "LandBench1" / "0.1",
                home / "data" / "landbench" / "adhub",
                home / "data" / "landbench" / "adhub" / "LandBench1" / "0.1",
            ]
            for candidate in candidates:
                if candidate and candidate.exists():
                    resolved_data_root = candidate
                    break
            else:
                resolved_data_root = base / "landbench" / "adhub"

        resolved_outputs_root = Path(env_outputs_root) if env_outputs_root else outputs_root
        if resolved_outputs_root is None:
            data_home = Path.home() / "data"
            if resolved_data_root and data_home.exists() and data_home in resolved_data_root.resolve().parents:
                resolved_outputs_root = data_home / "landgpt_outputs"
            else:
                resolved_outputs_root = base / "outputs"

        resolution_candidate = env_resolution or resolution or gptcast_cfg.get("resolution") or "1"
        resolved_resolution = str(resolution_candidate).strip()
        resolved_device = (env_device or device or gptcast_cfg.get("device") or "gpu").strip().lower()
        resolved_device_count = int(env_device_count or device_count or gptcast_cfg.get("device_count") or 1)

        if env_years:
            parsed_years = cls._parse_years(env_years)
            resolved_years = parsed_years if parsed_years else None
        elif years:
            resolved_years = tuple(sorted({int(year) for year in years}))
        else:
            yaml_years = gptcast_cfg.get("years")
            resolved_years = tuple(int(year) for year in yaml_years) if yaml_years else tuple(range(2015, 2021))

        return cls(
            repo_root=base,
            data_root=resolved_data_root,
            outputs_root=resolved_outputs_root,
            default_resolution=resolved_resolution,
            default_device=resolved_device,
            default_device_count=resolved_device_count,
            default_years=resolved_years,
            settings=settings,
        )

    @staticmethod
    def _load_settings(base: Path) -> Dict[str, object]:
        settings_path = base / "config" / "settings.yaml"
        if settings_path.exists():
            with settings_path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        return {}


CONFIG = ProjectConfig.build()
