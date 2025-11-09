"""Evaluation entry-point for the LandGPT soil moisture model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import xarray as xr
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
from gptcast.config import CONFIG

from gptcast.data import SoilMoistureDataset
from gptcast.models import SoilMoistureGPTCast


def parse_args() -> argparse.Namespace:
    """Parse command-line flags."""

    parser = argparse.ArgumentParser(description="Evaluate soil moisture GPTCast model")
    parser.add_argument(
        "--nc-root",
        type=str,
        default=None,
        help="Base LandBench directory (defaults to config.py).",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    default_years = list(CONFIG.default_years) if CONFIG.default_years else None
    if CONFIG.year_span:
        years_help = f"Subset of years to evaluate (default {CONFIG.year_span[0]}-{CONFIG.year_span[1]})"
    elif CONFIG.default_years:
        formatted = ", ".join(str(year) for year in CONFIG.default_years)
        years_help = f"Subset of years to evaluate (default: {formatted})"
    else:
        years_help = "Subset of years to evaluate"
    parser.add_argument("--years", type=int, nargs="*", default=default_years, help=years_help)
    parser.add_argument("--input-steps", type=int, default=6)
    parser.add_argument("--forecast-steps", type=int, default=3)
    parser.add_argument(
        "--resolution",
        type=str,
        default=CONFIG.default_resolution,
        help="LandBench spatial resolution (e.g. '1' or '0.5').",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=CONFIG.torch_device)
    parser.add_argument("--subset", type=str, default="test", choices=["train", "test"], help="Which memmap split to load")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CONFIG.outputs_root / "predictions"),
    )
    parser.add_argument("--save-predictions", action="store_true", help="Persist NetCDF predictions to disk")
    parser.add_argument(
        "--use-area-weighted-metrics",
        action="store_true",
        help="Report additional metrics that weight errors by cos(latitude).",
    )
    return parser.parse_args()


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move all tensor entries in a batch to the specified device."""

    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def save_predictions(preds: torch.Tensor, out_dir: Path, batch_index: int) -> None:
    """Persist predictions as NetCDF files for external analysis."""

    out_dir.mkdir(parents=True, exist_ok=True)
    arr = preds.squeeze(2).detach().cpu().numpy()  # (B, T, H, W)
    for sample_idx, sample in enumerate(arr):
        data_array = xr.DataArray(sample, dims=("time", "y", "x"), name="soil_moisture_forecast")
        data_array.to_netcdf(out_dir / f"prediction_{batch_index:04d}_{sample_idx:02d}.nc")


def _select_point_forecast(model: SoilMoistureGPTCast, preds: torch.Tensor) -> torch.Tensor:
    if getattr(model, "uncertainty", "none") == "quantile":
        median_index = getattr(model, "median_index", 0)
        return preds[:, :, median_index : median_index + 1, ...]
    return preds


def evaluate(
    model: SoilMoistureGPTCast,
    loader: DataLoader,
    device: torch.device,
    save_dir: Path,
    persist: bool,
    area_weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Run inference and accumulate RMSE/MAE metrics."""

    model.eval()
    mse_total = 0.0
    mae_total = 0.0
    count = 0
    area_weights_tensor = None
    area_count = 0
    mse_area_total = 0.0
    mae_area_total = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            preds = model(batch)
            point_preds = _select_point_forecast(model, preds)
            target = batch["target"].permute(0, 1, 4, 2, 3)
            diff = point_preds - target
            mse_total += torch.sum(diff ** 2).item()
            mae_total += torch.sum(torch.abs(diff)).item()
            count += diff.numel()
            if area_weights is not None:
                if area_weights_tensor is None:
                    aw = torch.as_tensor(area_weights, device=device, dtype=torch.float32)
                    aw = aw / aw.sum()
                    area_weights_tensor = aw.view(1, 1, 1, aw.shape[0], aw.shape[1])
                squared = (diff ** 2) * area_weights_tensor
                per_sample_mse = squared.sum(dim=(-1, -2)).squeeze(-1)
                mse_area_total += per_sample_mse.sum().item()
                area_count += per_sample_mse.numel()
                weighted_abs = torch.abs(diff) * area_weights_tensor
                per_sample_mae = weighted_abs.sum(dim=(-1, -2)).squeeze(-1)
                mae_area_total += per_sample_mae.sum().item()
            if persist:
                save_predictions(point_preds, save_dir, batch_idx)

    rmse = (mse_total / count) ** 0.5
    mae = mae_total / count
    metrics = {"rmse": rmse, "mae": mae}
    if area_weights is not None and area_count > 0:
        mse_area_mean = mse_area_total / area_count
        mae_area_mean = mae_area_total / area_count
        metrics["rmse_area_weighted"] = mse_area_mean ** 0.5
        metrics["mae_area_weighted"] = mae_area_mean
    return metrics


def main() -> None:
    args = parse_args()

    base_root = Path(args.nc_root) if args.nc_root else CONFIG.data_root
    if base_root.name == args.resolution:
        dataset_root = base_root
    elif (base_root / args.resolution).exists():
        dataset_root = base_root / args.resolution
    else:
        dataset_root = base_root / args.resolution
    dataset_root = dataset_root.resolve()

    print(f"正在加载评估数据目录：{dataset_root}")
    # Evaluate on the requested memmap split (usually the held-out test set).
    dataset = SoilMoistureDataset(
        root=str(dataset_root),
        subset=args.subset,
        input_steps=args.input_steps,
        forecast_steps=args.forecast_steps,
    )
    print(f"评估数据准备完成，样本总数：{len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model = SoilMoistureGPTCast.load_from_checkpoint(args.checkpoint, area_weights=dataset.area_weights)
    if dataset.feature_dim != model.hparams["in_channels"]:
        raise ValueError(
            f"Dataset feature dimension ({dataset.feature_dim}) does not match model checkpoint ({model.hparams['in_channels']})."
        )
    model.to(device)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("开始执行推理评估，请稍候...")
    area_weights = dataset.area_weights if args.use_area_weighted_metrics else None
    metrics = evaluate(model, loader, device, output_dir, args.save_predictions, area_weights=area_weights)

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, sort_keys=True)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
