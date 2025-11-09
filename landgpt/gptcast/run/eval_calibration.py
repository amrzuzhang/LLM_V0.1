"""Evaluate calibration diagnostics (PIT, reliability, CRPS) for quantile models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from gptcast.config import CONFIG
from gptcast.data import SoilMoistureDataset
from gptcast.metrics.calibration import crps_from_quantiles, pit_histogram, reliability_curve
from gptcast.models import SoilMoistureGPTCast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibration evaluation for quantile GPTCast models")
    parser.add_argument("--nc-root", type=str, default=None, help="Base LandBench directory (defaults to config.py).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Quantile model checkpoint path.")
    parser.add_argument("--resolution", type=str, default=CONFIG.default_resolution)
    parser.add_argument("--subset", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--input-steps", type=int, default=6)
    parser.add_argument("--forecast-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=CONFIG.torch_device)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.05, 0.5, 0.95])
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CONFIG.outputs_root / "calibration"),
        help="Directory for calibration figures and metrics.",
    )
    parser.add_argument("--bins", type=int, default=20, help="Number of bins in the PIT histogram.")
    parser.add_argument("--forecast-step", type=int, default=0, help="Forecast step index to evaluate (0-based).")
    return parser.parse_args()


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def flatten_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    valid_mask: np.ndarray,
    quantile_levels: list[float],
    forecast_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten spatial grids into sample-wise vectors."""

    mask = valid_mask.astype(bool).ravel()
    target_step = targets[:, forecast_step, 0].reshape(targets.shape[0], -1)
    target_values = target_step[:, mask].reshape(-1)

    quantile_step = preds[:, forecast_step].reshape(preds.shape[0], len(quantile_levels), -1)
    quantile_values = np.transpose(quantile_step[:, :, mask], (0, 2, 1)).reshape(-1, len(quantile_levels))
    return target_values, quantile_values


def plot_pit(hist: np.ndarray, edges: np.ndarray, out_path: Path) -> None:
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.figure(figsize=(6, 4))
    plt.bar(centers, hist, width=edges[1] - edges[0], edgecolor="black", color="#4C78A8")
    plt.axhline(np.mean(hist), color="red", linestyle="--", label="Ideal")
    plt.xlabel("PIT")
    plt.ylabel("Count")
    plt.title("PIT Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_reliability(quantiles: np.ndarray, empirical: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    plt.scatter(quantiles, empirical, color="#F58518", s=40, label="Observed")
    plt.xlabel("Nominal quantile")
    plt.ylabel("Observed frequency")
    plt.title("Quantile Reliability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()

    base_root = Path(args.nc_root) if args.nc_root else CONFIG.data_root
    dataset_root = base_root / args.resolution if base_root.name != args.resolution else base_root
    dataset_root = dataset_root.resolve()

    dataset = SoilMoistureDataset(
        root=str(dataset_root),
        subset=args.subset,
        input_steps=args.input_steps,
        forecast_steps=args.forecast_steps,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = SoilMoistureGPTCast.load_from_checkpoint(args.checkpoint)
    if getattr(model, "uncertainty", "none") != "quantile":
        raise ValueError("Loaded checkpoint does not contain quantile outputs; set --uncertainty=quantile during training.")
    model.to(device)
    model.eval()

    expected_quantiles = list(getattr(model, "quantile_levels", args.quantiles))
    if len(expected_quantiles) != len(args.quantiles):
        quantiles = expected_quantiles
    else:
        quantiles = args.quantiles

    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            preds = model(batch).cpu().numpy()
            targets = batch["target"].permute(0, 1, 4, 2, 3).cpu().numpy()
            all_targets.append(targets)
            all_preds.append(preds)

    targets_np = np.concatenate(all_targets, axis=0)
    preds_np = np.concatenate(all_preds, axis=0)

    forecast_step = int(np.clip(args.forecast_step, 0, args.forecast_steps - 1))
    obs_values, quantile_values = flatten_predictions(
        preds_np,
        targets_np,
        dataset.valid_mask,
        quantiles,
        forecast_step,
    )
    valid = np.isfinite(obs_values)
    obs_values = obs_values[valid]
    quantile_values = quantile_values[valid]

    crps = crps_from_quantiles(obs_values, quantiles, quantile_values)
    crps_mean = float(np.mean(crps))

    hist, edges, pit_values = pit_histogram(obs_values, quantiles, quantile_values, bins=args.bins)
    rel_x, rel_y = reliability_curve(obs_values, quantiles, quantile_values)

    output_dir = Path(args.output_dir).resolve()
    calib_dir = output_dir / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)

    plot_pit(hist, edges, calib_dir / "pit_histogram.png")
    plot_reliability(rel_x, rel_y, calib_dir / "reliability.png")

    metrics = {
        "quantiles": quantiles,
        "forecast_step": forecast_step,
        "crps_mean": crps_mean,
        "reliability": {f"{q:.3f}": float(freq) for q, freq in zip(rel_x, rel_y)},
        "pit_histogram": {
            "bins": edges.tolist(),
            "counts": hist.tolist(),
        },
        "pit_summary": {
            "mean": float(np.mean(pit_values)),
            "std": float(np.std(pit_values)),
            "num_samples": int(pit_values.size),
        },
    }
    metrics_path = calib_dir / "metrics_calibration.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Calibration metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
