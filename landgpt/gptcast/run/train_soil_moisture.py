"""Training entry-point for the LandGPT soil moisture model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
from gptcast.config import CONFIG

from gptcast.data import SoilMoistureDataset
from gptcast.models import SoilMoistureGPTCast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train soil moisture GPTCast model on LandBench data")
    parser.add_argument(
        "--nc-root",
        type=str,
        default=None,
        help="Base LandBench directory (defaults to config.py).",
    )
    default_years = list(CONFIG.default_years) if CONFIG.default_years else None
    if CONFIG.year_span:
        years_help = f"Subset of years to load (default {CONFIG.year_span[0]}-{CONFIG.year_span[1]})"
    elif CONFIG.default_years:
        formatted = ", ".join(str(year) for year in CONFIG.default_years)
        years_help = f"Subset of years to load (default: {formatted})"
    else:
        years_help = "Subset of years to load"
    parser.add_argument("--years", type=int, nargs="*", default=default_years, help=years_help)
    parser.add_argument("--input-steps", type=int, default=6, help="Number of past timesteps as input")
    parser.add_argument("--forecast-steps", type=int, default=3, help="Forecast horizon steps")
    parser.add_argument(
        "--resolution",
        type=str,
        default=CONFIG.default_resolution,
        help="LandBench spatial resolution (e.g. '1' or '0.5').",
    )
    parser.add_argument("--target-channel", type=int, default=0, help="Channel index of soil moisture in dataset")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-hyd", type=float, default=0.0, help="Weight for ΔSM≈P-ET-R regularization.")
    parser.add_argument("--cnn-channels", type=int, default=64)
    parser.add_argument("--fusion-channels", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--accelerator", type=str, default=CONFIG.accelerator)
    parser.add_argument("--devices", type=int, default=CONFIG.default_device_count)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--subset", type=str, default="train", choices=["train", "test"], help="Which memmap split to load")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CONFIG.outputs_root / "soil_moisture"),
    )
    parser.add_argument(
        "--use-area-weighted-loss",
        action="store_true",
        help="Apply cos(latitude) area weights when computing the training loss.",
    )
    parser.add_argument(
        "--uncertainty",
        type=str,
        default="none",
        choices=["none", "quantile"],
        help="Enable probabilistic outputs (quantile regression).",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.05, 0.5, 0.95],
        help="Quantile levels used when --uncertainty=quantile.",
    )
    return parser.parse_args()


def make_dataloaders(
    dataset: SoilMoistureDataset,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError("Dataset split produced non-positive training set size; adjust ratios.")

    print(f"数据集总样本 {n_total}，划分为：训练 {n_train}，验证 {n_val}，测试 {n_test}。")

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    def _loader(ds):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=ds is train_set,
            num_workers=num_workers,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )

    return _loader(train_set), _loader(val_set), _loader(test_set)


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    base_root = Path(args.nc_root) if args.nc_root else CONFIG.data_root
    if base_root.name == args.resolution:
        dataset_root = base_root
    elif (base_root / args.resolution).exists():
        dataset_root = base_root / args.resolution
    else:
        dataset_root = base_root / args.resolution
    dataset_root = dataset_root.resolve()

    print(f"正在使用数据目录：{dataset_root}")
    # LandBench preprocessing exports separate train/test memmaps; subset selects which split to read.
    dataset = SoilMoistureDataset(
        root=str(dataset_root),
        subset=args.subset,
        input_steps=args.input_steps,
        forecast_steps=args.forecast_steps,
    )
    print(f"数据加载完成，样本总数：{len(dataset)}")

    train_loader, val_loader, test_loader = make_dataloaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print("数据划分完成，训练/验证/测试 DataLoader 已就绪。")

    model = SoilMoistureGPTCast(
        in_channels=dataset.feature_dim,
        input_steps=args.input_steps,
        forecast_steps=args.forecast_steps,
        target_channel=args.target_channel,
        cnn_channels=args.cnn_channels,
        fusion_channels=args.fusion_channels,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lambda_hyd=args.lambda_hyd,
        use_area_weighted_loss=args.use_area_weighted_loss,
        area_weights=dataset.area_weights,
        uncertainty=args.uncertainty,
        quantiles=args.quantiles,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=output_dir,
        filename="soil-moisture-{epoch:02d}-{val_loss:.4f}",
        auto_insert_metric_name=False,
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=str(output_dir),
        log_every_n_steps=1,
        callbacks=[checkpoint_cb],
    )

    print("开始训练土壤湿度模型...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("训练完成，开始测试集评估...")
    trainer.test(model, dataloaders=test_loader)
    print(f"全部流程结束，模型权重及日志已保存至：{output_dir}")


if __name__ == "__main__":
    main()

