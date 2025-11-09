"""Utility script to visualize LandGPT soil moisture outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot soil moisture maps from NetCDF files.")
    parser.add_argument(
        "--nc-file",
        type=str,
        required=True,
        help="Path to the NetCDF file containing soil moisture predictions or raw data.",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="soil_moisture_forecast",
        help="Variable name inside the NetCDF file.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Time index to visualize (defaults to the first timestep).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the figure instead of showing it interactively.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap to use for rendering.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom title for the plot.",
    )
    return parser.parse_args()


def _extract_coords(data: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the X/Y coordinate arrays used for plotting."""

    if "lon" in data.coords and "lat" in data.coords:
        return data["lon"].to_numpy(), data["lat"].to_numpy()
    if "longitude" in data.coords and "latitude" in data.coords:
        return data["longitude"].to_numpy(), data["latitude"].to_numpy()
    if "x" in data.coords and "y" in data.coords:
        return data["x"].to_numpy(), data["y"].to_numpy()
    return np.arange(data.shape[-1]), np.arange(data.shape[-2])


def main() -> None:
    args = parse_args()
    nc_path = Path(args.nc_file)
    if not nc_path.exists():
        raise FileNotFoundError(f"Unable to locate NetCDF file: {nc_path}")

    ds = xr.open_dataset(nc_path)
    if args.variable not in ds:
        raise KeyError(f"Variable '{args.variable}' not found in {nc_path.name}. Available: {list(ds.data_vars)}")

    data = ds[args.variable]
    if "time" in data.dims:
        if args.time_index < 0 or args.time_index >= data.sizes["time"]:
            raise IndexError(f"time-index {args.time_index} out of range for dataset with {data.sizes['time']} steps.")
        slice_data = data.isel(time=args.time_index)
        timestamp = data["time"].isel(time=args.time_index).values
    else:
        slice_data = data
        timestamp = None

    lon, lat = _extract_coords(slice_data)
    values = slice_data.to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    mesh = ax.pcolormesh(lon, lat, values, cmap=args.cmap, shading="auto")
    ax.set_xlabel("Longitude" if slice_data.dims[-1] in {"lon", "longitude"} else "X")
    ax.set_ylabel("Latitude" if slice_data.dims[-2] in {"lat", "latitude"} else "Y")

    plot_title = args.title or f"{args.variable}"
    if timestamp is not None:
        plot_title += f" @ {np.datetime_as_string(timestamp, unit='m')}"
    ax.set_title(plot_title)
    fig.colorbar(mesh, ax=ax, label=args.variable)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
    else:
        plt.show()

    plt.close(fig)
    ds.close()


if __name__ == "__main__":
    main()
