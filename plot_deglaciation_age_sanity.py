#!/usr/bin/env python3
from __future__ import annotations

"""
Sanity-check plots for gridded deglaciation age NetCDF files.

This script is intended for quickly verifying that the NetCDFs produced by
`h5_to_deglaciation_age_netcdf.py` look reasonable.

It iterates through a directory of NetCDF files and, for each file:
  - loads `x`, `y`, and the deglaciation age variable (default: deglaciation_age)
  - plots the 2D age map
  - optionally saves a PNG (recommended if you have many runs or are in headless mode)

By default, color limits are auto-scaled per file. Use --vmin/--vmax if you
want consistent limits across runs.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _plot_one(
    *,
    nc_path: Path,
    var: str,
    vmin: float | None,
    vmax: float | None,
    cmap: str,
    save_path: Path | None,
    show: bool,
) -> None:
    ds = xr.open_dataset(nc_path, decode_times=False)
    try:
        if "x" not in ds or "y" not in ds:
            raise KeyError(f"{nc_path} is missing coordinate variables 'x'/'y'. Available: {list(ds.variables)}")
        if var not in ds:
            raise KeyError(f"{nc_path} is missing variable {var!r}. Available: {list(ds.data_vars)}")

        x = ds["x"].values
        y = ds["y"].values
        z = ds[var].values

        # Prefer imshow for speed. Handle y axis direction by choosing origin.
        origin = "upper" if (len(y) >= 2 and float(y[0]) > float(y[-1])) else "lower"
        extent = (float(np.nanmin(x)), float(np.nanmax(x)), float(np.nanmin(y)), float(np.nanmax(y)))

        z_masked = np.ma.masked_invalid(z)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(
            z_masked,
            origin=origin,
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(nc_path.name)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")

        units = ds[var].attrs.get("units")
        label = f"{var} ({units})" if units else var
        fig.colorbar(im, ax=ax, label=label, shrink=0.85)

        fig.tight_layout()

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)
    finally:
        ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterate through deglaciation-age NetCDFs and plot each one.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/issm_deglaciation_age_nc"),
        help="Directory containing per-run deglaciation-age NetCDF files.",
    )
    parser.add_argument("--pattern", type=str, default="*_deglaciation_age.nc", help="Glob pattern for NetCDF files.")
    parser.add_argument("--var", type=str, default="deglaciation_age", help="Name of the deglaciation age variable.")
    parser.add_argument("--cmap", type=str, default="viridis_r", help="Matplotlib colormap name.")
    parser.add_argument("--vmin", type=float, default=None, help="Optional fixed color min (years).")
    parser.add_argument("--vmax", type=float, default=None, help="Optional fixed color max (years).")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, save each plot as a PNG in this directory (recommended for many files).",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display plots (use with --save-dir).")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit on number of files to plot.")
    args = parser.parse_args()

    nc_files = sorted(args.input_dir.glob(args.pattern))
    if not nc_files:
        raise FileNotFoundError(f"No files matched {args.pattern!r} in {args.input_dir}")

    if args.max_files is not None:
        nc_files = nc_files[: int(args.max_files)]

    for i, p in enumerate(nc_files, start=1):
        print(f"[{i}/{len(nc_files)}] Plotting {p} ...", flush=True)
        save_path = None
        if args.save_dir is not None:
            save_path = args.save_dir / f"{p.stem}.png"
        _plot_one(
            nc_path=p,
            var=str(args.var),
            vmin=None if args.vmin is None else float(args.vmin),
            vmax=None if args.vmax is None else float(args.vmax),
            cmap=str(args.cmap),
            save_path=save_path,
            show=not bool(args.no_show),
        )


if __name__ == "__main__":
    main()

