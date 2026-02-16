#!/usr/bin/env python3
from __future__ import annotations

"""
Sanity-check plots for `snapshot_pca_deglaciation.py` output NetCDF.

Creates a small set of figures to visually validate:
  1) The two mean definitions side-by-side:
     - deglaciation_age_mean_norm_all
     - deglaciation_age_mean_norm_deglac_only
  2) The first N PCA modes for each mean definition:
     - pca_mode_norm_all[mode]
     - pca_mode_norm_deglac_only[mode]
  3) Context rasters:
     - bed_elevation
     - modern_thickness

By default, plots are saved to a directory (recommended for headless use).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _imshow(ax, da: xr.DataArray, *, title: str, cmap: str, vmin=None, vmax=None) -> None:
    x = da["x"].values
    y = da["y"].values
    z = da.values
    origin = "upper" if (len(y) >= 2 and float(y[0]) > float(y[-1])) else "lower"
    extent = (float(np.nanmin(x)), float(np.nanmax(x)), float(np.nanmin(y)), float(np.nanmax(y)))
    zm = np.ma.masked_invalid(z)
    im = ax.imshow(zm, origin=origin, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    return im


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=160)
    plt.close(fig)


def plot_means(ds: xr.Dataset, *, out_dir: Path, cmap: str) -> None:
    a = ds["deglaciation_age_mean_norm_all"]
    b = ds["deglaciation_age_mean_norm_deglac_only"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    im0 = _imshow(axes[0], a, title="Mean (all runs, includes 0s)", cmap=cmap, vmin=0.0, vmax=1.0)
    im1 = _imshow(axes[1], b, title="Mean (deglac-only on modern ice-free)", cmap=cmap, vmin=0.0, vmax=1.0)
    fig.colorbar(im0, ax=axes, label="Normalized age (0..1)", shrink=0.85)
    _save(fig, out_dir, "means_side_by_side.png")


def plot_modes(ds: xr.Dataset, *, out_dir: Path, n_modes: int, cmap: str) -> None:
    modes_all = ds["pca_mode_norm_all"]
    modes_deg = ds["pca_mode_norm_deglac_only"]
    available = int(min(modes_all.sizes.get("mode", 0), modes_deg.sizes.get("mode", 0)))
    n = int(min(n_modes, available))
    if n <= 0:
        raise ValueError("No PCA modes found in dataset.")

    for i in range(n):
        da_all = modes_all.isel(mode=i)
        da_deg = modes_deg.isel(mode=i)

        # Symmetric limits help interpret modes.
        vmax = float(np.nanmax([np.nanmax(np.abs(da_all.values)), np.nanmax(np.abs(da_deg.values))]))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        im0 = _imshow(axes[0], da_all, title=f"Mode {i+1} (all-mean)", cmap=cmap, vmin=-vmax, vmax=vmax)
        im1 = _imshow(axes[1], da_deg, title=f"Mode {i+1} (deglac-only mean)", cmap=cmap, vmin=-vmax, vmax=vmax)
        fig.colorbar(im0, ax=axes, label="Mode amplitude (normalized units)", shrink=0.85)
        _save(fig, out_dir, f"mode_{i+1:02d}_side_by_side.png")


def plot_context(ds: xr.Dataset, *, out_dir: Path) -> None:
    bed = ds["bed_elevation"]
    thk = ds["modern_thickness"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    im0 = _imshow(axes[0], bed, title="Bedrock elevation (m)", cmap="terrain")
    im1 = _imshow(axes[1], thk, title="Modern thickness (m)", cmap="Blues")
    fig.colorbar(im0, ax=axes[0], label="m", shrink=0.85)
    fig.colorbar(im1, ax=axes[1], label="m", shrink=0.85)
    _save(fig, out_dir, "bedrock_and_thickness.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sanity-check figures for deglaciation snapshot PCA output.")
    parser.add_argument(
        "--nc",
        type=Path,
        default=Path("data/deglaciation_snapshot_pca.nc"),
        help="Path to snapshot PCA NetCDF.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/snapshot_pca_sanity_plots"),
        help="Directory to write PNGs.",
    )
    parser.add_argument("--modes", type=int, default=5, help="Number of PCA modes to plot for each method.")
    parser.add_argument("--mean-cmap", type=str, default="viridis_r", help="Colormap for mean maps.")
    parser.add_argument("--mode-cmap", type=str, default="coolwarm", help="Colormap for PCA modes.")
    args = parser.parse_args()

    ds = xr.open_dataset(args.nc, decode_times=False)
    try:
        required = [
            "deglaciation_age_mean_norm_all",
            "deglaciation_age_mean_norm_deglac_only",
            "pca_mode_norm_all",
            "pca_mode_norm_deglac_only",
            "bed_elevation",
            "modern_thickness",
        ]
        missing = [v for v in required if v not in ds]
        if missing:
            raise KeyError(f"{args.nc} missing required variables: {missing}")

        plot_means(ds, out_dir=args.out_dir, cmap=str(args.mean_cmap))
        plot_modes(ds, out_dir=args.out_dir, n_modes=int(args.modes), cmap=str(args.mode_cmap))
        plot_context(ds, out_dir=args.out_dir)
    finally:
        ds.close()

    print(f"Wrote plots to: {args.out_dir}")


if __name__ == "__main__":
    main()

