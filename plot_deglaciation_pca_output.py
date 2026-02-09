#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    default_path = Path("data/deglaciation_snapshot_pca.nc")

    parser = argparse.ArgumentParser(
        description="Load the snapshot PCA NetCDF, print the xarray.Dataset, and plot all PCA modes."
    )
    parser.add_argument("nc_file", nargs="?", type=Path, default=default_path, help=f"Path to PCA NetCDF (default: {default_path})")
    parser.add_argument("--cmap", type=str, default="RdBu_r", help="Colormap for modes (default: RdBu_r)")
    parser.add_argument("--mean-cmap", type=str, default="magma", help="Colormap for mean field (default: magma)")
    parser.add_argument("--out-dir", type=Path, default=None, help="If set, save one PNG per mode here")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for saved figures (default: 200)")
    parser.add_argument("--vlim-pctl", type=float, default=99.0, help="Symmetric color limits from percentile (default: 99)")
    parser.add_argument("--plot-mean", action="store_true", help="Also plot deglaciation_age_mean_norm if present")
    args = parser.parse_args()

    path = args.nc_file
    if not path.exists():
        raise FileNotFoundError(path)

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt

    ds = xr.open_dataset(path, decode_times=False)
    print(ds)
    if "cumulative_explained_variance_ratio" in ds:
        print(ds["cumulative_explained_variance_ratio"].values)

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    # Mean deglaciation history across runs (normalized 0..1), if requested.
    if args.plot_mean and "deglaciation_age_mean_norm" in ds:
        mean_map = ds["deglaciation_age_mean_norm"].values
        fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
        im = ax.pcolormesh(
            ds["x"].values,
            ds["y"].values,
            mean_map,
            shading="auto",
            cmap=args.mean_cmap,
            vmin=0.0,
            vmax=1.0,
        )
        fig.colorbar(im, ax=ax, shrink=0.9, label="mean deglaciation (normalized)")
        ax.set_title("Mean deglaciation history (normalized 0..1)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if args.out_dir is not None:
            out = args.out_dir / "mean_deglaciation_norm.png"
            fig.savefig(out, dpi=int(args.dpi))
            plt.close(fig)
            print(f"Saved {out}")
        else:
            plt.show()

    if "pca_mode_norm" not in ds:
        raise KeyError(f"{path} missing 'pca_mode_norm'. Available vars: {list(ds.data_vars)}")

    modes = ds["pca_mode_norm"]  # (mode, y, x)
    evr = ds.get("explained_variance_ratio")
    cev = ds.get("cumulative_explained_variance_ratio")

    # Determine a common symmetric color scale for all modes.
    mvals = modes.values
    finite = np.isfinite(mvals)
    if not np.any(finite):
        raise ValueError("No finite values in pca_mode_norm to plot.")

    p = float(args.vlim_pctl)
    if not (0.0 < p <= 100.0):
        raise ValueError("--vlim-pctl must be in (0, 100].")
    vmax = float(np.nanpercentile(np.abs(mvals[finite]), p))
    if not (vmax > 0):
        vmax = 1.0

    for i, mode_id in enumerate(modes["mode"].values):
        z = modes.sel(mode=mode_id)
        title = f"Mode {int(mode_id)} (anom)"
        if evr is not None:
            try:
                title += f"  (EVR={float(evr.sel(mode=mode_id).values):.3f}"
                if cev is not None:
                    title += f", CEVR={float(cev.sel(mode=mode_id).values):.3f}"
                title += ")"
            except Exception:
                pass

        fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
        im = ax.pcolormesh(ds["x"].values, ds["y"].values, z.values, shading="auto", cmap=args.cmap, vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.9, label="mode amplitude (norm - mean_norm)")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if args.out_dir is not None:
            out = args.out_dir / f"mode_{int(mode_id):02d}.png"
            fig.savefig(out, dpi=int(args.dpi))
            plt.close(fig)
            print(f"Saved {out}")
        else:
            plt.show()

    ds.close()


if __name__ == "__main__":
    main()
