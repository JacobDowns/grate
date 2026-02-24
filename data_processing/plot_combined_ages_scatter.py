#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _finite_xy(df: pd.DataFrame, *, x_col: str, y_col: str) -> pd.Series:
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    return x.notna() & y.notna() & np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scatter plot of combined_ages.csv points, colored by age using discrete 1000-year bins, "
            "with different markers for radiocarbon vs cosmogenic rows."
        )
    )
    parser.add_argument(
        "--in",
        dest="inp",
        type=Path,
        default=Path("data/age_data/combined_ages.csv"),
        help="Input combined ages CSV (default: data/age_data/combined_ages.csv).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/combined_ages_scatter_age_bins_1000yr.png"),
        help="Output image path (default: output/combined_ages_scatter_age_bins_1000yr.png).",
    )
    parser.add_argument("--x-col", type=str, default="x_3413", help="X coordinate column (default: x_3413).")
    parser.add_argument("--y-col", type=str, default="y_3413", help="Y coordinate column (default: y_3413).")
    parser.add_argument("--age-col", type=str, default="age_mean", help="Age column (default: age_mean).")
    parser.add_argument("--type-col", type=str, default="obs_type", help="Type column (default: obs_type).")
    parser.add_argument(
        "--bin-years",
        type=float,
        default=1000.0,
        help="Discrete age bin width in years (default: 1000).",
    )
    parser.add_argument("--cmap", type=str, default="turbo", help="Matplotlib colormap name (default: turbo).")
    parser.add_argument("--s", type=float, default=18.0, help="Marker size (default: 18).")
    parser.add_argument("--alpha", type=float, default=0.9, help="Marker alpha (default: 0.9).")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI (default: 200).")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(7.5, 8.5),
        metavar=("W", "H"),
        help="Figure size in inches (default: 7.5 8.5).",
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    df = pd.read_csv(args.inp)

    required = {args.x_col, args.y_col, args.age_col, args.type_col}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in {args.inp}: {sorted(missing)}")

    df = df.copy()
    df[args.age_col] = pd.to_numeric(df[args.age_col], errors="coerce")
    df[args.type_col] = df[args.type_col].astype(str).str.strip().str.lower()

    valid = df[args.age_col].notna() & np.isfinite(df[args.age_col].to_numpy()) & _finite_xy(df, x_col=args.x_col, y_col=args.y_col)
    df = df.loc[valid].reset_index(drop=True)
    if df.empty:
        raise SystemExit("No finite rows to plot after filtering on x/y/age.")

    bin_years = float(args.bin_years)
    if not np.isfinite(bin_years) or bin_years <= 0:
        raise SystemExit(f"--bin-years must be positive and finite; got {args.bin_years!r}")

    ages = df[args.age_col].to_numpy(dtype=float)
    age_min = float(np.nanmin(ages))
    age_max = float(np.nanmax(ages))
    start = bin_years * np.floor(age_min / bin_years)
    stop = bin_years * np.ceil(age_max / bin_years)
    boundaries = np.arange(start, stop + bin_years, bin_years, dtype=float)
    if boundaries.size < 2:
        boundaries = np.array([start, start + bin_years], dtype=float)

    n_bins = int(boundaries.size - 1)
    cmap = plt.get_cmap(args.cmap, 10)
    norm = BoundaryNorm(boundaries, ncolors=n_bins, clip=True)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    markers = {
        "radiocarbon": "o",
        "cosmogenic": "^",
    }
    type_order = ["cosmogenic", "radiocarbon"]

    mappable = None
    for obs_type in type_order:
        sub = df.loc[df[args.type_col] == obs_type]
        if sub.empty:
            continue
        sc = ax.scatter(
            sub[args.x_col],
            sub[args.y_col],
            c=sub[args.age_col],
            cmap=cmap,
            norm=norm,
            s=float(args.s),
            alpha=float(args.alpha),
            marker=markers.get(obs_type, "o"),
            linewidths=0.0,
            label=obs_type,
        )
        mappable = sc

    if mappable is None:
        raise SystemExit(f"No rows with {args.type_col!r} in {type_order}.")

    # Discrete colorbar with 1000-year bin edges.
    cbar = fig.colorbar(mappable, ax=ax, boundaries=boundaries, ticks=boundaries)
    cbar.set_label(f"{args.age_col} (years); {int(bin_years):d}-year bins")
    if boundaries.size > 25:
        step = int(np.ceil(boundaries.size / 15))
        cbar.set_ticks(boundaries[::step])

    ax.set_xlabel(args.x_col)
    ax.set_ylabel(args.y_col)
    ax.set_title("Combined ages colored by age (discrete bins)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5, alpha=0.3)

    # Marker legend (types only; colors are handled by the colorbar).
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title=args.type_col, loc="best", frameon=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=int(args.dpi))
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()

