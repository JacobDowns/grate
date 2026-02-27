#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _to_float(x: str) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def _pick_first(existing: set[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in existing:
            return c
    return None


def _load_age_points(
    path: Path,
    *,
    x_col: str,
    y_col: str,
    age_col: str,
    type_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    ages: list[float] = []
    is_carbon: list[bool] = []

    with path.open(newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise SystemExit(f"Missing header: {path}")

        missing = {x_col, y_col, age_col, type_col} - set(r.fieldnames)
        if missing:
            raise SystemExit(f"Missing required columns in {path}: {sorted(missing)}")

        for row in r:
            x = _to_float(row.get(x_col, ""))
            y = _to_float(row.get(y_col, ""))
            age = _to_float(row.get(age_col, ""))
            if x is None or y is None or age is None:
                continue
            t = (row.get(type_col) or "").strip().lower()
            xs.append(x)
            ys.append(y)
            ages.append(age)
            is_carbon.append(t == "radiocarbon")

    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(ages, dtype=float),
        np.asarray(is_carbon, dtype=bool),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot modern Greenland ice thickness (from NetCDF) and overlay age points from combined_ages.csv, "
            "highlighting ages below a threshold in red."
        )
    )
    parser.add_argument(
        "--thickness-nc",
        type=Path,
        default=Path("data/modern_fields_native.nc"),
        help="Modern fields NetCDF path (default: data/modern_fields_native.nc).",
    )
    parser.add_argument(
        "--combined-ages",
        type=Path,
        default=Path("data/age_data/combined_ages.csv"),
        help="Combined ages CSV path (default: data/age_data/combined_ages.csv).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/modern_thickness_with_ages.png"),
        help="Output image path (default: output/modern_thickness_with_ages.png).",
    )
    parser.add_argument(
        "--age-threshold-years",
        type=float,
        default=6000.0,
        help="Highlight ages strictly below this (default: 6000).",
    )
    parser.add_argument("--x-col", type=str, default="x_3413", help="Age CSV X column (default: x_3413).")
    parser.add_argument("--y-col", type=str, default="y_3413", help="Age CSV Y column (default: y_3413).")
    parser.add_argument("--age-col", type=str, default="age_mean", help="Age CSV age column (default: age_mean).")
    parser.add_argument("--type-col", type=str, default="obs_type", help="Age CSV type column (default: obs_type).")
    parser.add_argument(
        "--thickness-var",
        type=str,
        default="",
        help="Thickness variable name in the NetCDF (default: auto-detect).",
    )
    parser.add_argument("--cmap", type=str, default="viridis", help="Thickness colormap (default: viridis).")
    parser.add_argument("--dpi", type=int, default=400, help="Output DPI (default: 400).")
    parser.add_argument("--figsize", type=float, nargs=2, default=(8.0, 8.0), metavar=("W", "H"))
    parser.add_argument("--s-all", type=float, default=3.0, help="Marker size for all points (default: 3).")
    parser.add_argument(
        "--s-highlight",
        type=float,
        default=12.0,
        help="Marker size for highlighted points (default: 12).",
    )
    parser.add_argument("--alpha-all", type=float, default=0.35, help="Alpha for all points (default: 0.35).")
    parser.add_argument("--alpha-highlight", type=float, default=0.9, help="Alpha for highlighted (default: 0.9).")
    parser.add_argument("--show", action="store_true", help="Show interactive window instead of only saving.")
    args = parser.parse_args()

    try:
        import xarray as xr
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency: xarray. Install the project deps (e.g. `uv sync`) "
            "or `python -m pip install xarray netCDF4 h5netcdf`."
        ) from e

    import matplotlib.pyplot as plt

    if not np.isfinite(float(args.age_threshold_years)):
        raise SystemExit(f"--age-threshold-years must be finite; got {args.age_threshold_years!r}")

    # Load thickness.
    if not args.thickness_nc.exists():
        raise SystemExit(f"NetCDF not found: {args.thickness_nc}")

    open_errors: list[str] = []
    ds = None
    for engine in (None, "h5netcdf", "netcdf4"):
        try:
            ds = xr.open_dataset(args.thickness_nc) if engine is None else xr.open_dataset(args.thickness_nc, engine=engine)
            break
        except Exception as e:  # noqa: BLE001
            open_errors.append(f"engine={engine!r}: {type(e).__name__}: {e}")
    if ds is None:
        raise SystemExit("Failed to open NetCDF with xarray:\n" + "\n".join(open_errors))

    try:
        if args.thickness_var.strip():
            if args.thickness_var not in ds.data_vars:
                raise SystemExit(f"--thickness-var {args.thickness_var!r} not found. Available: {sorted(ds.data_vars)}")
            thickness_name = args.thickness_var
        else:
            candidates = [
                "thickness",
                "ice_thickness",
                "IceThickness",
                "H",
                "thk",
                "thk_m",
                "thickness_m",
            ]
            thickness_name = _pick_first(set(ds.data_vars), candidates)
            if thickness_name is None:
                # Fallback: choose largest 2D numeric var.
                best = None
                best_size = -1
                for name, da in ds.data_vars.items():
                    if da.ndim < 2:
                        continue
                    if da.dtype.kind not in {"f", "i", "u"}:
                        continue
                    size = int(np.prod(da.shape))
                    if size > best_size:
                        best = name
                        best_size = size
                if best is None:
                    raise SystemExit(f"Could not auto-detect thickness var. Available: {sorted(ds.data_vars)}")
                thickness_name = best

        thk = ds[thickness_name]
        if thk.ndim < 2:
            raise SystemExit(f"Thickness variable {thickness_name!r} is not 2D+ (dims={thk.dims}).")

        # Determine plotting coordinates.
        coord_names = set(ds.coords)
        x_name = _pick_first(coord_names, ["x", "X", "easting", "xc", "lon", "longitude"])
        y_name = _pick_first(coord_names, ["y", "Y", "northing", "yc", "lat", "latitude"])

        # Prefer thickness dims if present as coords.
        if x_name is None:
            for d in reversed(thk.dims):
                if d in ds.coords and ds[d].ndim == 1:
                    x_name = d
                    break
        if y_name is None:
            for d in thk.dims:
                if d in ds.coords and ds[d].ndim == 1:
                    y_name = d
                    break

        if x_name is None or y_name is None:
            raise SystemExit(
                f"Could not determine x/y coordinates from NetCDF. Coords: {sorted(ds.coords)}. "
                f"Thickness dims: {thk.dims}. Pass a different NetCDF or adjust variable/coords."
            )

        x = ds[x_name].to_numpy()
        y = ds[y_name].to_numpy()
        z = thk.to_numpy()

    finally:
        ds.close()

    # Normalize thickness array shape to 2D for plotting.
    if z.ndim > 2:
        z = z.squeeze()
    if z.ndim != 2:
        raise SystemExit(f"Expected thickness to resolve to 2D after squeeze; got shape={z.shape}.")

    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y)
    elif x.ndim == 2 and y.ndim == 2:
        X, Y = x, y
    else:
        raise SystemExit(f"Unsupported x/y shapes: x={x.shape}, y={y.shape}")

    # Load ages.
    xs, ys, ages, _is_carbon = _load_age_points(
        args.combined_ages,
        x_col=args.x_col,
        y_col=args.y_col,
        age_col=args.age_col,
        type_col=args.type_col,
    )
    highlight = ages < float(args.age_threshold_years)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    m = ax.pcolormesh(X, Y, z, cmap=args.cmap, shading="auto")
    cbar = fig.colorbar(m, ax=ax)
    cbar.set_label(f"{thickness_name} (units as in NetCDF)")

    # All ages (gray), highlighted subset (red).
    ax.scatter(xs, ys, s=float(args.s_all), c="k", alpha=float(args.alpha_all), linewidths=0.0, label="all ages")
    if np.any(highlight):
        ax.scatter(
            xs[highlight],
            ys[highlight],
            s=float(args.s_highlight),
            c="red",
            alpha=float(args.alpha_highlight),
            edgecolors="k",
            linewidths=0.3,
            label=f"age < {float(args.age_threshold_years):g} yr",
        )

    ax.set_title("Modern Greenland ice thickness with age observations")
    ax.set_xlabel(f"{x_name} (EPSG:3413 assumed)")
    ax.set_ylabel(f"{y_name} (EPSG:3413 assumed)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=int(args.dpi))
    print(f"Wrote: {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
