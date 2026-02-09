#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _run_label(path: Path) -> str:
    # e.g. run_01_tot.nc -> run_01
    stem = path.stem
    if stem.endswith("_tot"):
        stem = stem[: -len("_tot")]
    return stem


def _load_age(nc_path: Path, *, var: str) -> "tuple[np.ndarray, np.ndarray, np.ndarray, dict]":
    import numpy as np
    import xarray as xr

    ds = xr.open_dataset(nc_path, decode_times=False)
    try:
        if var not in ds:
            raise KeyError(f"{nc_path} missing {var!r}. Available: {list(ds.data_vars)}")
        if "x" not in ds.coords or "y" not in ds.coords:
            raise KeyError(f"{nc_path} missing x/y coordinates.")

        age = ds[var]
        # Set zeros to NaN for transparent plotting.
        #age = age.where(age != 0)
        x = ds["x"].values.astype(np.float64)
        y = ds["y"].values.astype(np.float64)
        a = age.values.astype(np.float32)
        attrs = dict(age.attrs)
        return a, x, y, attrs
    finally:
        ds.close()


def _nan_minmax(a: "np.ndarray") -> tuple[float, float]:
    import numpy as np

    if not np.isfinite(a).any():
        return float("nan"), float("nan")
    return float(np.nanmin(a)), float(np.nanmax(a))


def main() -> None:
    default_input = Path("data/issm_extracted_nc")

    parser = argparse.ArgumentParser(
        description=(
            "Plot the deglaciation-age field for each run by looping through NetCDF files.\n"
            "Zeros are treated as NaN for plotting (transparent)."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=default_input, help="Directory containing run NetCDF files")
    parser.add_argument("--pattern", type=str, default="run_*_tot.nc", help="Glob pattern (default: run_*_tot.nc)")
    parser.add_argument("--var", type=str, default="deglaciation_age", help="Variable name to plot")
    parser.add_argument("--levels", type=int, default=32, help="Number of colormap levels (default: 32)")
    parser.add_argument(
        "--scale",
        choices=("global", "per-run"),
        default="global",
        help="Color scale: shared across runs or per-run (default: global)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="If set, write PNGs here instead of showing windows")
    parser.add_argument("--dpi", type=int, default=200, help="PNG DPI when saving (default: 200)")
    args = parser.parse_args()

    nc_files = sorted(args.input_dir.glob(args.pattern))
    if not nc_files:
        raise FileNotFoundError(f"No files matched {args.pattern!r} in {args.input_dir}")

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Load all ages (for min/max, and to avoid re-reading in global mode).
    loaded: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray, dict]] = []
    for p in nc_files:
        a, x, y, attrs = _load_age(p, var=args.var)
        loaded.append((p, a, x, y, attrs))

    global_vmin, global_vmax = float("nan"), float("nan")
    if args.scale == "global":
        mins = []
        maxs = []
        for _, a, _, _, _ in loaded:
            vmin, vmax = _nan_minmax(a)
            if np.isfinite(vmin) and np.isfinite(vmax):
                mins.append(vmin)
                maxs.append(vmax)
        if mins and maxs:
            global_vmin = float(np.min(mins))
            global_vmax = float(np.max(maxs))

    cmap = plt.get_cmap("magma", int(args.levels)).copy()
    cmap.set_bad(alpha=0.0)

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for p, a, x, y, attrs in loaded:
        label = _run_label(p)
        vmin, vmax = _nan_minmax(a)
        if args.scale == "global" and np.isfinite(global_vmin) and np.isfinite(global_vmax):
            vmin, vmax = global_vmin, global_vmax

        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            print(f"Skipping {p.name}: no finite data to plot.")
            continue
        if vmax == vmin:
            # Avoid a degenerate color scale (e.g. very short runs).
            vmax = vmin + 1.0

        boundaries = np.linspace(vmin, vmax, int(args.levels) + 1)
        norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

        fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
        mesh = ax.pcolormesh(x, y, a, cmap=cmap, norm=norm, shading="auto")
        cb = fig.colorbar(mesh, ax=ax, shrink=0.9)
        cb.set_label(attrs.get("units", ""))

        title = f"{label}: {args.var}"
        thr = attrs.get("threshold_m")
        if thr is not None:
            try:
                title += f" (threshold={float(thr):g} m)"
            except Exception:
                pass
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if args.out_dir is not None:
            out = args.out_dir / f"{label}_{args.var}.png"
            fig.savefig(out, dpi=int(args.dpi))
            plt.close(fig)
            print(f"Saved {out}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
