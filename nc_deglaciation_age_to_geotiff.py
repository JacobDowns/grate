#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _import_or_explain(module: str, install_hint: str) -> None:
    try:
        __import__(module)
    except ModuleNotFoundError as e:
        if e.name != module:
            raise
        print(
            f"Missing dependency: {module}\n\n"
            "Install dependencies, then re-run. For this repo:\n"
            f"{install_hint}\n",
            file=sys.stderr,
        )
        raise


def _parse_nodata(value: str) -> float:
    v = value.strip().lower()
    if v in {"nan", "+nan", "-nan"}:
        return float("nan")
    try:
        return float(value)
    except ValueError as e:
        raise SystemExit(f"Invalid --nodata value: {value!r}") from e


def _roughly_regular_spacing(coords) -> bool:
    import numpy as np

    c = np.asarray(coords, dtype=np.float64)
    if c.size < 2:
        return True
    diffs = np.diff(c)
    if not np.isfinite(diffs).all():
        return False
    step = float(np.median(diffs))
    if step == 0:
        return False
    return bool(np.allclose(diffs, step, rtol=1e-6, atol=1e-6 * abs(step)))


def _infer_crs(ds) -> str | None:
    # Prefer CF-style `crs` variable attrs if present.
    if "crs" in ds.variables:
        attrs = dict(ds["crs"].attrs)
        epsg_code = attrs.get("epsg_code")
        if isinstance(epsg_code, str) and epsg_code.strip():
            return epsg_code.strip()
        spatial_ref = attrs.get("spatial_ref")
        if isinstance(spatial_ref, str) and spatial_ref.strip():
            return spatial_ref.strip()
        crs_wkt = attrs.get("crs_wkt")
        if isinstance(crs_wkt, str) and crs_wkt.strip():
            return crs_wkt.strip()

    # Fall back to a common global attr written by this repo's pipeline.
    mesh_epsg = ds.attrs.get("mesh_epsg")
    if isinstance(mesh_epsg, (str, int)) and str(mesh_epsg).strip():
        return f"EPSG:{int(mesh_epsg)}"
    return None


def _prepare_da_for_geotiff(da, *, zero_as_nodata: bool):
    import numpy as np

    if tuple(da.dims) != ("y", "x"):
        raise ValueError(f"Expected dims ('y','x'); got {tuple(da.dims)!r}")

    x = da["x"].values
    y = da["y"].values
    if not _roughly_regular_spacing(x) or not _roughly_regular_spacing(y):
        raise ValueError("x/y coordinates are not regularly spaced; cannot write GeoTIFF safely.")

    # Ensure GIS-friendly axis ordering: x ascending; y descending (north-up, negative y pixel size).
    da = da.sortby("x")
    da = da.sortby("y", ascending=False)

    da = da.astype(np.float32, copy=False)
    if zero_as_nodata:
        da = da.where(da != 0)
    return da


def _write_geotiff(da, *, out: Path, crs: str, nodata: float, compress: str) -> None:
    import numpy as np

    if math.isnan(nodata):
        da_filled = da
    else:
        da_filled = da.where(np.isfinite(da), other=np.float32(nodata))

    da_filled = da_filled.rio.set_spatial_dims(x_dim="x", y_dim="y")
    da_filled = da_filled.rio.write_crs(crs)
    da_filled = da_filled.rio.write_nodata(nodata)
    da_filled = da_filled.rio.write_transform(da_filled.rio.transform(recalc=True))

    da_filled.rio.to_raster(
        out,
        driver="GTiff",
        dtype="float32",
        compress=str(compress),
        tiled=True,
        BIGTIFF="IF_SAFER",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-run deglaciation age maps from NetCDF as GeoTIFFs (for QGIS).\n"
            "By default, writes one GeoTIFF per input NetCDF into the same folder."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/issm_extracted_nc"))
    parser.add_argument("--pattern", type=str, default="run_*_tot.nc")
    parser.add_argument("--var", type=str, default="deglaciation_age", help="2D variable to export (y,x).")
    parser.add_argument(
        "--no-mean",
        action="store_true",
        help="Do not write the across-run mean GeoTIFF (default: write it).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as --input-dir).",
    )
    parser.add_argument("--suffix", type=str, default=None, help="Optional output filename suffix (default: _<var>).")
    parser.add_argument("--crs", type=str, default=None, help="Override CRS (e.g. EPSG:3413).")
    parser.add_argument(
        "--nodata",
        type=str,
        default="-9999",
        help="NoData value to write (default: -9999). Use 'nan' to keep NaNs.",
    )
    parser.add_argument(
        "--zero-as-nodata",
        action="store_true",
        help="Treat zeros as NoData (off by default; zeros can be meaningful in these maps).",
    )
    parser.add_argument("--compress", type=str, default="DEFLATE", help="GeoTIFF compression (default: DEFLATE).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing GeoTIFFs.")
    args = parser.parse_args()

    install_hint = (
        "  uv sync\n"
        "  uv run python nc_deglaciation_age_to_geotiff.py\n"
        "\n"
        "Or (conda):\n"
        "  conda env create -f environment.yml\n"
        "  conda activate grate\n"
        "  python nc_deglaciation_age_to_geotiff.py\n"
    )
    _import_or_explain("xarray", install_hint)
    _import_or_explain("rioxarray", install_hint)

    import numpy as np  # noqa: E402
    import xarray as xr  # noqa: E402
    import rioxarray  # noqa: F401,E402

    input_dir: Path = args.input_dir
    out_dir: Path = input_dir if args.out_dir is None else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        print(f"No files matched {args.pattern!r} in {input_dir}", file=sys.stderr)
        return 2

    nodata = _parse_nodata(args.nodata)

    # Determine CRS once.
    crs: str | None = args.crs
    if crs is None:
        ds0 = xr.open_dataset(files[0], decode_times=False)
        try:
            crs = _infer_crs(ds0)
        finally:
            ds0.close()
    if not crs:
        raise ValueError(
            f"Could not infer CRS from {files[0]}. Pass --crs (e.g. --crs EPSG:3413) to write georeferenced GeoTIFFs."
        )

    import numpy as np  # noqa: E402
    import xarray as xr  # noqa: E402

    mean_sum = None
    mean_count = None
    mean_coords = None

    for p in files:
        out_name = p.stem
        suffix = args.suffix if args.suffix is not None else f"_{args.var}"
        out = out_dir / f"{out_name}{suffix}.tif"

        if out.exists() and not args.force:
            print(f"Skipping {p.name}: {out.name} already exists (use --force).")
            # Still include in mean computation (unless disabled).
            if args.no_mean:
                continue

        ds = xr.open_dataset(p, decode_times=False)
        try:
            if args.var not in ds:
                raise KeyError(f"{p} missing {args.var!r}. Available: {list(ds.data_vars)}")
            if "x" not in ds.coords or "y" not in ds.coords:
                raise KeyError(f"{p} missing x/y coordinates.")

            da = _prepare_da_for_geotiff(ds[args.var], zero_as_nodata=bool(args.zero_as_nodata))

            if not (out.exists() and not args.force):
                _write_geotiff(da, out=out, crs=crs, nodata=nodata, compress=str(args.compress))
                print(f"Wrote {out}")

            if not args.no_mean:
                a = da.values.astype(np.float64, copy=False)
                finite = np.isfinite(a)
                if mean_sum is None:
                    mean_sum = np.zeros_like(a, dtype=np.float64)
                    mean_count = np.zeros_like(a, dtype=np.int32)
                    mean_coords = {"x": da["x"].values.copy(), "y": da["y"].values.copy()}
                else:
                    if not (
                        np.array_equal(mean_coords["x"], da["x"].values)
                        and np.array_equal(mean_coords["y"], da["y"].values)
                    ):
                        raise ValueError(f"{p}: x/y grid differs from earlier files; cannot compute a mean map.")

                mean_sum[finite] += a[finite]
                mean_count[finite] += 1
        finally:
            ds.close()

    if not args.no_mean and mean_sum is not None and mean_count is not None and mean_coords is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = mean_sum / mean_count
        mean = np.where(mean_count > 0, mean, np.nan).astype(np.float32, copy=False)

        mean_da = xr.DataArray(
            mean,
            dims=("y", "x"),
            coords={"y": mean_coords["y"], "x": mean_coords["x"]},
            name=f"mean_{args.var}",
        )
        out_mean = out_dir / f"mean_{args.var}.tif"
        if out_mean.exists() and not args.force:
            print(f"Skipping mean: {out_mean.name} already exists (use --force).")
        else:
            _write_geotiff(
                mean_da, out=out_mean, crs=crs, nodata=nodata, compress=str(args.compress)
            )
            print(f"Wrote {out_mean}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
