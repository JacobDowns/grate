#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunMap:
    run_id: int
    path: Path
    age: "np.ndarray"  # (y, x) float32 with NaNs
    age_norm: "np.ndarray"  # (y, x) float32 with NaNs
    scale: float


def _parse_run_id(path: Path) -> int:
    m = re.search(r"run_(\d+)", path.name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not parse run id from {path.name}")
    return int(m.group(1))


def _load_deglaciation_map(nc_path: Path, *, var: str) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", dict]:
    import numpy as np
    import h5netcdf

    with h5netcdf.File(nc_path, "r") as f:
        if var not in f.variables:
            raise KeyError(f"{nc_path} missing {var!r}. Available: {sorted(f.variables)}")
        if "x" not in f.variables or "y" not in f.variables:
            raise KeyError(f"{nc_path} missing x/y coordinate variables.")

        age = np.asarray(f.variables[var][:], dtype=np.float32)
        x = np.asarray(f.variables["x"][:], dtype=np.float64)
        y = np.asarray(f.variables["y"][:], dtype=np.float64)
        attrs = dict(f.variables[var].attrs)
        return age, x, y, attrs


def _normalize_age(age: "np.ndarray") -> tuple["np.ndarray", float]:
    import numpy as np

    finite = np.isfinite(age)
    if not np.any(finite):
        return age.copy(), float("nan")
    scale = float(np.nanmax(age))
    if not (scale > 0):
        # All-zero or negative (shouldn't happen): keep as-is.
        return age.copy(), scale
    out = age.astype(np.float32, copy=True)
    out[finite] = out[finite] / scale
    return out, scale


def _snapshot_pca(
    X: "np.ndarray", n_modes: int, *, center: bool = True
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
    """
    Snapshot PCA for X shaped (n_runs, n_pix), returns:
      mean (n_pix,),
      modes (n_modes, n_pix),
      scores (n_runs, n_modes),
      explained_variance_ratio (n_modes,),
      explained_variance (n_modes,)
    """
    import numpy as np

    n_runs, n_pix = X.shape
    n_modes = int(min(n_modes, n_runs))

    if center:
        mean = X.mean(axis=0)
        Xc = X - mean
    else:
        mean = X[:1, :].mean(axis=0) * 0.0
        Xc = X

    # Snapshot covariance in run-space
    C = Xc @ Xc.T
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals = np.maximum(eigvals, 0.0)
    svals = np.sqrt(eigvals)

    total = float(np.sum(eigvals))
    if total > 0:
        evr = eigvals[:n_modes] / total
    else:
        evr = np.zeros((n_modes,), dtype=np.float64)

    modes = np.zeros((n_modes, n_pix), dtype=np.float64)
    scores = np.zeros((n_runs, n_modes), dtype=np.float64)

    for i in range(n_modes):
        s = float(svals[i])
        u = eigvecs[:, i]
        scores[:, i] = u * s
        if s > 0:
            modes[i, :] = (Xc.T @ u) / s
        else:
            modes[i, :] = 0.0

    return mean, modes, scores, evr, eigvals[:n_modes]


def main() -> None:
    default_input = Path("data/issm_deglaciation_age_nc")
    default_out = Path("data/deglaciation_snapshot_pca.nc")

    parser = argparse.ArgumentParser(
        description=(
            "Snapshot PCA on per-run deglaciation maps (2D). Reads 'deglaciation_age' from run_*.nc files,\n"
            "normalizes each map so oldest time -> 1 and most recent -> 0, masks out NaNs (intersection mask),\n"
            "and writes maps + PCA modes to a single NetCDF.\n"
            "\n"
            "This script also supports an alternate mean definition that, for modern ice-free pixels,\n"
            "averages only over runs where the pixel deglaciated (i.e. deglaciation_age_norm > 0),\n"
            "ignoring runs where the pixel never deglaciated (0)."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=default_input, help="Directory containing per-run NetCDF files")
    parser.add_argument(
        "--pattern", type=str, default="*_deglaciation_age.nc", help="Glob pattern (default: *_deglaciation_age.nc)"
    )
    parser.add_argument("--var", type=str, default="deglaciation_age", help="Variable name to use (default: deglaciation_age)")
    parser.add_argument("--modes", type=int, default=20, help="Number of modes to save (default: 20)")
    parser.add_argument("--out", type=Path, default=default_out, help="Output NetCDF path")
    parser.add_argument("--compression-level", type=int, default=4, help="Gzip compression level (0-9) for output (default: 4)")
    parser.add_argument(
        "--qgreenland-bed",
        type=Path,
        default=Path("data/qgreenland/bedmachine_bed.tif"),
        help="Bed elevation GeoTIFF (will be reprojected to the output grid).",
    )
    parser.add_argument(
        "--qgreenland-thickness",
        type=Path,
        default=Path("data/qgreenland/bedmap_thickness.tif"),
        help="Modern thickness GeoTIFF (will be reprojected to the output grid).",
    )
    args = parser.parse_args()

    import numpy as np
    import rasterio.enums
    import rasterio.transform
    import rioxarray  # noqa: F401
    import xarray as xr

    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {args.pattern!r} in {args.input_dir}")

    runs: list[RunMap] = []
    x0 = y0 = None
    var_attrs: dict = {}

    for p in files:
        run_id = _parse_run_id(p)
        age, x, y, attrs = _load_deglaciation_map(p, var=args.var)
        age_norm, scale = _normalize_age(age)

        if x0 is None:
            x0, y0 = x, y
            var_attrs = attrs
        else:
            if x.shape != x0.shape or y.shape != y0.shape or not (np.allclose(x, x0) and np.allclose(y, y0)):
                raise ValueError(f"{p} has different x/y grid than the first file.")

        runs.append(RunMap(run_id=run_id, path=p, age=age, age_norm=age_norm, scale=scale))

    runs.sort(key=lambda r: r.run_id)
    n_runs = len(runs)

    # Validate NaN mask consistency and build intersection mask.
    masks = [np.isfinite(r.age_norm) for r in runs]
    valid_mask = masks[0].copy()
    for m in masks[1:]:
        valid_mask &= m

    same_mask = all(np.array_equal(m, masks[0]) for m in masks[1:])
    if not same_mask:
        n0 = int(np.size(masks[0]) - np.count_nonzero(masks[0]))
        ni = int(np.size(valid_mask) - np.count_nonzero(valid_mask))
        print(f"Warning: NaN masks differ across runs. Using intersection valid mask. (nan first={n0}, nan intersection={ni})")

    # Stack maps (run, y, x)
    ny, nx = runs[0].age.shape
    age_stack = np.stack([r.age for r in runs], axis=0).astype(np.float32, copy=False)
    age_norm_stack = np.stack([r.age_norm for r in runs], axis=0).astype(np.float32, copy=False)
    scales = np.array([r.scale for r in runs], dtype=np.float64)
    mins = np.array([float(np.nanmin(r.age)) for r in runs], dtype=np.float64)
    run_ids = np.array([r.run_id for r in runs], dtype=np.int32)
    global_max_age_years = float(np.nanmax(scales))

    # Reproject qgreenland rasters (bed elevation + modern thickness) to this grid for context and masking.
    ny, nx = runs[0].age.shape
    dx = float(x0[1] - x0[0]) if x0.size > 1 else float("nan")
    dy = float(y0[1] - y0[0]) if y0.size > 1 else float("nan")
    left = float(np.nanmin(x0) - 0.5 * dx)
    right = float(np.nanmax(x0) + 0.5 * dx)
    bottom = float(np.nanmin(y0) - 0.5 * dy)
    top = float(np.nanmax(y0) + 0.5 * dy)
    transform = rasterio.transform.from_bounds(left, bottom, right, top, width=int(nx), height=int(ny))

    template = xr.DataArray(
        np.zeros((ny, nx), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": ("y", y0), "x": ("x", x0)},
        name="template",
    ).rio.write_crs("EPSG:3413", inplace=False).rio.write_transform(transform, inplace=False)

    def _reproject_tif(path: Path, *, name: str) -> xr.DataArray:
        da = rioxarray.open_rasterio(path, masked=True).squeeze()
        # rioxarray will carry CRS/transform from the GeoTIFF; reproject to our template grid.
        out = da.rio.reproject_match(template, resampling=rasterio.enums.Resampling.bilinear)
        out = out.rename(name).astype("float32")
        out = out.assign_coords({"y": ("y", y0), "x": ("x", x0)})
        return out

    bed_elevation = _reproject_tif(args.qgreenland_bed, name="bed_elevation")
    modern_thickness = _reproject_tif(args.qgreenland_thickness, name="modern_thickness")

    # Build PCA matrix (run, pix) using the intersection mask.
    flat_idx = np.nonzero(valid_mask.reshape(-1))[0]
    X = age_norm_stack.reshape(n_runs, -1)[:, flat_idx].astype(np.float64, copy=False)

    # Mean deglaciation history (normalized 0..1) across simulations (original approach: include 0s).
    mean_hist_flat_all = X.mean(axis=0)

    # Alternate mean for modern ice-free pixels:
    # average only over runs where the pixel deglaciated (age_norm > 0), ignoring 0s.
    thk_flat = modern_thickness.values.reshape(-1)[flat_idx].astype(np.float64, copy=False)
    modern_ice_free = np.isfinite(thk_flat) & (thk_flat <= 0.0)
    mean_hist_flat_deglac_only = mean_hist_flat_all.copy()
    if np.any(modern_ice_free):
        # Compute mean over only the "deglaciated" runs (age_norm > 0), without triggering warnings
        # for all-missing columns.
        X_ice_free = X[:, modern_ice_free]
        deglac = X_ice_free > 0
        counts = deglac.sum(axis=0).astype(np.float64)
        sums = np.where(deglac, X_ice_free, 0.0).sum(axis=0).astype(np.float64)
        m = np.zeros_like(sums, dtype=np.float64)
        ok = counts > 0
        m[ok] = sums[ok] / counts[ok]
        mean_hist_flat_deglac_only[modern_ice_free] = m

    # Deviations from each mean history.
    X_anom_all = X - mean_hist_flat_all
    X_anom_deglac_only = X - mean_hist_flat_deglac_only

    _, modes_flat_all, scores_all, evr_all, ev_all = _snapshot_pca(X_anom_all, n_modes=int(args.modes), center=False)
    _, modes_flat_deglac, scores_deglac, evr_deglac, ev_deglac = _snapshot_pca(
        X_anom_deglac_only, n_modes=int(args.modes), center=False
    )
    n_modes = int(min(modes_flat_all.shape[0], modes_flat_deglac.shape[0]))
    cev_all = np.cumsum(evr_all[:n_modes], dtype=np.float64)
    cev_deglac = np.cumsum(evr_deglac[:n_modes], dtype=np.float64)

    # Expand back to (mode, y, x) with NaNs outside valid_mask.
    mean_hist_map_all = np.full((ny * nx,), np.nan, dtype=np.float32)
    mean_hist_map_all[flat_idx] = mean_hist_flat_all.astype(np.float32)
    mean_hist_map_all = mean_hist_map_all.reshape(ny, nx)

    mean_hist_map_deglac = np.full((ny * nx,), np.nan, dtype=np.float32)
    mean_hist_map_deglac[flat_idx] = mean_hist_flat_deglac_only.astype(np.float32)
    mean_hist_map_deglac = mean_hist_map_deglac.reshape(ny, nx)

    anom_stack_all = age_norm_stack.copy()
    anom_stack_all[:, ~valid_mask] = np.nan
    anom_stack_all = anom_stack_all - mean_hist_map_all[None, :, :]

    anom_stack_deglac = age_norm_stack.copy()
    anom_stack_deglac[:, ~valid_mask] = np.nan
    anom_stack_deglac = anom_stack_deglac - mean_hist_map_deglac[None, :, :]

    modes_map_all = np.full((n_modes, ny * nx), np.nan, dtype=np.float32)
    for i in range(n_modes):
        modes_map_all[i, flat_idx] = modes_flat_all[i, :].astype(np.float32)
    modes_map_all = modes_map_all.reshape(n_modes, ny, nx)

    modes_map_deglac = np.full((n_modes, ny * nx), np.nan, dtype=np.float32)
    for i in range(n_modes):
        modes_map_deglac[i, flat_idx] = modes_flat_deglac[i, :].astype(np.float32)
    modes_map_deglac = modes_map_deglac.reshape(n_modes, ny, nx)

    ds = xr.Dataset(
        coords={
            "run": ("run", run_ids),
            "mode": ("mode", np.arange(1, n_modes + 1, dtype=np.int32)),
            "y": ("y", y0),
            "x": ("x", x0),
        },
        data_vars={
            "deglaciation_age": (("run", "y", "x"), age_stack),
            "deglaciation_age_norm": (("run", "y", "x"), age_norm_stack),
            "deglaciation_age_mean_norm_all": (("y", "x"), mean_hist_map_all),
            "deglaciation_age_anom_norm_all": (("run", "y", "x"), anom_stack_all),
            "pca_mode_norm_all": (("mode", "y", "x"), modes_map_all),
            "pca_score_all": (("run", "mode"), scores_all[:, :n_modes].astype(np.float32)),
            "explained_variance_all": (("mode",), ev_all[:n_modes].astype(np.float64)),
            "explained_variance_ratio_all": (("mode",), evr_all[:n_modes].astype(np.float64)),
            "cumulative_explained_variance_ratio_all": (("mode",), cev_all.astype(np.float64)),

            "deglaciation_age_mean_norm_deglac_only": (("y", "x"), mean_hist_map_deglac),
            "deglaciation_age_anom_norm_deglac_only": (("run", "y", "x"), anom_stack_deglac),
            "pca_mode_norm_deglac_only": (("mode", "y", "x"), modes_map_deglac),
            "pca_score_deglac_only": (("run", "mode"), scores_deglac[:, :n_modes].astype(np.float32)),
            "explained_variance_deglac_only": (("mode",), ev_deglac[:n_modes].astype(np.float64)),
            "explained_variance_ratio_deglac_only": (("mode",), evr_deglac[:n_modes].astype(np.float64)),
            "cumulative_explained_variance_ratio_deglac_only": (("mode",), cev_deglac.astype(np.float64)),

            "deglaciation_age_scale": (("run",), scales),
            "deglaciation_age_min_years": (("run",), mins),
            "deglaciation_age_max_years": (("run",), scales),
            "valid_mask": (("y", "x"), valid_mask),
            "bed_elevation": (("y", "x"), bed_elevation.values.astype(np.float32, copy=False)),
            "modern_thickness": (("y", "x"), modern_thickness.values.astype(np.float32, copy=False)),
        },
        attrs={
            "source_dir": str(args.input_dir),
            "source_pattern": args.pattern,
            "source_var": args.var,
            "normalization": "per-run: deglaciation_age_norm = deglaciation_age / max_finite(deglaciation_age); oldest=1, most recent=0",
            "pca_input_all": "PCA is performed on deglaciation_age_anom_norm_all (norm - mean_norm_all)",
            "pca_input_deglac_only": (
                "PCA is performed on deglaciation_age_anom_norm_deglac_only (norm - mean_norm_deglac_only); "
                "mean_norm_deglac_only ignores 0-values for modern ice-free pixels"
            ),
            "nan_mask": "PCA uses intersection of finite pixels across runs",
            "age_norm_min_years": 0.0,
            "age_norm_max_years": global_max_age_years,
            "qgreenland_bed": str(args.qgreenland_bed),
            "qgreenland_thickness": str(args.qgreenland_thickness),
        },
    )

    # Carry through a few variable attrs, if present.
    for k in ("units", "threshold_m", "note", "long_name"):
        if k in var_attrs:
            ds["deglaciation_age"].attrs[k] = var_attrs[k]
            ds["deglaciation_age_norm"].attrs[k] = var_attrs[k]
            ds["deglaciation_age_mean_norm_all"].attrs[k] = var_attrs[k]
            ds["deglaciation_age_anom_norm_all"].attrs[k] = var_attrs[k]
            ds["deglaciation_age_mean_norm_deglac_only"].attrs[k] = var_attrs[k]
            ds["deglaciation_age_anom_norm_deglac_only"].attrs[k] = var_attrs[k]

    ds["deglaciation_age_norm"].attrs["units"] = "1"
    ds["deglaciation_age_mean_norm_all"].attrs["units"] = "1"
    ds["deglaciation_age_mean_norm_all"].attrs["long_name"] = "mean deglaciation history across runs (normalized 0..1), including 0s"
    ds["deglaciation_age_anom_norm_all"].attrs["units"] = "1"
    ds["deglaciation_age_anom_norm_all"].attrs["long_name"] = "deglaciation anomaly (all-mean) = norm - mean_norm_all"
    ds["pca_mode_norm_all"].attrs["units"] = "1"
    ds["pca_mode_norm_all"].attrs["long_name"] = "snapshot PCA spatial modes of (norm - mean_norm_all)"
    ds["pca_score_all"].attrs["long_name"] = "snapshot PCA scores per run (all-mean)"
    ds["explained_variance_all"].attrs["long_name"] = "variance explained by each PCA component (all-mean)"
    ds["explained_variance_ratio_all"].attrs["long_name"] = "fraction of variance explained by each PCA component (all-mean)"
    ds["cumulative_explained_variance_ratio_all"].attrs["long_name"] = "cumulative fraction of variance explained (all-mean)"

    ds["deglaciation_age_mean_norm_deglac_only"].attrs["units"] = "1"
    ds["deglaciation_age_mean_norm_deglac_only"].attrs["long_name"] = (
        "mean deglaciation history across runs (normalized 0..1); for modern ice-free pixels, "
        "ignore runs where the pixel never deglaciated (0)"
    )
    ds["deglaciation_age_anom_norm_deglac_only"].attrs["units"] = "1"
    ds["deglaciation_age_anom_norm_deglac_only"].attrs["long_name"] = "deglaciation anomaly (deglac-only mean) = norm - mean_norm_deglac_only"
    ds["pca_mode_norm_deglac_only"].attrs["units"] = "1"
    ds["pca_mode_norm_deglac_only"].attrs["long_name"] = "snapshot PCA spatial modes of (norm - mean_norm_deglac_only)"
    ds["pca_score_deglac_only"].attrs["long_name"] = "snapshot PCA scores per run (deglac-only mean)"
    ds["explained_variance_deglac_only"].attrs["long_name"] = "variance explained by each PCA component (deglac-only mean)"
    ds["explained_variance_ratio_deglac_only"].attrs["long_name"] = "fraction of variance explained by each PCA component (deglac-only mean)"
    ds["cumulative_explained_variance_ratio_deglac_only"].attrs["long_name"] = "cumulative fraction of variance explained (deglac-only mean)"

    ds["bed_elevation"].attrs["units"] = "m"
    ds["bed_elevation"].attrs["long_name"] = "bedrock elevation (reprojected to model grid)"
    ds["modern_thickness"].attrs["units"] = "m"
    ds["modern_thickness"].attrs["long_name"] = "modern ice thickness (reprojected to model grid)"

    ds["deglaciation_age_min_years"].attrs["units"] = "years"
    ds["deglaciation_age_max_years"].attrs["units"] = "years"
    ds["deglaciation_age_min_years"].attrs["long_name"] = "per-run minimum deglaciation age (years)"
    ds["deglaciation_age_max_years"].attrs["long_name"] = "per-run maximum deglaciation age (years)"

    args.out.parent.mkdir(parents=True, exist_ok=True)

    clevel = int(args.compression_level)
    if not (0 <= clevel <= 9):
        raise ValueError("--compression-level must be between 0 and 9.")

    encoding = {}
    if clevel > 0:
        # Reasonable chunking for (run, y, x) and (mode, y, x).
        encoding = {
            "deglaciation_age": {"compression": "gzip", "compression_opts": clevel, "chunksizes": (1, 256, 256)},
            "deglaciation_age_norm": {"compression": "gzip", "compression_opts": clevel, "chunksizes": (1, 256, 256)},
            "deglaciation_age_mean_norm_all": {"compression": "gzip", "compression_opts": clevel, "chunksizes": (256, 256)},
            "deglaciation_age_anom_norm_all": {"compression": "gzip", "compression_opts": clevel, "chunksizes": (1, 256, 256)},
            "pca_mode_norm_all": {"compression": "gzip", "compression_opts": clevel, "chunksizes": (1, 256, 256)},
            "pca_score_all": {"compression": "gzip", "compression_opts": clevel},
            "explained_variance_all": {"compression": "gzip", "compression_opts": clevel},
            "explained_variance_ratio_all": {"compression": "gzip", "compression_opts": clevel},
            "cumulative_explained_variance_ratio_all": {"compression": "gzip", "compression_opts": clevel},

            "deglaciation_age_mean_norm_deglac_only": {"compression": "gzip", "compression_opts": clevel, "chunksizes": (256, 256)},
            "deglaciation_age_anom_norm_deglac_only": {"compression": "gzip", "compression_opts": clevel, "chunksizes": (1, 256, 256)},
            "pca_mode_norm_deglac_only": {"compression": "gzip", "compression_opts": clevel, "chunksizes": (1, 256, 256)},
            "pca_score_deglac_only": {"compression": "gzip", "compression_opts": clevel},
            "explained_variance_deglac_only": {"compression": "gzip", "compression_opts": clevel},
            "explained_variance_ratio_deglac_only": {"compression": "gzip", "compression_opts": clevel},
            "cumulative_explained_variance_ratio_deglac_only": {"compression": "gzip", "compression_opts": clevel},
            "deglaciation_age_min_years": {"compression": "gzip", "compression_opts": clevel},
            "deglaciation_age_max_years": {"compression": "gzip", "compression_opts": clevel},
            "valid_mask": {"compression": "gzip", "compression_opts": clevel, "chunksizes": (256, 256)},
            "bed_elevation": {"compression": "gzip", "compression_opts": clevel, "chunksizes": (256, 256)},
            "modern_thickness": {"compression": "gzip", "compression_opts": clevel, "chunksizes": (256, 256)},
        }

    ds.to_netcdf(args.out, engine="h5netcdf", encoding=encoding)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
