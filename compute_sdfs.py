#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RasterSpec:
    path: Path
    crs_wkt: str
    transform: "Affine"
    width: int
    height: int
    nodata: float | int | None


def _open_spec(path: Path) -> RasterSpec:
    import rasterio

    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"{path} has {src.count} bands; expected 1.")
        if src.crs is None:
            raise ValueError(f"{path} missing CRS.")
        return RasterSpec(
            path=path,
            crs_wkt=src.crs.to_wkt(),
            transform=src.transform,
            width=int(src.width),
            height=int(src.height),
            nodata=src.nodata,
        )


def _require_same_grid(a: RasterSpec, b: RasterSpec) -> None:
    if a.width != b.width or a.height != b.height:
        raise ValueError(
            f"Grid shapes differ: {a.path.name} is (h={a.height}, w={a.width}) but "
            f"{b.path.name} is (h={b.height}, w={b.width})."
        )
    if a.crs_wkt != b.crs_wkt:
        raise ValueError(f"CRS differs between {a.path} and {b.path}.")
    if a.transform != b.transform:
        raise ValueError(f"Affine transform differs between {a.path} and {b.path}.")


def _clip_window_to_bounds(
    *,
    col_off: int,
    row_off: int,
    width: int,
    height: int,
    max_width: int,
    max_height: int,
) -> tuple[int, int, int, int]:
    col_off = max(0, min(int(col_off), max_width - 1))
    row_off = max(0, min(int(row_off), max_height - 1))
    width = max(1, min(int(width), max_width - col_off))
    height = max(1, min(int(height), max_height - row_off))
    return col_off, row_off, width, height


def _infer_valid_data_window(
    path: Path,
    *,
    sample_factor: int,
    pad_pixels: int,
) -> "Window":
    """
    Find a tight bounding window around valid (non-nodata) pixels, using a downsampled mask.
    """
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    if sample_factor < 1:
        raise ValueError("--sample-factor must be >= 1")
    if pad_pixels < 0:
        raise ValueError("--pad-km must be >= 0")

    with rasterio.open(path) as src:
        out_h = max(1, src.height // sample_factor)
        out_w = max(1, src.width // sample_factor)
        mask = src.read_masks(1, out_shape=(out_h, out_w))
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return Window(0, 0, src.width, src.height)

        miny, maxy = int(ys.min()), int(ys.max())
        minx, maxx = int(xs.min()), int(xs.max())

        row_off = miny * sample_factor - pad_pixels
        col_off = minx * sample_factor - pad_pixels
        height = (maxy - miny + 1) * sample_factor + 2 * pad_pixels
        width = (maxx - minx + 1) * sample_factor + 2 * pad_pixels

        col_off, row_off, width, height = _clip_window_to_bounds(
            col_off=col_off,
            row_off=row_off,
            width=width,
            height=height,
            max_width=src.width,
            max_height=src.height,
        )

        return Window(col_off, row_off, width, height)


def _read_window_float32(path: Path, *, window: "Window") -> tuple["np.ndarray", "Affine", str]:
    import numpy as np
    import rasterio
    from rasterio.windows import transform as window_transform

    with rasterio.open(path) as src:
        a = src.read(1, window=window, masked=True)
        out = a.astype(np.float32, copy=False).filled(np.nan)
        return out, window_transform(window, src.transform), src.crs.to_wkt()


def _read_window_resampled_float32(
    path: Path,
    *,
    window: "Window",
    out_height: int,
    out_width: int,
    resampling: "Resampling",
) -> tuple["np.ndarray", "Affine", str]:
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import transform as window_transform

    if out_height < 1 or out_width < 1:
        raise ValueError("out_height/out_width must be >= 1")
    if not isinstance(resampling, Resampling):
        raise TypeError("resampling must be a rasterio.enums.Resampling value")

    with rasterio.open(path) as src:
        window_tfm = window_transform(window, src.transform)
        a = src.read(
            1,
            window=window,
            out_shape=(int(out_height), int(out_width)),
            masked=True,
            resampling=resampling,
        )
        out = a.astype(np.float32, copy=False).filled(np.nan)

        scale_x = float(window.width) / float(out_width)
        scale_y = float(window.height) / float(out_height)
        out_tfm = window_tfm * rasterio.Affine.scale(scale_x, scale_y)

        return out, out_tfm, src.crs.to_wkt()


def _coords_from_affine(transform: "Affine", *, width: int, height: int) -> tuple["np.ndarray", "np.ndarray"]:
    import numpy as np

    # North-up rasters should have no rotation.
    if not (transform.b == 0 and transform.d == 0):
        raise ValueError(f"Rotated/sheared transforms are not supported: {transform}")

    x0 = transform.c + 0.5 * transform.a
    y0 = transform.f + 0.5 * transform.e
    x = x0 + np.arange(width, dtype=np.float64) * transform.a
    y = y0 + np.arange(height, dtype=np.float64) * transform.e
    return x, y


def _compute_signed_distance_to_margin(
    thickness: "np.ndarray",
    *,
    dx: float,
    dy: float,
    thickness_threshold_m: float,
    connectivity: int,
) -> tuple["np.ndarray", "np.ndarray"]:
    import numpy as np
    from scipy import ndimage

    if connectivity not in (4, 8):
        raise ValueError("--connectivity must be 4 or 8")

    finite = np.isfinite(thickness)
    ice = finite & (thickness > float(thickness_threshold_m))

    if connectivity == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    else:
        structure = np.ones((3, 3), dtype=bool)

    eroded = ndimage.binary_erosion(ice, structure=structure, border_value=0)
    margin = ice & ~eroded

    # Distance to nearest margin pixel (0 on margin).
    dist = ndimage.distance_transform_edt(~margin, sampling=(dy, dx)).astype(np.float32, copy=False)

    # Match sign convention used elsewhere in this repo: SDF < 0 => ice, SDF > 0 => ice-free.
    signed = dist.copy()
    signed[ice] *= -1.0
    signed[~finite] = np.nan
    return signed, ice


def _compute_distance_to_coast(
    surface: "np.ndarray",
    *,
    dx: float,
    dy: float,
    treat_nodata_as_water: bool,
) -> "np.ndarray":
    import numpy as np
    from scipy import ndimage

    finite = np.isfinite(surface)
    water = (surface <= 0.0) & finite
    if treat_nodata_as_water:
        water = water | ~finite

    dist = ndimage.distance_transform_edt(~water, sampling=(dy, dx)).astype(np.float32, copy=False)
    dist[~finite] = np.nan
    return dist


def _to_netcdf_attr_value(value: object) -> object:
    """
    Convert Python values to NetCDF/HDF5-safe scalar attribute values.

    In particular, h5netcdf rejects boolean-typed attributes (and classic NetCDF
    doesn't support boolean dtypes). We store booleans as 0/1 integers.
    """
    if isinstance(value, bool):
        return int(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a NetCDF on the native BedMachine/BedMap GeoTIFF grid with:\n"
            "  (1) bedrock elevation + thickness,\n"
            "  (2) signed distance to the present-day ice margin (SDF < 0 inside ice),\n"
            "  (3) distance to coast (nearest pixel with surface = bed + thickness <= 0).\n"
            "Also plots these fields at the end."
        )
    )
    parser.add_argument(
        "--bed-tif",
        type=Path,
        default=Path("data/qgreenland/bedmachine_bed.tif"),
        help="Bedrock elevation GeoTIFF (native grid).",
    )
    parser.add_argument(
        "--thickness-tif",
        type=Path,
        default=Path("data/qgreenland/bedmap_thickness.tif"),
        help="Modern thickness GeoTIFF (native grid).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/modern_fields_native.nc"),
        help="Output NetCDF path.",
    )
    parser.add_argument(
        "--thickness-threshold-m",
        type=float,
        default=0.0,
        help="Ice mask threshold (m); thickness > threshold is treated as ice (default: 0).",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=8,
        help="Connectivity used to define the margin (4 or 8; default: 8).",
    )
    parser.add_argument(
        "--treat-nodata-as-water",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat nodata pixels as water for the coast distance transform (default: true).",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable cropping to the valid-data window (may be very large).",
    )
    parser.add_argument(
        "--sample-factor",
        type=int,
        default=32,
        help="Downsampling factor used to estimate the valid-data window (default: 32).",
    )
    parser.add_argument(
        "--pad-km",
        type=float,
        default=100.0,
        help="Padding added around the valid-data window when cropping (km; default: 100).",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="Gzip compression level (0-9) for NetCDF (default: 4).",
    )
    parser.add_argument(
        "--resolution-m",
        type=float,
        default=1000.0,
        help=(
            "Target output grid resolution (meters). Default 500. "
            "Set to 150 to keep native resolution, or any positive value."
        ),
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot the compiled fields (default: true).",
    )
    parser.add_argument(
        "--plot-downsample",
        type=int,
        default=12,
        help="Coarsening factor for plotting only (default: 12).",
    )
    args = parser.parse_args()

    import numpy as np
    import xarray as xr

    bed_spec = _open_spec(args.bed_tif)
    thk_spec = _open_spec(args.thickness_tif)
    _require_same_grid(bed_spec, thk_spec)

    # Cropping window (helps avoid reading huge empty margins in the GeoTIFFs).
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(args.bed_tif) as src:
        xres, yres = src.res
        native_dx = float(abs(xres))
        native_dy = float(abs(yres))
        dx = native_dx
        dy = native_dy
        if not (dx > 0 and dy > 0):
            raise ValueError(f"Invalid raster resolution: res={src.res}")

    pad_pixels = int(round((float(args.pad_km) * 1000.0) / min(dx, dy)))

    if args.no_crop:
        window = Window(0, 0, bed_spec.width, bed_spec.height)
    else:
        window = _infer_valid_data_window(args.thickness_tif, sample_factor=int(args.sample_factor), pad_pixels=pad_pixels)

    target_res_m = float(args.resolution_m)
    if not (target_res_m > 0):
        raise ValueError("--resolution-m must be > 0")

    # Read + resample bed/thickness to the target resolution on the same CRS and within the cropped window.
    if np.isclose(target_res_m, native_dx) and np.isclose(target_res_m, native_dy):
        bed, transform, crs_wkt = _read_window_float32(args.bed_tif, window=window)
        thickness, transform2, crs_wkt2 = _read_window_float32(args.thickness_tif, window=window)
    else:
        from rasterio.enums import Resampling

        out_width = max(1, int(round(float(window.width) * native_dx / target_res_m)))
        out_height = max(1, int(round(float(window.height) * native_dy / target_res_m)))

        bed, transform, crs_wkt = _read_window_resampled_float32(
            args.bed_tif, window=window, out_height=out_height, out_width=out_width, resampling=Resampling.bilinear
        )
        thickness, transform2, crs_wkt2 = _read_window_resampled_float32(
            args.thickness_tif,
            window=window,
            out_height=out_height,
            out_width=out_width,
            resampling=Resampling.bilinear,
        )

        dx = float(abs(transform.a))
        dy = float(abs(transform.e))

    if transform != transform2 or crs_wkt != crs_wkt2:
        raise RuntimeError("Internal error: windowed grids mismatch.")

    height, width = bed.shape
    x, y = _coords_from_affine(transform, width=width, height=height)

    surface = (bed + thickness).astype(np.float32, copy=False)

    signed_dist_margin, ice_mask = _compute_signed_distance_to_margin(
        thickness,
        dx=dx,
        dy=dy,
        thickness_threshold_m=float(args.thickness_threshold_m),
        connectivity=int(args.connectivity),
    )

    dist_to_coast = _compute_distance_to_coast(
        surface,
        dx=dx,
        dy=dy,
        treat_nodata_as_water=bool(args.treat_nodata_as_water),
    )

    # NetCDF doesn't have a standard boolean dtype; store as 0/1.
    ice_mask_u8 = ice_mask.astype(np.uint8, copy=False)

    ds = xr.Dataset(
        coords={"y": ("y", y), "x": ("x", x)},
        data_vars={
            "bed_elevation": (("y", "x"), bed),
            "thickness": (("y", "x"), thickness),
            "surface": (("y", "x"), surface),
            "signed_distance_to_margin": (("y", "x"), signed_dist_margin),
            "distance_to_coast": (("y", "x"), dist_to_coast),
            "ice_mask": (("y", "x"), ice_mask_u8),
        },
        attrs={
            "crs_wkt": _to_netcdf_attr_value(crs_wkt),
            "source_bed_tif": _to_netcdf_attr_value(str(args.bed_tif)),
            "source_thickness_tif": _to_netcdf_attr_value(str(args.thickness_tif)),
            "native_resolution_m": _to_netcdf_attr_value(float(native_dx)),
            "output_resolution_m": _to_netcdf_attr_value(float(dx)),
            "thickness_threshold_m": _to_netcdf_attr_value(float(args.thickness_threshold_m)),
            "connectivity": _to_netcdf_attr_value(int(args.connectivity)),
            "treat_nodata_as_water": _to_netcdf_attr_value(bool(args.treat_nodata_as_water)),
            "crop_window_col_off": _to_netcdf_attr_value(int(window.col_off)),
            "crop_window_row_off": _to_netcdf_attr_value(int(window.row_off)),
            "crop_window_width": _to_netcdf_attr_value(int(window.width)),
            "crop_window_height": _to_netcdf_attr_value(int(window.height)),
        },
    )

    # Attach CRS/transform in a way that downstream tooling can use.
    import rioxarray  # noqa: F401

    ds = ds.rio.write_crs(crs_wkt, inplace=False)
    ds = ds.rio.write_transform(transform, inplace=False)

    ds["bed_elevation"].attrs.update({"units": "m", "long_name": "bedrock elevation"})
    ds["thickness"].attrs.update({"units": "m", "long_name": "modern ice thickness"})
    ds["surface"].attrs.update({"units": "m", "long_name": "surface elevation (bed + thickness)"})
    ds["signed_distance_to_margin"].attrs.update(
        {
            "units": "m",
            "long_name": "signed distance to present-day ice margin (negative inside ice, positive outside)",
        }
    )
    ds["distance_to_coast"].attrs.update({"units": "m", "long_name": "distance to coast (surface <= 0)"})
    ds["ice_mask"].attrs.update(
        {
            "long_name": "ice mask used for margin (thickness > threshold)",
            "flag_values": np.array([0, 1], dtype=np.uint8),
            "flag_meanings": "ice_free ice",
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    clevel = int(args.compression_level)
    if not (0 <= clevel <= 9):
        raise ValueError("--compression-level must be between 0 and 9.")

    encoding: dict[str, dict] = {}
    if clevel > 0:
        chunks = (1024, 1024)
        encoding = {
            "bed_elevation": {"compression": "gzip", "compression_opts": clevel, "chunksizes": chunks},
            "thickness": {"compression": "gzip", "compression_opts": clevel, "chunksizes": chunks},
            "surface": {"compression": "gzip", "compression_opts": clevel, "chunksizes": chunks},
            "signed_distance_to_margin": {"compression": "gzip", "compression_opts": clevel, "chunksizes": chunks},
            "distance_to_coast": {"compression": "gzip", "compression_opts": clevel, "chunksizes": chunks},
            "ice_mask": {"compression": "gzip", "compression_opts": clevel, "chunksizes": chunks, "dtype": "u1"},
        }

    ds.to_netcdf(args.out, engine="h5netcdf", encoding=encoding)
    print(f"Wrote {args.out}")

    if not args.plot:
        return

    import matplotlib.pyplot as plt

    down = int(args.plot_downsample)
    plot_ds = ds[["bed_elevation", "thickness", "signed_distance_to_margin", "distance_to_coast"]]
    if down > 1:
        plot_ds = plot_ds.coarsen(x=down, y=down, boundary="trim").mean()

    extent = [float(plot_ds.x.min()), float(plot_ds.x.max()), float(plot_ds.y.min()), float(plot_ds.y.max())]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    ax = axes[0, 0]
    im = ax.imshow(plot_ds["bed_elevation"].values, origin="upper", extent=extent, cmap="terrain")
    ax.set_title("Bedrock elevation (m)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    im = ax.imshow(plot_ds["thickness"].values, origin="upper", extent=extent, cmap="Blues")
    ax.set_title("Thickness (m)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 0]
    abs_sdf = np.abs(plot_ds["signed_distance_to_margin"].values)
    v = float(np.nanpercentile(abs_sdf, 98)) if np.any(np.isfinite(abs_sdf)) else 1.0
    im = ax.imshow(
        plot_ds["signed_distance_to_margin"].values,
        origin="upper",
        extent=extent,
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
    )
    ax.set_title("Signed distance to margin (m)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 1]
    im = ax.imshow(plot_ds["distance_to_coast"].values, origin="upper", extent=extent, cmap="viridis")
    ax.set_title("Distance to coast (m)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    for ax in axes.ravel():
        ax.set_xlabel("x (m, EPSG:3413)")
        ax.set_ylabel("y (m, EPSG:3413)")

    plt.show()


if __name__ == "__main__":
    main()
