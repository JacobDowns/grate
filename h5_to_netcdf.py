#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GridSpec:
    x: "np.ndarray"  # (nx,)
    y: "np.ndarray"  # (ny,)
    X: "np.ndarray"  # (ny, nx)
    Y: "np.ndarray"  # (ny, nx)


@dataclass(frozen=True)
class Mesh2D:
    x: "np.ndarray"  # (n2d,)
    y: "np.ndarray"  # (n2d,)
    triangles: "np.ndarray"  # (ntri, 3) int, 0-based
    n2d: int
    nlayers: int
    epsg: int | None = None


@dataclass(frozen=True)
class LinearMap:
    inside_flat_idx: "np.ndarray"  # (n_inside,) int64 indices into flat grid
    v0: "np.ndarray"  # (n_inside,) int64
    v1: "np.ndarray"  # (n_inside,) int64
    v2: "np.ndarray"  # (n_inside,) int64
    w0: "np.ndarray"  # (n_inside,) float64
    w1: "np.ndarray"  # (n_inside,) float64
    w2: "np.ndarray"  # (n_inside,) float64
    grid_shape: tuple[int, int]  # (ny, nx)


def _ravel_col(v) -> "np.ndarray":
    import numpy as np

    arr = np.asarray(v)
    return arr.reshape(-1)


def _infer_layering(x: "np.ndarray", y: "np.ndarray", max_layers: int = 12) -> tuple[int, int]:
    import numpy as np

    nvert = int(x.size)
    for nlayers in range(max_layers, 1, -1):
        if nvert % nlayers != 0:
            continue
        n2d = nvert // nlayers
        x0, y0 = x[:n2d], y[:n2d]
        ok = True
        for k in range(1, nlayers):
            if not (
                np.allclose(x0, x[k * n2d : (k + 1) * n2d])
                and np.allclose(y0, y[k * n2d : (k + 1) * n2d])
            ):
                ok = False
                break
        if ok:
            return n2d, nlayers
    return nvert, 1


def load_mesh(mesh_file: Path) -> Mesh2D:
    import numpy as np
    from scipy.io import loadmat

    mesh = loadmat(mesh_file, squeeze_me=False, struct_as_record=False)
    keys = {k for k in mesh.keys() if not k.startswith("__")}
    required = {"x", "y", "elements"}
    missing = sorted(required - keys)
    if missing:
        raise KeyError(f"{mesh_file} missing keys: {missing}. Available keys: {sorted(keys)}")

    x = _ravel_col(mesh["x"]).astype(float)
    y = _ravel_col(mesh["y"]).astype(float)

    elements = np.asarray(mesh["elements"])
    if elements.ndim != 2 or elements.shape[1] < 3:
        raise ValueError(f"Unexpected 'elements' shape {elements.shape}; expected (nelem, >=3).")

    n2d, nlayers = _infer_layering(x, y)
    x2d = x[:n2d]
    y2d = y[:n2d]

    # Collapse layered vertex ids to base 2D vertex ids.
    tri_raw_1based = elements[:, :3].astype(np.int64)
    tri_1based = ((tri_raw_1based - 1) % n2d) + 1
    triangles = tri_1based - 1

    # Deduplicate triangles (elements may include prism connectivity, repeats, etc.).
    tri_sorted = np.sort(triangles, axis=1)
    _, uniq_idx = np.unique(tri_sorted, axis=0, return_index=True)
    triangles = triangles[np.sort(uniq_idx)].astype(np.int32)

    epsg = None
    if "epsg" in mesh:
        try:
            epsg = int(np.asarray(mesh["epsg"]).reshape(-1)[0])
        except Exception:
            epsg = None

    return Mesh2D(x=x2d, y=y2d, triangles=triangles, n2d=n2d, nlayers=nlayers, epsg=epsg)


def build_grid(mesh: Mesh2D, *, dx: float, dy: float, margin: float = 0.0) -> GridSpec:
    import numpy as np

    xmin, xmax = float(mesh.x.min()), float(mesh.x.max())
    ymin, ymax = float(mesh.y.min()), float(mesh.y.max())

    xmin -= margin
    xmax += margin
    ymin -= margin
    ymax += margin

    nx = int(np.floor((xmax - xmin) / dx)) + 1
    ny = int(np.floor((ymax - ymin) / dy)) + 1
    x = xmin + dx * np.arange(nx, dtype=np.float64)
    y = ymin + dy * np.arange(ny, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
    return GridSpec(x=x, y=y, X=X, Y=Y)


def precompute_linear_map(mesh: Mesh2D, grid: GridSpec) -> LinearMap:
    import numpy as np
    import matplotlib.tri as mtri

    triang = mtri.Triangulation(mesh.x, mesh.y, mesh.triangles)
    trifinder = triang.get_trifinder()

    Xf = grid.X.reshape(-1)
    Yf = grid.Y.reshape(-1)
    tri_idx = trifinder(Xf, Yf).astype(np.int64)  # -1 outside

    inside = tri_idx >= 0
    inside_flat_idx = np.nonzero(inside)[0].astype(np.int64)
    tri_idx_inside = tri_idx[inside]

    tri_verts = mesh.triangles[tri_idx_inside].astype(np.int64)  # (n_inside, 3)
    v0 = tri_verts[:, 0]
    v1 = tri_verts[:, 1]
    v2 = tri_verts[:, 2]

    x0 = mesh.x[v0]
    y0 = mesh.y[v0]
    x1 = mesh.x[v1]
    y1 = mesh.y[v1]
    x2 = mesh.x[v2]
    y2 = mesh.y[v2]

    xp = Xf[inside]
    yp = Yf[inside]

    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    # Avoid divide-by-zero; for degenerate triangles (shouldn't happen), mark weights as NaN.
    bad = den == 0
    den = np.where(bad, np.nan, den)

    w0 = ((y1 - y2) * (xp - x2) + (x2 - x1) * (yp - y2)) / den
    w1 = ((y2 - y0) * (xp - x2) + (x0 - x2) * (yp - y2)) / den
    w2 = 1.0 - w0 - w1

    return LinearMap(
        inside_flat_idx=inside_flat_idx,
        v0=v0,
        v1=v1,
        v2=v2,
        w0=w0.astype(np.float64),
        w1=w1.astype(np.float64),
        w2=w2.astype(np.float64),
        grid_shape=grid.X.shape,
    )


def interp_nodal_to_grid(values: "np.ndarray", *, linmap: LinearMap, out_dtype: str) -> "np.ndarray":
    import numpy as np

    vals = np.asarray(values)
    out = np.full((linmap.grid_shape[0] * linmap.grid_shape[1],), np.nan, dtype=np.float64)
    out[linmap.inside_flat_idx] = (
        linmap.w0 * vals[linmap.v0] + linmap.w1 * vals[linmap.v1] + linmap.w2 * vals[linmap.v2]
    )
    out = out.reshape(linmap.grid_shape)
    return out.astype(np.float32 if out_dtype == "float32" else np.float64, copy=False)


def _sanitize_name(name: str) -> str:
    # NetCDF variable names must be valid identifiers-ish.
    safe = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if safe and safe[0].isdigit():
        safe = f"v_{safe}"
    return safe or "var"


def _repeated_across_layers(
    v_all: "np.ndarray",
    *,
    n2d: int,
    nlayers: int,
    rtol: float = 1e-10,
    atol: float = 1e-8,
) -> bool:
    import numpy as np

    if nlayers <= 1:
        return True
    if v_all.size != n2d * nlayers:
        return False
    base = np.asarray(v_all[:n2d])
    for k in range(1, nlayers):
        blk = np.asarray(v_all[k * n2d : (k + 1) * n2d])
        if not np.allclose(base, blk, rtol=rtol, atol=atol, equal_nan=True):
            return False
    return True


def write_run_netcdf(
    *,
    mesh: Mesh2D,
    grid: GridSpec,
    linmap: LinearMap,
    run_h5: Path,
    out_nc: Path,
    out_dtype: str,
    compression_level: int,
    add_latlon: bool,
) -> None:
    import numpy as np
    import h5py
    import h5netcdf

    out_nc.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(run_h5, "r") as src, h5netcdf.File(out_nc, "w") as dst:
        keys = sorted(src.keys())
        if "time" not in src:
            raise KeyError(f"{run_h5} missing 'time' dataset")

        time = np.asarray(src["time"][:], dtype=np.float64)
        nt = int(time.size)

        # Dimensions
        dst.dimensions = {
            "time": nt,
            "x": grid.x.size,
            "y": grid.y.size,
            **({"layer": mesh.nlayers} if mesh.nlayers > 1 else {}),
        }

        # Coordinates
        vtime = dst.create_variable("time", ("time",), dtype="f8")
        vtime[:] = time
        vtime.attrs["units"] = "years since 1850-01-01 00:00:00"
        vtime.attrs["long_name"] = "time (converted; may be negative for years before 1850)"

        vx = dst.create_variable("x", ("x",), dtype="f8")
        vx[:] = grid.x
        vx.attrs["units"] = "m"
        vx.attrs["standard_name"] = "projection_x_coordinate"

        vy = dst.create_variable("y", ("y",), dtype="f8")
        vy[:] = grid.y
        vy.attrs["units"] = "m"
        vy.attrs["standard_name"] = "projection_y_coordinate"

        if mesh.nlayers > 1:
            vlayer = dst.create_variable("layer", ("layer",), dtype="i4")
            vlayer[:] = np.arange(mesh.nlayers, dtype=np.int32)
            vlayer.attrs["long_name"] = "vertical layer index (block order from the original mesh)"

        # CRS metadata
        crs_var = dst.create_variable("crs", (), dtype="i4")
        if mesh.epsg is not None:
            try:
                from pyproj import CRS

                crs = CRS.from_epsg(mesh.epsg)
                cf = crs.to_cf()
                for k, v in cf.items():
                    crs_var.attrs[k] = v
                crs_var.attrs["spatial_ref"] = crs.to_wkt()
                crs_var.attrs["epsg_code"] = f"EPSG:{mesh.epsg}"
            except Exception:
                crs_var.attrs["epsg_code"] = f"EPSG:{mesh.epsg}"
        else:
            crs_var.attrs["long_name"] = "unknown CRS (epsg not provided in mesh_info.mat)"

        # Optional lat/lon grids
        if add_latlon and mesh.epsg is not None:
            try:
                from pyproj import Transformer

                transformer = Transformer.from_crs(mesh.epsg, 4326, always_xy=True)
                lon2d, lat2d = transformer.transform(grid.X, grid.Y)
                vlat = dst.create_variable(
                    "lat",
                    ("y", "x"),
                    dtype="f4",
                    chunks=(min(512, grid.y.size), min(512, grid.x.size)),
                    compression="gzip",
                    compression_opts=compression_level,
                    fillvalue=np.float32(np.nan),
                )
                vlon = dst.create_variable(
                    "lon",
                    ("y", "x"),
                    dtype="f4",
                    chunks=(min(512, grid.y.size), min(512, grid.x.size)),
                    compression="gzip",
                    compression_opts=compression_level,
                    fillvalue=np.float32(np.nan),
                )
                vlat[:] = lat2d.astype(np.float32)
                vlon[:] = lon2d.astype(np.float32)
                vlat.attrs["units"] = "degrees_north"
                vlon.attrs["units"] = "degrees_east"
            except Exception:
                pass

        # Global attrs
        dst.attrs["source"] = str(run_h5)
        dst.attrs["mesh_epsg"] = str(mesh.epsg) if mesh.epsg is not None else "unknown"
        dst.attrs["mesh_n2d"] = int(mesh.n2d)
        dst.attrs["mesh_nlayers"] = int(mesh.nlayers)
        dst.attrs["dx_m"] = float(grid.x[1] - grid.x[0]) if grid.x.size > 1 else float("nan")
        dst.attrs["dy_m"] = float(grid.y[1] - grid.y[0]) if grid.y.size > 1 else float("nan")

        n2d = mesh.n2d
        nvert = n2d * mesh.nlayers
        # Variables
        chunk_y = min(256, grid.y.size)
        chunk_x = min(256, grid.x.size)

        for key in keys:
            ds = src[key]
            if key == "time":
                continue
            if not hasattr(ds, "shape"):
                continue

            name = _sanitize_name(key)
            shape = tuple(ds.shape)

            # Scalars-per-time (e.g. IceVolume) -> write as (time,)
            if len(shape) == 1 and shape[0] == nt:
                v = dst.create_variable(name, ("time",), dtype="f8")
                v[:] = np.asarray(ds[:], dtype=np.float64)
                continue

            # Nodal-per-time (already 2D) -> interpolate to grid and write as (time, y, x)
            if len(shape) == 2 and shape[0] == nt and shape[1] == n2d:
                v = dst.create_variable(
                    name,
                    ("time", "y", "x"),
                    dtype="f4" if out_dtype == "float32" else "f8",
                    chunks=(1, chunk_y, chunk_x),
                    compression="gzip",
                    compression_opts=compression_level,
                    fillvalue=np.float32(np.nan) if out_dtype == "float32" else np.float64(np.nan),
                )
                v.attrs["grid_mapping"] = "crs"
                v.attrs["coordinates"] = "time y x"

                for ti in range(nt):
                    vals = ds[ti, :]
                    grid2d = interp_nodal_to_grid(vals, linmap=linmap, out_dtype=out_dtype)
                    v[ti, :, :] = grid2d
                continue

            # Layered nodal-per-time -> either collapse (if identical across layers) or write (time, layer, y, x)
            if mesh.nlayers > 1 and len(shape) == 2 and shape[0] == nt and shape[1] == nvert:
                # Heuristic: if layer blocks are identical at both ends of the simulation, treat it as a 2D field.
                repeated = _repeated_across_layers(ds[0, :], n2d=n2d, nlayers=mesh.nlayers) and _repeated_across_layers(
                    ds[-1, :], n2d=n2d, nlayers=mesh.nlayers
                )

                if repeated:
                    v = dst.create_variable(
                        name,
                        ("time", "y", "x"),
                        dtype="f4" if out_dtype == "float32" else "f8",
                        chunks=(1, chunk_y, chunk_x),
                        compression="gzip",
                        compression_opts=compression_level,
                        fillvalue=np.float32(np.nan) if out_dtype == "float32" else np.float64(np.nan),
                    )
                    v.attrs["grid_mapping"] = "crs"
                    v.attrs["coordinates"] = "time y x"
                    v.attrs["note"] = "collapsed from (time,layer,vertex) because values are identical across layers"

                    for ti in range(nt):
                        vals = ds[ti, :n2d]
                        grid2d = interp_nodal_to_grid(vals, linmap=linmap, out_dtype=out_dtype)
                        v[ti, :, :] = grid2d
                    continue

                v = dst.create_variable(
                    name,
                    ("time", "layer", "y", "x"),
                    dtype="f4" if out_dtype == "float32" else "f8",
                    chunks=(1, 1, chunk_y, chunk_x),
                    compression="gzip",
                    compression_opts=compression_level,
                    fillvalue=np.float32(np.nan) if out_dtype == "float32" else np.float64(np.nan),
                )
                v.attrs["grid_mapping"] = "crs"
                v.attrs["coordinates"] = "time layer y x"

                for ti in range(nt):
                    vals_all = ds[ti, :]
                    for li in range(mesh.nlayers):
                        vals = vals_all[li * n2d : (li + 1) * n2d]
                        grid2d = interp_nodal_to_grid(vals, linmap=linmap, out_dtype=out_dtype)
                        v[ti, li, :, :] = grid2d
                continue

            # Fallback: write raw
            dim_names = []
            for i, dim_len in enumerate(shape):
                dim = f"{name}_dim{i}"
                if dim not in dst.dimensions:
                    dst.dimensions[dim] = dim_len
                dim_names.append(dim)
            v = dst.create_variable(name, tuple(dim_names), dtype="f8")
            v[:] = np.asarray(ds[:], dtype=np.float64)


def main() -> None:
    default_mesh = Path(__file__).resolve().parent / "issm_outputs" / "mesh_info.mat"
    default_in = Path(__file__).resolve().parent / "data" / "issm_extracted"
    default_out = Path(__file__).resolve().parent / "data" / "issm_extracted_nc"

    parser = argparse.ArgumentParser(description="Convert extracted ISSM run_XX_tot.h5 to gridded NetCDF.")
    parser.add_argument("--mesh-file", type=Path, default=default_mesh, help="Path to mesh_info.mat")
    parser.add_argument("--input-dir", type=Path, default=default_in, help="Directory containing run_XX_tot.h5 files")
    parser.add_argument("--pattern", type=str, default="run_*_tot.h5", help="Glob pattern for input .h5 files")
    parser.add_argument("--output-dir", type=Path, default=default_out, help="Directory to write NetCDF files")
    parser.add_argument("--dx", type=float, default=2500.0, help="Grid spacing in x (meters) (default: 2500)")
    parser.add_argument("--dy", type=float, default=2500.0, help="Grid spacing in y (meters) (default: 2500)")
    parser.add_argument("--margin", type=float, default=0.0, help="Extra margin around mesh bounds (meters)")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Output dtype for gridded fields (default: float32)",
    )
    parser.add_argument("--compression-level", type=int, default=4, help="NetCDF deflate compression level (0-9)")
    parser.add_argument("--add-latlon", action="store_true", help="Also write 2D lat/lon grids (if EPSG is known)")
    args = parser.parse_args()

    import numpy as np

    mesh = load_mesh(args.mesh_file)
    grid = build_grid(mesh, dx=float(args.dx), dy=float(args.dy), margin=float(args.margin))
    print(
        f"Mesh: n2d={mesh.n2d:,} nlayers={mesh.nlayers} nvert={mesh.n2d * mesh.nlayers:,} "
        f"triangles={mesh.triangles.shape[0]:,} EPSG={mesh.epsg}"
    )
    print(f"Grid: nx={grid.x.size:,} ny={grid.y.size:,} points={grid.x.size * grid.y.size:,} dx={args.dx} dy={args.dy}")

    linmap = precompute_linear_map(mesh, grid)
    print(f"Grid points inside mesh: {linmap.inside_flat_idx.size:,}")

    in_files = sorted(args.input_dir.glob(args.pattern))
    if not in_files:
        raise FileNotFoundError(f"No input files matched {args.pattern!r} in {args.input_dir}")

    for h5_path in in_files:
        out_nc = args.output_dir / (h5_path.stem + ".nc")
        print(f"Writing {out_nc} ...")
        write_run_netcdf(
            mesh=mesh,
            grid=grid,
            linmap=linmap,
            run_h5=h5_path,
            out_nc=out_nc,
            out_dtype=args.dtype,
            compression_level=int(args.compression_level),
            add_latlon=bool(args.add_latlon),
        )


if __name__ == "__main__":
    main()
