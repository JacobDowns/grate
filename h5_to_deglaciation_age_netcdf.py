#!/usr/bin/env python3
from __future__ import annotations

"""
Convert ISSM extracted run files (.h5) directly into a *single* gridded deglaciation-age map per run.

Why this exists
---------------
`h5_to_netcdf.py` can write many time-dependent variables to NetCDF. At high output resolution this can
become extremely slow and produce very large files, because it writes (time, y, x) arrays for each
variable.

This script combines the essential parts of:
  - `h5_to_netcdf.py` (mesh loading, grid creation, mesh->grid interpolation)
  - `add_deglaciation_age.py` (definition of "deglaciation age" and time->age conversion)

…but it only writes a 2D deglaciation-age map, not the full time series.

Inputs
------
- A mesh description (`issm_outputs/mesh_info.mat`) providing:
  - vertex coordinates (x, y)
  - triangle connectivity (elements)
  - optional EPSG code for CRS metadata
- One or more extracted ISSM run files (`run_XX_tot.h5`) containing:
  - `time`: 1D array of timesteps
  - `Thickness`: 2D array (time, vertex) with ice thickness at mesh vertices

Algorithm (high level)
----------------------
For each run:
1) Load the run time vector.
2) Convert time values into "age" values (typically years before 1850) using `--age-mode`.
3) Compute a deglaciation age at *mesh nodes* (not on the output grid) by streaming through time:
   - For each node, find the **last** timestep where Thickness > threshold.
   - If a node never exceeds the threshold, treat it as "initially deglaciated" and set age to the first time.
   - If a node is still above the threshold at the final timestep, set age to 0 (never deglaciates within the run).
   - Preserve NaNs where thickness is NaN.
4) Interpolate the final per-node age values to the requested regular output grid (y, x) once.
5) Write a compact NetCDF containing only:
   - coordinates: x, y
   - CRS metadata variable: crs
   - data variable: deglaciation_age (y, x)
   - optional 2D lat/lon grids if EPSG is known and `--add-latlon` is set

Notes
-----
- This approach is much smaller/faster than writing full time-dependent fields because it performs
  the expensive mesh->grid interpolation only once per run (on the final deglaciation-age field),
  rather than once per timestep per variable.
"""

import argparse
import re
import time as time_mod
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
    """
    Infer whether the mesh file stores layered coordinates.

    If x/y contain repeated blocks (one per vertical layer), return (n2d, nlayers),
    otherwise fall back to (nvert, 1).
    """
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
    """
    Load mesh connectivity and vertex coordinates from a MATLAB .mat file.

    The `mesh_info.mat` used in this project stores:
      - x, y: vertex coordinate arrays
      - elements: 1-based triangle connectivity, possibly including layered connectivity
      - epsg: optional EPSG integer
    """
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
    """
    Precompute a linear (barycentric) mapping from mesh triangles to a regular (x, y) grid.

    For each grid point that lies inside the mesh, we store:
      - the triangle vertices (v0,v1,v2)
      - the barycentric weights (w0,w1,w2)
    This lets us interpolate any nodal field to the grid cheaply.
    """
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


def interp_nodal_to_grid(values_2d: "np.ndarray", *, linmap: LinearMap, out_dtype: str) -> "np.ndarray":
    """
    Interpolate a 2D nodal field to the output grid using the precomputed barycentric weights.

    The output is NaN outside the mesh.
    """
    import numpy as np

    vals = np.asarray(values_2d)
    out = np.full((linmap.grid_shape[0] * linmap.grid_shape[1],), np.nan, dtype=np.float64)
    out[linmap.inside_flat_idx] = (
        linmap.w0 * vals[linmap.v0] + linmap.w1 * vals[linmap.v1] + linmap.w2 * vals[linmap.v2]
    )
    out = out.reshape(linmap.grid_shape)
    return out.astype(np.float32 if out_dtype == "float32" else np.float64, copy=False)


def _sanitize_name(name: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if safe and safe[0].isdigit():
        safe = f"v_{safe}"
    return safe or "var"


def _compute_age_steps(time_vals: "np.ndarray", *, age_mode: str, time_zero: float | None) -> tuple["np.ndarray", str, str]:
    """
    Convert the run `time` vector into an "age per timestep" vector.

    This mirrors the logic in `add_deglaciation_age.py` so the resulting units/meaning match.
    """
    import numpy as np

    t = np.asarray(time_vals, dtype=np.float64)

    if age_mode == "auto":
        # If time is already converted (usually <= 0 at/after 1850), use age = -time.
        # If time is raw/absolute (positive), use age = time_zero - time.
        if np.nanmin(t) < 0:
            age_steps = -t
            age_units = "years before 1850"
            age_note = "auto: detected negative time; used age = -time"
        else:
            tz = float(np.nanmax(t) if time_zero is None else time_zero)
            age_steps = tz - t
            age_units = "years before 1850"
            age_note = f"auto: detected non-negative time; used age = time_zero - time (time_zero={tz:g})"
    elif age_mode == "before1850":
        age_steps = -t
        age_units = "years before 1850"
        age_note = "used age = -time"
    elif age_mode == "raw_to_before1850":
        tz = float(np.nanmax(t) if time_zero is None else time_zero)
        age_steps = tz - t
        age_units = "years before 1850"
        age_note = f"used age = time_zero - time (time_zero={tz:g})"
    elif age_mode == "time":
        age_steps = t
        age_units = "same as time"
        age_note = "used age = time"
    else:
        raise ValueError(f"Unknown age_mode={age_mode!r}")

    return age_steps, age_units, age_note


def _slice_to_2d_nodes(
    slab: "np.ndarray",
    *,
    n2d: int,
    nlayers: int,
    collapse: str,
) -> "np.ndarray":
    """
    Convert a 1D per-vertex slab into a 1D per-2D-node slab.

    Some extracted ISSM arrays are stored with vertical layering (nvert = n2d*nlayers),
    but in this project most surface-based fields are identical across layers.
    """
    import numpy as np

    slab = np.asarray(slab)
    if slab.size == n2d:
        return slab
    if slab.size != n2d * nlayers:
        raise ValueError(f"Unexpected vertex count: got {slab.size}, expected {n2d} or {n2d*nlayers}.")

    if collapse == "first":
        return slab[:n2d]
    if collapse == "nanmax":
        return np.nanmax(slab.reshape(nlayers, n2d), axis=0)
    raise ValueError(f"Unknown collapse={collapse!r}; expected 'first' or 'nanmax'.")


def compute_deglaciation_age_nodes(
    *,
    time_vals: "np.ndarray",
    thickness_ds,
    n2d: int,
    nlayers: int,
    threshold_m: float,
    age_mode: str,
    time_zero: float | None,
    collapse_layers: str,
    progress_every: int,
) -> tuple["np.ndarray", str, str]:
    """
    Compute deglaciation age at 2D mesh nodes by streaming through time.

    Returns:
      - age_nodes: (n2d,) float32 array
      - age_units: string
      - age_note: string describing how time->age conversion was done
    """
    import numpy as np

    age_steps, age_units, age_note = _compute_age_steps(time_vals, age_mode=age_mode, time_zero=time_zero)
    nt = int(age_steps.size)

    last_idx = np.full((n2d,), -1, dtype=np.int32)
    nan_mask = np.zeros((n2d,), dtype=bool)
    ever_above = np.zeros((n2d,), dtype=bool)
    final_above = np.zeros((n2d,), dtype=bool)

    t0 = time_mod.perf_counter()
    for ti in range(nt):
        slab = thickness_ds[ti, :]
        slab2d = _slice_to_2d_nodes(slab, n2d=n2d, nlayers=nlayers, collapse=collapse_layers)

        nan_mask |= np.isnan(slab2d)
        slab2d = np.nan_to_num(slab2d, nan=0.0)
        mask = slab2d > threshold_m

        last_idx[mask] = ti
        ever_above |= mask
        if ti == nt - 1:
            final_above = mask.copy()

        if progress_every and ((ti + 1) % progress_every == 0 or (ti + 1) == nt):
            elapsed = time_mod.perf_counter() - t0
            rate = (ti + 1) / elapsed if elapsed > 0 else float("inf")
            eta = (nt - (ti + 1)) / rate if rate > 0 else float("inf")
            print(f"  Thickness: {ti+1}/{nt} ({rate:.2f} steps/s, ETA {eta/60:.1f} min)", flush=True)

    out = np.zeros((n2d,), dtype=np.float32)
    valid = last_idx >= 0
    if np.any(valid):
        out[valid] = age_steps[last_idx[valid]].astype(np.float32)

    never_above = (~ever_above) & (~nan_mask)
    out[never_above] = float(age_steps[0]) if age_steps.size else 0.0

    out[final_above] = 0.0
    out[nan_mask] = np.nan

    return out, age_units, age_note


def write_deglaciation_age_netcdf(
    *,
    out_nc: Path,
    grid: GridSpec,
    mesh: Mesh2D,
    age_grid: "np.ndarray",
    out_var: str,
    out_dtype: str,
    compression_level: int,
    add_latlon: bool,
    source_h5: Path,
    threshold_m: float,
    age_units: str,
    age_note: str,
) -> None:
    import numpy as np
    import h5netcdf

    out_nc.parent.mkdir(parents=True, exist_ok=True)

    with h5netcdf.File(out_nc, "w") as dst:
        dst.dimensions = {"x": grid.x.size, "y": grid.y.size}

        vx = dst.create_variable("x", ("x",), dtype="f8")
        vx[:] = grid.x
        vx.attrs["units"] = "m"
        vx.attrs["standard_name"] = "projection_x_coordinate"

        vy = dst.create_variable("y", ("y",), dtype="f8")
        vy[:] = grid.y
        vy.attrs["units"] = "m"
        vy.attrs["standard_name"] = "projection_y_coordinate"

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

        # Small helper mask to make downstream processing easier.
        valid_mask = np.isfinite(age_grid)
        vmask = dst.create_variable("valid_mask", ("y", "x"), dtype="i1")
        vmask[:] = valid_mask.astype(np.int8)
        vmask.attrs["long_name"] = "1 where deglaciation_age is finite (inside mesh), else 0"

        ny, nx = age_grid.shape
        chunks = (min(512, ny), min(512, nx))
        v = dst.create_variable(
            _sanitize_name(out_var),
            ("y", "x"),
            dtype="f4" if out_dtype == "float32" else "f8",
            chunks=chunks,
            compression="gzip",
            compression_opts=compression_level,
            fillvalue=np.float32(np.nan) if out_dtype == "float32" else np.float64(np.nan),
        )
        v[:, :] = age_grid.astype(np.float32 if out_dtype == "float32" else np.float64, copy=False)
        v.attrs["grid_mapping"] = "crs"
        v.attrs["coordinates"] = "y x"
        v.attrs["long_name"] = "deglaciation age"
        v.attrs["units"] = age_units
        v.attrs["threshold_m"] = float(threshold_m)
        v.attrs["note"] = (
            "Computed by scanning thickness through time at mesh nodes, finding the last timestep where "
            f"Thickness > {float(threshold_m):g} m, converting that timestep to an age, and interpolating "
            "the resulting per-node ages to a regular grid. "
            "Nodes that never exceed threshold are set to the first time (initially deglaciated). "
            "Nodes still above threshold at the final timestep are set to 0 (never deglaciates within run). "
            f"Age conversion: {age_note}."
        )

        dst.attrs["source_h5"] = str(source_h5)
        dst.attrs["mesh_epsg"] = str(mesh.epsg) if mesh.epsg is not None else "unknown"
        dst.attrs["mesh_n2d"] = int(mesh.n2d)
        dst.attrs["mesh_nlayers"] = int(mesh.nlayers)
        dst.attrs["dx_m"] = float(grid.x[1] - grid.x[0]) if grid.x.size > 1 else float("nan")
        dst.attrs["dy_m"] = float(grid.y[1] - grid.y[0]) if grid.y.size > 1 else float("nan")


def main() -> None:
    default_mesh = Path(__file__).resolve().parent / "issm_outputs" / "mesh_info.mat"
    default_in = Path(__file__).resolve().parent / "data" / "issm_extracted"
    default_out = Path(__file__).resolve().parent / "data" / "issm_deglaciation_age_nc"

    parser = argparse.ArgumentParser(
        description=(
            "Convert extracted ISSM run_XX_tot.h5 to a compact NetCDF containing ONLY a gridded "
            "deglaciation_age map (no time-dependent variables)."
        )
    )
    parser.add_argument("--mesh-file", type=Path, default=default_mesh, help="Path to mesh_info.mat")
    parser.add_argument("--input-dir", type=Path, default=default_in, help="Directory containing run_XX_tot.h5 files")
    parser.add_argument("--pattern", type=str, default="run_*_tot.h5", help="Glob pattern for input .h5 files")
    parser.add_argument("--output-dir", type=Path, default=default_out, help="Directory to write NetCDF files")
    parser.add_argument("--dx", type=float, default=1000.0, help="Grid spacing in x (meters)")
    parser.add_argument("--dy", type=float, default=1000.0, help="Grid spacing in y (meters)")
    parser.add_argument("--margin", type=float, default=0.0, help="Extra margin around mesh bounds (meters)")
    parser.add_argument("--thickness-key", type=str, default="Thickness", help="Thickness dataset name in the H5 file")
    parser.add_argument("--out-var", type=str, default="deglaciation_age", help="Output variable name in NetCDF")
    parser.add_argument("--threshold-m", type=float, default=10.0, help="Thickness threshold in meters (default: 10)")
    parser.add_argument(
        "--age-mode",
        choices=("auto", "before1850", "raw_to_before1850", "time"),
        default="auto",
        help=(
            "How to interpret time values into an 'age'. "
            "auto: if time has negatives use -time, else use (time_zero - time)."
        ),
    )
    parser.add_argument(
        "--time-zero",
        type=float,
        default=None,
        help="Reference time used by --age-mode=raw_to_before1850 or auto when time is non-negative (default: max(time)).",
    )
    parser.add_argument(
        "--collapse-layers",
        choices=("first", "nanmax"),
        default="first",
        help=(
            "How to collapse (n2d*nlayers,) vertex slabs to (n2d,) node slabs. "
            "'first' assumes layers are identical and takes the first block; 'nanmax' is safer but slower."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Output dtype for the deglaciation age map (default: float32)",
    )
    parser.add_argument("--compression-level", type=int, default=4, help="NetCDF deflate compression level (0-9)")
    parser.add_argument("--add-latlon", action="store_true", help="Also write 2D lat/lon grids (if EPSG is known)")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="If >0, print progress every N timesteps while scanning thickness (default: 0).",
    )
    args = parser.parse_args()

    import h5py
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
        out_nc = args.output_dir / f"{h5_path.stem}_deglaciation_age.nc"
        print(f"Writing {out_nc} ...", flush=True)

        with h5py.File(h5_path, "r") as src:
            if "time" not in src:
                raise KeyError(f"{h5_path} missing 'time' dataset. Available: {sorted(src.keys())}")
            if args.thickness_key not in src:
                raise KeyError(
                    f"{h5_path} missing thickness dataset {args.thickness_key!r}. Available: {sorted(src.keys())}"
                )

            time_vals = np.asarray(src["time"][:], dtype=np.float64)
            thickness_ds = src[args.thickness_key]

            age_nodes, age_units, age_note = compute_deglaciation_age_nodes(
                time_vals=time_vals,
                thickness_ds=thickness_ds,
                n2d=int(mesh.n2d),
                nlayers=int(mesh.nlayers),
                threshold_m=float(args.threshold_m),
                age_mode=str(args.age_mode),
                time_zero=None if args.time_zero is None else float(args.time_zero),
                collapse_layers=str(args.collapse_layers),
                progress_every=int(args.progress_every),
            )

        age_grid = interp_nodal_to_grid(age_nodes, linmap=linmap, out_dtype=str(args.dtype))

        write_deglaciation_age_netcdf(
            out_nc=out_nc,
            grid=grid,
            mesh=mesh,
            age_grid=age_grid,
            out_var=str(args.out_var),
            out_dtype=str(args.dtype),
            compression_level=int(args.compression_level),
            add_latlon=bool(args.add_latlon),
            source_h5=h5_path,
            threshold_m=float(args.threshold_m),
            age_units=age_units,
            age_note=age_note,
        )


if __name__ == "__main__":
    main()

