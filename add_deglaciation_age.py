#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _compute_deglaciation_age_2d(
    *,
    thk_var,
    time: "np.ndarray",
    threshold_m: float,
    age_mode: str,
    time_zero: float | None,
) -> "np.ndarray":
    import numpy as np

    t = np.asarray(time, dtype=np.float64)

    if age_mode == "auto":
        # If time is already converted (usually <= 0 at/after 1850), use age = -time.
        # If time is raw/absolute (positive, typically ending at 124980), use age = time_zero - time.
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

    nt = int(thk_var.shape[0])
    ny = int(thk_var.shape[-2])
    nx = int(thk_var.shape[-1])

    last_idx = np.full((ny, nx), -1, dtype=np.int32)
    nan_mask = np.zeros((ny, nx), dtype=bool)
    final_above = np.zeros((ny, nx), dtype=bool)
    ever_above = np.zeros((ny, nx), dtype=bool)

    # Stream one timestep at a time to keep memory usage bounded.
    if len(thk_var.shape) == 3:
        # (time, y, x)
        for ti in range(nt):
            slab = thk_var[ti, :, :]
            nan_mask |= np.isnan(slab)
            slab = np.nan_to_num(slab, nan=0.0)
            mask = slab > threshold_m  # NaNs treated as 0 for thresholding
            last_idx[mask] = ti
            ever_above |= mask
            if ti == nt - 1:
                final_above = mask.copy()
    elif len(thk_var.shape) == 4:
        # (time, layer, y, x) or (time, y, x, layer) (we only support the former, produced by our pipeline)
        for ti in range(nt):
            slab = thk_var[ti, :, :, :]  # (layer, y, x)
            # Thickness should be identical across layers; use nanmax to be safe.
            slab = np.asarray(slab)
            slab2d = np.nanmax(slab, axis=0)  # NaN if all layers are NaN
            nan_mask |= np.isnan(slab2d)
            slab2d = np.nan_to_num(slab2d, nan=0.0)
            mask = slab2d > threshold_m
            last_idx[mask] = ti
            ever_above |= mask
            if ti == nt - 1:
                final_above = mask.copy()
    else:
        raise ValueError(f"Unexpected Thickness shape {thk_var.shape}; expected 3D or 4D with time.")

    out = np.zeros((ny, nx), dtype=np.float32)
    valid = last_idx >= 0
    if np.any(valid):
        out[valid] = age_steps[last_idx[valid]].astype(np.float32)

    # Pixels that never exceed the threshold are "initially deglaciated" for this run -> set to the first time.
    never_above = (~ever_above) & (~nan_mask)
    out[never_above] = float(age_steps[0]) if age_steps.size else 0.0

    # Regions still glaciated at the end of the run have not deglaciated within the run -> age = 0.
    out[final_above] = 0.0

    # Retain NaNs anywhere thickness is NaN (commonly outside the valid mesh/domain).
    out[nan_mask] = np.nan

    return out, age_units, age_note


def process_file(
    nc_path: Path,
    *,
    thickness_var: str,
    out_var: str,
    threshold_m: float,
    age_mode: str,
    time_zero: float | None,
    overwrite: bool,
) -> None:
    import numpy as np
    import h5netcdf

    with h5netcdf.File(nc_path, "a") as f:
        if thickness_var not in f.variables:
            raise KeyError(f"{nc_path} missing variable {thickness_var!r}. Available: {sorted(f.variables)}")
        if "time" not in f.variables:
            raise KeyError(f"{nc_path} missing variable 'time'.")

        thk = f.variables[thickness_var]
        time = np.asarray(f.variables["time"][:], dtype=np.float64)

        existing = out_var in f.variables
        if existing and not overwrite:
            print(f"Skipping {nc_path.name}: {out_var} already exists (use --overwrite).")
            return

        deglac_age, age_units, age_note = _compute_deglaciation_age_2d(
            thk_var=thk,
            time=time,
            threshold_m=float(threshold_m),
            age_mode=age_mode,
            time_zero=time_zero,
        )

        # Write output variable (prefer in-place update when possible).
        if existing:
            v = f.variables[out_var]
            if getattr(v, "dimensions", None) == ("y", "x") and tuple(v.shape) == tuple(deglac_age.shape):
                v[:, :] = deglac_age
            else:
                # Fall back to deleting/recreating via underlying h5py handle.
                del f._h5file[out_var]
                existing = False

        if not existing:
            ny, nx = deglac_age.shape
            chunks = (min(512, ny), min(512, nx))
            v = f.create_variable(
                out_var,
                ("y", "x"),
                dtype="f4",
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
                fillvalue=np.float32(np.nan),
            )
            v[:, :] = deglac_age

        v.attrs["long_name"] = f"deglaciation age from {thickness_var}"
        v.attrs["units"] = age_units
        v.attrs["threshold_m"] = float(threshold_m)
        v.attrs["note"] = (
            "Computed as age(time[last index where thickness > threshold]); "
            "NaN thickness preserved as NaN in the output, but treated as 0 for thresholding. "
            "Pixels that never exceed threshold are set to the first time (initially deglaciated). "
            "Pixels still above threshold at the final timestep are set to 0 (never deglaciates within run). "
            f"Age conversion: {age_note}."
        )


def main() -> None:
    default_dir = Path("data/issm_extracted_nc")

    parser = argparse.ArgumentParser(
        description=(
            "Append a deglaciation-age map to each NetCDF file.\n"
            "For each pixel: find the last timestep where thickness > threshold, then store the corresponding age.\n"
            "NaN thickness values are treated as 0 for thresholding."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=default_dir, help="Directory containing NetCDF files")
    parser.add_argument("--pattern", type=str, default="run_*_tot.nc", help="Glob pattern for NetCDF files")
    parser.add_argument("--thickness-var", type=str, default="Thickness", help="Thickness variable name (default: Thickness)")
    parser.add_argument("--out-var", type=str, default="deglaciation_age", help="Output variable name to append")
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
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output variable if it already exists")
    args = parser.parse_args()

    nc_files = sorted(args.input_dir.glob(args.pattern))
    if not nc_files:
        raise FileNotFoundError(f"No files matched {args.pattern!r} in {args.input_dir}")

    for p in nc_files:
        print(f"Updating {p} ...")
        process_file(
            p,
            thickness_var=args.thickness_var,
            out_var=args.out_var,
            threshold_m=float(args.threshold_m),
            age_mode=str(args.age_mode),
            time_zero=None if args.time_zero is None else float(args.time_zero),
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    main()
