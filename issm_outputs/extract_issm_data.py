#!/usr/bin/env python3
"""
Extract time-series fields from ISSM run .mat files (MATLAB v7.3/HDF5) into per-run HDF5 outputs.

The newer ISSM outputs have a structure like:

  data.run.tot  (1xN struct array with fields)

In the underlying HDF5, this typically appears as the group:

  /run/tot

with datasets like /run/tot/Thickness, /run/tot/Vx, ... where each dataset is a (nt, 1)
object/reference array, and each entry points to the actual numeric array for that time step.

This script follows those references and writes one output file per run with datasets shaped
like (nt, ...), plus a numeric 1D "time" dataset.

Example:
  .venv/bin/python issm_outputs/extract_issm_data.py --input-dir issm_outputs --pattern 'run_*.mat'
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


DEFAULT_FIELDS = [
    "Thickness",
    "MaskIceLevelset",
    "Surface",
    "Base",
    "time",
    "Vel",
    "Vx",
    "Vy",
    "Temperature",
    "IceVolume",
    "FrictionCoefficient",
    "SmbMassBalance",
]


def is_mat_v73(path: Path) -> bool:
    # MATLAB v7.3 files are HDF5 and include "MATLAB 7.3" in the header text.
    with path.open("rb") as f:
        header = f.read(256)
    return b"MATLAB 7.3" in header


def parse_run_id(path: Path) -> int | None:
    m = re.match(r"(?i)run_(\d+)\.mat$", path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _squeeze_matlab_array(arr):
    import numpy as np

    a = np.asarray(arr)
    a = np.squeeze(a)
    return a


def _as_time_series_refs(ds):
    import numpy as np

    raw = ds[()]
    raw = np.asarray(raw)
    raw = np.squeeze(raw)
    if raw.ndim != 1:
        raise ValueError(f"Expected a 1D reference array for {ds.name}, got shape={raw.shape}")
    return raw


def _resolve_ref(f, ref):
    import h5py

    if isinstance(ref, h5py.Reference):
        if not ref:
            return None
        return f[ref]
    return None


def _read_ref_value(f, ref):
    import numpy as np

    obj = _resolve_ref(f, ref)
    if obj is None:
        return None
    val = np.array(obj)
    val = _squeeze_matlab_array(val)
    return val


def _first_non_null_value(f, refs):
    for ref in refs:
        v = _read_ref_value(f, ref)
        if v is not None:
            return v
    return None


def _dataset_chunks(nt: int, sample_shape: tuple[int, ...]) -> tuple[int, ...]:
    if not sample_shape:
        return (min(nt, 1024),)
    return (1,) + sample_shape


def extract_run_tot_v73(
    run_file: Path,
    *,
    out_file: Path,
    fields: list[str],
    group_path: str = "/run/tot",
    compression: str | None = "gzip",
    compression_level: int = 4,
    shuffle: bool = True,
    max_steps: int | None = None,
    time_zero: float = 124_980.0,
    convert_time: bool = True,
) -> None:
    import h5py
    import numpy as np

    with h5py.File(run_file, "r") as f:
        if group_path not in f:
            raise KeyError(f"{run_file} missing group {group_path}")

        tot = f[group_path]
        missing = [name for name in fields if name not in tot]
        if missing:
            raise KeyError(f"{run_file} missing fields in {group_path}: {missing}")

        time_refs = _as_time_series_refs(tot["time"])
        nt = int(time_refs.size)
        if max_steps is not None:
            nt = min(nt, int(max_steps))
            time_refs = time_refs[:nt]

        out_file.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_file, "w") as out:
            out.attrs["source_file"] = str(run_file)
            out.attrs["source_group"] = group_path
            out.attrs["nt"] = nt
            out.attrs["time_zero"] = float(time_zero)
            out.attrs["time_convert"] = bool(convert_time)

            time_raw = np.empty((nt,), dtype=np.float64)
            for i, ref in enumerate(time_refs):
                v = _read_ref_value(f, ref)
                if v is None:
                    time_raw[i] = np.nan
                else:
                    v = np.asarray(v).reshape(-1)
                    time_raw[i] = float(v[0]) if v.size else np.nan

            # Convention requested: time_out = time_raw - 124980, which yields negative values for years before 1850.
            time_out = (time_raw - float(time_zero)) if convert_time else time_raw.copy()
            out.create_dataset("time", data=time_out)
            out.create_dataset("time_raw", data=time_raw)
            out.attrs["time_note"] = "time = time_raw - time_zero (negative implies years before 1850)" if convert_time else "time is unmodified (same as time_raw)"

            for field in fields:
                if field == "time":
                    continue

                refs = _as_time_series_refs(tot[field])[:nt]
                sample = _first_non_null_value(f, refs)
                if sample is None:
                    print(f"Warning: {run_file.name} field {field} had no readable entries; skipping.")
                    continue

                sample = np.asarray(sample)
                sample_shape = tuple(sample.shape)
                dtype = sample.dtype

                dset = out.create_dataset(
                    field,
                    shape=(nt,) + sample_shape,
                    dtype=dtype,
                    chunks=_dataset_chunks(nt, sample_shape),
                    compression=compression,
                    compression_opts=compression_level if compression == "gzip" else None,
                    shuffle=shuffle,
                )

                for i, ref in enumerate(refs):
                    v = _read_ref_value(f, ref)
                    if v is None:
                        continue
                    v = np.asarray(v)
                    v = np.squeeze(v)
                    if v.shape != sample_shape:
                        raise ValueError(
                            f"{run_file.name} field {field} step {i} shape mismatch: {v.shape} != {sample_shape}"
                        )
                    dset[i, ...] = v

            out.attrs["fields"] = list(out.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract /run/tot time-series fields from ISSM run_*.mat files")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing run_*.mat files (default: issm_outputs/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "issm_extracted",
        help="Directory to write extracted .h5 files (default: data/issm_extracted/)",
    )
    parser.add_argument("--pattern", type=str, default="run_*.mat", help="Glob pattern for run files (default: run_*.mat)")
    parser.add_argument(
        "--fields",
        type=str,
        default=",".join(DEFAULT_FIELDS),
        help="Comma-separated list of fields to extract from /run/tot",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="If set, only extract the first N time steps")
    parser.add_argument(
        "--time-zero",
        type=float,
        default=124_980.0,
        help="Value to subtract from the raw model time when writing 'time' (default: 124980).",
    )
    parser.add_argument(
        "--convert-time",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If true, write time = time_raw - time_zero (default: true).",
    )
    args = parser.parse_args()

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    run_files = sorted(args.input_dir.glob(args.pattern))

    if not run_files:
        print(f"Error: No files matched {args.pattern!r} in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(run_files)} run file(s) in {args.input_dir} matching {args.pattern!r}")

    extracted = 0
    for run_file in run_files:
        run_id = parse_run_id(run_file)
        tag = f"{run_id:02d}" if run_id is not None else run_file.stem
        out_file = args.output_dir / f"run_{tag}_tot.h5"

        print(f"\nExtracting {run_file.name} -> {out_file}")

        if not is_mat_v73(run_file):
            raise RuntimeError(f"{run_file} does not look like a MATLAB v7.3 file (expected HDF5).")

        try:
            extract_run_tot_v73(
                run_file,
                out_file=out_file,
                fields=fields,
                max_steps=args.max_steps,
                time_zero=float(args.time_zero),
                convert_time=bool(args.convert_time),
            )
        except ImportError as e:
            print(f"Error: Missing dependency for v7.3 extraction: {e}")
            print("Install with: pip install numpy h5py")
            sys.exit(1)

        extracted += 1

    print(f"\nDone. Wrote {extracted} file(s) into {args.output_dir}")
    example = args.output_dir / "run_01_tot.h5"
    print("\nTo load in Python:")
    print("  import h5py")
    print(f"  f = h5py.File(r'{example}', 'r')")
    print("  print(list(f.keys()))")
    print("  time = f['time'][:]")
    print("  thickness = f['Thickness'][:]")


if __name__ == "__main__":
    main()
