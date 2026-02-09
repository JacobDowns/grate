#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    default_path = Path("data/deglaciation_snapshot_pca.nc")

    parser = argparse.ArgumentParser(
        description=(
            "Ensure deglaciation_snapshot_pca.nc records the min/max ages used for normalization.\n"
            "Adds global attrs and per-run min/max variables if missing."
        )
    )
    parser.add_argument("nc_file", nargs="?", type=Path, default=default_path, help=f"Path (default: {default_path})")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing min/max variables/attrs")
    args = parser.parse_args()

    import numpy as np
    import h5netcdf

    path = args.nc_file
    if not path.exists():
        raise FileNotFoundError(path)

    with h5netcdf.File(path, "a") as f:
        if "deglaciation_age" not in f.variables:
            raise KeyError(f"{path} missing 'deglaciation_age'. Available: {sorted(f.variables)}")
        if "run" not in f.dimensions:
            raise KeyError(f"{path} missing 'run' dimension.")

        age = f.variables["deglaciation_age"]
        nrun = int(f.dimensions["run"].size)

        mins = np.empty((nrun,), dtype=np.float64)
        maxs = np.empty((nrun,), dtype=np.float64)
        for i in range(nrun):
            slab = np.asarray(age[i, :, :], dtype=np.float32)
            mins[i] = float(np.nanmin(slab))
            maxs[i] = float(np.nanmax(slab))

        global_min = float(np.nanmin(mins))
        global_max = float(np.nanmax(maxs))

        # The normalization used in the PCA pipeline is 0..1 where 0 is most recent and 1 is oldest.
        # In physical units (years), the minimum is 0 by construction; maximum is per-run max age.
        age_norm_min_years = 0.0
        age_norm_max_years = global_max

        def set_attr(k: str, v):
            if (k in f.attrs) and (not args.overwrite):
                return
            f.attrs[k] = v

        set_attr("age_norm_min_years", float(age_norm_min_years))
        set_attr("age_norm_max_years", float(age_norm_max_years))
        set_attr("age_norm_note", "Normalization reference: age_norm = clip(age_years, min, max) / max; 0=most recent, 1=oldest.")

        # Per-run min/max (in years)
        for name, data in (("deglaciation_age_min_years", mins), ("deglaciation_age_max_years", maxs)):
            if name in f.variables and args.overwrite:
                del f._h5file[name]
            if name not in f.variables:
                v = f.create_variable(name, ("run",), dtype="f8")
                v[:] = data
                v.attrs["units"] = "years"
                v.attrs["long_name"] = name.replace("_", " ")

        print(f"Updated {path}")


if __name__ == "__main__":
    main()

