#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np 

def main() -> None:
    default_path = Path("data/issm_extracted_nc/run_01_tot.nc")

    parser = argparse.ArgumentParser(description="Load a NetCDF file with xarray and print the Dataset repr.")
    parser.add_argument("nc_file", nargs="?", type=Path, default=default_path, help=f"Path to NetCDF (default: {default_path})")
    args = parser.parse_args()

    path = args.nc_file
    if not path.exists():
        raise FileNotFoundError(path)

    import xarray as xr

    # xarray/cftime do not support "years since ..." units; treat time as numeric.
    ds = xr.open_dataset(path, decode_times=False)
    print(ds)
    #ds.deglaciation_age.data[ds.deglaciation_age.data == 0] = np.nan

    ds.deglaciation_age.plot()
    plt.show()

    plt.subplot(2,1,1)
    ds.Thickness[0].plot()

    plt.subplot(2,1,2)
    ds.Thickness[-1].plot()
    plt.show()


if __name__ == "__main__":
    main()
