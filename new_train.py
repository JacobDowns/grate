import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import pandas as pd

def load_pca_data(path):
    ds = xr.open_dataset(path, decode_times=False)

    xg = ds["x"].values.astype(np.float64)
    yg = ds["y"].values.astype(np.float64)
    mean = ds["deglaciation_age_mean_norm"].values.astype(np.float32)
    modes = ds["pca_mode_norm"].values.astype(np.float32)  


    min_age = ds.attrs['age_norm_min_years']
    max_age = ds.attrs['age_norm_max_years']

    return xg, yg, mean, modes, min_age, max_age

def load_age_data(path, min_age, max_age):
    df = pd.read_csv(path)

    x = df["x_3413"].to_numpy(dtype=np.float32)
    y = df["y_3413"].to_numpy(dtype=np.float32)
    ages = df["ages"].to_numpy(dtype=np.float32)
    errs = df["errors"].to_numpy(dtype=np.float32)

    ages = (ages - min_age) / (max_age - min_age)

    # Clip ages to min/max
    ages = np.clip(ages, 0, 1)
    errs /= (max_age - min_age)

    plt.hist(errs)
    plt.show()

    return ages, errs

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GP for deglaciation age with PCA-parameterized mean (gpytorch kernel + torch).")
    parser.add_argument("--pca_path", type=Path, default=Path("data/deglaciation_snapshot_pca.nc"), help="PCA output NetCDF")
    parser.add_argument("--ages_path", type=Path, default=Path("data/age_data_epsg3413.csv"), help="Age observations CSV (EPSG:3413)")
    args = parser.parse_args()


    xg, yg, mean, modes, min_age, max_age = load_pca_data(args.pca_path)
    load_age_data(args.ages_path, min_age, max_age)
   



if __name__ == "__main__":
    main()

