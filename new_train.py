import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
from pathlib import Path
import argparse
import pandas as pd

# --- 1. GPyTorch Model Definition ---

class PhysicsInformedGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_pca_modes):
        super(PhysicsInformedGP, self).__init__(train_x, train_y, likelihood)
        
        # Mean Module: Learns the weights for the PCA modes
        self.mean_module = gpytorch.means.LinearMean(input_size=num_pca_modes)
        
        # Covariance Module: Models the spatial residual using only (x, y) coordinates
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5, active_dims=[0, 1])
        )

    def forward(self, x):
        pca_features = x[:, 2:]  
        mean_pred = self.mean_module(pca_features)
        covar_pred = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_pred, covar_pred)

# --- 2. Data Loading & Preparation ---

def load_pca_dataset(path):
    ds = xr.open_dataset(path, decode_times=False)
    min_age = ds.attrs.get('age_norm_min_years', 0.0)
    max_age = ds.attrs.get('age_norm_max_years', 1.0)
    print(f"Loaded NetCDF with CRS: {ds.rio.crs}")
    return ds, min_age, max_age

def load_age_data(path, min_age, max_age):
    df = pd.read_csv(path)
    x = df["x_3413"].to_numpy(dtype=np.float32)
    y = df["y_3413"].to_numpy(dtype=np.float32)
    ages = df["ages"].to_numpy(dtype=np.float32)
    errs = df["errors"].to_numpy(dtype=np.float32)

    ages = (ages - min_age) / (max_age - min_age)
    ages = np.clip(ages, 0, 1)
    errs = errs / (max_age - min_age)

    return x, y, ages, errs

def prepare_training_tensors(ds, x_obs, y_obs, ages_obs, errs_obs, num_pca_modes):
    x_xr = xr.DataArray(x_obs, dims="points")
    y_xr = xr.DataArray(y_obs, dims="points")
    
    ds_sampled = ds.interp(x=x_xr, y=y_xr, method="linear")
    
    sampled_mean = ds_sampled["deglaciation_age_mean_norm"].values.astype(np.float32)
    sampled_modes = ds_sampled["pca_mode_norm"].values.astype(np.float32) 
    
    sampled_modes = sampled_modes.T[:, :num_pca_modes]
    
    valid_mask = ~np.isnan(sampled_mean) & ~np.isnan(sampled_modes).any(axis=1)
    
    # Extract valid coordinates
    coords_raw = np.column_stack((x_obs[valid_mask], y_obs[valid_mask]))
    
    # *** CRITICAL FIX: STANDARDIZE SPATIAL COORDINATES ***
    coords_mean = coords_raw.mean(axis=0)
    coords_std = coords_raw.std(axis=0)
    coords_scaled = (coords_raw - coords_mean) / coords_std
    
    coords_tensor = torch.tensor(coords_scaled, dtype=torch.float32)
    pca_features = torch.tensor(sampled_modes[valid_mask], dtype=torch.float32)
    
    train_x = torch.cat([coords_tensor, pca_features], dim=1)
    train_y = torch.tensor(ages_obs[valid_mask] - sampled_mean[valid_mask], dtype=torch.float32)
    train_errs = torch.tensor(errs_obs[valid_mask], dtype=torch.float32)
    
    return train_x, train_y, train_errs, valid_mask, coords_mean, coords_std

# --- 3. Visualization Tools ---

def plot_interpolation_sanity_check(train_x, train_y, ages_obs, valid_mask, coords_mean, coords_std):
    # Un-scale coordinates just for plotting
    x_coords = (train_x[:, 0].numpy() * coords_std[0]) + coords_mean[0]
    y_coords = (train_x[:, 1].numpy() * coords_std[1]) + coords_mean[1]
    modes = train_x[:, 2:].numpy()
    
    valid_ages = ages_obs[valid_mask]
    sampled_mean = valid_ages - train_y.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sc0 = axes[0].scatter(x_coords, y_coords, c=valid_ages, cmap='viridis', s=15, alpha=0.8)
    axes[0].set_title("1. Observed Ages (Normalized)")
    axes[0].set_aspect('equal')
    plt.colorbar(sc0, ax=axes[0])
    
    sc1 = axes[1].scatter(x_coords, y_coords, c=sampled_mean, cmap='viridis', s=15, alpha=0.8)
    axes[1].set_title("2. Interpolated Physics Mean")
    axes[1].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[1])
    
    if modes.shape[1] > 0:
        sc2 = axes[2].scatter(x_coords, y_coords, c=modes[:, 0], cmap='coolwarm', s=15, alpha=0.8)
        axes[2].set_title("3. Interpolated PCA Mode 0")
        axes[2].set_aspect('equal')
        plt.colorbar(sc2, ax=axes[2])
        
    plt.tight_layout()
    plt.show()

def predict_and_plot_grid(model, ds, num_pca_modes, min_age, max_age, coords_mean, coords_std):
    model.eval()
    
    xg = ds["x"].values
    yg = ds["y"].values
    X_grid, Y_grid = np.meshgrid(xg, yg)
    
    mean_grid = ds["deglaciation_age_mean_norm"].values.astype(np.float32)
    modes_grid = ds["pca_mode_norm"].values.astype(np.float32)
    
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()
    mean_flat = mean_grid.flatten()
    modes_flat = modes_grid.reshape(modes_grid.shape[0], -1).T[:, :num_pca_modes]
    
    valid_mask = ~np.isnan(mean_flat) & ~np.isnan(modes_flat).any(axis=1)
    
    # Scale test grid coordinates using the exact same mean/std from training
    coords_raw = np.column_stack((X_flat[valid_mask], Y_flat[valid_mask]))
    coords_scaled = (coords_raw - coords_mean) / coords_std
    
    coords_tensor = torch.tensor(coords_scaled, dtype=torch.float32)
    modes_tensor = torch.tensor(modes_flat[valid_mask], dtype=torch.float32)
    test_x = torch.cat([coords_tensor, modes_tensor], dim=1)
    
    print(f"\nPredicting on {test_x.shape[0]} valid grid pixels in batches...")
    batch_size = 10000
    pred_means = []
    pred_vars = []
    
    # fast_pred_var prevents OOM issues when computing variances over grids
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, test_x.shape[0], batch_size):
            batch_x = test_x[i : i + batch_size]
            output = model(batch_x)
            pred_means.append(output.mean.numpy())
            pred_vars.append(output.variance.numpy())
            
    pred_mean_flat = np.concatenate(pred_means)
    pred_var_flat = np.concatenate(pred_vars)
    
    # Reconstruct final normalized age, then un-normalize to Years
    final_age_norm = pred_mean_flat + mean_flat[valid_mask]
    final_age_years = final_age_norm * (max_age - min_age) + min_age
    uncertainty_years = np.sqrt(pred_var_flat) * (max_age - min_age)
    
    age_map = np.full_like(mean_flat, np.nan)
    age_map[valid_mask] = final_age_years
    age_map = age_map.reshape(mean_grid.shape)
    
    unc_map = np.full_like(mean_flat, np.nan)
    unc_map[valid_mask] = uncertainty_years
    unc_map = unc_map.reshape(mean_grid.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    im1 = axes[0].pcolormesh(X_grid, Y_grid, age_map, cmap='viridis_r', shading='auto')
    axes[0].set_title("Reconstructed Deglaciation Age (Years)")
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0], label="Age (Years)")
    
    im2 = axes[1].pcolormesh(X_grid, Y_grid, unc_map, cmap='plasma', shading='auto')
    axes[1].set_title("Prediction Uncertainty (1 Std Dev, Years)")
    axes[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1], label="Uncertainty (Years)")
    
    plt.tight_layout()
    plt.show()

# --- 4. Main Execution ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GP for deglaciation age with PCA-parameterized mean.")
    parser.add_argument("--pca_path", type=Path, default=Path("data/deglaciation_snapshot_pca.nc"), help="PCA output NetCDF")
    parser.add_argument("--ages_path", type=Path, default=Path("data/age_data_epsg3413.csv"), help="Age observations CSV")
    parser.add_argument("--num_pca_modes", type=int, default=10, help="Number of PCA modes to use in the mean function")
    args = parser.parse_args()

    ds, min_age, max_age = load_pca_dataset(args.pca_path)
    x, y, ages, errs = load_age_data(args.ages_path, min_age, max_age)
    
    print(f"\nInterpolating data using {args.num_pca_modes} PCA modes...")
    train_x, train_y, train_errs, valid_mask, coords_mean, coords_std = prepare_training_tensors(
        ds, x, y, ages, errs, args.num_pca_modes
    )
    
    plot_interpolation_sanity_check(train_x, train_y, ages, valid_mask, coords_mean, coords_std)
    
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=train_errs**2, 
        learn_additional_noise=True 
    )
    
    model = PhysicsInformedGP(train_x, train_y, likelihood, num_pca_modes=args.num_pca_modes)
    
    # --- Explicit Initialization ---
    # In standardized coordinate space (N(0,1)), a lengthscale of 0.1 
    # roughly corresponds to ~10% of the Greenland domain.
    model.covar_module.base_kernel.lengthscale = 0.1
    
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.033)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    training_iterations = 1000
    print("\nStarting GP Training...")
    
    # Temporarily suppress CG warnings if they still pop up during early optimization
    with gpytorch.settings.max_cg_iterations(2000), gpytorch.settings.cholesky_jitter(1e-4):
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 10 == 0:
                lengthscale = model.covar_module.base_kernel.lengthscale.item()
                print(f"Iter {i+1:>3}/{training_iterations} - Loss: {loss.item():.3f}  |  Spatial Lengthscale (Standardized): {lengthscale:.3f}")

    print("\nTraining complete.")
    
    predict_and_plot_grid(model, ds, args.num_pca_modes, min_age, max_age, coords_mean, coords_std)

if __name__ == "__main__":
    main()