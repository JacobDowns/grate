import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
            gpytorch.kernels.MaternKernel(nu=1.5, active_dims=[0, 1])
        )

    def forward(self, x):
        pca_features = x[:, 2:]  
        mean_pred = self.mean_module(pca_features)
        covar_pred = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_pred, covar_pred)

# --- 2. Data Loading & Preparation ---

def load_pca_dataset(path):
    ds = xr.open_dataset(path, decode_times=False)
    #print(ds)
    #ds['deglaciation_age_norm'][0].plot()
    #plt.show()
    #quit()
    min_age = ds.attrs.get('age_norm_min_years', 0.0)
    max_age = ds.attrs.get('age_norm_max_years', 1.0)
    print(f"Loaded NetCDF with CRS: {ds.rio.crs}")
    return ds, min_age, max_age

def load_age_data(path, min_age, max_age):
    df = pd.read_csv(path)
    if "age_mean" not in df.columns or "age_sd" not in df.columns:
        raise ValueError(f"{path} must contain columns 'age_mean' and 'age_sd'")
    if "x_3413" not in df.columns or "y_3413" not in df.columns:
        raise ValueError(f"{path} must contain columns 'x_3413' and 'y_3413'")

    return df


def filter_age_data(
    df: pd.DataFrame,
    *,
    cosmogenic_only: bool,
    min_quality: str,
) -> pd.DataFrame:
    out = df.copy()

    if cosmogenic_only:
        if "obs_type" not in out.columns:
            raise ValueError("Requested --cosmogenic-only but input CSV has no 'obs_type' column.")
        out = out[out["obs_type"].astype(str).str.lower() == "cosmogenic"].copy()
        return out

    if "quality" not in out.columns:
        return out

    quality_rank = {"high": 0, "mid": 1, "low": 2}
    q = str(min_quality).strip().lower()
    if q not in quality_rank:
        raise ValueError(f"Unknown min_quality={min_quality!r}; expected one of High/Mid/Low.")

    out["_quality_rank"] = out["quality"].astype(str).str.strip().str.lower().map(quality_rank)
    out = out[out["_quality_rank"].notna() & (out["_quality_rank"] <= quality_rank[q])].copy()
    out = out.drop(columns=["_quality_rank"])
    return out


def normalize_age_data(
    df: pd.DataFrame,
    min_age: float,
    max_age: float,
    *,
    ages_bp_ref_year: float,
    model_bp_ref_year: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = df["x_3413"].to_numpy(dtype=np.float32)
    y = df["y_3413"].to_numpy(dtype=np.float32)
    ages = df["age_mean"].to_numpy(dtype=np.float32)
    errs = df["age_sd"].to_numpy(dtype=np.float32)

    # CSV ages are years before `ages_bp_ref_year` (typically 1950).
    # The model expects years before `model_bp_ref_year` (1850).
    # Example: 10,000 yr BP (1950) == 9,900 yr BP (1850).
    ages = ages - np.float32(float(ages_bp_ref_year) - float(model_bp_ref_year))

    ages = (ages - min_age) / (max_age - min_age)
    ages = np.clip(ages, 0, 1)
    errs = errs / (max_age - min_age)

    return x, y, ages, errs

def prepare_training_tensors(ds, x_obs, y_obs, ages_obs, errs_obs, num_pca_modes, *, include_bedrock: bool):
    x_xr = xr.DataArray(x_obs, dims="points")
    y_xr = xr.DataArray(y_obs, dims="points")
    
    ds_sampled = ds.interp(x=x_xr, y=y_xr, method="linear")
    
    # Use the "all" mean/PCA variant produced by snapshot_pca_deglaciation.py.
    sampled_mean = ds_sampled["deglaciation_age_mean_norm_all"].values.astype(np.float32)
    sampled_modes = ds_sampled["pca_mode_norm_all"].values.astype(np.float32) 
    
    sampled_modes = sampled_modes.T[:, :num_pca_modes]
    
    if include_bedrock:
        if "bed_elevation" not in ds_sampled:
            raise KeyError("Requested --include-bedrock but dataset has no 'bed_elevation' variable.")
        sampled_bed = ds_sampled["bed_elevation"].values.astype(np.float32)
        valid_mask = (
            ~np.isnan(sampled_mean)
            & ~np.isnan(sampled_modes).any(axis=1)
            & np.isfinite(sampled_bed)
        )
    else:
        sampled_bed = None
        valid_mask = ~np.isnan(sampled_mean) & ~np.isnan(sampled_modes).any(axis=1)
    
    # Extract valid coordinates
    coords_raw = np.column_stack((x_obs[valid_mask], y_obs[valid_mask]))
    
    # *** CRITICAL FIX: STANDARDIZE SPATIAL COORDINATES ***
    coords_mean = coords_raw.mean(axis=0)
    coords_std = coords_raw.std(axis=0)
    coords_scaled = (coords_raw - coords_mean) / coords_std
    
    coords_tensor = torch.tensor(coords_scaled, dtype=torch.float32)
    pca_features = torch.tensor(sampled_modes[valid_mask], dtype=torch.float32)
    
    if include_bedrock:
        bed_raw = sampled_bed[valid_mask].astype(np.float32)
        bed_mean = float(np.nanmean(bed_raw))
        bed_std = float(np.nanstd(bed_raw))
        if not (bed_std > 0):
            bed_std = 1.0
        bed_scaled = (bed_raw - bed_mean) / bed_std
        bed_tensor = torch.tensor(bed_scaled[:, None], dtype=torch.float32)
        train_x = torch.cat([coords_tensor, bed_tensor, pca_features], dim=1)
    else:
        bed_mean = None
        bed_std = None
        train_x = torch.cat([coords_tensor, pca_features], dim=1)
    train_y = torch.tensor(ages_obs[valid_mask] - sampled_mean[valid_mask], dtype=torch.float32)
    train_errs = torch.tensor(errs_obs[valid_mask], dtype=torch.float32)
    
    return train_x, train_y, train_errs, valid_mask, coords_mean, coords_std, bed_mean, bed_std

# --- 3. Visualization Tools ---

def plot_interpolation_sanity_check(train_x, train_y, ages_obs, valid_mask, coords_mean, coords_std, *, include_bedrock: bool):
    # Un-scale coordinates just for plotting
    x_coords = (train_x[:, 0].numpy() * coords_std[0]) + coords_mean[0]
    y_coords = (train_x[:, 1].numpy() * coords_std[1]) + coords_mean[1]
    features = train_x[:, 2:].numpy()
    
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
    
    if features.shape[1] > 0:
        # If bedrock is included, feature 0 is bedrock; feature 1 is PCA mode 0.
        feature_idx = 1 if include_bedrock and features.shape[1] > 1 else 0
        sc2 = axes[2].scatter(x_coords, y_coords, c=features[:, feature_idx], cmap='coolwarm', s=15, alpha=0.8)
        axes[2].set_title("3. Interpolated PCA Mode 0" if feature_idx != 0 or not include_bedrock else "3. Bedrock Feature")
        axes[2].set_aspect('equal')
        plt.colorbar(sc2, ax=axes[2])
        
    plt.tight_layout()
    plt.show()

def predict_and_plot_grid(
    model,
    ds,
    num_pca_modes,
    min_age,
    max_age,
    coords_mean,
    coords_std,
    *,
    include_bedrock: bool,
    bed_mean: float | None,
    bed_std: float | None,
):
    model.eval()
    
    xg = ds["x"].values
    yg = ds["y"].values
    X_grid, Y_grid = np.meshgrid(xg, yg)
    
    # Use the "all" mean/PCA variant produced by snapshot_pca_deglaciation.py.
    mean_grid = ds["deglaciation_age_mean_norm_all"].values.astype(np.float32)
    modes_grid = ds["pca_mode_norm_all"].values.astype(np.float32)
    bed_grid = ds["bed_elevation"].values.astype(np.float32) if include_bedrock else None

    if "deglaciation_age_norm" not in ds:
        raise KeyError("Dataset is missing 'deglaciation_age_norm' needed to mask output to Holocene max extent.")
    # Define Holocene maximum ice-sheet extent as pixels where any run has deglaciation_age_norm < 1.
    holocene_extent = ds["deglaciation_age_norm"].min(dim="run").values.astype(np.float32)  # (y, x)

    # Mask out modern ice cover. If an explicit mask isn't present in the NetCDF, infer it:
    # pixels with deglaciation_age_norm == 0 for every run are still ice-covered at the end of the runs.
    if "modern_thickness" in ds:
        thk = ds["modern_thickness"].values.astype(np.float32)
        modern_ice_mask = np.isfinite(thk) & (thk > 0.0)
    elif "modern_ice_mask" in ds:
        modern_ice_mask = ds["modern_ice_mask"].values.astype(bool)
    else:
        modern_ice_mask = (ds["deglaciation_age_norm"].max(dim="run").values.astype(np.float32) <= 0.0)
    
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()
    mean_flat = mean_grid.flatten()
    modes_flat = modes_grid.reshape(modes_grid.shape[0], -1).T[:, :num_pca_modes]
    extent_flat = holocene_extent.flatten()
    modern_ice_flat = modern_ice_mask.flatten()
    
    valid_mask = ~np.isnan(mean_flat) & ~np.isnan(modes_flat).any(axis=1)
    valid_mask &= np.isfinite(extent_flat) & (extent_flat < 1.0)
    valid_mask &= ~modern_ice_flat

    if include_bedrock:
        if bed_grid is None:
            raise KeyError("Requested --include-bedrock but dataset has no 'bed_elevation' variable.")
        if bed_mean is None or bed_std is None:
            raise ValueError("include_bedrock=True requires bed_mean/bed_std from training.")
        bed_flat = bed_grid.flatten()
        valid_mask &= np.isfinite(bed_flat)
    
    # Scale test grid coordinates using the exact same mean/std from training
    coords_raw = np.column_stack((X_flat[valid_mask], Y_flat[valid_mask]))
    coords_scaled = (coords_raw - coords_mean) / coords_std
    
    coords_tensor = torch.tensor(coords_scaled, dtype=torch.float32)
    modes_tensor = torch.tensor(modes_flat[valid_mask], dtype=torch.float32)
    if include_bedrock:
        bed_scaled = (bed_flat[valid_mask].astype(np.float32) - float(bed_mean)) / float(bed_std)
        bed_tensor = torch.tensor(bed_scaled[:, None], dtype=torch.float32)
        test_x = torch.cat([coords_tensor, bed_tensor, modes_tensor], dim=1)
    else:
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

    modeled_mean_years = mean_grid * (max_age - min_age) + min_age
    modeled_mean_mask = np.isfinite(modeled_mean_years) & (holocene_extent < 1.0) & (~modern_ice_mask)
    if "valid_mask" in ds:
        modeled_mean_mask &= ds["valid_mask"].values.astype(bool)
    modeled_mean_years = np.where(modeled_mean_mask, modeled_mean_years, np.nan)
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
    n_levels = 32
    age_bounds = np.linspace(min_age, max_age, n_levels + 1, dtype=np.float32)
    age_cmap = plt.get_cmap("seismic_r", n_levels)
    age_norm = mcolors.BoundaryNorm(age_bounds, age_cmap.N, clip=True)
    
    im1 = axes[0].pcolormesh(
        X_grid,
        Y_grid,
        age_map,
        cmap=age_cmap,
        norm=age_norm,
        shading="auto",
    )
    axes[0].set_title("Reconstructed Deglaciation Age (Years)")
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0], label="Age (Years)", boundaries=age_bounds)
    
    im2 = axes[1].pcolormesh(
        X_grid,
        Y_grid,
        modeled_mean_years,
        cmap=age_cmap,
        norm=age_norm,
        shading="auto",
    )
    axes[1].set_title("Modeled Mean Deglaciation Age (Years)")
    axes[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1], label="Age (Years)", boundaries=age_bounds)

    im3 = axes[2].pcolormesh(X_grid, Y_grid, unc_map, cmap='plasma', shading='auto')
    axes[2].set_title("Prediction Uncertainty (1 Std Dev, Years)")
    axes[2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[2], label="Uncertainty (Years)")
    
    plt.tight_layout()
    plt.show()

# --- 4. Main Execution ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GP for deglaciation age with PCA-parameterized mean.")
    parser.add_argument("--pca_path", type=Path, default=Path("data/deglaciation_snapshot_pca.nc"), help="PCA output NetCDF")
    parser.add_argument("--ages_path", type=Path, default=Path("data/ryan_data/all_data.csv"), help="Age observations CSV")
    parser.add_argument("--num_pca_modes", type=int, default=18, help="Number of PCA modes to use in the mean function")
    parser.add_argument(
        "--cosmogenic-only",
        action="store_true",
        help="Use only cosmogenic observations (requires obs_type column).",
    )
    parser.add_argument(
        "--min-quality",
        type=str,
        default="Low",
        choices=["High", "Mid", "Low"],
        help="Include all observations at least this good (High ⊂ Mid ⊂ Low). Ignored with --cosmogenic-only.",
    )
    parser.add_argument(
        "--ages-bp-ref-year",
        type=float,
        default=1950.0,
        help="Reference year for CSV ages (years before this year). Default: 1950.",
    )
    parser.add_argument(
        "--model-bp-ref-year",
        type=float,
        default=1850.0,
        help="Reference year expected by the model (years before this year). Default: 1850.",
    )
    parser.add_argument(
        "--include-bedrock",
        action="store_true",
        help="Include bedrock elevation as an additional mean-function feature (requires bed_elevation in the NetCDF).",
    )
    args = parser.parse_args()

    ds, min_age, max_age = load_pca_dataset(args.pca_path)
    ages_df = load_age_data(args.ages_path, min_age, max_age)
    ages_df = filter_age_data(ages_df, cosmogenic_only=bool(args.cosmogenic_only), min_quality=str(args.min_quality))
    x, y, ages, errs = normalize_age_data(
        ages_df,
        min_age,
        max_age,
        ages_bp_ref_year=float(args.ages_bp_ref_year),
        model_bp_ref_year=float(args.model_bp_ref_year),
    )
    
    print(f"\nInterpolating data using {args.num_pca_modes} PCA modes...")
    train_x, train_y, train_errs, valid_mask, coords_mean, coords_std, bed_mean, bed_std = prepare_training_tensors(
        ds, x, y, ages, errs, args.num_pca_modes, include_bedrock=bool(args.include_bedrock)
    )
    
    plot_interpolation_sanity_check(
        train_x,
        train_y,
        ages,
        valid_mask,
        coords_mean,
        coords_std,
        include_bedrock=bool(args.include_bedrock),
    )
    
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=train_errs**2, 
        learn_additional_noise=True 
    )
    
    mean_feature_dim = int(args.num_pca_modes) + (1 if args.include_bedrock else 0)
    model = PhysicsInformedGP(train_x, train_y, likelihood, num_pca_modes=mean_feature_dim)
    
    # --- Explicit Initialization ---
    # In standardized coordinate space (N(0,1)), a lengthscale of 0.1 
    # roughly corresponds to ~10% of the Greenland domain.
    model.covar_module.base_kernel.lengthscale = 0.1
    
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.033)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    training_iterations = 2500
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
    
    predict_and_plot_grid(
        model,
        ds,
        args.num_pca_modes,
        5e3,
        14.5e3,
        coords_mean,
        coords_std,
        include_bedrock=bool(args.include_bedrock),
        bed_mean=bed_mean,
        bed_std=bed_std,
    )

if __name__ == "__main__":
    main()
