import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import gpytorch
from pathlib import Path
import argparse
import pandas as pd

# --- 1. Neural Network & GP Model Definition ---

class MLPMean(nn.Module):
    def __init__(self, input_dim):
        super(MLPMean, self).__init__()
        # Small network to prevent overfitting on sparse data
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.ReLU(),
            nn.Linear(1, 1)
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)

class DataDrivenGP(gpytorch.models.ExactGP):
    # GPyTorch can natively handle multiple input tensors by passing them as a tuple
    def __init__(self, train_x_mlp, train_x_gp, train_y, likelihood):
        super(DataDrivenGP, self).__init__((train_x_mlp, train_x_gp), train_y, likelihood)
        
        #self.mean_module = MLPMean(input_dim=train_x_mlp.shape[1])
        self.mean_module = gpytorch.means.LinearMean(input_size=train_x_mlp.shape[1], bias=True)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5, 
                ard_num_dims=train_x_gp.shape[1] 
            )
        )

    def forward(self, x_mlp, x_gp):
        # The Mean only sees the MLP features
        mean_pred = self.mean_module(x_mlp)
        
        # The Kernel only sees the GP features
        covar_pred = self.covar_module(x_gp)
        
        return gpytorch.distributions.MultivariateNormal(mean_pred, covar_pred)

# --- 2. Data Loading & Feature Engineering ---

def filter_age_data(df: pd.DataFrame, cosmogenic_only: bool, min_quality: str) -> pd.DataFrame:
    out = df.copy()
    if cosmogenic_only:
        out = out[out["obs_type"].astype(str).str.lower() == "cosmogenic"].copy()
        
    if "quality" not in out.columns:
        return out

    quality_rank = {"high": 0, "mid": 1, "low": 2}
    q = str(min_quality).strip().lower()
    if q in quality_rank:
        out["_quality_rank"] = out["quality"].astype(str).str.strip().str.lower().map(quality_rank)
        out = out[out["_quality_rank"].notna() & (out["_quality_rank"] <= quality_rank[q])].copy()
        out = out.drop(columns=["_quality_rank"])
    return out

def compute_dynamic_features(ds: xr.Dataset, requested_features: list):
    bed = np.nan_to_num(ds["bed_elevation"].values, nan=0.0)

    if "bed_slope" in requested_features and "bed_slope" not in ds:
        print("  -> Computing dynamic feature: bed_slope")
        slope = ndimage.gaussian_gradient_magnitude(bed, sigma=1.0)
        ds["bed_slope"] = (("y", "x"), slope.astype(np.float32))

    if "bed_roughness" in requested_features and "bed_roughness" not in ds:
        print("  -> Computing dynamic feature: bed_roughness")
        c1 = ndimage.uniform_filter(bed, size=3)
        c2 = ndimage.uniform_filter(bed * bed, size=3)
        roughness = np.sqrt(np.clip(c2 - c1 * c1, 0, None))
        ds["bed_roughness"] = (("y", "x"), roughness.astype(np.float32))

    return ds

def prepare_training_tensors(
    ds: xr.Dataset, df: pd.DataFrame, mlp_features: list, gp_features: list,
    min_age: float, max_age: float, ages_bp_ref_year: float, model_bp_ref_year: float
):
    x_obs = df["x_3413"].to_numpy(dtype=np.float32)
    y_obs = df["y_3413"].to_numpy(dtype=np.float32)
    ages_raw = df["age_mean"].to_numpy(dtype=np.float32)
    errs_raw = df["age_sd"].to_numpy(dtype=np.float32)

    ages_shifted = ages_raw - np.float32(float(ages_bp_ref_year) - float(model_bp_ref_year))
    ages_norm = (ages_shifted - min_age) / (max_age - min_age)
    ages_norm = np.clip(ages_norm, 0, 1)
    errs_norm = errs_raw / (max_age - min_age)

    x_xr = xr.DataArray(x_obs, dims="points")
    y_xr = xr.DataArray(y_obs, dims="points")
    ds_sampled = ds.interp(x=x_xr, y=y_xr, method="linear")
    
    valid_mask = np.isfinite(ages_norm)
    all_features = list(set(mlp_features + gp_features))
    
    coords_raw = np.column_stack((x_obs, y_obs))
    feature_stats = {}
    extracted_features = {}
    
    for feat in all_features:
        if feat not in ds_sampled:
            raise KeyError(f"Feature '{feat}' not found in the dataset.")
        feat_data = ds_sampled[feat].values.astype(np.float32)
        valid_mask &= np.isfinite(feat_data)
        extracted_features[feat] = feat_data
        
    coords_valid = coords_raw[valid_mask]
    ages_valid = ages_norm[valid_mask]
    errs_valid = errs_norm[valid_mask]
    
    coords_mean = coords_valid.mean(axis=0)
    coords_std = coords_valid.std(axis=0)
    coords_scaled = (coords_valid - coords_mean) / coords_std
    feature_stats['coords'] = {'mean': coords_mean, 'std': coords_std}

    # Standardize all requested custom features
    scaled_features = {}
    for feat in all_features:
        feat_valid = extracted_features[feat][valid_mask]
        feat_mean = float(np.nanmean(feat_valid))
        feat_std = float(np.nanstd(feat_valid))
        if feat_std == 0: feat_std = 1.0
        
        scaled_features[feat] = (feat_valid - feat_mean) / feat_std
        feature_stats[feat] = {'mean': feat_mean, 'std': feat_std}

    # Build MLP Tensor (Coords + Requested MLP Features)
    mlp_tensor_list = [torch.tensor(coords_scaled, dtype=torch.float32)]
    for feat in mlp_features:
        mlp_tensor_list.append(torch.tensor(scaled_features[feat][:, None], dtype=torch.float32))
    train_x_mlp = torch.cat(mlp_tensor_list, dim=1)
    
    # Build GP Tensor (Coords + Requested GP Features)
    gp_tensor_list = [torch.tensor(coords_scaled, dtype=torch.float32)]
    for feat in gp_features:
        gp_tensor_list.append(torch.tensor(scaled_features[feat][:, None], dtype=torch.float32))
    train_x_gp = torch.cat(gp_tensor_list, dim=1)

    train_y = torch.tensor(ages_valid, dtype=torch.float32)
    train_errs = torch.tensor(errs_valid, dtype=torch.float32)
    
    return train_x_mlp, train_x_gp, train_y, train_errs, valid_mask, feature_stats

# --- 3. Visualization & Validation ---

def get_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    cod = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
    else:
        pearson_r2 = 0.0
        
    return mse, cod, pearson_r2

def run_checkerboard_cv(train_x_mlp, train_x_gp, train_y, train_errs, feature_stats, min_age, max_age, size, iterations):
    coords_mean = feature_stats['coords']['mean']
    coords_std = feature_stats['coords']['std']
    x_coords = (train_x_mlp[:, 0].numpy() * coords_std[0]) + coords_mean[0]
    y_coords = (train_x_mlp[:, 1].numpy() * coords_std[1]) + coords_mean[1]

    ix = np.floor(x_coords / size).astype(int)
    iy = np.floor(y_coords / size).astype(int)
    is_even = ((ix + iy) % 2) == 0
    mask_A, mask_B = is_even, ~is_even
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for idx, (tr_mask, te_mask, name) in enumerate([
        (mask_A, mask_B, "Fold 1: Train A / Test B"), 
        (mask_B, mask_A, "Fold 2: Train B / Test A")
    ]):
        if np.sum(tr_mask) == 0 or np.sum(te_mask) == 0: continue
            
        print(f"--- Running CV: {name} (Train: {np.sum(tr_mask)}, Test: {np.sum(te_mask)}) ---")
        
        tx_mlp, tx_gp = train_x_mlp[tr_mask], train_x_gp[tr_mask]
        ty, te = train_y[tr_mask], train_errs[tr_mask]
        
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=te**2, learn_additional_noise=True)
        model = DataDrivenGP(tx_mlp, tx_gp, ty, likelihood)
        model.covar_module.base_kernel.lengthscale = torch.tensor([[0.5] * tx_gp.shape[1]])
        
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        with gpytorch.settings.max_cg_iterations(2000), gpytorch.settings.cholesky_jitter(1e-4):
            for i in range(iterations):
                optimizer.zero_grad()
                output = model(tx_mlp, tx_gp)
                loss = -mll(output, ty)
                loss.backward()
                optimizer.step()
                
        model.eval()
        test_x_mlp, test_x_gp = train_x_mlp[te_mask], train_x_gp[te_mask]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_age_norm = model(test_x_mlp, test_x_gp).mean.numpy()
            
        pred_age_yrs = pred_age_norm * (max_age - min_age) + min_age
        true_age_yrs = train_y[te_mask].numpy() * (max_age - min_age) + min_age
        
        mse, cod, pearson_r2 = get_metrics(true_age_yrs, pred_age_yrs)
        print(f"  Result -> RMSE: {np.sqrt(mse):.0f} yrs | R^2: {cod:.3f}")
        
        ax_scatter = axes[0, idx]
        ax_scatter.scatter(true_age_yrs, pred_age_yrs, alpha=0.7, edgecolors='k')
        ax_scatter.plot([min_age, max_age], [min_age, max_age], 'r--', lw=2)
        ax_scatter.set_title(f"{name}\nCoef. of Determination ($R^2$): {cod:.3f}\nRMSE: {np.sqrt(mse):.0f} yrs")
        ax_scatter.set_xlabel("Observed Age (Years)")
        ax_scatter.set_ylabel("Predicted Age (Years)")
        
        residuals = pred_age_yrs - true_age_yrs
        ax_map = axes[1, idx]
        sc = ax_map.scatter(x_coords[te_mask], y_coords[te_mask], c=residuals, cmap='RdBu', vmin=-2500, vmax=2500, alpha=0.8, edgecolors='k')
        ax_map.set_title("Residuals Map (Predicted - Observed)")
        ax_map.set_aspect('equal')
        plt.colorbar(sc, ax=ax_map, label="Residual (Years)")

    plt.tight_layout()
    plt.show()

def predict_and_plot_grid(model, ds, mlp_features, gp_features, feature_stats, min_age, max_age):
    model.eval()
    
    xg = ds["x"].values
    yg = ds["y"].values
    X_grid, Y_grid = np.meshgrid(xg, yg)
    
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()

    # 1. Base Validity Mask (Exclude NaNs)
    valid_mask = np.ones_like(X_flat, dtype=bool)
    all_features = list(set(mlp_features + gp_features))
    
    flat_features = {}
    for feat in all_features:
        feat_flat = ds[feat].values.flatten()
        valid_mask &= np.isfinite(feat_flat)
        flat_features[feat] = feat_flat

    # 2. Ice Masking
    if "thickness" in ds:
        ice_mask = ds["thickness"].values.flatten() > 0.0
    elif "ice_mask" in ds:
        ice_mask = ds["ice_mask"].values.flatten() == 1
    else:
        ice_mask = np.zeros_like(X_flat, dtype=bool)
    valid_mask &= ~ice_mask
        
    coords_raw = np.column_stack((X_flat[valid_mask], Y_flat[valid_mask]))
    coords_mean, coords_std = feature_stats['coords']['mean'], feature_stats['coords']['std']
    coords_scaled = (coords_raw - coords_mean) / coords_std
    
    # Build MLP Test Tensor
    mlp_tensor_list = [torch.tensor(coords_scaled, dtype=torch.float32)]
    for feat in mlp_features:
        feat_raw = flat_features[feat][valid_mask].astype(np.float32)
        f_mean, f_std = feature_stats[feat]['mean'], feature_stats[feat]['std']
        feat_scaled = (feat_raw - f_mean) / f_std
        mlp_tensor_list.append(torch.tensor(feat_scaled[:, None], dtype=torch.float32))
    test_x_mlp = torch.cat(mlp_tensor_list, dim=1)
    
    # Build GP Test Tensor
    gp_tensor_list = [torch.tensor(coords_scaled, dtype=torch.float32)]
    for feat in gp_features:
        feat_raw = flat_features[feat][valid_mask].astype(np.float32)
        f_mean, f_std = feature_stats[feat]['mean'], feature_stats[feat]['std']
        feat_scaled = (feat_raw - f_mean) / f_std
        gp_tensor_list.append(torch.tensor(feat_scaled[:, None], dtype=torch.float32))
    test_x_gp = torch.cat(gp_tensor_list, dim=1)
    
    print(f"\nPredicting on {test_x_mlp.shape[0]} valid grid pixels in batches...")
    batch_size = 20000
    pred_means = []
    pred_vars = []
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, test_x_mlp.shape[0], batch_size):
            b_mlp = test_x_mlp[i : i + batch_size]
            b_gp = test_x_gp[i : i + batch_size]
            output = model(b_mlp, b_gp)
            pred_means.append(output.mean.numpy())
            pred_vars.append(output.variance.numpy())
            
    pred_mean_flat = np.concatenate(pred_means)
    pred_var_flat = np.concatenate(pred_vars)
    
    final_age_years = pred_mean_flat * (max_age - min_age) + min_age
    uncertainty_years = np.sqrt(pred_var_flat) * (max_age - min_age)
    
    age_map = np.full_like(X_flat, np.nan, dtype=np.float32)
    age_map[valid_mask] = final_age_years
    age_map = age_map.reshape(X_grid.shape)
    
    unc_map = np.full_like(X_flat, np.nan, dtype=np.float32)
    unc_map[valid_mask] = uncertainty_years
    unc_map = unc_map.reshape(X_grid.shape)
    
    # Prediction from the MLP Alone
    with torch.no_grad():
        mlp_only_flat = model.mean_module(test_x_mlp).numpy()
    mlp_only_years = mlp_only_flat * (max_age - min_age) + min_age
    mlp_map = np.full_like(X_flat, np.nan, dtype=np.float32)
    mlp_map[valid_mask] = mlp_only_years
    mlp_map = mlp_map.reshape(X_grid.shape)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
    n_levels = 64
    age_bounds = np.linspace(min_age, max_age, n_levels + 1, dtype=np.float32)
    age_cmap = plt.get_cmap("seismic_r", n_levels)
    age_norm = mcolors.BoundaryNorm(age_bounds, age_cmap.N, clip=True)
    
    im1 = axes[0].pcolormesh(X_grid, Y_grid, age_map, cmap=age_cmap, norm=age_norm, shading="auto")
    axes[0].set_title("Full Model Prediction (MLP + GP)")
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0], label="Age (Years)", boundaries=age_bounds)
    
    im2 = axes[1].pcolormesh(X_grid, Y_grid, mlp_map, cmap=age_cmap, norm=age_norm, shading="auto")
    axes[1].set_title("Neural Network Output (Global Trend Only)")
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
    parser = argparse.ArgumentParser(description="Train a Data-Driven MLP+GP for deglaciation age.")
    
    parser.add_argument("--nc_path", type=Path, default=Path("data/modern_fields_native.nc"))
    parser.add_argument("--ages_path", type=Path, default=Path("data/age_data/combined_ages.csv"))
    
    # --- SPLIT FEATURE SELECTION ---
    parser.add_argument("--mlp-features", nargs="*", 
                        default=["bed_elevation", "signed_distance_to_margin"],
                        help="Features passed to the Neural Network (mean function). Coordinates (x,y) are always included automatically.")
    parser.add_argument("--gp-features", nargs="*", 
                        default=['bed_elevation', 'bed_slope'],
                        help="Features passed to the Gaussian Process (covariance function). Coordinates (x,y) are always included automatically.")
    
    parser.add_argument("--cosmogenic-only", action="store_true")
    parser.add_argument("--min-quality", type=str, default="Low", choices=["High", "Mid", "Low"])
    parser.add_argument("--min-age", type=float, default=0.0)
    parser.add_argument("--max-age", type=float, default=14000.0)
    parser.add_argument("--ages-bp-ref-year", type=float, default=1950.0)
    parser.add_argument("--model-bp-ref-year", type=float, default=1850.0)
    
    parser.add_argument("--checkerboard-cv", action="store_true")
    parser.add_argument("--checkerboard-size", type=float, default=50000.0)
    parser.add_argument("--epochs", type=int, default=4000)

    args = parser.parse_args()

    # Ensure empty lists instead of None if no arguments are passed
    mlp_feats = args.mlp_features if args.mlp_features else []
    gp_feats = args.gp_features if args.gp_features else []

    print(f"Loading Raster Features: {args.nc_path}")
    ds = xr.open_dataset(args.nc_path, decode_times=False)
    
    all_requested = list(set(mlp_feats + gp_feats))
    ds = compute_dynamic_features(ds, all_requested)
    
    print(f"Loading Observations: {args.ages_path}")
    df = pd.read_csv(args.ages_path)
    df = filter_age_data(df, cosmogenic_only=bool(args.cosmogenic_only), min_quality=str(args.min_quality))
    
    print(f"\nInterpolating and standardizing features...")
    print(f"  MLP Features: ['x', 'y'] + {mlp_feats}")
    print(f"  GP Features:  ['x', 'y'] + {gp_feats}")
    
    train_x_mlp, train_x_gp, train_y, train_errs, valid_mask, feature_stats = prepare_training_tensors(
        ds, df, mlp_feats, gp_feats, args.min_age, args.max_age, 
        args.ages_bp_ref_year, args.model_bp_ref_year
    )
    
    print(f"Total valid training points: {len(train_y)}")
    
    if args.checkerboard_cv:
        print(f"\n--- Running Spatial Checkerboard CV (Size: {args.checkerboard_size/1000:.0f} km) ---")
        run_checkerboard_cv(
            train_x_mlp, train_x_gp, train_y, train_errs, feature_stats, 
            args.min_age, args.max_age, args.checkerboard_size, iterations=4000
        )
        print("--- CV Complete. Proceeding to train on FULL dataset. ---\n")

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=train_errs**2, learn_additional_noise=True)
    model = DataDrivenGP(train_x_mlp, train_x_gp, train_y, likelihood)
    
    model.covar_module.base_kernel.lengthscale = torch.tensor([[0.5] * train_x_gp.shape[1]])
    
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    print("Starting Final GP + MLP Training...")
    with gpytorch.settings.max_cg_iterations(2000), gpytorch.settings.cholesky_jitter(1e-4):
        for i in range(args.epochs):
            optimizer.zero_grad()
            output = model(train_x_mlp, train_x_gp)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                lengthscale = model.covar_module.base_kernel.lengthscale.detach().numpy()[0]
                ls_str = ", ".join([f"{ls:.3f}" for ls in lengthscale])
                print(f"Iter {i+1:>4}/{args.epochs} - Loss: {loss.item():.3f} | GP Lengthscales: [{ls_str}]")

    print("\nTraining complete.")
    predict_and_plot_grid(model, ds, mlp_feats, gp_feats, feature_stats, args.min_age, args.max_age)

if __name__ == "__main__":
    main()