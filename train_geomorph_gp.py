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
from tqdm import tqdm

# --- 1. Neural Network & GP Model Definition ---

class DataDrivenGP(gpytorch.models.ExactGP):
    def __init__(self, train_x_mlp, train_x_gp, train_y, likelihood):
        super(DataDrivenGP, self).__init__((train_x_mlp, train_x_gp), train_y, likelihood)
        
        self.mean_module = gpytorch.means.LinearMean(input_size=train_x_mlp.shape[1], bias=True)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            # Reverted to nu=2.5 to match the original smoothness assumption
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x_gp.shape[1])
        )

    def forward(self, x_mlp, x_gp):
        mean_pred = self.mean_module(x_mlp)
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
            raise KeyError(f"Feature '{feat}' not found in dataset.")
        feat_data = ds_sampled[feat].values.astype(np.float32)
        valid_mask &= np.isfinite(feat_data)
        extracted_features[feat] = feat_data
        
    coords_valid = coords_raw[valid_mask]
    coords_mean = coords_valid.mean(axis=0)
    coords_std = coords_valid.std(axis=0)
    coords_scaled = (coords_valid - coords_mean) / coords_std
    feature_stats['coords'] = {'mean': coords_mean, 'std': coords_std}

    scaled_features = {}
    for feat in all_features:
        feat_valid = extracted_features[feat][valid_mask]
        feat_mean = float(np.nanmean(feat_valid))
        feat_std = float(np.nanstd(feat_valid))
        if feat_std == 0: feat_std = 1.0
        scaled_features[feat] = (feat_valid - feat_mean) / feat_std
        feature_stats[feat] = {'mean': feat_mean, 'std': feat_std}

    mlp_tensor_list = [torch.tensor(coords_scaled, dtype=torch.float32)]
    for feat in mlp_features:
        mlp_tensor_list.append(torch.tensor(scaled_features[feat][:, None], dtype=torch.float32))
    train_x_mlp = torch.cat(mlp_tensor_list, dim=1)
    
    gp_tensor_list = [torch.tensor(coords_scaled, dtype=torch.float32)]
    for feat in gp_features:
        gp_tensor_list.append(torch.tensor(scaled_features[feat][:, None], dtype=torch.float32))
    train_x_gp = torch.cat(gp_tensor_list, dim=1)

    train_y = torch.tensor(ages_norm[valid_mask], dtype=torch.float32)
    train_errs = torch.tensor(errs_norm[valid_mask], dtype=torch.float32)
    
    return train_x_mlp, train_x_gp, train_y, train_errs, valid_mask, feature_stats

def prepare_moraine_tensors(ds, df_moraines, mlp_features, gp_features, feature_stats, step_size_m=500.0):
    """Calculates forward and backward step coordinates and scales them identically to the training data."""
    x_m = df_moraines["x_3413"].values
    y_m = df_moraines["y_3413"].values
    vx = df_moraines["vx"].values
    vy = df_moraines["vy"].values
    
    # Step forward and backward
    x_fwd, y_fwd = x_m + (step_size_m * vx), y_m + (step_size_m * vy)
    x_bwd, y_bwd = x_m - (step_size_m * vx), y_m - (step_size_m * vy)
    
    all_features = list(set(mlp_features + gp_features))
    
    def extract_and_scale(x_coords, y_coords):
        x_xr, y_xr = xr.DataArray(x_coords, dims="points"), xr.DataArray(y_coords, dims="points")
        ds_sampled = ds.interp(x=x_xr, y=y_xr, method="linear")
        
        valid_mask = np.ones(len(x_coords), dtype=bool)
        scaled_dict = {}
        for feat in all_features:
            feat_data = ds_sampled[feat].values.astype(np.float32)
            valid_mask &= np.isfinite(feat_data)
            f_mean, f_std = feature_stats[feat]['mean'], feature_stats[feat]['std']
            scaled_dict[feat] = (feat_data - f_mean) / f_std
            
        c_mean, c_std = feature_stats['coords']['mean'], feature_stats['coords']['std']
        coords_raw = np.column_stack((x_coords, y_coords))
        coords_scaled = (coords_raw - c_mean) / c_std
        
        mlp_list = [torch.tensor(coords_scaled, dtype=torch.float32)]
        for feat in mlp_features: mlp_list.append(torch.tensor(scaled_dict[feat][:, None], dtype=torch.float32))
        x_mlp = torch.cat(mlp_list, dim=1)
        
        gp_list = [torch.tensor(coords_scaled, dtype=torch.float32)]
        for feat in gp_features: gp_list.append(torch.tensor(scaled_dict[feat][:, None], dtype=torch.float32))
        x_gp = torch.cat(gp_list, dim=1)
        
        return x_mlp, x_gp, valid_mask
        
    fwd_mlp, fwd_gp, fwd_mask = extract_and_scale(x_fwd, y_fwd)
    bwd_mlp, bwd_gp, bwd_mask = extract_and_scale(x_bwd, y_bwd)
    
    valid_pair_mask = fwd_mask & bwd_mask
    return fwd_mlp[valid_pair_mask], fwd_gp[valid_pair_mask], bwd_mlp[valid_pair_mask], bwd_gp[valid_pair_mask]

# --- 3. Unified Training & Validation ---

def get_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    cod = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    pearson_r2 = np.corrcoef(y_true, y_pred)[0, 1]**2 if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0 else 0.0
    return mse, cod, pearson_r2

def train_pinn_loop(
    model, likelihood, optimizer, mll, iterations,
    tx_mlp, tx_gp, ty,
    mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd,
    lambda_geom, desc="Training"
):
    model.train()
    likelihood.train()
    
    pbar = tqdm(range(iterations), desc=desc)
    for i in pbar:
        optimizer.zero_grad()
        output = model(tx_mlp, tx_gp)
        loss_mll = -mll(output, ty)
        
        loss_geom = torch.tensor(0.0)
        if lambda_geom > 0 and mx_mlp_fwd is not None:
            model.eval()
            with gpytorch.settings.fast_pred_var(False):
                pred_fwd = model(mx_mlp_fwd, mx_gp_fwd).mean
                pred_bwd = model(mx_mlp_bwd, mx_gp_bwd).mean
                loss_geom = torch.mean((pred_fwd - pred_bwd)**2)
            model.train()
            
        loss_total = loss_mll + (lambda_geom * loss_geom)
        loss_total.backward()
        optimizer.step()
        
        pbar.set_postfix({
            "Loss": f"{loss_total.item():.2f}", 
            "MLL": f"{loss_mll.item():.2f}", 
            "Geom": f"{loss_geom.item():.4f}"
        })

def run_checkerboard_cv(
    train_x_mlp, train_x_gp, train_y, train_errs, feature_stats, 
    mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd, lambda_geom,
    min_age, max_age, size, iterations
):
    coords_mean, coords_std = feature_stats['coords']['mean'], feature_stats['coords']['std']
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
            
        print(f"\n--- Running CV: {name} (Train: {np.sum(tr_mask)}, Test: {np.sum(te_mask)}) ---")
        tx_mlp, tx_gp, ty, te = train_x_mlp[tr_mask], train_x_gp[tr_mask], train_y[tr_mask], train_errs[tr_mask]
        
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=te**2, learn_additional_noise=True)
        model = DataDrivenGP(tx_mlp, tx_gp, ty, likelihood)
        model.covar_module.base_kernel.lengthscale = torch.tensor([[0.5] * tx_gp.shape[1]])
        
        # Original LR = 0.02
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        # Original CG settings enforced
        with gpytorch.settings.max_cg_iterations(2000), gpytorch.settings.cholesky_jitter(1e-4):
            train_pinn_loop(
                model, likelihood, optimizer, mll, iterations,
                tx_mlp, tx_gp, ty,
                mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd,
                lambda_geom, desc=f"CV {name}"
            )
                
        model.eval()
        test_x_mlp, test_x_gp = train_x_mlp[te_mask], train_x_gp[te_mask]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_age_norm = model(test_x_mlp, test_x_gp).mean.numpy()
            
        pred_age_yrs = pred_age_norm * (max_age - min_age) + min_age
        true_age_yrs = train_y[te_mask].numpy() * (max_age - min_age) + min_age
        
        mse, cod, pearson_r2 = get_metrics(true_age_yrs, pred_age_yrs)
        print(f"  Result -> RMSE: {np.sqrt(mse):.0f} yrs | R^2: {cod:.3f}")
        
        # Plotting the 2x2 grid from original script
        ax_scatter = axes[0, idx]
        ax_scatter.scatter(true_age_yrs, pred_age_yrs, alpha=0.7, edgecolors='k')
        ax_scatter.plot([min_age, max_age], [min_age, max_age], 'r--', lw=2)
        ax_scatter.set_title(f"{name}\n$R^2$: {cod:.3f} | RMSE: {np.sqrt(mse):.0f} yrs")
        ax_scatter.set_xlabel("Observed Age (Years BP)")
        ax_scatter.set_ylabel("Predicted Age (Years BP)")
        
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
    xg, yg = ds["x"].values, ds["y"].values
    X_grid, Y_grid = np.meshgrid(xg, yg)
    X_flat, Y_flat = X_grid.flatten(), Y_grid.flatten()

    valid_mask = np.ones_like(X_flat, dtype=bool)
    all_features = list(set(mlp_features + gp_features))
    flat_features = {}
    for feat in all_features:
        feat_flat = ds[feat].values.flatten()
        valid_mask &= np.isfinite(feat_flat)
        flat_features[feat] = feat_flat

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
    
    mlp_tensor_list = [torch.tensor(coords_scaled, dtype=torch.float32)]
    for feat in mlp_features:
        feat_raw = flat_features[feat][valid_mask].astype(np.float32)
        f_mean, f_std = feature_stats[feat]['mean'], feature_stats[feat]['std']
        mlp_tensor_list.append(torch.tensor(((feat_raw - f_mean) / f_std)[:, None], dtype=torch.float32))
    test_x_mlp = torch.cat(mlp_tensor_list, dim=1)
    
    gp_tensor_list = [torch.tensor(coords_scaled, dtype=torch.float32)]
    for feat in gp_features:
        feat_raw = flat_features[feat][valid_mask].astype(np.float32)
        f_mean, f_std = feature_stats[feat]['mean'], feature_stats[feat]['std']
        gp_tensor_list.append(torch.tensor(((feat_raw - f_mean) / f_std)[:, None], dtype=torch.float32))
    test_x_gp = torch.cat(gp_tensor_list, dim=1)
    
    print(f"\nPredicting on {test_x_mlp.shape[0]} valid grid pixels in batches...")
    batch_size = 20000
    pred_means, pred_vars = [], []
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, test_x_mlp.shape[0], batch_size):
            b_mlp, b_gp = test_x_mlp[i : i + batch_size], test_x_gp[i : i + batch_size]
            output = model(b_mlp, b_gp)
            pred_means.append(output.mean.numpy())
            pred_vars.append(output.variance.numpy())
            
    final_age_years = np.concatenate(pred_means) * (max_age - min_age) + min_age
    uncertainty_years = np.sqrt(np.concatenate(pred_vars)) * (max_age - min_age)
    
    age_map = np.full_like(X_flat, np.nan, dtype=np.float32)
    age_map[valid_mask] = final_age_years
    age_map = age_map.reshape(X_grid.shape)
    
    unc_map = np.full_like(X_flat, np.nan, dtype=np.float32)
    unc_map[valid_mask] = uncertainty_years
    unc_map = unc_map.reshape(X_grid.shape)
    
    with torch.no_grad():
        mlp_only_years = model.mean_module(test_x_mlp).numpy() * (max_age - min_age) + min_age
    mlp_map = np.full_like(X_flat, np.nan, dtype=np.float32)
    mlp_map[valid_mask] = mlp_only_years
    mlp_map = mlp_map.reshape(X_grid.shape)

    # 1x3 Plotting from original script
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    n_levels = 64
    age_bounds = np.linspace(min_age, max_age, n_levels + 1, dtype=np.float32)
    age_cmap = plt.get_cmap("inferno", n_levels)
    age_norm = mcolors.BoundaryNorm(age_bounds, age_cmap.N, clip=True)
    
    im1 = axes[0].pcolormesh(X_grid, Y_grid, age_map, cmap=age_cmap, norm=age_norm, shading="auto")
    axes[0].set_title("Full Model Prediction (Mean + GP)")
    axes[0].set_aspect('equal')
    axes[0].set_facecolor('lightgray')
    plt.colorbar(im1, ax=axes[0], label="Age (Years)", boundaries=age_bounds)
    
    im2 = axes[1].pcolormesh(X_grid, Y_grid, mlp_map, cmap=age_cmap, norm=age_norm, shading="auto")
    axes[1].set_title("Linear Mean Output (Global Trend Only)")
    axes[1].set_aspect('equal')
    axes[1].set_facecolor('lightgray')
    plt.colorbar(im2, ax=axes[1], label="Age (Years)", boundaries=age_bounds)

    im3 = axes[2].pcolormesh(X_grid, Y_grid, unc_map, cmap='plasma', shading='auto')
    axes[2].set_title("Prediction Uncertainty (1 Std Dev, Years)")
    axes[2].set_aspect('equal')
    axes[2].set_facecolor('lightgray')
    plt.colorbar(im3, ax=axes[2], label="Uncertainty (Years)")
    
    plt.tight_layout()
    plt.show()

# --- 4. Main Execution ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Geomorphology-Informed Data-Driven GP.")
    parser.add_argument("--nc_path", type=Path, default=Path("data/modern_fields_native.nc"))
    parser.add_argument("--ages_path", type=Path, default=Path("data/age_data/combined_ages.csv"))
    parser.add_argument("--moraines_path", type=Path, default=Path("data/age_data/moraine_tangents.csv"))
    
    parser.add_argument("--mlp-features", nargs="*", default=["signed_distance_to_margin"])
    parser.add_argument("--gp-features", nargs="*", default=["bed_elevation"])
    
    parser.add_argument("--lambda-geom", type=float, default=5.0)
    parser.add_argument("--moraine-step", type=float, default=500.0)
    parser.add_argument("--moraine-frac", type=float, default=1.0)
    
    parser.add_argument("--cosmogenic-only", action="store_true")
    parser.add_argument("--min-quality", type=str, default="Low", choices=["High", "Mid", "Low"])
    parser.add_argument("--min-age", type=float, default=0.0)
    parser.add_argument("--max-age", type=float, default=15000.0)
    parser.add_argument("--ages-bp-ref-year", type=float, default=1950.0)
    parser.add_argument("--model-bp-ref-year", type=float, default=1850.0)
    
    parser.add_argument("--checkerboard-cv", action="store_true")
    parser.add_argument("--checkerboard-size", type=float, default=66000.0)
    parser.add_argument("--epochs", type=int, default=750)
    args = parser.parse_args()

    mlp_feats = args.mlp_features if args.mlp_features else []
    gp_feats = args.gp_features if args.gp_features else []

    print(f"Loading Raster Features: {args.nc_path}")
    ds = compute_dynamic_features(xr.open_dataset(args.nc_path, decode_times=False), list(set(mlp_feats + gp_feats)))
    
    print(f"Loading Observations: {args.ages_path}")
    df = filter_age_data(pd.read_csv(args.ages_path), cosmogenic_only=bool(args.cosmogenic_only), min_quality=str(args.min_quality))
    
    print("\nInterpolating and standardizing features...")
    train_x_mlp, train_x_gp, train_y, train_errs, valid_mask, feature_stats = prepare_training_tensors(
        ds, df, mlp_feats, gp_feats, args.min_age, args.max_age, 
        args.ages_bp_ref_year, args.model_bp_ref_year
    )
    
    mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd = None, None, None, None
    if args.lambda_geom > 0 and args.moraines_path.exists():
        print("Processing Moraine Tensors (Geomorphology-Informed Mode)...")
        df_moraines = pd.read_csv(args.moraines_path)
        if args.moraine_frac < 1.0:
            df_moraines = df_moraines.sample(frac=args.moraine_frac, random_state=42)
            
        mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd = prepare_moraine_tensors(
            ds, df_moraines, mlp_feats, gp_feats, feature_stats, step_size_m=args.moraine_step
        )
        print(f"  -> Extracted {len(mx_mlp_fwd):,} valid PINN constraints.")

    if args.checkerboard_cv:
        run_checkerboard_cv(
            train_x_mlp, train_x_gp, train_y, train_errs, feature_stats,
            mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd, args.lambda_geom,
            args.min_age, args.max_age, args.checkerboard_size, args.epochs
        )
        print("--- CV Complete. Proceeding to train on FULL dataset. ---\n")

    print("Starting Final GP + Mean Training...")
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=train_errs**2, learn_additional_noise=True)
    model = DataDrivenGP(train_x_mlp, train_x_gp, train_y, likelihood)
    model.covar_module.base_kernel.lengthscale = torch.tensor([[0.5] * train_x_gp.shape[1]])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    with gpytorch.settings.max_cg_iterations(2000), gpytorch.settings.cholesky_jitter(1e-4):
        train_pinn_loop(
            model, likelihood, optimizer, mll, args.epochs,
            train_x_mlp, train_x_gp, train_y,
            mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd,
            args.lambda_geom, desc="Final Model"
        )

    print("\nTraining complete.")
    predict_and_plot_grid(model, ds, mlp_feats, gp_feats, feature_stats, args.min_age, args.max_age)

if __name__ == "__main__":
    main()