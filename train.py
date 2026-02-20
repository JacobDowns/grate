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

try:
    import rioxarray
    HAS_RIOXARRAY = True
except ImportError:
    HAS_RIOXARRAY = False
    print("Warning: 'rioxarray' not found. GeoTIFF export will be disabled. (Run: pip install rioxarray)")

# --- 1. Neural Network & GP Model Definition ---

class DataDrivenGP(gpytorch.models.ExactGP):
    def __init__(self, train_x_mlp, train_x_gp, train_y, likelihood):
        super(DataDrivenGP, self).__init__((train_x_mlp, train_x_gp), train_y, likelihood)
        self.mean_module = gpytorch.means.LinearMean(input_size=train_x_mlp.shape[1], bias=True)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x_gp.shape[1])
        )

    def forward(self, x_mlp, x_gp):
        mean_pred = self.mean_module(x_mlp)
        covar_pred = self.covar_module(x_gp)
        return gpytorch.distributions.MultivariateNormal(mean_pred, covar_pred)

# --- 2. Data Loading & Feature Engineering ---
# (Kept exactly identical to the previous script for mathematical consistency)

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
        feat_data = ds_sampled[feat].values.astype(np.float32)
        valid_mask &= np.isfinite(feat_data)
        extracted_features[feat] = feat_data
        
    coords_valid = coords_raw[valid_mask]
    coords_mean, coords_std = coords_valid.mean(axis=0), coords_valid.std(axis=0)
    coords_scaled = (coords_valid - coords_mean) / coords_std
    feature_stats['coords'] = {'mean': coords_mean, 'std': coords_std}

    scaled_features = {}
    for feat in all_features:
        feat_valid = extracted_features[feat][valid_mask]
        feat_mean, feat_std = float(np.nanmean(feat_valid)), float(np.nanstd(feat_valid))
        if feat_std == 0: feat_std = 1.0
        scaled_features[feat] = (feat_valid - feat_mean) / feat_std
        feature_stats[feat] = {'mean': feat_mean, 'std': feat_std}

    mlp_tensor_list = [torch.tensor(coords_scaled, dtype=torch.float32)]
    for feat in mlp_features: mlp_tensor_list.append(torch.tensor(scaled_features[feat][:, None], dtype=torch.float32))
    train_x_mlp = torch.cat(mlp_tensor_list, dim=1)
    
    gp_tensor_list = [torch.tensor(coords_scaled, dtype=torch.float32)]
    for feat in gp_features: gp_tensor_list.append(torch.tensor(scaled_features[feat][:, None], dtype=torch.float32))
    train_x_gp = torch.cat(gp_tensor_list, dim=1)

    train_y = torch.tensor(ages_norm[valid_mask], dtype=torch.float32)
    train_errs = torch.tensor(errs_norm[valid_mask], dtype=torch.float32)
    
    return train_x_mlp, train_x_gp, train_y, train_errs, valid_mask, feature_stats

def prepare_moraine_tensors(ds, df_moraines, mlp_features, gp_features, feature_stats, step_size_m=500.0):
    x_m, y_m = df_moraines["x_3413"].values, df_moraines["y_3413"].values
    vx, vy = df_moraines["vx"].values, df_moraines["vy"].values
    
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
            scaled_dict[feat] = (feat_data - feature_stats[feat]['mean']) / feature_stats[feat]['std']
            
        c_mean, c_std = feature_stats['coords']['mean'], feature_stats['coords']['std']
        coords_scaled = (np.column_stack((x_coords, y_coords)) - c_mean) / c_std
        
        mlp_list = [torch.tensor(coords_scaled, dtype=torch.float32)]
        for feat in mlp_features: mlp_list.append(torch.tensor(scaled_dict[feat][:, None], dtype=torch.float32))
        
        gp_list = [torch.tensor(coords_scaled, dtype=torch.float32)]
        for feat in gp_features: gp_list.append(torch.tensor(scaled_dict[feat][:, None], dtype=torch.float32))
        
        return torch.cat(mlp_list, dim=1), torch.cat(gp_list, dim=1), valid_mask
        
    fwd_mlp, fwd_gp, fwd_mask = extract_and_scale(x_fwd, y_fwd)
    bwd_mlp, bwd_gp, bwd_mask = extract_and_scale(x_bwd, y_bwd)
    
    valid_pair_mask = fwd_mask & bwd_mask
    return fwd_mlp[valid_pair_mask], fwd_gp[valid_pair_mask], bwd_mlp[valid_pair_mask], bwd_gp[valid_pair_mask]

# --- 3. Unified Training & CV Plotting ---

def get_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    ss_res, ss_tot = np.sum((y_true - y_pred)**2), np.sum((y_true - np.mean(y_true))**2)
    cod = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return mse, cod

def train_pinn_loop(
    model, likelihood, optimizer, mll, iterations,
    tx_mlp, tx_gp, ty, mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd,
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
        
        pbar.set_postfix({"Loss": f"{loss_total.item():.2f}", "MLL": f"{loss_mll.item():.2f}", "Geom": f"{loss_geom.item():.4f}"})

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
    mask_A = ((ix + iy) % 2) == 0
    mask_B = ~mask_A
    
    cv_x, cv_y, cv_true, cv_pred = [], [], [], []
    
    for idx, (tr_mask, te_mask, name) in enumerate([
        (mask_A, mask_B, "Fold 1"), 
        (mask_B, mask_A, "Fold 2")
    ]):
        if np.sum(tr_mask) == 0 or np.sum(te_mask) == 0: continue
            
        print(f"\n--- Running CV: {name} (Train: {np.sum(tr_mask)}, Test: {np.sum(te_mask)}) ---")
        tx_mlp, tx_gp, ty, te = train_x_mlp[tr_mask], train_x_gp[tr_mask], train_y[tr_mask], train_errs[tr_mask]
        
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=te**2, learn_additional_noise=True)
        model = DataDrivenGP(tx_mlp, tx_gp, ty, likelihood)
        model.covar_module.base_kernel.lengthscale = torch.tensor([[0.5] * tx_gp.shape[1]])
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        with gpytorch.settings.max_cg_iterations(2000), gpytorch.settings.cholesky_jitter(1e-4):
            train_pinn_loop(
                model, likelihood, optimizer, mll, iterations,
                tx_mlp, tx_gp, ty, mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd,
                lambda_geom, desc=f"CV {name}"
            )
                
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_age_norm = model(train_x_mlp[te_mask], train_x_gp[te_mask]).mean.numpy()
            
        pred_age_yrs = pred_age_norm * (max_age - min_age) + min_age
        true_age_yrs = train_y[te_mask].numpy() * (max_age - min_age) + min_age
        
        cv_x.extend(x_coords[te_mask])
        cv_y.extend(y_coords[te_mask])
        cv_true.extend(true_age_yrs)
        cv_pred.extend(pred_age_yrs)
        
        mse, cod = get_metrics(true_age_yrs, pred_age_yrs)
        print(f"  Result -> RMSE: {np.sqrt(mse):.0f} yrs | R^2: {cod:.3f}")
        
    # --- New Unified Spatial CV Map ---
    cv_x, cv_y = np.array(cv_x), np.array(cv_y)
    cv_true, cv_pred = np.array(cv_true), np.array(cv_pred)
    sq_errors = (cv_pred - cv_true)**2
    mse_total, cod_total = get_metrics(cv_true, cv_pred)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: 1:1 Scatter
    axes[0].scatter(cv_true, cv_pred, alpha=0.6, edgecolors='k', color='royalblue')
    min_val, max_val = min(cv_true.min(), cv_pred.min()), max(cv_true.max(), cv_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_title(f"Out-of-Sample CV Predictions\nTotal $R^2$: {cod_total:.3f} | Total RMSE: {np.sqrt(mse_total):.0f} yrs")
    axes[0].set_xlabel("Observed Age (Years BP)")
    axes[0].set_ylabel("Predicted Age (Years BP)")
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Right: Spatial MSE Map with Checkerboard Overlay
    vmax_cap = np.percentile(sq_errors, 95) # Cap colors at 95th percentile to prevent blowout from 1 outlier
    sc = axes[1].scatter(cv_x, cv_y, c=sq_errors, cmap='Reds', s=40, edgecolors='k', linewidth=0.5, vmin=0, vmax=vmax_cap)
    
    # Draw Grid Lines
    x_grids = np.arange(np.floor(cv_x.min()/size)*size, np.ceil(cv_x.max()/size)*size + size, size)
    y_grids = np.arange(np.floor(cv_y.min()/size)*size, np.ceil(cv_y.max()/size)*size + size, size)
    for xg in x_grids: axes[1].axvline(xg, color='gray', linestyle='--', alpha=0.5, zorder=0)
    for yg in y_grids: axes[1].axhline(yg, color='gray', linestyle='--', alpha=0.5, zorder=0)

    axes[1].set_title(f"Spatial Distribution of Squared Errors\n(Checkerboard Size: {size/1000:.0f} km)")
    axes[1].set_aspect('equal')
    axes[1].set_facecolor('whitesmoke')
    plt.colorbar(sc, ax=axes[1], label="Squared Error (Years^2)")

    plt.tight_layout()
    plt.show()

# --- 4. Final Grid Prediction & GeoTIFF Export ---

def predict_and_export_grid(model, ds, args, feature_stats, min_age, max_age):
    model.eval()
    xg, yg = ds["x"].values, ds["y"].values
    X_grid, Y_grid = np.meshgrid(xg, yg)
    X_flat, Y_flat = X_grid.flatten(), Y_grid.flatten()

    all_features = list(set(args.mlp_features + args.gp_features))
    valid_mask = np.ones_like(X_flat, dtype=bool)
    flat_features = {}
    
    for feat in all_features:
        feat_flat = ds[feat].values.flatten()
        valid_mask &= np.isfinite(feat_flat)
        flat_features[feat] = feat_flat

    if "thickness" in ds: ice_mask = ds["thickness"].values.flatten() > 10.0
    elif "ice_mask" in ds: ice_mask = ds["ice_mask"].values.flatten() == 1
    else: ice_mask = np.zeros_like(X_flat, dtype=bool)
    valid_mask &= ~ice_mask
        
    coords_scaled = (np.column_stack((X_flat[valid_mask], Y_flat[valid_mask])) - feature_stats['coords']['mean']) / feature_stats['coords']['std']
    
    mlp_tensor_list = [torch.tensor(coords_scaled, dtype=torch.float32)]
    for feat in args.mlp_features:
        f_scaled = (flat_features[feat][valid_mask].astype(np.float32) - feature_stats[feat]['mean']) / feature_stats[feat]['std']
        mlp_tensor_list.append(torch.tensor(f_scaled[:, None], dtype=torch.float32))
    test_x_mlp = torch.cat(mlp_tensor_list, dim=1)
    
    gp_tensor_list = [torch.tensor(coords_scaled, dtype=torch.float32)]
    for feat in args.gp_features:
        f_scaled = (flat_features[feat][valid_mask].astype(np.float32) - feature_stats[feat]['mean']) / feature_stats[feat]['std']
        gp_tensor_list.append(torch.tensor(f_scaled[:, None], dtype=torch.float32))
    test_x_gp = torch.cat(gp_tensor_list, dim=1)
    
    print(f"\nPredicting on {test_x_mlp.shape[0]:,} valid grid pixels...")
    batch_size = 20000
    pred_means, pred_vars = [], []
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in tqdm(range(0, test_x_mlp.shape[0], batch_size), desc="Predicting Map"):
            output = model(test_x_mlp[i : i + batch_size], test_x_gp[i : i + batch_size])
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

    # Export to GeoTIFF if rioxarray is available and requested
    if HAS_RIOXARRAY and args.export_dir is not None:
        out_path = Path(args.export_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        print(f"\nExporting GeoTIFFs to {out_path}...")
        
        da_age = xr.DataArray(age_map, coords=[("y", yg), ("x", xg)], dims=["y", "x"], name="age_mean_yrs")
        da_unc = xr.DataArray(unc_map, coords=[("y", yg), ("x", xg)], dims=["y", "x"], name="age_std_yrs")
        
        # Write CRS and save
        da_age.rio.write_crs("EPSG:3413", inplace=True).rio.to_raster(out_path / "predicted_age.tif")
        da_unc.rio.write_crs("EPSG:3413", inplace=True).rio.to_raster(out_path / "predicted_uncertainty.tif")
        print("GeoTIFF export complete.")

    # 1x2 Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    im1 = axes[0].pcolormesh(X_grid, Y_grid, age_map, cmap="turbo_r", shading="auto", vmin=0, vmax=max_age)
    axes[0].set_title("Full Model Prediction (Age BP)")
    axes[0].set_aspect('equal')
    axes[0].set_facecolor('lightgray')
    plt.colorbar(im1, ax=axes[0], label="Age (Years)")

    im2 = axes[1].pcolormesh(X_grid, Y_grid, unc_map, cmap='magma', shading='auto')
    axes[1].set_title("Prediction Uncertainty (1 Std Dev)")
    axes[1].set_aspect('equal')
    axes[1].set_facecolor('lightgray')
    plt.colorbar(im2, ax=axes[1], label="Uncertainty (Years)")
    
    plt.tight_layout()
    plt.show()

# --- 5. Main Execution ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Geomorphology-Informed Data-Driven GP.")
    parser.add_argument("--nc_path", type=Path, default=Path("data/modern_fields_native.nc"))
    parser.add_argument("--ages_path", type=Path, default=Path("data/age_data/combined_ages.csv"))
    parser.add_argument("--moraines_path", type=Path, default=Path("data/age_data/moraine_tangents.csv"))
    
    parser.add_argument("--mlp-features", nargs="*", default=["signed_distance_to_margin"])
    parser.add_argument("--gp-features", nargs="*", default=["bed_elevation"])
    
    parser.add_argument("--lambda-geom", type=float, default=1.0)
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
    
    # NEW ARGUMENTS FOR I/O
    parser.add_argument("--save-model", type=str, default=None, help="Path to save the trained PyTorch state_dict (.pth)")
    parser.add_argument("--load-model", type=str, default=None, help="Path to load a pre-trained state_dict (.pth) and skip training")
    parser.add_argument("--export-dir", type=str, default=None, help="Directory to save output GeoTIFFs (requires rioxarray)")

    args = parser.parse_args()
    mlp_feats = args.mlp_features if args.mlp_features else []
    gp_feats = args.gp_features if args.gp_features else []

    print(f"Loading datasets...")
    ds = compute_dynamic_features(xr.open_dataset(args.nc_path, decode_times=False), list(set(mlp_feats + gp_feats)))
    df = filter_age_data(pd.read_csv(args.ages_path), cosmogenic_only=bool(args.cosmogenic_only), min_quality=str(args.min_quality))
    
    train_x_mlp, train_x_gp, train_y, train_errs, valid_mask, feature_stats = prepare_training_tensors(
        ds, df, mlp_feats, gp_feats, args.min_age, args.max_age, args.ages_bp_ref_year, args.model_bp_ref_year
    )
    
    mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd = None, None, None, None
    if args.lambda_geom > 0 and args.moraines_path.exists():
        df_moraines = pd.read_csv(args.moraines_path)
        if args.moraine_frac < 1.0: df_moraines = df_moraines.sample(frac=args.moraine_frac, random_state=42)
        mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd = prepare_moraine_tensors(
            ds, df_moraines, mlp_feats, gp_feats, feature_stats, step_size_m=args.moraine_step
        )

    if args.checkerboard_cv and args.load_model is None:
        run_checkerboard_cv(
            train_x_mlp, train_x_gp, train_y, train_errs, feature_stats,
            mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd, args.lambda_geom,
            args.min_age, args.max_age, args.checkerboard_size, args.epochs
        )

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=train_errs**2, learn_additional_noise=True)
    model = DataDrivenGP(train_x_mlp, train_x_gp, train_y, likelihood)
    
    # Checkpoint Logic
    if args.load_model:
        print(f"\nLoading pre-trained model from {args.load_model}...")
        model.load_state_dict(torch.load(args.load_model))
    else:
        print("\nStarting Final GP + Mean Training...")
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
            
        if args.save_model:
            print(f"Saving trained model to {args.save_model}...")
            torch.save(model.state_dict(), args.save_model)

    print("\nGenerating final maps...")
    predict_and_export_grid(model, ds, args, feature_stats, args.min_age, args.max_age)

if __name__ == "__main__":
    main()