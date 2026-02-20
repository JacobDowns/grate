import xarray as xr
import numpy as np
import scipy.ndimage as ndimage
import torch
import gpytorch
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 1. Linear Mean & GP Model Definition ---

class LinearMeanGP(gpytorch.models.ExactGP):
    def __init__(self, train_x_mean, train_x_gp, train_y, likelihood):
        super(LinearMeanGP, self).__init__((train_x_mean, train_x_gp), train_y, likelihood)
        self.mean_module = gpytorch.means.LinearMean(input_size=train_x_mean.shape[1], bias=True)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x_gp.shape[1])
        )

    def forward(self, x_mean, x_gp):
        mean_pred = self.mean_module(x_mean)
        covar_pred = self.covar_module(x_gp)
        return gpytorch.distributions.MultivariateNormal(mean_pred, covar_pred)

# --- 2. Data Preparation ---

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
        ds["bed_slope"] = (("y", "x"), ndimage.gaussian_gradient_magnitude(bed, sigma=1.0).astype(np.float32))
    if "bed_roughness" in requested_features and "bed_roughness" not in ds:
        c1 = ndimage.uniform_filter(bed, size=3)
        c2 = ndimage.uniform_filter(bed * bed, size=3)
        ds["bed_roughness"] = (("y", "x"), np.sqrt(np.clip(c2 - c1 * c1, 0, None)).astype(np.float32))
    return ds

def extract_and_scale_features(x_coords, y_coords, ds, all_features, feature_stats=None):
    x_xr = xr.DataArray(x_coords, dims="points")
    y_xr = xr.DataArray(y_coords, dims="points")
    ds_sampled = ds.interp(x=x_xr, y=y_xr, method="linear")
    
    valid_mask = np.ones(len(x_coords), dtype=bool)
    feature_dict = {}
    
    for feat in all_features:
        feat_data = ds_sampled[feat].values.astype(np.float32)
        valid_mask &= np.isfinite(feat_data)
        feature_dict[feat] = feat_data
        
    coords_raw = np.column_stack((x_coords, y_coords))
    
    if feature_stats is None:
        feature_stats = {}
        c_mean, c_std = coords_raw[valid_mask].mean(axis=0), coords_raw[valid_mask].std(axis=0)
        feature_stats['coords'] = {'mean': c_mean, 'std': c_std}
        for feat in all_features:
            f_val = feature_dict[feat][valid_mask]
            f_std = np.nanstd(f_val) if np.nanstd(f_val) > 0 else 1.0
            feature_stats[feat] = {'mean': np.nanmean(f_val), 'std': f_std}
            
    c_mean, c_std = feature_stats['coords']['mean'], feature_stats['coords']['std']
    scaled_features = {"coords": torch.tensor((coords_raw - c_mean) / c_std, dtype=torch.float32)}
    
    for feat in all_features:
        f_mean, f_std = feature_stats[feat]['mean'], feature_stats[feat]['std']
        f_scaled = (feature_dict[feat] - f_mean) / f_std
        scaled_features[feat] = torch.tensor(f_scaled[:, None], dtype=torch.float32)
        
    return scaled_features, valid_mask, feature_stats

def build_tensor(feat_list, scaled_features, mask):
    tensors = [scaled_features["coords"][mask]]
    for f in feat_list:
        tensors.append(scaled_features[f][mask])
    return torch.cat(tensors, dim=1)

def prepare_moraine_tensors(ds, moraine_df, all_features, feature_stats, step_size_m=500.0):
    x_m = moraine_df["x_3413"].values
    y_m = moraine_df["y_3413"].values
    vx = moraine_df["vx"].values
    vy = moraine_df["vy"].values
    
    x_fwd, y_fwd = x_m + (step_size_m * vx), y_m + (step_size_m * vy)
    x_bwd, y_bwd = x_m - (step_size_m * vx), y_m - (step_size_m * vy)
    
    fwd_scaled, fwd_mask, _ = extract_and_scale_features(x_fwd, y_fwd, ds, all_features, feature_stats)
    bwd_scaled, bwd_mask, _ = extract_and_scale_features(x_bwd, y_bwd, ds, all_features, feature_stats)
    
    valid_pair_mask = fwd_mask & bwd_mask
    return fwd_scaled, bwd_scaled, valid_pair_mask

# --- 3. Training & Plotting Engines ---

def train_model(
    model, likelihood, optimizer, mll, epochs,
    tx_mean, tx_gp, ty,
    mx_mean_fwd, mx_gp_fwd, mx_mean_bwd, mx_gp_bwd,
    lambda_geom, desc="Training"
):
    model.train()
    likelihood.train()
    
    pbar = tqdm(range(epochs), desc=desc)
    for i in pbar:
        optimizer.zero_grad()
        
        output = model(tx_mean, tx_gp)
        loss_mll = -mll(output, ty)
        
        loss_geom = torch.tensor(0.0)
        if lambda_geom > 0 and mx_mean_fwd is not None:
            model.eval()
            with gpytorch.settings.fast_pred_var(False):
                pred_fwd = model(mx_mean_fwd, mx_gp_fwd).mean
                pred_bwd = model(mx_mean_bwd, mx_gp_bwd).mean
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

def plot_cv_results(true_ages, pred_ages, avg_r2, avg_rmse):
    """Plots a 1:1 scatter plot of the out-of-sample Cross Validation results."""
    plt.figure(figsize=(8, 8))
    plt.scatter(true_ages, pred_ages, alpha=0.6, edgecolors='k', color='royalblue')
    
    # 1:1 Line
    min_val = min(min(true_ages), min(pred_ages))
    max_val = max(max(true_ages), max(pred_ages))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="1:1 Perfect Prediction")
    
    plt.title(f"Cross-Validation Results\n$R^2$: {avg_r2:.3f} | RMSE: {avg_rmse:.0f} yrs")
    plt.xlabel("Observed Age (Years BP)")
    plt.ylabel("Predicted Age (Years BP)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

def predict_and_plot_map(model, likelihood, ds, args, feature_stats, min_age, max_age):
    """Generates a full spatial prediction for deglaciated Greenland and plots Age + Uncertainty."""
    print("\nPreparing full grid for final spatial prediction...")
    
    xg = ds["x"].values
    yg = ds["y"].values
    X_grid, Y_grid = np.meshgrid(xg, yg)
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()

    # Mask out ONLY the modern ice sheet. 
    # We want to predict on the continental shelf and deep fjords!
    if "thickness" in ds:
        ice_mask = ds["thickness"].values.flatten() > 0.0 
    elif "ice_mask" in ds:
        ice_mask = ds["ice_mask"].values.flatten() == 1
    else:
        ice_mask = np.zeros_like(X_flat, dtype=bool)
        
    valid_mask = ~ice_mask
    
    X_valid = X_flat[valid_mask]
    Y_valid = Y_flat[valid_mask]
    
    all_feats = list(set(args.mlp_features + args.gp_features))
    scaled_grid, grid_mask_valid, _ = extract_and_scale_features(X_valid, Y_valid, ds, all_feats, feature_stats)
    
    tx_mean_full = build_tensor(args.mlp_features, scaled_grid, grid_mask_valid)
    tx_gp_full = build_tensor(args.gp_features, scaled_grid, grid_mask_valid)

    print(f"Predicting ages for {len(tx_mean_full):,} valid grid pixels (including marine regions)...")
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Push through likelihood to get total predictive variance
        predictions = likelihood(model(tx_mean_full, tx_gp_full))
        pred_mean_norm = predictions.mean.numpy()
        pred_std_norm = predictions.stddev.numpy()

    # Denormalize
    pred_mean_yrs = pred_mean_norm * (max_age - min_age) + min_age
    pred_std_yrs = pred_std_norm * (max_age - min_age)
    
    # Reconstruct 2D maps
    map_mean = np.full_like(X_flat, np.nan, dtype=np.float32)
    map_std = np.full_like(X_flat, np.nan, dtype=np.float32)
    
    # Apply the secondary mask from extraction
    final_valid_indices = np.where(valid_mask)[0][grid_mask_valid]
    map_mean[final_valid_indices] = pred_mean_yrs
    map_std[final_valid_indices] = pred_std_yrs
    
    map_mean = map_mean.reshape(X_grid.shape)
    map_std = map_std.reshape(X_grid.shape)
    
    # --- Plotting ---
    print("Generating maps...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Mean Age
    im1 = axes[0].pcolormesh(X_grid, Y_grid, map_mean, cmap='turbo_r', shading='auto', vmin=0, vmax=15000)
    axes[0].set_aspect('equal')
    axes[0].set_title("Predicted Deglaciation Age (Years BP)")
    axes[0].set_facecolor('lightgray') # Ice mask remains gray
    plt.colorbar(im1, ax=axes[0], label="Age (yrs)", fraction=0.046, pad=0.04)
    
    # Plot 2: Uncertainty (Standard Deviation)
    im2 = axes[1].pcolormesh(X_grid, Y_grid, map_std, cmap='magma', shading='auto')
    axes[1].set_aspect('equal')
    axes[1].set_title("Prediction Uncertainty (1σ Years)")
    axes[1].set_facecolor('lightgray')
    plt.colorbar(im2, ax=axes[1], label="Uncertainty (yrs)", fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# --- 4. Main Execution ---

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_path", type=Path, default=Path("data/modern_fields_native.nc"))
    parser.add_argument("--ages_path", type=Path, default=Path("data/age_data/combined_ages.csv"))
    parser.add_argument("--moraines_path", type=Path, default=Path("data/age_data/moraine_tangents.csv"))
    
    parser.add_argument("--mlp-features", nargs="*", default=["bed_elevation", "signed_distance_to_margin"])
    parser.add_argument("--gp-features", nargs="*", default=["bed_elevation", "bed_slope"])
    
    parser.add_argument("--lambda-geom", type=float, default=5.0)
    parser.add_argument("--moraine-step", type=float, default=500.0)
    parser.add_argument("--moraine-frac", type=float, default=1.0)
    
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--checkerboard-size", type=float, default=66000.0)
    args = parser.parse_args()

    all_feats = list(set(args.mlp_features + args.gp_features))
    
    print("Loading datasets...")
    ds = compute_dynamic_features(xr.open_dataset(args.nc_path, decode_times=False), all_feats)
    df_ages = filter_age_data(pd.read_csv(args.ages_path), cosmogenic_only=False, min_quality="Low")
    
    min_age, max_age = 0.0, 15000.0
    ages_norm = np.clip((df_ages["age_mean"].values - 100.0 - min_age) / (max_age - min_age), 0, 1)

    print("Processing Radiocarbon Tensors...")
    scaled_radiocarbon, radio_mask, feature_stats = extract_and_scale_features(
        df_ages["x_3413"].values, df_ages["y_3413"].values, ds, all_feats
    )
    
    train_x_mean = build_tensor(args.mlp_features, scaled_radiocarbon, radio_mask)
    train_x_gp = build_tensor(args.gp_features, scaled_radiocarbon, radio_mask)
    train_y = torch.tensor(ages_norm[radio_mask], dtype=torch.float32)
    train_errs = torch.tensor(df_ages["age_sd"].values[radio_mask] / (max_age - min_age), dtype=torch.float32)
    coords_raw = np.column_stack((df_ages["x_3413"].values[radio_mask], df_ages["y_3413"].values[radio_mask]))

    mx_mean_fwd, mx_gp_fwd, mx_mean_bwd, mx_gp_bwd = None, None, None, None
    if args.lambda_geom > 0 and args.moraines_path.exists():
        print("Processing Moraine Tensors (Geomorphology-Informed Mode)...")
        df_moraines = pd.read_csv(args.moraines_path)
        
        if args.moraine_frac < 1.0:
            df_moraines = df_moraines.sample(frac=args.moraine_frac, random_state=42)
            
        fwd_scaled, bwd_scaled, moraine_mask = prepare_moraine_tensors(
            ds, df_moraines, all_feats, feature_stats, step_size_m=args.moraine_step
        )
        
        mx_mean_fwd = build_tensor(args.mlp_features, fwd_scaled, moraine_mask)
        mx_gp_fwd = build_tensor(args.gp_features, fwd_scaled, moraine_mask)
        mx_mean_bwd = build_tensor(args.mlp_features, bwd_scaled, moraine_mask)
        mx_gp_bwd = build_tensor(args.gp_features, bwd_scaled, moraine_mask)

    print(f"\n--- Starting Spatial Cross-Validation (Checkerboard: {args.checkerboard_size/1000:.0f}km) ---")
    ix = np.floor(coords_raw[:, 0] / args.checkerboard_size).astype(int)
    iy = np.floor(coords_raw[:, 1] / args.checkerboard_size).astype(int)
    mask_A = ((ix + iy) % 2) == 0
    mask_B = ~mask_A
    
    metrics = []
    cv_true_all = []
    cv_pred_all = []
    fold = 1
    
    for tr_mask, te_mask in [(mask_A, mask_B), (mask_B, mask_A)]:
        if np.sum(tr_mask) == 0 or np.sum(te_mask) == 0: continue
            
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=train_errs[tr_mask]**2, learn_additional_noise=True)
        model = LinearMeanGP(train_x_mean[tr_mask], train_x_gp[tr_mask], train_y[tr_mask], likelihood)
        model.covar_module.base_kernel.lengthscale = torch.tensor([[0.5] * train_x_gp.shape[1]])
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        with gpytorch.settings.cholesky_jitter(1e-4):
            train_model(
                model, likelihood, optimizer, mll, args.epochs,
                train_x_mean[tr_mask], train_x_gp[tr_mask], train_y[tr_mask],
                mx_mean_fwd, mx_gp_fwd, mx_mean_bwd, mx_gp_bwd,
                args.lambda_geom, desc=f"CV Fold {fold}/2"
            )
                
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_norm = model(train_x_mean[te_mask], train_x_gp[te_mask]).mean.numpy()
            
        pred_yrs = pred_norm * (max_age - min_age) + min_age
        true_yrs = train_y[te_mask].numpy() * (max_age - min_age) + min_age
        
        cv_true_all.extend(true_yrs)
        cv_pred_all.extend(pred_yrs)
        
        mse = np.mean((true_yrs - pred_yrs)**2)
        ss_res = np.sum((true_yrs - pred_yrs)**2)
        ss_tot = np.sum((true_yrs - np.mean(true_yrs))**2)
        cod = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        metrics.append((np.sqrt(mse), cod))
        fold += 1
        
    avg_rmse = np.mean([m[0] for m in metrics])
    avg_r2 = np.mean([m[1] for m in metrics])
    
    print("\nPlotting CV Scatter...")
    plot_cv_results(cv_true_all, cv_pred_all, avg_r2, avg_rmse)

    print("\n--- Final Training Phase (100% of Data) ---")
    likelihood_final = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=train_errs**2, learn_additional_noise=True)
    model_final = LinearMeanGP(train_x_mean, train_x_gp, train_y, likelihood_final)
    model_final.covar_module.base_kernel.lengthscale = torch.tensor([[0.5] * train_x_gp.shape[1]])
    
    optimizer_final = torch.optim.Adam(model_final.parameters(), lr=0.05)
    mll_final = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_final, model_final)
    
    with gpytorch.settings.cholesky_jitter(1e-4):
        train_model(
            model_final, likelihood_final, optimizer_final, mll_final, args.epochs,
            train_x_mean, train_x_gp, train_y,
            mx_mean_fwd, mx_gp_fwd, mx_mean_bwd, mx_gp_bwd,
            args.lambda_geom, desc="Final Model"
        )
        
    # Generate and plot the final map
    predict_and_plot_map(model_final, likelihood_final, ds, args, feature_stats, min_age, max_age)

if __name__ == "__main__":
    main()