import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.collections as mc
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import gpytorch
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm
import palettable.cubehelix as ch

try:
    import rioxarray
    HAS_RIOXARRAY = True
except ImportError:
    HAS_RIOXARRAY = False
    print("Warning: 'rioxarray' not found. GeoTIFF export will be disabled. (Run: pip install rioxarray)")

# --- 1. Neural Network & GP Model Definition ---

class MLPMean(nn.Module):
    def __init__(self, input_dim):
        super(MLPMean, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2),
            nn.Tanh(),
            nn.Linear(2, 2),
            nn.Tanh(),
             nn.Linear(2, 2),
            nn.Tanh(),
            nn.Linear(2, 1)
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)
    
class WarpingNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(WarpingNetwork, self).__init__()
        # A tiny network to stretch/squish the spatial features
        self.warper = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(), # Tanh is smooth and strictly bounded, preventing exploded spaces
            nn.Linear(16, latent_dim)
        )
        
    def forward(self, x):
        return self.warper(x)


class GatedPiecewiseMean(nn.Module):
    def __init__(self, input_dim, dist_idx=2):
        super(GatedPiecewiseMean, self).__init__()
        
        # 1. Base Continuous Retreat (The Poly2 structure)
        self.poly_dim = input_dim * 2 + (input_dim * (input_dim - 1)) // 2
        self.base_linear = nn.Linear(self.poly_dim, 1)
        
        # 2. The Temporal Jump Parameters
        # jump_magnitude: How many scaled years of history were erased?
        self.jump_magnitude = nn.Parameter(torch.tensor(0.0))
        # jump_threshold: At what scaled distance does the readvance moraine sit?
        self.jump_threshold = nn.Parameter(torch.tensor(0.0))
        
        # 3. The Spatial Gate (Determines WHERE the readvance happened)
        # Takes in only scaled x and y (indices 0 and 1)
        self.spatial_gate = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid() # Squashes output to exactly [0, 1]
        )
        
        # Which column in train_x_mlp is 'signed_distance_to_margin'?
        # By default, [x, y, dist, bed] means dist is index 2.
        self.dist_idx = dist_idx

    def forward(self, x):
        input_dim = x.shape[-1]
        
        # --- A. Calculate Base Polynomial ---
        terms = [x, x ** 2]
        cross_terms = []
        for i in range(input_dim):
            for j in range(i + 1, input_dim):
                cross_terms.append((x[..., i] * x[..., j]).unsqueeze(-1))
        if cross_terms:
            terms.append(torch.cat(cross_terms, dim=-1))
            
        poly_x = torch.cat(terms, dim=-1)
        base_pred = self.base_linear(poly_x).squeeze(-1)
        
        # --- B. Calculate the Discontinuity Jump ---
        dist = x[..., self.dist_idx]
        # Sharp sigmoid acts as a differentiable step function.
        # If dist > threshold, step approaches 1. If dist < threshold, step approaches 0.
        sharpness = 15.0 
        step = torch.sigmoid(sharpness * (dist - self.jump_threshold))
        
        # --- C. Calculate the Spatial Gate ---
        coords = x[..., 0:2]
        gate = self.spatial_gate(coords).squeeze(-1)
        
        # --- D. Final Prediction ---
        # Base retreat + (Jump Size * Is_Past_Distance_Threshold * Did_Readvance_Happen_Here)
        return base_pred + (self.jump_magnitude * step * gate)

class SecondOrderPolyMean(nn.Module):
    def __init__(self, input_dim):
        super(SecondOrderPolyMean, self).__init__()
        self.poly_dim = input_dim * 2 + (input_dim * (input_dim - 1)) // 2
        self.linear = nn.Linear(self.poly_dim, 1)

    def forward(self, x):
        input_dim = x.shape[-1]
        terms = [x, x ** 2]
        
        cross_terms = []
        for i in range(input_dim):
            for j in range(i + 1, input_dim):
                cross_terms.append((x[..., i] * x[..., j]).unsqueeze(-1))
                
        if cross_terms:
            terms.append(torch.cat(cross_terms, dim=-1))
            
        poly_x = torch.cat(terms, dim=-1)
        return self.linear(poly_x).squeeze(-1)

class DataDrivenGP(gpytorch.models.ExactGP):
    def __init__(self, train_x_mlp, train_x_gp, train_y, likelihood, mean_type="linear"):
        super(DataDrivenGP, self).__init__((train_x_mlp, train_x_gp), train_y, likelihood)
        
        if mean_type.lower() == "mlp":
            self.mean_module = MLPMean(input_dim=train_x_mlp.shape[1])
        elif mean_type.lower() == "poly2":
            self.mean_module = SecondOrderPolyMean(input_dim=train_x_mlp.shape[1])
        elif mean_type.lower() == "gated_piecewise":
            # Assuming signed_distance_to_margin is the first feature you pass in args.mlp_features, 
            # it will sit at index 2 (after scaled x and scaled y).
            self.mean_module = GatedPiecewiseMean(input_dim=train_x_mlp.shape[1], dist_idx=2)
        else:
            self.mean_module = gpytorch.means.LinearMean(input_size=train_x_mlp.shape[1], bias=True)
    
        #self.covar_module = gpytorch.kernels.ScaleKernel(
        #    gpytorch.kernels.RQKernel(ard_num_dims=train_x_gp.shape[1])
        #)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=train_x_gp.shape[1])
        )

    def forward(self, x_mlp, x_gp):
        mean_pred = self.mean_module(x_mlp)
        covar_pred = self.covar_module(x_gp)
        return gpytorch.distributions.MultivariateNormal(mean_pred, covar_pred)

# --- 2. Data Loading & Feature Engineering ---

def filter_age_data(df: pd.DataFrame, cosmogenic_only: bool, min_quality: str, filter_min: float, filter_max: float) -> pd.DataFrame:
    out = df.copy()
    
    # 1. Apply the Age Range Filter (Drops outliers completely)
    out = out[(out["age_mean"] >= filter_min) & (out["age_mean"] <= filter_max)].copy()
    
    # 2. Apply Cosmogenic Filter
    if cosmogenic_only:
        out = out[out["obs_type"].astype(str).str.lower() == "cosmogenic"].copy()
        
    # 3. Apply Quality Filter
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
    if "bed_curvature" in requested_features and "bed_curvature" not in ds:
        print("  -> Computing dynamic feature: bed_curvature")
        # We use a larger sigma (e.g., 2.0 or 3.0) to aggressively smooth 
        # out high-frequency DEM radar noise before taking the 2nd derivative!
        curvature = ndimage.gaussian_laplace(bed, sigma=2.0)
        ds["bed_curvature"] = (("y", "x"), curvature.astype(np.float32))

    return ds

def generate_pseudo_margin_points(ds: xr.Dataset, n_points: int, assumed_age: float = 0.0, assumed_sd: float = 100.0):
    """
    Isolates the main contiguous ice sheet, finds its perimeter, 
    and samples N random points to act as modern-day (age=0) anchors.
    """
    print(f"  -> Generating {n_points} dynamic pseudo-observations on the modern margin...")
    
    # 1. Get the binary ice mask
    if "thickness" in ds:
        ice_mask = ds["thickness"].values > 0.0
    elif "ice_mask" in ds:
        ice_mask = ds["ice_mask"].values == 1
    else:
        raise ValueError("Dataset must contain 'thickness' or 'ice_mask' to find the margin.")

    # 2. Isolate the main ice sheet (largest connected component)
    labeled_array, num_features = ndimage.label(ice_mask)
    if num_features == 0:
        raise ValueError("No ice found in the dataset.")
        
    sizes = ndimage.sum(ice_mask, labeled_array, range(1, num_features + 1))
    main_label = np.argmax(sizes) + 1
    main_ice = (labeled_array == main_label)

    # 3. Find the perimeter (pixels that are ice, but touch non-ice)
    eroded_ice = ndimage.binary_erosion(main_ice)
    margin_mask = main_ice ^ eroded_ice  # XOR gives the exact 1-pixel boundary
    
    # 4. Extract coordinates
    y_idx, x_idx = np.where(margin_mask)
    x_coords = ds["x"].values[x_idx]
    y_coords = ds["y"].values[y_idx]
    
    # 5. Randomly sample N points
    if len(x_coords) > n_points:
        indices = np.random.choice(len(x_coords), size=n_points, replace=False)
        x_coords = x_coords[indices]
        y_coords = y_coords[indices]
        
    # 6. Format as a DataFrame identical to the age CSV
    df_pseudo = pd.DataFrame({
        "x_3413": x_coords,
        "y_3413": y_coords,
        "age_mean": assumed_age,
        "age_sd": assumed_sd, 
        "obs_type": "pseudo",
        "quality": "High"  # Ensures it passes any downstream filters
    })
    
    print(f"     Added {len(df_pseudo)} margin anchors to the training set.")
    return df_pseudo

def prepare_training_tensors(
    ds: xr.Dataset, df: pd.DataFrame, mlp_features: list, gp_features: list,
    clip_min: float, clip_max: float, ages_bp_ref_year: float, model_bp_ref_year: float
):
    x_obs, y_obs = df["x_3413"].to_numpy(dtype=np.float32), df["y_3413"].to_numpy(dtype=np.float32)
    ages_raw = df["age_mean"].to_numpy(dtype=np.float32)
    errs_raw = df["age_sd"].to_numpy(dtype=np.float32)

    ages_shifted = ages_raw - np.float32(float(ages_bp_ref_year) - float(model_bp_ref_year))
    
    # Clip and Normalize based on the fixed neural network boundaries
    ages_norm = np.clip((ages_shifted - clip_min) / (clip_max - clip_min), 0, 1)
    errs_norm = errs_raw / (clip_max - clip_min)

    ds_sampled = ds.interp(x=xr.DataArray(x_obs, dims="points"), y=xr.DataArray(y_obs, dims="points"), method="linear")
    
    valid_mask = np.isfinite(ages_norm)
    all_features = list(set(mlp_features + gp_features))
    
    extracted_features = {}
    for feat in all_features:
        feat_data = ds_sampled[feat].values.astype(np.float32)
        valid_mask &= np.isfinite(feat_data)
        extracted_features[feat] = feat_data
        
    coords_raw = np.column_stack((x_obs, y_obs))
    coords_valid = coords_raw[valid_mask]
    coords_mean, coords_std = coords_valid.mean(axis=0), coords_valid.std(axis=0)
    coords_scaled = (coords_valid - coords_mean) / coords_std
    
    feature_stats = {'coords': {'mean': coords_mean, 'std': coords_std}}
    scaled_features = {}
    for feat in all_features:
        feat_valid = extracted_features[feat][valid_mask]
        f_mean, f_std = float(np.nanmean(feat_valid)), float(np.nanstd(feat_valid))
        if f_std == 0: f_std = 1.0
        scaled_features[feat] = (feat_valid - f_mean) / f_std
        feature_stats[feat] = {'mean': f_mean, 'std': f_std}

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
        ds_sampled = ds.interp(x=xr.DataArray(x_coords, dims="points"), y=xr.DataArray(y_coords, dims="points"), method="linear")
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
    clip_min, clip_max, size, iterations, out_dir, show_plot, mean_type
):
    coords_mean, coords_std = feature_stats['coords']['mean'], feature_stats['coords']['std']
    x_coords = (train_x_mlp[:, 0].numpy() * coords_std[0]) + coords_mean[0]
    y_coords = (train_x_mlp[:, 1].numpy() * coords_std[1]) + coords_mean[1]

    ix = np.floor(x_coords / size).astype(int)
    iy = np.floor(y_coords / size).astype(int)
    mask_A = ((ix + iy) % 2) == 0
    mask_B = ~mask_A
    
    cv_x, cv_y, cv_true, cv_pred = [], [], [], []
    cv_true_err, cv_pred_err = [], []
    
    for tr_mask, te_mask, name in [(mask_A, mask_B, "Fold 1"), (mask_B, mask_A, "Fold 2")]:
        if np.sum(tr_mask) == 0 or np.sum(te_mask) == 0: continue
            
        print(f"\n--- Running CV: {name} (Train: {np.sum(tr_mask)}, Test: {np.sum(te_mask)}) ---")
        tx_mlp, tx_gp, ty, te = train_x_mlp[tr_mask], train_x_gp[tr_mask], train_y[tr_mask], train_errs[tr_mask]
        
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=te**2, learn_additional_noise=True)
        model = DataDrivenGP(tx_mlp, tx_gp, ty, likelihood, mean_type=mean_type)
        model.covar_module.base_kernel.lengthscale = torch.tensor([[0.5] * tx_gp.shape[1]])
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        with gpytorch.settings.max_cg_iterations(2000), gpytorch.settings.cholesky_jitter(1e-4):
            train_pinn_loop(
                model, likelihood, optimizer, mll, iterations,
                tx_mlp, tx_gp, ty, mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd,
                lambda_geom, desc=f"CV {name}"
            )
                
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = model(train_x_mlp[te_mask], train_x_gp[te_mask])
            pred_age_norm = pred_dist.mean.numpy()
            pred_var_norm = pred_dist.variance.numpy()
            
        pred_age_yrs = pred_age_norm * (clip_max - clip_min) + clip_min
        pred_std_yrs = np.sqrt(pred_var_norm) * (clip_max - clip_min)
        
        true_age_yrs = train_y[te_mask].numpy() * (clip_max - clip_min) + clip_min
        true_err_yrs = train_errs[te_mask].numpy() * (clip_max - clip_min)
        
        cv_x.extend(x_coords[te_mask])
        cv_y.extend(y_coords[te_mask])
        cv_true.extend(true_age_yrs)
        cv_pred.extend(pred_age_yrs)
        cv_true_err.extend(true_err_yrs)
        cv_pred_err.extend(pred_std_yrs)
        
        mse, cod = get_metrics(true_age_yrs, pred_age_yrs)
        print(f"  Result -> RMSE: {np.sqrt(mse):.0f} yrs | R^2: {cod:.3f}")
        
    cv_x, cv_y = np.array(cv_x), np.array(cv_y)
    cv_true, cv_pred = np.array(cv_true), np.array(cv_pred)
    cv_true_err, cv_pred_err = np.array(cv_true_err), np.array(cv_pred_err)
    
    sq_errors = (cv_pred - cv_true)**2
    mse_total, cod_total = get_metrics(cv_true, cv_pred)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    axes[0].errorbar(
        cv_true, cv_pred, 
        xerr=2 * cv_true_err, 
        yerr=2 * cv_pred_err, 
        fmt='o', alpha=0.6, ecolor='silver', elinewidth=1, 
        markeredgecolor='k', markerfacecolor='royalblue', markersize=5, zorder=2
    )
    
    min_val = min(cv_true.min(), cv_pred.min())
    max_val = max(cv_true.max(), cv_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, zorder=3)
    
    axes[0].set_title(f"Out-of-Sample CV Predictions\nTotal $R^2$: {cod_total:.3f} | Total RMSE: {np.sqrt(mse_total):.0f} yrs")
    axes[0].set_xlabel("Observed Age (Years BP)")
    axes[0].set_ylabel("Predicted Age (Years BP)")
    axes[0].grid(True, linestyle=':', alpha=0.6)

    vmax_cap = np.percentile(sq_errors, 95)
    sc = axes[1].scatter(cv_x, cv_y, c=sq_errors, cmap='Reds', s=40, edgecolors='k', linewidth=0.5, vmin=0, vmax=vmax_cap)
    
    x_grids = np.arange(np.floor(cv_x.min()/size)*size, np.ceil(cv_x.max()/size)*size + size, size)
    y_grids = np.arange(np.floor(cv_y.min()/size)*size, np.ceil(cv_y.max()/size)*size + size, size)
    for xg in x_grids: axes[1].axvline(xg, color='gray', linestyle='--', alpha=0.5, zorder=0)
    for yg in y_grids: axes[1].axhline(yg, color='gray', linestyle='--', alpha=0.5, zorder=0)

    axes[1].set_title(f"Spatial Distribution of Squared Errors\n(Checkerboard Size: {size/1000:.0f} km)")
    axes[1].set_aspect('equal')
    axes[1].set_facecolor('whitesmoke')
    plt.colorbar(sc, ax=axes[1], label="Squared Error (Years^2)")

    plt.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cv_spatial_errors.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n[+] Saved high-res CV plot to: {out_path}")
    
    if show_plot: plt.show()
    else: plt.close(fig)

def run_lobo_cv(
    train_x_mlp, train_x_gp, train_y, train_errs, feature_stats, 
    mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd, lambda_geom,
    clip_min, clip_max, size, iterations, out_dir, show_plot, mean_type
):
    coords_mean, coords_std = feature_stats['coords']['mean'], feature_stats['coords']['std']
    x_coords = (train_x_mlp[:, 0].numpy() * coords_std[0]) + coords_mean[0]
    y_coords = (train_x_mlp[:, 1].numpy() * coords_std[1]) + coords_mean[1]

    ix = np.floor(x_coords / size).astype(int)
    iy = np.floor(y_coords / size).astype(int)
    
    blocks = np.column_stack((ix, iy))
    unique_blocks = np.unique(blocks, axis=0)
    
    print(f"\n--- Starting Leave-One-Block-Out (LOBO) CV ({len(unique_blocks)} unique blocks) ---")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "lobo_cv_results.csv"
    print(f"Live results will be continuously saved to: {csv_path}")
    
    block_results = []
    lobo_true_all, lobo_pred_all = [], []
    lobo_true_err_all, lobo_pred_err_all = [], []
    
    for b_idx, b in enumerate(unique_blocks):
        te_mask = (ix == b[0]) & (iy == b[1])
        tr_mask = ~te_mask
        
        if np.sum(tr_mask) == 0: continue
            
        print(f"\n[Block {b_idx+1}/{len(unique_blocks)}] Testing on {np.sum(te_mask)} points | Training on {np.sum(tr_mask)} points")
        
        tx_mlp, tx_gp, ty, te = train_x_mlp[tr_mask], train_x_gp[tr_mask], train_y[tr_mask], train_errs[tr_mask]
        
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=te**2, learn_additional_noise=True)
        model = DataDrivenGP(tx_mlp, tx_gp, ty, likelihood, mean_type=mean_type)
        model.covar_module.base_kernel.lengthscale = torch.tensor([[0.5] * tx_gp.shape[1]])
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        with gpytorch.settings.max_cg_iterations(2000), gpytorch.settings.cholesky_jitter(1e-4):
            train_pinn_loop(
                model, likelihood, optimizer, mll, iterations,
                tx_mlp, tx_gp, ty, mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd,
                lambda_geom, desc=f"LOBO {b_idx+1}/{len(unique_blocks)}"
            )
                
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = model(train_x_mlp[te_mask], train_x_gp[te_mask])
            pred_age_norm = pred_dist.mean.numpy()
            pred_var_norm = pred_dist.variance.numpy()
            
        pred_age_yrs = pred_age_norm * (clip_max - clip_min) + clip_min
        pred_std_yrs = np.sqrt(pred_var_norm) * (clip_max - clip_min)
        
        true_age_yrs = train_y[te_mask].numpy() * (clip_max - clip_min) + clip_min
        true_err_yrs = train_errs[te_mask].numpy() * (clip_max - clip_min)
        
        lobo_true_all.extend(true_age_yrs)
        lobo_pred_all.extend(pred_age_yrs)
        lobo_true_err_all.extend(true_err_yrs)
        lobo_pred_err_all.extend(pred_std_yrs)
        
        mse, cod = get_metrics(true_age_yrs, pred_age_yrs)
        rmse = np.sqrt(mse)
        
        block_results.append({
            'block_ix': b[0], 'block_iy': b[1],
            'x_center_3413': (b[0] + 0.5) * size,
            'y_center_3413': (b[1] + 0.5) * size,
            'n_test_points': int(np.sum(te_mask)),
            'mse': mse, 'rmse': rmse, 'local_r_squared': cod
        })
        pd.DataFrame(block_results).to_csv(csv_path, index=False)
        
    lobo_true_arr, lobo_pred_arr = np.array(lobo_true_all), np.array(lobo_pred_all)
    lobo_true_err_arr, lobo_pred_err_arr = np.array(lobo_true_err_all), np.array(lobo_pred_err_all)
    
    global_mse, global_r2 = get_metrics(lobo_true_arr, lobo_pred_arr)
    global_rmse = np.sqrt(global_mse)
    
    print("\n" + "="*50)
    print("FINAL AGGREGATED LOBO METRICS:")
    print(f"Total Points Tested: {len(lobo_true_arr)}")
    print(f"Aggregated RMSE:     {global_rmse:.0f} years")
    print(f"Aggregated R^2:      {global_r2:.3f}")
    print("="*50 + "\n")
        
    print("\nGenerating LOBO Spatial Maps...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    axes[0].errorbar(
        lobo_true_arr, lobo_pred_arr,
        xerr=2 * lobo_true_err_arr,  
        yerr=2 * lobo_pred_err_arr, 
        fmt='o', alpha=0.6, ecolor='silver', elinewidth=1,
        markeredgecolor='k', markerfacecolor='royalblue', markersize=5, zorder=2
    )
    min_val = min(lobo_true_arr.min(), lobo_pred_arr.min())
    max_val = max(lobo_true_arr.max(), lobo_pred_arr.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, zorder=3)
    
    axes[0].set_title(f"Aggregated LOBO Predictions\nTotal $R^2$: {global_r2:.3f} | Total RMSE: {global_rmse:.0f} yrs")
    axes[0].set_xlabel("Observed Age (Years BP)")
    axes[0].set_ylabel("Predicted Age (Years BP)")
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    rmses = [r['rmse'] for r in block_results]
    norm_rmse = mcolors.Normalize(vmin=0, vmax=np.percentile(rmses, 95))
    cmap = plt.get_cmap('Reds')
    
    axes[1].set_aspect('equal')
    axes[1].set_title(f"LOBO Block Errors (RMSE)")
    axes[1].set_facecolor('whitesmoke')
    axes[1].scatter(x_coords, y_coords, c='k', s=5, alpha=0.5, zorder=2)
    
    for res in block_results:
        rect = patches.Rectangle(
            (res['block_ix'] * size, res['block_iy'] * size),
            size, size, linewidth=1, edgecolor='gray',
            facecolor=cmap(norm_rmse(res['rmse'])), alpha=0.8, zorder=1
        )
        axes[1].add_patch(rect)
        
        text_color = 'white' if norm_rmse(res['rmse']) > 0.6 else 'black'
        axes[1].text(res['x_center_3413'], res['y_center_3413'], f"{res['rmse']:.0f}", 
                ha='center', va='center', fontsize=9, color=text_color, fontweight='bold', zorder=3)
                
    axes[1].set_xlim(x_coords.min() - size, x_coords.max() + size)
    axes[1].set_ylim(y_coords.min() - size, y_coords.max() + size)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_rmse)
    sm.set_array([])
    plt.colorbar(sm, ax=axes[1], label='RMSE (Years)')
        
    plt.tight_layout()
    out_path = out_dir / "lobo_cv_blocks.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[+] Saved high-res LOBO plot to: {out_path}")
    
    if show_plot: plt.show()
    else: plt.close(fig)

# --- 4. Final Grid Prediction & Output Export ---

def predict_and_export_grid(model, ds, args, feature_stats, clip_min, clip_max, out_dir, df_moraines=None, mean_type="linear"):
    model.eval()
    xg, yg = ds["x"].values, ds["y"].values
    X_grid, Y_grid = np.meshgrid(xg, yg)
    X_flat, Y_grid_flat = X_grid.flatten(), Y_grid.flatten()

    valid_mask = np.ones_like(X_flat, dtype=bool)
    all_features = list(set(args.mlp_features + args.gp_features))
    flat_features = {}
    
    for feat in all_features:
        feat_flat = ds[feat].values.flatten()
        valid_mask &= np.isfinite(feat_flat)
        flat_features[feat] = feat_flat

    if "thickness" in ds: ice_mask = ds["thickness"].values.flatten() > 0.0
    elif "ice_mask" in ds: ice_mask = ds["ice_mask"].values.flatten() == 1
    else: ice_mask = np.zeros_like(X_flat, dtype=bool)
    valid_mask &= ~ice_mask
        
    coords_scaled = (np.column_stack((X_flat[valid_mask], Y_grid_flat[valid_mask])) - feature_stats['coords']['mean']) / feature_stats['coords']['std']
    
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
            
    final_age_years = np.concatenate(pred_means) * (clip_max - clip_min) + clip_min
    uncertainty_years = np.sqrt(np.concatenate(pred_vars)) * (clip_max - clip_min)
    
    age_map = np.full_like(X_flat, np.nan, dtype=np.float32)
    age_map[valid_mask] = final_age_years
    age_map = age_map.reshape(X_grid.shape)
    
    unc_map = np.full_like(X_flat, np.nan, dtype=np.float32)
    unc_map[valid_mask] = uncertainty_years
    unc_map = unc_map.reshape(X_grid.shape)

    out_dir.mkdir(parents=True, exist_ok=True)
    if HAS_RIOXARRAY:
        print(f"\n[+] Exporting GeoTIFFs to {out_dir}/ ...")
        da_age = xr.DataArray(age_map, coords=[("y", yg), ("x", xg)], dims=["y", "x"], name="age_mean_yrs")
        da_unc = xr.DataArray(unc_map, coords=[("y", yg), ("x", xg)], dims=["y", "x"], name="age_std_yrs")
        da_age.rio.write_crs("EPSG:3413", inplace=True).rio.to_raster(out_dir / "predicted_age.tif")
        da_unc.rio.write_crs("EPSG:3413", inplace=True).rio.to_raster(out_dir / "predicted_uncertainty.tif")

    fig1, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    im1 = axes[0].pcolormesh(X_grid, Y_grid, age_map, cmap='seismic_r', shading="auto", vmin=clip_min, vmax=clip_max)
    axes[0].set_title(f"Final Model Prediction\n(Mean Architecture: {mean_type.upper()})")
    axes[0].set_aspect('equal')
    axes[0].set_facecolor('lightgray')
    plt.colorbar(im1, ax=axes[0], label="Age (Years)")

    im2 = axes[1].pcolormesh(X_grid, Y_grid, unc_map, cmap='magma', shading='auto')
    axes[1].set_title("Prediction Uncertainty (1 Std Dev)")
    axes[1].set_aspect('equal')
    axes[1].set_facecolor('lightgray')
    plt.colorbar(im2, ax=axes[1], label="Uncertainty (Years)")
    
    plt.tight_layout()
    fig1.savefig(out_dir / "final_prediction_maps.png", dpi=500, bbox_inches='tight')
    
    print("\nGenerating Isochrone Alignment Map...")
    fig2, ax_iso = plt.subplots(figsize=(10, 10))
    
    step = 1000
    levels = np.arange(0, clip_max + step, step)
    cmap_discrete = plt.get_cmap("turbo_r")
    norm = mcolors.BoundaryNorm(levels, ncolors=cmap_discrete.N, clip=True)
    
    im_iso = ax_iso.pcolormesh(X_grid, Y_grid, age_map, cmap=cmap_discrete, norm=norm, shading="auto")
    ax_iso.set_facecolor('lightgray')
    
    contours = ax_iso.contour(X_grid, Y_grid, age_map, levels=levels, colors='black', linewidths=1.2, alpha=0.7)
    ax_iso.clabel(contours, inline=True, fontsize=9, fmt='%1.0f')
    
    if df_moraines is not None:
        x = df_moraines["x_3413"].values
        y = df_moraines["y_3413"].values
        vx = df_moraines["vx"].values
        vy = df_moraines["vy"].values
        
        L = 2500.0 
        
        segments = [
            [(x[i] - L*vx[i], y[i] - L*vy[i]), (x[i] + L*vx[i], y[i] + L*vy[i])]
            for i in range(len(x))
        ]
        
        lc = mc.LineCollection(segments, colors='magenta', linewidths=1.5, alpha=0.9)
        ax_iso.add_collection(lc)
        ax_iso.scatter(x, y, s=2, c='black', zorder=4, alpha=0.8)
        
        ax_iso.plot([], [], color='magenta', linewidth=1.5, label="Moraine Tangents")
        ax_iso.legend(loc="upper right")
        
    ax_iso.set_aspect('equal')
    ax_iso.set_title("Model Isochrones & Moraine Tangent Alignment\n(2,000-Year Intervals)")
    
    cbar = plt.colorbar(im_iso, ax=ax_iso, fraction=0.046, pad=0.04, ticks=levels)
    cbar.set_label("Age (Years BP)")
    
    plt.tight_layout()
    out_path_iso = out_dir / "isochrone_alignment.png"
    fig2.savefig(out_path_iso, dpi=300, bbox_inches='tight')
    print(f"[+] Saved high-res isochrone map to: {out_path_iso}")

    if args.show: 
        plt.show()
    else: 
        plt.close(fig1)
        plt.close(fig2)

# --- 5. Main Execution ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Geomorphology-Informed Data-Driven GP.")
    parser.add_argument("--nc_path", type=Path, default=Path("data/modern_fields_native.nc"))
    parser.add_argument("--ages_path", type=Path, default=Path("data/age_data/combined_ages.csv"))
    parser.add_argument("--moraines_path", type=Path, default=Path("data/age_data/moraine_tangents.csv"))
    
    parser.add_argument("--mlp-features", nargs="*", default=["signed_distance_to_margin", "distance_to_coast"])
    parser.add_argument("--gp-features", nargs="*", default=["bed_elevation"])
    
    parser.add_argument("--mean-type", type=str, default="linear", choices=["linear", "mlp", "poly2", "gated_piecewise"])
    
    parser.add_argument("--lambda-geom", type=float, default=10.0)
    parser.add_argument("--moraine-step", type=float, default=500.0)
    parser.add_argument("--moraine-frac", type=float, default=1.0)
    
    parser.add_argument("--cosmogenic-only", action="store_true")
    parser.add_argument("--min-quality", type=str, default="Low", choices=["High", "Mid", "Low"])
    
    parser.add_argument("--add-pseudo-margin", action="store_true", help="Anchor the modern ice margin with age=0 pseudo-points.")
    parser.add_argument("--n-pseudo-margin", type=int, default=1000, help="Number of pseudo-points to place on the margin.")
    
    parser.add_argument("--filter-min-age", type=float, default=0.0, help="Drop data points younger than this threshold.")
    parser.add_argument("--filter-max-age", type=float, default=20000.0, help="Drop data points older than this threshold.")
    parser.add_argument("--clip-min-age", type=float, default=0.0, help="Minimum bound for PyTorch [0,1] normalization.")
    parser.add_argument("--clip-max-age", type=float, default=20000.0, help="Maximum bound for PyTorch [0,1] normalization.")
    
    parser.add_argument("--ages-bp-ref-year", type=float, default=1950.0)
    parser.add_argument("--model-bp-ref-year", type=float, default=1850.0)
    
    parser.add_argument("--checkerboard-cv", action="store_true", help="Run 2-Fold Checkerboard CV")
    parser.add_argument("--lobo-cv", action="store_true", help="Run exhaustive Leave-One-Block-Out CV")
    parser.add_argument("--checkerboard-size", type=float, default=120000.0)
    parser.add_argument("--epochs", type=int, default=2000)
    
    parser.add_argument("--out-dir", type=Path, default=Path("output"), help="Directory for all plots and GeoTIFFs")
    parser.add_argument("--save-model", type=str, default=None, help="Path to save the trained PyTorch state_dict (.pth)")
    parser.add_argument("--load-model", type=str, default=None, help="Path to load a pre-trained state_dict (.pth) and skip training")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True, help="Show interactive plot windows")

    args = parser.parse_args()
    mlp_feats = args.mlp_features if args.mlp_features else []
    gp_feats = args.gp_features if args.gp_features else []

    print(f"Loading datasets...")
    ds = compute_dynamic_features(xr.open_dataset(args.nc_path, decode_times=False), list(set(mlp_feats + gp_feats)))
    
    df = filter_age_data(
        pd.read_csv(args.ages_path), 
        cosmogenic_only=bool(args.cosmogenic_only), 
        min_quality=str(args.min_quality),
        filter_min=args.filter_min_age,
        filter_max=args.filter_max_age
    )
    
    if args.add_pseudo_margin:
        df_pseudo = generate_pseudo_margin_points(ds, n_points=args.n_pseudo_margin)
        df = pd.concat([df, df_pseudo], ignore_index=True)
    
    train_x_mlp, train_x_gp, train_y, train_errs, valid_mask, feature_stats = prepare_training_tensors(
        ds, df, mlp_feats, gp_feats, args.clip_min_age, args.clip_max_age, args.ages_bp_ref_year, args.model_bp_ref_year
    )
    
    df_moraines_full = None
    if args.moraines_path.exists():
        df_moraines_full = pd.read_csv(args.moraines_path)
        
    mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd = None, None, None, None
    if args.lambda_geom > 0 and df_moraines_full is not None:
        df_moraines_sampled = df_moraines_full
        if args.moraine_frac < 1.0: 
            df_moraines_sampled = df_moraines_full.sample(frac=args.moraine_frac, random_state=42)
            
        print(f"Processing Moraine Tensors (Geomorphology-Informed Mode)...")
        mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd = prepare_moraine_tensors(
            ds, df_moraines_sampled, mlp_feats, gp_feats, feature_stats, step_size_m=args.moraine_step
        )

    if args.load_model is None:
        if args.checkerboard_cv:
            run_checkerboard_cv(
                train_x_mlp, train_x_gp, train_y, train_errs, feature_stats,
                mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd, args.lambda_geom,
                args.clip_min_age, args.clip_max_age, args.checkerboard_size, args.epochs, 
                args.out_dir, args.show, args.mean_type
            )

        if args.lobo_cv:
            run_lobo_cv(
                train_x_mlp, train_x_gp, train_y, train_errs, feature_stats,
                mx_mlp_fwd, mx_gp_fwd, mx_mlp_bwd, mx_gp_bwd, args.lambda_geom,
                args.clip_min_age, args.clip_max_age, args.checkerboard_size, args.epochs, 
                args.out_dir, args.show, args.mean_type
            )

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=train_errs**2, learn_additional_noise=True)
    model = DataDrivenGP(train_x_mlp, train_x_gp, train_y, likelihood, mean_type=args.mean_type)
    
    if args.load_model:
        print(f"\nLoading pre-trained model from {args.load_model}...")
        model.load_state_dict(torch.load(args.load_model))
    else:
        print(f"\nStarting Final GP + {args.mean_type.upper()} Mean Training...")
        model.covar_module.base_kernel.lengthscale = torch.tensor([[0.5] * train_x_gp.shape[1]])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
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
    predict_and_export_grid(
        model, ds, args, feature_stats, 
        args.clip_min_age, args.clip_max_age, args.out_dir, 
        df_moraines=df_moraines_full,
        mean_type=args.mean_type
    )

if __name__ == "__main__":
    main()