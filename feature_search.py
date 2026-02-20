import xarray as xr
import numpy as np
import scipy.ndimage as ndimage
import torch
import gpytorch
from pathlib import Path
import argparse
import pandas as pd
import time
import itertools
import os

# --- 1. Linear Mean & GP Model Definition ---

class LinearMeanGP(gpytorch.models.ExactGP):
    def __init__(self, train_x_mean, train_x_gp, train_y, likelihood):
        super(LinearMeanGP, self).__init__((train_x_mean, train_x_gp), train_y, likelihood)
        
        self.mean_module = gpytorch.means.LinearMean(input_size=train_x_mean.shape[1], bias=True)
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=1.5, 
                ard_num_dims=train_x_gp.shape[1] 
            )
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

def prepare_master_tensors(ds, df, all_features, min_age, max_age, ages_bp_ref, model_bp_ref):
    x_obs = df["x_3413"].to_numpy(dtype=np.float32)
    y_obs = df["y_3413"].to_numpy(dtype=np.float32)
    ages_raw = df["age_mean"].to_numpy(dtype=np.float32)

    ages_norm = (ages_raw - (ages_bp_ref - model_bp_ref) - min_age) / (max_age - min_age)
    ages_norm = np.clip(ages_norm, 0, 1)

    x_xr, y_xr = xr.DataArray(x_obs, dims="points"), xr.DataArray(y_obs, dims="points")
    ds_sampled = ds.interp(x=x_xr, y=y_xr, method="linear")
    
    valid_mask = np.isfinite(ages_norm)
    feature_dict = {}
    
    for feat in all_features:
        feat_data = ds_sampled[feat].values.astype(np.float32)
        valid_mask &= np.isfinite(feat_data)
        feature_dict[feat] = feat_data
        
    coords_raw = np.column_stack((x_obs[valid_mask], y_obs[valid_mask]))
    coords_scaled = (coords_raw - coords_raw.mean(axis=0)) / coords_raw.std(axis=0)
    
    scaled_features = {"coords": torch.tensor(coords_scaled, dtype=torch.float32)}
    
    for feat in all_features:
        f_val = feature_dict[feat][valid_mask]
        f_std = np.nanstd(f_val) if np.nanstd(f_val) > 0 else 1.0
        f_scaled = (f_val - np.nanmean(f_val)) / f_std
        scaled_features[feat] = torch.tensor(f_scaled[:, None], dtype=torch.float32)

    train_y = torch.tensor(ages_norm[valid_mask], dtype=torch.float32)
    train_errs = torch.tensor(df["age_sd"].to_numpy(dtype=np.float32)[valid_mask] / (max_age - min_age), dtype=torch.float32)
    
    return scaled_features, train_y, train_errs, coords_raw

# --- 3. Evaluation Engine ---

def get_all_combinations(features):
    """Generates the power set (all possible combinations) of a list of features."""
    all_combos = []
    for r in range(len(features) + 1):
        combinations = itertools.combinations(features, r)
        all_combos.extend([list(c) for c in combinations])
    return all_combos

def build_tensor(feat_list, scaled_features):
    tensors = [scaled_features["coords"]]
    for f in feat_list:
        tensors.append(scaled_features[f])
    return torch.cat(tensors, dim=1)

def run_single_experiment(mean_feats, gp_feats, scaled_features, train_y, train_errs, coords_raw, size, min_age, max_age):
    tx_mean = build_tensor(mean_feats, scaled_features)
    tx_gp = build_tensor(gp_feats, scaled_features)
    
    ix = np.floor(coords_raw[:, 0] / size).astype(int)
    iy = np.floor(coords_raw[:, 1] / size).astype(int)
    mask_A = ((ix + iy) % 2) == 0
    mask_B = ~mask_A
    
    metrics = []
    
    for tr_mask, te_mask in [(mask_A, mask_B), (mask_B, mask_A)]:
        if np.sum(tr_mask) == 0 or np.sum(te_mask) == 0: continue
            
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=train_errs[tr_mask]**2, learn_additional_noise=True)
        model = LinearMeanGP(tx_mean[tr_mask], tx_gp[tr_mask], train_y[tr_mask], likelihood)
        model.covar_module.base_kernel.lengthscale = torch.tensor([[0.5] * tx_gp.shape[1]])
        
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        # Keep iterations around 800 to balance speed and convergence for grid searching
        with gpytorch.settings.max_cg_iterations(1000), gpytorch.settings.cholesky_jitter(1e-4):
            for _ in range(2500):
                optimizer.zero_grad()
                loss = -mll(model(tx_mean[tr_mask], tx_gp[tr_mask]), train_y[tr_mask])
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_norm = model(tx_mean[te_mask], tx_gp[te_mask]).mean.numpy()
            
        pred_yrs = pred_norm * (max_age - min_age) + min_age
        true_yrs = train_y[te_mask].numpy() * (max_age - min_age) + min_age
        
        mse = np.mean((true_yrs - pred_yrs)**2)
        ss_res = np.sum((true_yrs - pred_yrs)**2)
        ss_tot = np.sum((true_yrs - np.mean(true_yrs))**2)
        cod = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        metrics.append((np.sqrt(mse), cod))
        
    avg_rmse = np.mean([m[0] for m in metrics])
    avg_r2 = np.mean([m[1] for m in metrics])
    
    return avg_rmse, avg_r2

# --- 4. Main Execution ---

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_path", type=Path, default=Path("data/modern_fields_native.nc"))
    parser.add_argument("--ages_path", type=Path, default=Path("data/age_data/combined_ages.csv"))
    parser.add_argument("--output_csv", type=str, default="feature_grid_search_results.csv")
    args = parser.parse_args()

    # ---------------------------------------------------------
    # DEFINE THE FEATURES TO BRUTE-FORCE
    # (3 features = 64 models, 4 features = 256 models, 5 features = 1024 models)
    # ---------------------------------------------------------
    features_to_test = [
        "bed_elevation", 
        "signed_distance_to_margin", 
        "distance_to_coast",
        "bed_slope",      # Uncomment these to add to the combinatorial explosion
        "bed_roughness"
    ]

    print(f"Generating combinations for {len(features_to_test)} features...")
    all_combos = get_all_combinations(features_to_test)
    
    experiments = []
    for mean_feats in all_combos:
        for gp_feats in all_combos:
            experiments.append({
                "mean": mean_feats,
                "gp": gp_feats
            })
            
    total_models = len(experiments)
    print(f"Total Unique Models to Evaluate: {total_models}")

    print("Loading datasets and standardizing all features...")
    ds = compute_dynamic_features(xr.open_dataset(args.nc_path, decode_times=False), features_to_test)
    df = filter_age_data(pd.read_csv(args.ages_path), cosmogenic_only=False, min_quality="Low")
    
    min_age, max_age = 0.0, 15000.0
    scaled_features, train_y, train_errs, coords_raw = prepare_master_tensors(
        ds, df, features_to_test, min_age, max_age, 1950.0, 1850.0
    )

    print(f"\nStarting Exhaustive Search...")
    results = []
    
    # Check if the output CSV already exists, if so, delete it so we start fresh
    if os.path.exists(args.output_csv):
        os.remove(args.output_csv)
    
    start_time = time.time()
    
    for i, exp in enumerate(experiments):
        mean_str = " + ".join(exp["mean"]) if exp["mean"] else "None"
        gp_str = " + ".join(exp["gp"]) if exp["gp"] else "None"
        
        print(f"\n[ {i+1} / {total_models} ]")
        print(f"  Mean: {mean_str}")
        print(f"  GP:   {gp_str}")
        
        rmse, r2 = run_single_experiment(
            exp["mean"], exp["gp"], 
            scaled_features, train_y, train_errs, coords_raw, 200000.0, min_age, max_age
        )
        
        print(f"  Result -> RMSE: {rmse:.0f} | R^2: {r2:.3f}")
        
        # Save results iteratively
        results.append({
            "Mean_Features": mean_str,
            "GP_Features": gp_str,
            "RMSE": rmse,
            "R2": r2
        })
        
        # Dump to CSV every iteration so nothing is lost if interrupted
        pd.DataFrame(results).to_csv(args.output_csv, index=False)

    print(f"\nExhaustive Search Complete! (Took {(time.time() - start_time)/60:.1f} minutes)")
    print(f"Results saved to {args.output_csv}")
    
    # Print Top 10 Leaderboard
    leaderboard = pd.DataFrame(results).sort_values(by="R2", ascending=False).reset_index(drop=True)
    print("\n===========================================================")
    print("                  TOP 10 FEATURE SETS")
    print("===========================================================")
    print(leaderboard.head(10).to_string(index=False, formatters={'RMSE': '{:,.0f}'.format, 'R2': '{:.3f}'.format}))
    print("===========================================================\n")

if __name__ == "__main__":
    main()