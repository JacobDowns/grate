import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import scipy.ndimage as ndimage
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm

try:
    import rioxarray
    HAS_RIOXARRAY = True
except ImportError:
    HAS_RIOXARRAY = False
    print("Warning: 'rioxarray' not found. GeoTIFF export will be disabled.")

# --- 1. Random Forest Uncertainty Wrapper ---

def predict_rf_with_uncertainty(model, X):
    """
    Extracts mean and variance from a trained scikit-learn RandomForestRegressor 
    by evaluating the predictions of all individual decision trees in the ensemble.
    """
    # predictions shape: (n_estimators, n_samples)
    preds = np.array([tree.predict(X) for tree in model.estimators_])
    return preds.mean(axis=0), preds.var(axis=0)

# --- 2. Data Loading & Feature Engineering ---

def filter_age_data(df: pd.DataFrame, cosmogenic_only: bool, min_quality: str, filter_min: float, filter_max: float) -> pd.DataFrame:
    out = df.copy()
    out = out[(out["age_mean"] >= filter_min) & (out["age_mean"] <= filter_max)].copy()
    
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
        slope = ndimage.gaussian_gradient_magnitude(bed, sigma=1.0)
        ds["bed_slope"] = (("y", "x"), slope.astype(np.float32))
    if "bed_roughness" in requested_features and "bed_roughness" not in ds:
        c1 = ndimage.uniform_filter(bed, size=3)
        c2 = ndimage.uniform_filter(bed * bed, size=3)
        roughness = np.sqrt(np.clip(c2 - c1 * c1, 0, None))
        ds["bed_roughness"] = (("y", "x"), roughness.astype(np.float32))
    return ds

def generate_pseudo_margin_points(ds: xr.Dataset, n_points: int, assumed_age: float = 0.0, assumed_sd: float = 25.0):
    print(f"  -> Generating {n_points} dynamic pseudo-observations on the modern margin...")
    if "thickness" in ds: ice_mask = ds["thickness"].values > 0.0
    elif "ice_mask" in ds: ice_mask = ds["ice_mask"].values == 1
    else: raise ValueError("Dataset must contain 'thickness' or 'ice_mask'.")

    labeled_array, num_features = ndimage.label(ice_mask)
    if num_features == 0: raise ValueError("No ice found.")
        
    sizes = ndimage.sum(ice_mask, labeled_array, range(1, num_features + 1))
    main_label = np.argmax(sizes) + 1
    main_ice = (labeled_array == main_label)

    eroded_ice = ndimage.binary_erosion(main_ice)
    margin_mask = main_ice ^ eroded_ice 
    
    y_idx, x_idx = np.where(margin_mask)
    x_coords, y_coords = ds["x"].values[x_idx], ds["y"].values[y_idx]
    
    if len(x_coords) > n_points:
        indices = np.random.choice(len(x_coords), size=n_points, replace=False)
        x_coords, y_coords = x_coords[indices], y_coords[indices]
        
    return pd.DataFrame({
        "x_3413": x_coords, "y_3413": y_coords,
        "age_mean": assumed_age, "age_sd": assumed_sd, 
        "obs_type": "pseudo", "quality": "High"
    })

def prepare_training_arrays(
    ds: xr.Dataset, df: pd.DataFrame, features: list,
    clip_min: float, clip_max: float, ages_bp_ref_year: float, model_bp_ref_year: float
):
    x_obs, y_obs = df["x_3413"].to_numpy(dtype=np.float32), df["y_3413"].to_numpy(dtype=np.float32)
    ages_raw = df["age_mean"].to_numpy(dtype=np.float32)
    errs_raw = df["age_sd"].to_numpy(dtype=np.float32)

    ages_shifted = ages_raw - np.float32(float(ages_bp_ref_year) - float(model_bp_ref_year))
    ages_norm = np.clip((ages_shifted - clip_min) / (clip_max - clip_min), 0, 1)
    errs_norm = errs_raw / (clip_max - clip_min)

    ds_sampled = ds.interp(x=xr.DataArray(x_obs, dims="points"), y=xr.DataArray(y_obs, dims="points"), method="linear")
    
    valid_mask = np.isfinite(ages_norm)
    
    extracted_features = {}
    for feat in features:
        feat_data = ds_sampled[feat].values.astype(np.float32)
        valid_mask &= np.isfinite(feat_data)
        extracted_features[feat] = feat_data
        
    coords_raw = np.column_stack((x_obs, y_obs))
    coords_valid = coords_raw[valid_mask]
    coords_mean, coords_std = coords_valid.mean(axis=0), coords_valid.std(axis=0)
    coords_scaled = (coords_valid - coords_mean) / coords_std
    
    feature_stats = {'coords': {'mean': coords_mean, 'std': coords_std}}
    scaled_features = {}
    for feat in features:
        feat_valid = extracted_features[feat][valid_mask]
        f_mean, f_std = float(np.nanmean(feat_valid)), float(np.nanstd(feat_valid))
        if f_std == 0: f_std = 1.0
        scaled_features[feat] = (feat_valid - f_mean) / f_std
        feature_stats[feat] = {'mean': f_mean, 'std': f_std}

    # Build the massive X array [x, y, feat1, feat2, ...]
    X_list = [coords_scaled]
    for feat in features:
        X_list.append(scaled_features[feat][:, None])
    
    X_train = np.hstack(X_list)
    y_train = ages_norm[valid_mask]
    err_train = errs_norm[valid_mask]
    
    return X_train, y_train, err_train, valid_mask, feature_stats

# --- 3. Unified Training & CV Plotting ---

def get_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    ss_res, ss_tot = np.sum((y_true - y_pred)**2), np.sum((y_true - np.mean(y_true))**2)
    cod = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return mse, cod

def run_checkerboard_cv(
    model, X_train, y_train, err_train, feature_stats, 
    clip_min, clip_max, size, out_dir, show_plot
):
    coords_mean, coords_std = feature_stats['coords']['mean'], feature_stats['coords']['std']
    x_coords = (X_train[:, 0] * coords_std[0]) + coords_mean[0]
    y_coords = (X_train[:, 1] * coords_std[1]) + coords_mean[1]

    ix = np.floor(x_coords / size).astype(int)
    iy = np.floor(y_coords / size).astype(int)
    mask_A = ((ix + iy) % 2) == 0
    mask_B = ~mask_A
    
    cv_x, cv_y, cv_true, cv_pred = [], [], [], []
    cv_true_err, cv_pred_err = [], []
    
    for tr_mask, te_mask, name in [(mask_A, mask_B, "Fold 1"), (mask_B, mask_A, "Fold 2")]:
        if np.sum(tr_mask) == 0 or np.sum(te_mask) == 0: continue
            
        print(f"\n--- Running CV: {name} (Train: {np.sum(tr_mask)}, Test: {np.sum(te_mask)}) ---")
        X_tr, y_tr = X_train[tr_mask], y_train[tr_mask]
        X_te, y_te, err_te = X_train[te_mask], y_train[te_mask], err_train[te_mask]
        
        # Fit the RF
        model.fit(X_tr, y_tr)
        
        # Predict with uncertainty
        pred_age_norm, pred_var_norm = predict_rf_with_uncertainty(model, X_te)
            
        pred_age_yrs = pred_age_norm * (clip_max - clip_min) + clip_min
        pred_std_yrs = np.sqrt(pred_var_norm) * (clip_max - clip_min)
        
        true_age_yrs = y_te * (clip_max - clip_min) + clip_min
        true_err_yrs = err_te * (clip_max - clip_min)
        
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
        cv_true, cv_pred, xerr=2 * cv_true_err, yerr=2 * cv_pred_err, 
        fmt='o', alpha=0.6, ecolor='silver', elinewidth=1, 
        markeredgecolor='k', markerfacecolor='forestgreen', markersize=5, zorder=2
    )
    min_val = min(cv_true.min(), cv_pred.min())
    max_val = max(cv_true.max(), cv_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, zorder=3)
    
    axes[0].set_title(f"RF Out-of-Sample CV Predictions\nTotal $R^2$: {cod_total:.3f} | Total RMSE: {np.sqrt(mse_total):.0f} yrs")
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
    out_path = out_dir / "rf_cv_spatial_errors.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    
    if show_plot: plt.show()
    else: plt.close(fig)

def run_lobo_cv(
    model, X_train, y_train, err_train, feature_stats, 
    clip_min, clip_max, size, out_dir, show_plot
):
    coords_mean, coords_std = feature_stats['coords']['mean'], feature_stats['coords']['std']
    x_coords = (X_train[:, 0] * coords_std[0]) + coords_mean[0]
    y_coords = (X_train[:, 1] * coords_std[1]) + coords_mean[1]

    ix = np.floor(x_coords / size).astype(int)
    iy = np.floor(y_coords / size).astype(int)
    
    blocks = np.column_stack((ix, iy))
    unique_blocks = np.unique(blocks, axis=0)
    
    print(f"\n--- Starting RF Leave-One-Block-Out (LOBO) CV ({len(unique_blocks)} unique blocks) ---")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rf_lobo_cv_results.csv"
    
    block_results = []
    lobo_true_all, lobo_pred_all = [], []
    lobo_true_err_all, lobo_pred_err_all = [], []
    
    for b_idx, b in enumerate(tqdm(unique_blocks, desc="LOBO Blocks")):
        te_mask = (ix == b[0]) & (iy == b[1])
        tr_mask = ~te_mask
        if np.sum(tr_mask) == 0: continue
            
        X_tr, y_tr = X_train[tr_mask], y_train[tr_mask]
        X_te, y_te, err_te = X_train[te_mask], y_train[te_mask], err_train[te_mask]
        
        # Train RF
        model.fit(X_tr, y_tr)
        
        pred_age_norm, pred_var_norm = predict_rf_with_uncertainty(model, X_te)
            
        pred_age_yrs = pred_age_norm * (clip_max - clip_min) + clip_min
        pred_std_yrs = np.sqrt(pred_var_norm) * (clip_max - clip_min)
        
        true_age_yrs = y_te * (clip_max - clip_min) + clip_min
        true_err_yrs = err_te * (clip_max - clip_min)
        
        lobo_true_all.extend(true_age_yrs)
        lobo_pred_all.extend(pred_age_yrs)
        lobo_true_err_all.extend(true_err_yrs)
        lobo_pred_err_all.extend(pred_std_yrs)
        
        mse, cod = get_metrics(true_age_yrs, pred_age_yrs)
        block_results.append({
            'block_ix': b[0], 'block_iy': b[1],
            'x_center_3413': (b[0] + 0.5) * size,
            'y_center_3413': (b[1] + 0.5) * size,
            'n_test_points': int(np.sum(te_mask)),
            'mse': mse, 'rmse': np.sqrt(mse), 'local_r_squared': cod
        })
        pd.DataFrame(block_results).to_csv(csv_path, index=False)
        
    lobo_true_arr, lobo_pred_arr = np.array(lobo_true_all), np.array(lobo_pred_all)
    lobo_true_err_arr, lobo_pred_err_arr = np.array(lobo_true_err_all), np.array(lobo_pred_err_all)
    
    global_mse, global_r2 = get_metrics(lobo_true_arr, lobo_pred_arr)
    global_rmse = np.sqrt(global_mse)
    
    print("\n" + "="*50)
    print("RF FINAL AGGREGATED LOBO METRICS:")
    print(f"Total Points Tested: {len(lobo_true_arr)}")
    print(f"Aggregated RMSE:     {global_rmse:.0f} years")
    print(f"Aggregated R^2:      {global_r2:.3f}")
    print("="*50 + "\n")
        
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].errorbar(
        lobo_true_arr, lobo_pred_arr, xerr=2 * lobo_true_err_arr, yerr=2 * lobo_pred_err_arr, 
        fmt='o', alpha=0.6, ecolor='silver', elinewidth=1,
        markeredgecolor='k', markerfacecolor='forestgreen', markersize=5, zorder=2
    )
    min_val = min(lobo_true_arr.min(), lobo_pred_arr.min())
    max_val = max(lobo_true_arr.max(), lobo_pred_arr.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, zorder=3)
    
    axes[0].set_title(f"RF Aggregated LOBO Predictions\nTotal $R^2$: {global_r2:.3f} | Total RMSE: {global_rmse:.0f} yrs")
    axes[0].set_xlabel("Observed Age (Years BP)")
    axes[0].set_ylabel("Predicted Age (Years BP)")
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    rmses = [r['rmse'] for r in block_results]
    norm_rmse = mcolors.Normalize(vmin=0, vmax=np.percentile(rmses, 95))
    cmap = plt.get_cmap('Reds')
    
    axes[1].set_aspect('equal')
    axes[1].set_title(f"RF LOBO Block Errors (RMSE)")
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
    out_path = out_dir / "rf_lobo_cv_blocks.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    
    if show_plot: plt.show()
    else: plt.close(fig)

# --- 4. Final Grid Prediction & Output Export ---

def predict_and_export_grid(model, ds, features, feature_stats, clip_min, clip_max, out_dir):
    xg, yg = ds["x"].values, ds["y"].values
    X_grid, Y_grid = np.meshgrid(xg, yg)
    X_flat, Y_grid_flat = X_grid.flatten(), Y_grid.flatten()

    valid_mask = np.ones_like(X_flat, dtype=bool)
    flat_features = {}
    
    for feat in features:
        feat_flat = ds[feat].values.flatten()
        valid_mask &= np.isfinite(feat_flat)
        flat_features[feat] = feat_flat

    if "thickness" in ds: ice_mask = ds["thickness"].values.flatten() > 0.0
    elif "ice_mask" in ds: ice_mask = ds["ice_mask"].values.flatten() == 1
    else: ice_mask = np.zeros_like(X_flat, dtype=bool)
    valid_mask &= ~ice_mask
        
    coords_scaled = (np.column_stack((X_flat[valid_mask], Y_grid_flat[valid_mask])) - feature_stats['coords']['mean']) / feature_stats['coords']['std']
    
    X_list = [coords_scaled]
    for feat in features:
        f_scaled = (flat_features[feat][valid_mask].astype(np.float32) - feature_stats[feat]['mean']) / feature_stats[feat]['std']
        X_list.append(f_scaled[:, None])
    
    X_test = np.hstack(X_list)
    
    print(f"\nPredicting on {X_test.shape[0]:,} valid grid pixels using Random Forest...")
    batch_size = 50000
    pred_means, pred_vars = [], []
    
    for i in tqdm(range(0, X_test.shape[0], batch_size), desc="Predicting Map"):
        mean_batch, var_batch = predict_rf_with_uncertainty(model, X_test[i : i + batch_size])
        pred_means.append(mean_batch)
        pred_vars.append(var_batch)
            
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
        da_age.rio.write_crs("EPSG:3413", inplace=True).rio.to_raster(out_dir / "rf_predicted_age.tif")
        da_unc.rio.write_crs("EPSG:3413", inplace=True).rio.to_raster(out_dir / "rf_predicted_uncertainty.tif")

    fig1, axes = plt.subplots(1, 2, figsize=(16, 8))
    im1 = axes[0].pcolormesh(X_grid, Y_grid, age_map, cmap="turbo_r", shading="auto", vmin=clip_min, vmax=clip_max)
    axes[0].set_title(f"Random Forest Final Model Prediction")
    axes[0].set_aspect('equal')
    axes[0].set_facecolor('lightgray')
    plt.colorbar(im1, ax=axes[0], label="Age (Years)")

    im2 = axes[1].pcolormesh(X_grid, Y_grid, unc_map, cmap='magma', shading='auto')
    axes[1].set_title("Random Forest Prediction Uncertainty (1 Std Dev)")
    axes[1].set_aspect('equal')
    axes[1].set_facecolor('lightgray')
    plt.colorbar(im2, ax=axes[1], label="Uncertainty (Years)")
    
    plt.tight_layout()
    fig1.savefig(out_dir / "rf_final_prediction_maps.png", dpi=300, bbox_inches='tight')
    
    plt.close(fig1)

# --- 5. Main Execution ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Random Forest Baseline Model.")
    parser.add_argument("--nc_path", type=Path, default=Path("data/modern_fields_native.nc"))
    parser.add_argument("--ages_path", type=Path, default=Path("data/age_data/combined_ages.csv"))
    
    # We combine both lists since RF doesn't care about mean vs covar split
    parser.add_argument("--features", nargs="*", default=["signed_distance_to_margin", "bed_elevation"])
    
    # Random Forest specific args
    parser.add_argument("--n-estimators", type=int, default=150, help="Number of trees in the forest.")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum depth of the trees.")
    
    parser.add_argument("--cosmogenic-only", action="store_true")
    parser.add_argument("--min-quality", type=str, default="Low", choices=["High", "Mid", "Low"])
    
    parser.add_argument("--add-pseudo-margin", action="store_true", help="Anchor the modern ice margin with age=0 pseudo-points.")
    parser.add_argument("--n-pseudo-margin", type=int, default=1000, help="Number of pseudo-points to place on the margin.")
    
    parser.add_argument("--filter-min-age", type=float, default=0.0, help="Drop data points younger than this threshold.")
    parser.add_argument("--filter-max-age", type=float, default=50000.0, help="Drop data points older than this threshold.")
    parser.add_argument("--clip-min-age", type=float, default=0.0, help="Minimum bound for [0,1] normalization.")
    parser.add_argument("--clip-max-age", type=float, default=20000.0, help="Maximum bound for [0,1] normalization.")
    
    parser.add_argument("--ages-bp-ref-year", type=float, default=1950.0)
    parser.add_argument("--model-bp-ref-year", type=float, default=1850.0)
    
    parser.add_argument("--checkerboard-cv", action="store_true", help="Run 2-Fold Checkerboard CV")
    parser.add_argument("--lobo-cv", action="store_true", help="Run exhaustive Leave-One-Block-Out CV")
    parser.add_argument("--checkerboard-size", type=float, default=120000.0)
    
    parser.add_argument("--out-dir", type=Path, default=Path("rf"), help="Directory for all plots and GeoTIFFs")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True, help="Show interactive plot windows")

    args = parser.parse_args()

    print(f"Loading datasets...")
    ds = compute_dynamic_features(xr.open_dataset(args.nc_path, decode_times=False), args.features)
    
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
    
    X_train, y_train, err_train, valid_mask, feature_stats = prepare_training_arrays(
        ds, df, args.features, args.clip_min_age, args.clip_max_age, args.ages_bp_ref_year, args.model_bp_ref_year
    )
    
    # Initialize the Random Forest Baseline
    # n_jobs=-1 uses all available CPU cores to train trees in parallel
    model = RandomForestRegressor(
        n_estimators=args.n_estimators, 
        max_depth=args.max_depth, 
        n_jobs=-1, 
        random_state=42
    )

    if args.checkerboard_cv:
        run_checkerboard_cv(
            model, X_train, y_train, err_train, feature_stats,
            args.clip_min_age, args.clip_max_age, args.checkerboard_size, 
            args.out_dir, args.show
        )

    if args.lobo_cv:
        run_lobo_cv(
            model, X_train, y_train, err_train, feature_stats,
            args.clip_min_age, args.clip_max_age, args.checkerboard_size, 
            args.out_dir, args.show
        )

    print(f"\nStarting Final Random Forest Training...")
    model.fit(X_train, y_train)

    print("\nGenerating final maps...")
    predict_and_export_grid(
        model, ds, args.features, feature_stats, 
        args.clip_min_age, args.clip_max_age, args.out_dir
    )

if __name__ == "__main__":
    main()