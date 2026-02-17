import xarray as xr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import argparse

def filter_age_data(df: pd.DataFrame, *, cosmogenic_only: bool, min_quality: str) -> pd.DataFrame:
    out = df.copy()
    if cosmogenic_only:
        out = out[out["age_type"].astype(str).str.lower() == "cosmogenic"].copy()
        return out

    if "quality" not in out.columns:
        return out

    quality_rank = {"high": 0, "mid": 1, "low": 2}
    q = str(min_quality).strip().lower()
    
    out["_quality_rank"] = out["quality"].astype(str).str.strip().str.lower().map(quality_rank)
    out = out[out["_quality_rank"].notna() & (out["_quality_rank"] <= quality_rank[q])].copy()
    out = out.drop(columns=["_quality_rank"])
    return out

def main():
    parser = argparse.ArgumentParser(description="Diagnostic: Linear Regression on PCA Modes")
    parser.add_argument("--pca_path", type=Path, default=Path("data/deglaciation_snapshot_pca.nc"), help="PCA output NetCDF")
    parser.add_argument("--ages_path", type=Path, default=Path("data/ryan_data/all_data.csv"), help="Age observations CSV")
    parser.add_argument("--num_pca_modes", type=int, default=10, help="Number of modes to test.")
    parser.add_argument("--cosmogenic-only", action="store_true")
    parser.add_argument("--min-quality", type=str, default="Low", choices=["High", "Mid", "Low"])
    args = parser.parse_args()

    # 1. Load Data
    ds = xr.open_dataset(args.pca_path, decode_times=False)
    min_age = ds.attrs.get('age_norm_min_years', 0.0)
    max_age = ds.attrs.get('age_norm_max_years', 1.0)

    df = pd.read_csv(args.ages_path)
    df = filter_age_data(df, cosmogenic_only=args.cosmogenic_only, min_quality=args.min_quality)
    
    x_col = "x_epsg3413" if "x_epsg3413" in df.columns else "x_3413"
    y_col = "y_epsg3413" if "y_epsg3413" in df.columns else "y_3413"
    age_col = "age_mean_cal_yr_bp" if "age_mean_cal_yr_bp" in df.columns else "age_mean"
    
    x_obs = df[x_col].to_numpy(dtype=np.float32)
    y_obs = df[y_col].to_numpy(dtype=np.float32)
    ages_raw = df[age_col].to_numpy(dtype=np.float32)

    # 2. Shift reference year (1950 -> 1850) and Normalize
    ages_bp_ref_year = 1950.0
    model_bp_ref_year = 1850.0
    y_true_years = ages_raw - np.float32(float(ages_bp_ref_year) - float(model_bp_ref_year))
    y_true_norm = (y_true_years - min_age) / (max_age - min_age)

    # 3. Interpolate Physics
    x_xr = xr.DataArray(x_obs, dims="points")
    y_xr = xr.DataArray(y_obs, dims="points")
    ds_sampled = ds.interp(x=x_xr, y=y_xr, method="linear")
    
    phys_mean_norm = ds_sampled["deglaciation_age_mean_norm_all"].values.astype(np.float32)
    pca_modes_norm = ds_sampled["pca_mode_norm_all"].values.astype(np.float32).T[:, :args.num_pca_modes]

    # 4. Filter NaNs
    valid_mask = ~np.isnan(phys_mean_norm) & ~np.isnan(pca_modes_norm).any(axis=1) & ~np.isnan(y_true_norm)
    
    y_true = y_true_norm[valid_mask]
    X_mean = phys_mean_norm[valid_mask]
    X_pca = pca_modes_norm[valid_mask]
    
    print(f"Total Valid Observations: {len(y_true)}")

    # ---------------------------------------------------------
    # TEST 1: Physics Mean Alone (Baseline)
    # ---------------------------------------------------------
    y_true_yrs = y_true * (max_age - min_age) + min_age
    X_mean_yrs = X_mean * (max_age - min_age) + min_age
    
    r2_mean = r2_score(y_true_yrs, X_mean_yrs)
    rmse_mean = np.sqrt(mean_squared_error(y_true_yrs, X_mean_yrs))
    
    print(f"\n=======================================================")
    print(f"  BASELINE: Physics Ensemble Mean Alone")
    print(f"=======================================================")
    print(f"  RMSE: {rmse_mean:,.0f} yr")
    print(f"  R^2:  {r2_mean:.3f}")

    # ---------------------------------------------------------
    # TEST 2: Physics Mean + PCA Modes (Linear Regression)
    # ---------------------------------------------------------
    # The GP tries to fit the residual: (Observed - Physics Mean)
    y_residual = y_true - X_mean
    
    # We fit a linear regression with an INTERCEPT. 
    # This acts exactly like setting `bias=True` in the GP!
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X_pca, y_residual)
    
    # Reconstruct: Predicted Residual + Physics Mean
    y_pred_residual = reg.predict(X_pca)
    y_pred_total_norm = y_pred_residual + X_mean
    
    y_pred_total_yrs = y_pred_total_norm * (max_age - min_age) + min_age
    
    r2_pca = r2_score(y_true_yrs, y_pred_total_yrs)
    rmse_pca = np.sqrt(mean_squared_error(y_true_yrs, y_pred_total_yrs))
    
    print(f"\n=======================================================")
    print(f"  OPTIMIZED: Mean + {args.num_pca_modes} PCA Modes + Global Shift")
    print(f"=======================================================")
    print(f"  RMSE: {rmse_pca:,.0f} yr")
    print(f"  R^2:  {r2_pca:.3f}")
    
    # Calculate the global shift in years
    global_shift_yrs = reg.intercept_ * (max_age - min_age)
    print(f"\n  -> The optimal global time-shift applied: {global_shift_yrs:,.0f} years")

if __name__ == "__main__":
    main()