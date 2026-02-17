import xarray as xr
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import argparse

def _pick_column(df: pd.DataFrame, candidates: list[str], *, label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find a {label} column. Tried {candidates}; available columns={list(df.columns)!r}"
    )

def filter_age_data(df: pd.DataFrame, *, cosmogenic_only: bool, min_quality: str) -> pd.DataFrame:
    out = df.copy()
    if cosmogenic_only:
        type_col = None
        for candidate in ("age_type", "obs_type"):
            if candidate in out.columns:
                type_col = candidate
                break
        if type_col is None:
            raise ValueError(
                "Requested --cosmogenic-only but input CSV has no 'age_type' or 'obs_type' column."
            )
        out = out[out[type_col].astype(str).str.lower() == "cosmogenic"].copy()
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

def main():
    parser = argparse.ArgumentParser(description="Diagnostic: Physics Mean vs Global Average")
    parser.add_argument("--pca_path", type=Path, default=Path("data/deglaciation_snapshot_pca.nc"))
    parser.add_argument("--ages_path", type=Path, default=Path("data/ryan_data/all_data.csv"))
    parser.add_argument("--cosmogenic-only", action="store_true")
    parser.add_argument("--min-quality", type=str, default="Low", choices=["High", "Mid", "Low"])
    parser.add_argument("--ages-bp-ref-year", type=float, default=1950.0)
    parser.add_argument("--model-bp-ref-year", type=float, default=1850.0)
    parser.add_argument("--phys-mean-var", type=str, default="deglaciation_age_mean_norm_all")
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading Physics from: {args.pca_path}")
    ds = xr.open_dataset(args.pca_path, decode_times=False)
    min_age = ds.attrs.get('age_norm_min_years', 0.0)
    max_age = ds.attrs.get('age_norm_max_years', 1.0)

    print(f"Loading Observations from: {args.ages_path}")
    df = pd.read_csv(args.ages_path)
    df = filter_age_data(df, cosmogenic_only=args.cosmogenic_only, min_quality=args.min_quality)
    
    # Handle column names flexibly (supports both "unified_ages_*.csv" and "age_data_*.csv" schemas)
    x_col = _pick_column(df, ["x_epsg3413", "x_3413", "x"], label="x coordinate")
    y_col = _pick_column(df, ["y_epsg3413", "y_3413", "y"], label="y coordinate")
    age_col = _pick_column(df, ["age_mean_cal_yr_bp", "age_mean", "ages"], label="age")
    
    x_obs = df[x_col].to_numpy(dtype=np.float32)
    y_obs = df[y_col].to_numpy(dtype=np.float32)
    ages_raw = df[age_col].to_numpy(dtype=np.float32)

    # 2. Shift reference year (default 1950 -> 1850) to match model
    y_true_years = ages_raw - np.float32(float(args.ages_bp_ref_year) - float(args.model_bp_ref_year))

    # 3. Interpolate Physics Mean to Observation Points
    x_xr = xr.DataArray(x_obs, dims="points")
    y_xr = xr.DataArray(y_obs, dims="points")
    ds_sampled = ds.interp(x=x_xr, y=y_xr, method="linear")
    
    # Extract normalized physics mean, and convert it back to YEARS
    if args.phys_mean_var not in ds_sampled:
        raise KeyError(
            f"--phys-mean-var {args.phys_mean_var!r} not found in dataset. "
            f"Available variables={list(ds_sampled.data_vars)!r}"
        )
    phys_mean_norm = ds_sampled[args.phys_mean_var].values.astype(np.float32)
    y_phys_years = phys_mean_norm * (max_age - min_age) + min_age

    # 4. Filter out NaNs (points outside the ice sheet model domain)
    valid_mask = ~np.isnan(y_phys_years) & ~np.isnan(y_true_years)
    
    y_true_valid = y_true_years[valid_mask]
    y_phys_valid = y_phys_years[valid_mask]
    
    print(f"\n--- Data Summary ---")
    print(f"Total Valid Observations: {len(y_true_valid)}")
    if len(y_true_valid) == 0:
        raise RuntimeError(
            "No valid observations after filtering/interpolation. "
            "Check coordinate columns, CRS/units, and whether points fall within the model grid."
        )
    print(f"Observed Age Range:       {y_true_valid.min():.0f} to {y_true_valid.max():.0f} years BP")
    
    # 5. Define Model 1: The "Dumb" Model (Global Average of Observations)
    global_mean_age = np.mean(y_true_valid)
    y_dumb_model = np.full_like(y_true_valid, global_mean_age)
    
    # 6. Calculate Metrics
    # Model 1 (Dumb Mean)
    mse_dumb = mean_squared_error(y_true_valid, y_dumb_model)
    rmse_dumb = np.sqrt(mse_dumb)
    r2_dumb = r2_score(y_true_valid, y_dumb_model)  # By definition, this is EXACTLY 0.0
    
    # Model 2 (Physics Mean)
    mse_phys = mean_squared_error(y_true_valid, y_phys_valid)
    rmse_phys = np.sqrt(mse_phys)
    r2_phys = r2_score(y_true_valid, y_phys_valid)
    
    print(f"\n=======================================================")
    print(f"  MODEL 1: Global Observation Average ({global_mean_age:.0f} years)")
    print(f"=======================================================")
    print(f"  MSE:  {mse_dumb:,.0f} yr^2")
    print(f"  RMSE: {rmse_dumb:,.0f} yr")
    print(f"  R^2:  {r2_dumb:.3f}   (Baseline)")

    print(f"\n=======================================================")
    print(f"  MODEL 2: Physics Simulation Mean")
    print(f"=======================================================")
    print(f"  MSE:  {mse_phys:,.0f} yr^2")
    print(f"  RMSE: {rmse_phys:,.0f} yr")
    print(f"  R^2:  {r2_phys:.3f}")
    print(f"=======================================================\n")
    
    if r2_phys < 0:
        print(">>> WARNING: The Physics Mean has a negative R^2.")
        print(">>> This means the physical simulations are systemically biased or ")
        print(">>> completely mismatched with the spatial pattern of the data.")
        print(">>> The model would literally be more accurate if it just guessed ")
        print(f">>> {global_mean_age:.0f} everywhere instead of using the simulations.")
    else:
        print(f">>> SUCCESS: The Physics Mean explains {r2_phys*100:.1f}% of the variance in the data.")

if __name__ == "__main__":
    main()
