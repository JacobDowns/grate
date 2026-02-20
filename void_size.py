import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Calculate spatial voids/gaps in age data on deglaciated land.")
    parser.add_argument("--nc_path", type=Path, default=Path("data/modern_fields_native.nc"))
    parser.add_argument("--ages_path", type=Path, default=Path("data/age_data/combined_ages.csv"))
    args = parser.parse_args()

    print("Loading datasets...")
    ds = xr.open_dataset(args.nc_path, decode_times=False)
    
    df = pd.read_csv(args.ages_path)
    df = df.dropna(subset=["x_3413", "y_3413"])
    obs_coords = np.column_stack((df["x_3413"].values, df["y_3413"].values))

    print(f"Loaded {len(obs_coords)} valid observation points.")

    xg = ds["x"].values
    yg = ds["y"].values
    X_grid, Y_grid = np.meshgrid(xg, yg)
    
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()

    # 1. Masking Logic: Isolate Deglaciated Land
    # Check for modern ice
    if "thickness" in ds:
        ice_mask = ds["thickness"].values.flatten() > 0.0
    elif "ice_mask" in ds:
        ice_mask = ds["ice_mask"].values.flatten() == 1
    else:
        ice_mask = np.zeros_like(X_flat, dtype=bool)
        
    # Check for ocean (ice-free and below sea level)
    bed_flat = np.nan_to_num(ds["bed_elevation"].values.flatten(), nan=-9999.0)
    
    # Valid pixels: NOT ice AND ON land
    valid_mask = (~ice_mask) & (bed_flat >= 0.0)
    
    grid_coords_valid = np.column_stack((X_flat[valid_mask], Y_flat[valid_mask]))
    
    print(f"Calculating distances for {len(grid_coords_valid):,} deglaciated land pixels...")

    # 2. Build KD-Tree and Query Nearest Neighbors
    tree = cKDTree(obs_coords)
    distances_meters, _ = tree.query(grid_coords_valid, k=1)
    distances_km = distances_meters / 1000.0

    # 3. Calculate Gap Statistics
    mean_gap = np.mean(distances_km)
    median_gap = np.median(distances_km)
    p90_gap = np.percentile(distances_km, 90)
    max_gap = np.max(distances_km)

    print("\n===========================================================")
    print("           SPATIAL VOID STATISTICS (LAND ONLY)")
    print("===========================================================")
    print(f"  Average Void (Mean):         {mean_gap:.1f} km")
    print(f"  Typical Void (Median):       {median_gap:.1f} km")
    print(f"  Extreme Void (90th Pctl):    {p90_gap:.1f} km")
    print(f"  Maximum Void (Worst Case):   {max_gap:.1f} km")
    print("===========================================================\n")

    # 4. Map the Distances Back to the Grid
    dist_map = np.full_like(X_flat, np.nan, dtype=np.float32)
    dist_map[valid_mask] = distances_km
    dist_map = dist_map.reshape(X_grid.shape)

    # 5. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Map Plot
    im = axes[0].pcolormesh(X_grid, Y_grid, dist_map, cmap='magma_r', shading='auto', vmin=0, vmax=p90_gap)
    axes[0].scatter(obs_coords[:, 0], obs_coords[:, 1], c='cyan', s=5, edgecolors='black', linewidths=0.5, label='Observations')
    axes[0].set_aspect('equal')
    axes[0].set_title("Distance to Nearest Observation (km)")
    
    # Make the background (ocean/ice) a distinct color so you can verify the mask worked
    axes[0].set_facecolor('lightgray') 
    
    axes[0].legend(loc='upper right')
    plt.colorbar(im, ax=axes[0], label="Distance (km)")

    # Histogram Plot
    axes[1].hist(distances_km, bins=50, color='royalblue', edgecolor='black')
    axes[1].axvline(mean_gap, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_gap:.0f} km')
    axes[1].axvline(p90_gap, color='orange', linestyle='dashed', linewidth=2, label=f'90th Pctl: {p90_gap:.0f} km')
    axes[1].set_title("Distribution of Void Sizes (Land Only)")
    axes[1].set_xlabel("Distance to Nearest Observation (km)")
    axes[1].set_ylabel("Number of Grid Pixels")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()