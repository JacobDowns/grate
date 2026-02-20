import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

try:
    import skgstat as skg
except ImportError:
    print("Error: The 'scikit-gstat' library is required.")
    print("Please install it by running: pip install scikit-gstat")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Plot Empirical Variogram for Age Data")
    parser.add_argument("--ages_path", type=Path, default=Path("data/age_data/combined_ages.csv"))
    parser.add_argument("--max_lag", type=float, default=300000.0, help="Maximum distance to plot in meters (Default: 300km)")
    args = parser.parse_args()

    # 1. Load the observations
    print(f"Loading data from {args.ages_path}...")
    df = pd.read_csv(args.ages_path)
    
    # Optional: Filter out low quality if you want the purest spatial signal
    # df = df[df["quality"].astype(str).str.lower() != "low"]
    
    df = df.dropna(subset=["x_3413", "y_3413", "age_mean"])
    
    coords = df[["x_3413", "y_3413"]].values
    ages = df["age_mean"].values

    print(f"Building variogram for {len(ages)} points...")
    print(f"Calculating up to a maximum distance of {args.max_lag / 1000:.0f} km...")

    # 2. Build the Variogram
    # We use the Matern model to match your GPyTorch kernel
    V = skg.Variogram(
        coords, 
        ages, 
        maxlag=args.max_lag, 
        n_lags=30,           # Number of distance bins
        model='matern', 
        normalize=False
    )

    # 3. Extract the key metrics
    range_m = V.parameters[0]
    sill = V.parameters[1]
    nugget = V.parameters[2] if len(V.parameters) > 2 else 0.0
    
    range_km = range_m / 1000.0

    print("\n===========================================================")
    print("                 VARIOGRAM STATISTICS")
    print("===========================================================")
    print(f"  Autocorrelation Range: {range_km:.1f} km")
    print(f"  Sill (Total Variance): {sill:,.0f} yr^2")
    print(f"  Nugget (Noise/Error):  {nugget:,.0f} yr^2")
    print("===========================================================\n")

    if range_km > 64.0:
        print(f">>> GOOD NEWS: The spatial correlation range ({range_km:.1f} km) ")
        print(f">>> is larger than your 90th percentile void (64.0 km). ")
        print(f">>> The GP can safely interpolate across your gaps!")
    else:
        print(f">>> CAUTION: The spatial correlation range ({range_km:.1f} km) ")
        print(f">>> is smaller than some of your physical gaps. ")
        print(f">>> The GP will rely heavily on the Topography/Mean in those voids.")

    # 4. Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    V.plot(axes=ax, hist=False)
    
    ax.set_title("Empirical Variogram of Deglaciation Ages")
    ax.set_xlabel("Distance (meters)")
    ax.set_ylabel("Semi-Variance (Years^2)")
    
    # Add a vertical line for your Mean Void
    ax.axvline(33000, color='green', linestyle='--', label='Mean Gap (33km)')
    # Add a vertical line for your 90th Pctl Void
    ax.axvline(64000, color='orange', linestyle='--', label='90th Pctl Gap (64km)')
    
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()