import rioxarray
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# --- Configuration Parameters ---
# 1. Downsampling (The Low-Pass Filter)
#    If input is ~150m (BedMachine), factor=33 -> ~5km resolution.
#    If input is ~1000m, factor=5 -> ~5km resolution.
DOWNSAMPLE_FACTOR = 25  

# 2. Margin Definition (on the Coarse Grid)
#    Since averaging blurs the edge, we accept slightly thicker ice as "margin".
#    0 < h < 50m is a good range for 5km pixels.
THICKNESS_MIN = 20.0    
THICKNESS_MAX = 50.0   

# 3. Sampling Budget
NUM_SAMPLES = 1000      # Total number of pseudo-observations to generate
MIN_COARSE_PIXELS = 10  # Ignore islands smaller than 5 coarse pixels (noise)

def main():
    # --- STEP 1: LOAD & DOWNSAMPLE ---
    print(f"Loading thickness data...")
    # masked=True converts -9999 nodata to NaN automatically
    data = rioxarray.open_rasterio('data/qgreenland/bedmap_thickness.tif', masked=True)
    thickness_high_res = data.squeeze()
    
    print(f"Original Resolution: {thickness_high_res.rio.resolution()}")
    print(f"Downsampling by factor of {DOWNSAMPLE_FACTOR}...")
    
    # Coarsen using mean (preserves volume, smooths edges)
    # boundary='trim' drops any partial blocks at the edges
    thickness_coarse = thickness_high_res.coarsen(
        x=DOWNSAMPLE_FACTOR, 
        y=DOWNSAMPLE_FACTOR, 
        boundary='trim'
    ).mean()
    
    print(f"Coarse Shape: {thickness_coarse.shape}")

    # --- STEP 2: IDENTIFY ICE COMPONENTS ---
    # Define "Ice" as anything with thickness > 0
    is_ice = thickness_coarse > 0.0
    
    # Label connected components on the COARSE grid
    labeled_array, num_features = ndimage.label(is_ice)
    
    # Calculate area of each component (in coarse pixels)
    component_areas = np.bincount(labeled_array.ravel())
    
    # Total area of valid ice (ignoring background 0)
    total_ice_area = np.sum(component_areas[1:])
    
    print(f"Found {num_features} disjoint ice features on coarse grid.")
    
    # --- STEP 3: STRATIFIED SAMPLING ---
    print(f"Sampling {NUM_SAMPLES} points weighted by component area...")
    
    final_x = []
    final_y = []
    rng = np.random.default_rng(42)
    
    # Identify potential margin pixels globally on the coarse grid
    # Must be > 0 (ice) and < MAX (margin zone)
    global_margin_mask = (thickness_coarse > THICKNESS_MIN) & \
                         (thickness_coarse < THICKNESS_MAX)
    
    # Get coordinates of ALL margin pixels
    margin_y_inds, margin_x_inds = np.where(global_margin_mask.values)
    
    # Get the label ID for every single margin pixel
    margin_labels = labeled_array[margin_y_inds, margin_x_inds]
    
    # Iterate through unique labels found in the margin
    present_labels = np.unique(margin_labels)
    present_labels = present_labels[present_labels != 0] # skip background
    
    for label_id in present_labels:
        area = component_areas[label_id]
        
        # Skip tiny coarse noise (e.g., 1-2 pixel islands)
        if area < MIN_COARSE_PIXELS:
            continue
            
        # A. Calculate Target Count (Weighted by Area)
        # The main ice sheet will have huge area -> gets most points
        weight = area / total_ice_area
        n_target = int(np.ceil(NUM_SAMPLES * weight))
        
        # B. Identify available margin pixels for THIS component
        component_indices = np.where(margin_labels == label_id)[0]
        n_available = len(component_indices)
        
        if n_available > 0:
            # Don't take more than exist
            n_take = min(n_target, n_available)
            
            # Randomly select indices
            chosen_indices = rng.choice(component_indices, size=n_take, replace=False)
            
            # Store the selected GLOBAL indices
            final_y.extend(margin_y_inds[chosen_indices])
            final_x.extend(margin_x_inds[chosen_indices])

    # --- STEP 4: CONVERT TO COORDINATES & PLOT ---
    
    # Look up the projected coordinates from the COARSE grid
    x_coords = thickness_coarse.x.values[final_x]
    y_coords = thickness_coarse.y.values[final_y]
    
    print(f"Generated {len(x_coords)} pseudo-observations.")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the coarse thickness field
    thickness_coarse.plot(ax=ax, cmap='Blues', add_colorbar=True, vmax=2000, cbar_kwargs={'label': 'Ice Thickness (m)'})
    
    # Overlay the sampled points
    ax.scatter(x_coords, y_coords, c='red', s=15, edgecolors='black', linewidth=0.5, label='Pseudo-Obs')
    
    ax.set_title(f"Coarse Margin Sampling (Factor {DOWNSAMPLE_FACTOR}x)\nMain Sheet Weighted by Area")
    ax.legend()
    plt.show()
    
    # Optional: Save to CSV for the GP script
    # import pandas as pd
    df = pd.DataFrame({'x': x_coords, 'y': y_coords})
    df.to_csv("margin_pseudo_obs.csv", index=False)

if __name__ == "__main__":
    main()