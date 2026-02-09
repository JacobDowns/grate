from pathlib import Path
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np 

DATASET_PATH = Path(__file__).with_name("issm_run1_2p5km_thk_bed.nc")
data = xr.open_dataset(DATASET_PATH)

H = data['thickness'].astype(float)   # ice thickness [m]
B = data['bed'].astype(float)        # bed elevation [m], sea level at 0

rho_i = 917.0   # ice density [kg/m^3]
rho_w = 1000.0  # seawater density [kg/m^3]

# Where there is actually ice
has_ice = H > 0.0   # adjust threshold if needed

# Water depth (only where bed is below sea level)
water_depth = xr.where(B < 0.0, -B, 0.0)

# Thickness required for flotation at that bed depth:
#   ρ_i * H_flot = ρ_w * water_depth  →  H_flot = (ρ_w / ρ_i) * water_depth
H_flot = (rho_w / rho_i) * water_depth

# Floating where:
#   - bed below sea level
#   - there is ice
#   - ice is thick enough to float
floating = (B < 0.0) & has_ice & (H <= H_flot)

# Hydrostatic flotation base:
B_float = - (rho_i / rho_w) * H

# Correct ice base:
# - bed where no ice OR grounded
# - flotation base where floating
B_base = xr.where(floating, B_float, B)

# Optionally add to dataset
data['ice_base_corrected'] = B_base

plt.subplot(2,1,1)
(H.isel(time=0)).plot()

plt.subplot(2,1,2)
(B.isel(time=0) + H.isel(time=0)).plot()
plt.show()


quit()

with xr.open_dataset(DATASET_PATH) as ds:
    thickness = ds["thickness"]
    bed = ds["bed"]

    # Simple buoyancy correction: if the ice would be floating (i.e., thickness is
    # less than flotation thickness for a bed below sea level), compute the surface
    # as freeboard; otherwise use bed + thickness.
    rho_i = 917.0  # ice density (kg/m^3)
    rho_w = 1028.0  # seawater density (kg/m^3)

    # Flotation thickness required to just touch the bed (positive where bed < 0).
    float_thickness = (-bed * rho_w / rho_i).where(bed < 0, other=0)

    def surface_elevation(thk, b):
        is_floating = (b < 0) & (thk <= float_thickness)
        freeboard = thk * (1 - rho_i / rho_w)
        grounded = b + thk
        return xr.where(is_floating, freeboard, grounded)

    first_surface = surface_elevation(thickness.isel(time=0), bed.isel(time=0)).load()
    last_surface = surface_elevation(thickness.isel(time=-1), bed.isel(time=-1)).load()

surfaces = (first_surface, last_surface)
titles = ("Surface elevation (first step)", "Surface elevation (last step)")

# Share color limits so the two plots use the same scale.
combined = xr.concat(surfaces, dim="surface")
vmin = float(combined.min(skipna=True))
vmax = float(combined.max(skipna=True))

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for ax, surface, title in zip(axes, surfaces, titles):
    surf2d = surface.squeeze(drop=True)  # ensure only (y, x) dims for plotting
    surf2d.plot(ax=ax, x="x", y="y", vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_title(title)
    ax.set_aspect("equal")

plt.show()
