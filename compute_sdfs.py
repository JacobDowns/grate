from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

sims = [1, 2, 3, 4, 5, 7, 8]
base_dir = Path(__file__).parent / "data"

for sim in sims:

    DATASET_PATH = base_dir / f"issm_run_{sim}.nc"
    OUTPUT_PATH = base_dir / f"{DATASET_PATH.stem}_sdf.nc"
    ICE_THRESHOLD = 5.0

    data = xr.open_dataset(DATASET_PATH)
    H = data["thickness"].astype(np.float32)

    # Use grid spacing to express distances in meters
    dx = float(abs(data["x"].diff("x").mean()))
    dy = float(abs(data["y"].diff("y").mean()))
    sampling = (dy, dx)

    ice_mask = H.values > ICE_THRESHOLD
    sdf = np.empty_like(H.values, dtype=np.float32)

    for t in range(ice_mask.shape[0]):
        if t % 10 == 0:
            print(f"Processing time step {t}")
        mask_t = ice_mask[t]
        dist_inside = distance_transform_edt(mask_t, sampling=sampling)
        dist_outside = distance_transform_edt(~mask_t, sampling=sampling)
        sdf[t] = np.where(mask_t, dist_inside, -dist_outside)
        #plt.imshow(sdf[t])
        #plt.colorbar()
        #plt.show()

    # Normalize sdf to [-0.5, 0.5]
    #sdf = (sdf - sdf.min()) / (sdf.max() - sdf.min()) - 0.5

    signed_distance = xr.DataArray(
        sdf,
        coords=H.coords,
        dims=H.dims,
        name="signed_distance",
        attrs={"description": "Signed distance to ice margin (positive inside ice)"},
    )

    data_out = data.assign(signed_distance=signed_distance)
    if "epsg" in data.attrs:
        data_out.attrs["epsg"] = int(data.attrs["epsg"])
    data_out.to_netcdf(OUTPUT_PATH)
    print(f"Signed distance field saved to {OUTPUT_PATH}")
