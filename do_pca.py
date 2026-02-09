from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Find all NetCDF files containing signed distance fields
SDF_FILES = sorted(Path(".").glob("data/*_sdf.nc"))
if not SDF_FILES:
    raise FileNotFoundError("No *_sdf.nc files found to stack.")

signed_distance_arrays = []
run_labels = []
first_snapshots = []
use_first_snapshot_mean = True  # Toggle baseline choice (first snapshots vs mean over all fields)
for sdf_path in SDF_FILES:
    with xr.open_dataset(sdf_path) as ds:
        if "signed_distance" not in ds:
            raise KeyError(f"'signed_distance' variable not found in {sdf_path}")
        # Ensure consistent dtype and eager load to memory for PCA
        sdf = ds["signed_distance"].astype(np.float32).load()

        signed_distance_arrays.append(sdf)
        run_labels.append(sdf_path.stem)
        # Save the first snapshot for the global first-timestep mean
        first_snapshots.append(sdf.isel(time=0) if "time" in sdf.dims else sdf[0])

if use_first_snapshot_mean:
    # Compute the spatial mean of the first time step across runs (2D field)
    if not first_snapshots:
        raise ValueError("No first snapshots found to build the baseline field.")
    baseline_first = xr.concat(first_snapshots, dim="run").mean("run")
    # Subtract the baseline from every snapshot
    signed_distance_arrays = [sdf - baseline_first for sdf in signed_distance_arrays]
    reference_field = baseline_first.values.astype(np.float32)
    center_pca = False  # data already centered relative to baseline
else:
    reference_field = None
    center_pca = True

# Convert to numpy for PCA
data = np.concatenate([sdf.values for sdf in signed_distance_arrays], axis=0)
template = signed_distance_arrays[0]

# Get the grid cell size
dx = float(abs(template["x"].diff("x").mean()))
dy = float(abs(template["y"].diff("y").mean()))

# Normalize to [-0.5, 0.5]
data /= dx # (data - data.min()) / (data.max() - data.min()) - 0.5
if reference_field is not None:
    reference_field = reference_field / dx

def snapshot_pca(
    data,             # numpy array of shape (n_snapshots, ny, nx)
    n_modes=None,     # number of modes to keep, or None for all
    mask=None,        # optional 2D boolean mask (ny, nx), True = keep
    dtype=np.float32, # use float32 to save memory if needed
    center=False       # subtract per-cell mean across snapshots before PCA
):
    """
    Snapshot PCA / EOF analysis on a stack of raster fields.

    Parameters
    ----------
    data : np.ndarray
        Array with shape (n_snapshots, ny, nx).
    n_modes : int or None
        Number of leading modes to return. If None, return all.
    mask : np.ndarray or None
        Optional boolean mask of shape (ny, nx). True = include cell in PCA.
        Cells where mask is False are ignored (treated as NaN).
    dtype : np.dtype
        Data type to cast to (e.g., np.float32 to reduce memory).
    center : bool
        If True, subtract the per-cell mean across snapshots (standard PCA
        centering). If False, assume the data is already centered.

    Returns
    -------
    mean_field : np.ndarray
        Mean field over snapshots, shape (ny, nx).
    modes : np.ndarray
        Spatial modes (basis functions), shape (n_modes, ny, nx).
    coeffs : np.ndarray
        Time coefficients for each mode, shape (n_snapshots, n_modes).
    eigvals : np.ndarray
        Eigenvalues corresponding to each mode, shape (n_modes,).
    """
    # Ensure we have a copy in the desired dtype
    data = np.array(data, dtype=dtype, copy=True)
    n_snapshots, ny, nx = data.shape

    # Apply mask if provided
    if mask is not None:
        if mask.shape != (ny, nx):
            raise ValueError(f"mask must have shape {(ny, nx)}, got {mask.shape}")
        # set masked-out cells to NaN so we can ignore them
        data[:, ~mask] = np.nan

    # Flatten spatial dimensions: (n_snapshots, ny*nx)
    data_2d = data.reshape(n_snapshots, -1)  # (T, M), T = 868, M = ny*nx

    # Identify columns (grid cells) that are all-NaN across time → drop them
    valid_mask = ~np.isnan(data_2d).all(axis=0)  # length M
    data_valid = data_2d[:, valid_mask]         # (T, M_valid)

    # Compute mean field over snapshots for valid cells
    if center:
        mean_valid = np.nanmean(data_valid, axis=0, keepdims=True)  # (1, M_valid)
        X = data_valid - mean_valid
    else:
        mean_valid = np.zeros((1, data_valid.shape[1]), dtype=dtype)
        X = data_valid

    # Replace any remaining NaNs with 0
    X = np.nan_to_num(X, nan=0.0)  # (T, M_valid)

    # Method of snapshots:
    # C = X X^T has shape (T, T), much smaller than (M_valid, M_valid)
    C = X @ X.T  # (T, T)

    # Eigen-decompose C (symmetric)
    eigvals, eigvecs = np.linalg.eigh(C)  # eigvecs: columns are eigenvectors

    # Sort eigenvalues/vectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Optionally truncate
    if n_modes is not None:
        eigvals = eigvals[:n_modes]
        eigvecs = eigvecs[:, :n_modes]

    # Spatial modes: phi_j = X^T v_j / sqrt(lambda_j)
    # X shape: (T, M_valid), eigvecs: (T, r)
    # -> spatial_modes_valid: (M_valid, r)
    spatial_modes_valid = X.T @ eigvecs  # (M_valid, r)
    # normalize by sqrt eigenvalues to get orthonormal modes
    spatial_modes_valid /= np.sqrt(eigvals + 1e-12)[None, :]  # broadcast

    # Time coefficients (scores): a = eigvecs * sqrt(lambda)
    # shape: (T, r)
    coeffs = eigvecs * np.sqrt(eigvals + 1e-12)[None, :]

    # Rebuild full-sized mean field and modes (fill invalid cells with NaN)
    M = ny * nx
    mean_full = np.full(M, np.nan, dtype=dtype)
    mean_full[valid_mask] = mean_valid.ravel()
    mean_field = mean_full.reshape(ny, nx)

    # Modes: (M_valid, r) -> (ny, nx, r)
    modes_full = np.full((M, eigvals.shape[0]), np.nan, dtype=dtype)
    modes_full[valid_mask, :] = spatial_modes_valid
    modes_full = modes_full.reshape(ny, nx, eigvals.shape[0])  # (ny, nx, r)

    # Reorder to (n_modes, ny, nx) for convenience
    modes = np.moveaxis(modes_full, -1, 0)  # (r, ny, nx)

    return mean_field, modes, coeffs, eigvals

def plot_eigenmodes(modes, n_plot=6, cmap="RdBu_r", share_colorbar=True):
    """
    Plot the first n_plot spatial modes as images.

    Parameters
    ----------
    modes : np.ndarray
        Array of spatial modes with shape (n_modes, ny, nx).
    n_plot : int
        Number of leading modes to plot.
    cmap : str
        Matplotlib colormap.
    share_colorbar : bool
        If True, use the same color scale for all plots.
    """
    n_modes, ny, nx = modes.shape
    n_plot = min(n_plot, n_modes)

    # Compute global vmin/vmax for symmetric color scale (optional but nice)
    if share_colorbar:
        vmax = np.nanmax(np.abs(modes[:n_plot]))
        vmin = -vmax
    else:
        vmin = vmax = None

    # Choose grid layout
    n_cols = int(np.ceil(np.sqrt(n_plot)))
    n_rows = int(np.ceil(n_plot / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False
    )

    for i in range(n_plot):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r, c]

        mode = modes[i]

        im = ax.imshow(
            mode,
            origin="lower",
            cmap=cmap,
            vmin=vmin if share_colorbar else None,
            vmax=vmax if share_colorbar else None,
        )
        ax.set_title(f"Mode {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])

        # Add colorbar per subplot if not sharing
        if not share_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # If sharing colorbar, make a single colorbar for the whole figure
    if share_colorbar:
        fig.colorbar(
            im, ax=axes, orientation="vertical",
            fraction=0.046, pad=0.04
        )

    plt.tight_layout()
    plt.show()


# thickness_all.shape == (868, 1425, 893)
mean_field, modes, coeffs, eigvals = snapshot_pca(
    data,
    n_modes=30,     
    mask=None,      # or a (1425, 893) boolean mask if you have one
    center=center_pca    # data already centered by first-timestep average when False
)

# Choose the reference field used for plotting/reconstruction
reference_baseline = reference_field if reference_field is not None else np.zeros_like(mean_field)

# Reconstruct snapshot t using first r modes
t = 0
r = 30
# PCA mean contributes only when centering is enabled
mean_component = mean_field if center_pca else np.zeros_like(mean_field)
recon_anom_flat = mean_component.reshape(-1) + (modes[:r].reshape(r, -1).T @ coeffs[t, :r])
recon = (reference_baseline.reshape(-1) + recon_anom_flat).reshape(mean_field.shape)
# Original (uncentered) field for snapshot t
original = (reference_baseline.reshape(-1) + data[t].reshape(-1)).reshape(mean_field.shape)


# Compare recon vs original
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 4))
plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(original, origin="lower")
plt.colorbar()

plt.subplot(1,3,2)
plt.title(f"Reconstruction (r={r}) with 0-contour")
im = plt.imshow(recon, origin="lower")
plt.contour(recon, levels=[0], colors="k", linewidths=1, origin="lower")
plt.colorbar(im)

plt.subplot(1,3,3)
plt.title("Reconstruction - Original")
plt.imshow(recon - original, origin="lower")
plt.colorbar()
plt.tight_layout()
plt.show()

plot_eigenmodes(modes, n_plot=6)

print("Mean field shape:", mean_field.shape)   # (1425, 893)
print("Modes shape:", modes.shape)             # (10, 1425, 893)
print("Coeffs shape:", coeffs.shape)           # (868, 10)
print("Eigenvalues:", eigvals)

var_explained = eigvals / eigvals.sum()
cum_var = np.cumsum(var_explained)

# Save the modes to NetCDF in the data directory
mode_numbers = np.arange(1, modes.shape[0] + 1)
output_nc = Path("data/signed_distance_pca_modes.nc")

modes_da = xr.DataArray(
    modes,
    dims=("mode", "y", "x"),
    coords={"mode": mode_numbers, "y": template["y"], "x": template["x"]},
    name="modes",
    attrs={"description": "Spatial PCA modes for signed distance fields"},
)

ds_out = xr.Dataset(
    data_vars={
        "modes": modes_da,
        "mean_field": xr.DataArray(
            mean_field,
            dims=("y", "x"),
            coords={"y": template["y"], "x": template["x"]},
            attrs={
                "description": (
                    "Mean signed distance field used for centering "
                    "(zeros when pre-centered by first-snapshot baseline)"
                )
            },
        ),
        "eigenvalues": xr.DataArray(
            eigvals.astype(np.float32),
            dims=("mode",),
            coords={"mode": mode_numbers},
        ),
        "reference_field": xr.DataArray(
            reference_baseline,
            dims=("y", "x"),
            coords={"y": template["y"], "x": template["x"]},
            attrs={
                "description": (
                    "Reference field subtracted before PCA "
                    "(first-snapshot mean across runs when enabled; zeros otherwise)"
                )
            },
        ),
        "variance_explained": xr.DataArray(
            var_explained.astype(np.float32),
            dims=("mode",),
            coords={"mode": mode_numbers},
        ),
    },
    attrs={
        "stacked_runs": ", ".join(run_labels),
        "normalization": (
            "Signed distance normalized to [-0.5, 0.5] and offset by the "
            "spatial mean of the first time-step image across runs before PCA "
            "when use_first_snapshot_mean=True; otherwise centered in PCA."
        ),
        "use_first_snapshot_mean": str(use_first_snapshot_mean),
    },
)

engine = None
try:
    import netCDF4  # noqa: F401

    engine = "netcdf4"
except ImportError:
    try:
        import h5netcdf  # noqa: F401

        engine = "h5netcdf"
    except ImportError:
        pass

to_netcdf_kwargs = {"engine": engine} if engine is not None else {}
ds_out.to_netcdf(output_nc, **to_netcdf_kwargs)
ds_out.close()
print(f"PCA modes written to {output_nc}")
