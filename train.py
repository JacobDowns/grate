#!/usr/bin/env python

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import torch


# ============================
# Helper: nearest index in 1D
# ============================

def nearest_index(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Return nearest integer indices of `values` within a 1D coordinate array."""
    return np.abs(coords[None, :] - values[:, None]).argmin(axis=1)


# ============================
# 1. Load PCA modes + mean SDF + grid
# ============================

modes_ds = xr.open_dataset("data/signed_distance_pca_modes.nc")

# Expecting: modes(mode, y, x), mean_field(y, x), x(x), y(y)
modes = modes_ds["modes"].values.astype(np.float32)       # (n_modes, ny, nx)
mean_sdf = modes_ds["mean_field"].values.astype(np.float32)  # (ny, nx)
reference_field = (
    modes_ds["reference_field"].values.astype(np.float32)
    if "reference_field" in modes_ds
    else np.zeros_like(mean_sdf, dtype=np.float32)
)  # baseline added back to anomalies
x = modes_ds["x"].values.astype(np.float32)               # (nx,)
y = modes_ds["y"].values.astype(np.float32)               # (ny,)

ny, nx = mean_sdf.shape
n_modes_total = modes.shape[0]

# Choose a subset of leading modes to use
n_modes_use = 20
modes = modes[:n_modes_use]  # (M, ny, nx)
M = n_modes_use

print(f"Using {M} PCA modes out of {n_modes_total}.")

alpha  = 100.0 

# ============================
# 2. Load age data (years BP -> ka)
# ============================

ages_df = pd.read_csv("data/age_data_epsg3413.csv")

age_xs = ages_df["x_3413"].to_numpy(dtype=np.float32)
age_ys = ages_df["y_3413"].to_numpy(dtype=np.float32)

# Observed ages & 1-sigma errors are in years BP -> convert to ka
age_obs_years = ages_df["ages"].to_numpy(dtype=np.float32)
age_err_years = ages_df["errors"].to_numpy(dtype=np.float32) 

age_obs = age_obs_years / 1000.0  # (J,) in ka
age_err = age_err_years / 1000.0  # (J,) in ka
#age_err /= 100.0

J = age_obs.shape[0]
print(f"Loaded {J} age observations (converted to ka).")

# Map age sites onto grid
ages_df["x_index"] = nearest_index(age_xs, x)
ages_df["y_index"] = nearest_index(age_ys, y)

ix = ages_df["x_index"].to_numpy()
iy = ages_df["y_index"].to_numpy()


# ============================
# 3. Build reconstruction time grid (in ka)
# ============================

# Make a time grid that covers the age range with some buffer (in ka)
age_min = float(age_obs.min())
age_max = float(age_obs.max())
buffer = 0.5  # ka
dt = 0.25     # ka step

t_start = age_min - buffer
t_end = age_max + buffer

time_grid = np.arange(t_start, t_end + 1e-6, dt, dtype=np.float32)  # (T,)
T = time_grid.shape[0]

print(f"Time grid: {T} points from {t_start:.2f} to {t_end:.2f} ka, step {dt} ka.")


# ============================
# 4. Precompute SDF basis values at age sites
# ============================

# Mean SDF at age sites
mean_sdf_sites = mean_sdf[iy, ix].astype(np.float32)  # (J,)

# Mode values at age sites: (J, M)
mode_vals_sites = np.zeros((J, M), dtype=np.float32)
for m in range(M):
    mode_vals_sites[:, m] = modes[m, iy, ix]

print("Precomputed SDF basis at age sites.")


# ============================
# 5. Move data to torch (float32)
# ============================

device = torch.device("cpu")  # or torch.device("cuda") if available

time_grid_t = torch.from_numpy(time_grid).to(device=device, dtype=torch.float32)          # (T,)
mean_sdf_sites_t = torch.from_numpy(mean_sdf_sites).to(device=device, dtype=torch.float32)  # (J,)
mode_vals_sites_t = torch.from_numpy(mode_vals_sites).to(device=device, dtype=torch.float32) # (J, M)
age_obs_t = torch.from_numpy(age_obs).to(device=device, dtype=torch.float32)              # (J,)
age_err_t = torch.from_numpy(age_err).to(device=device, dtype=torch.float32)              # (J,)


# ============================
# 6. Build soft labels y_star_t (J,T) from ages + uncertainties
# ============================

# We interpret ages as deglaciation times: before age -> likely ice-covered (0),
# after age -> likely ice-free (1). Use a Gaussian CDF in time to soften with age_err.

normal = torch.distributions.Normal(loc=torch.tensor(0.0, dtype=torch.float32, device=device),
                                    scale=torch.tensor(1.0, dtype=torch.float32, device=device))

t = time_grid_t[None, :]        # (1,T)
a = age_obs_t[:, None]          # (J,1)
s = age_err_t[:, None]  / 10.        # (J,1)
s = torch.clamp(s, min=1e-3)    # avoid zero std

z = (t - a) / s                 # (J,T)
y_star_t = normal.cdf(z)        # (J,T), soft target prob of being ice-free

print("Constructed soft labels y_star_t for Bernoulli likelihood.")


# ============================
# 7. GP prior on coefficients in time (float32)
# ============================

def rbf_kernel(t, length_scale, variance):
    """
    Squared-exponential (RBF) kernel in float32.

    t: (T,) 1D tensor of times (ka), dtype float32
    length_scale: float
    variance: float
    returns: (T, T) covariance matrix (float32)
    """
    dt = t[:, None] - t[None, :]
    return variance * torch.exp(-0.5 * (dt / length_scale)**2)


# Hyperparameters for GP prior (tune these)
length_scale = 0.05   # ka, correlation length
coeff_std   = 100.0   # prior std dev for coefficients (SDF units)
variance    = coeff_std**2

base_nugget = 1e-5
max_tries   = 7

# Build base covariance (without nugget)
K = rbf_kernel(time_grid_t, length_scale, variance)

# Force exact symmetry
K = 0.5 * (K + K.T)

# Try Cholesky with increasing jitter
L = None
logdet_K = None
nugget = base_nugget

for attempt in range(max_tries):
    try:
        K_jittered = K + nugget * torch.eye(T, device=device, dtype=torch.float32)
        L = torch.linalg.cholesky(K_jittered)
        logdet_K = 2.0 * torch.sum(torch.log(torch.diag(L)))
        print(f"Cholesky succeeded with nugget={nugget:.1e}")
        break
    except torch._C._LinAlgError:
        print(f"Cholesky failed with nugget={nugget:.1e}, increasing jitter...")
        nugget *= 10.0

if L is None:
    raise RuntimeError("Cholesky failed even with large jitter; check time grid / kernel settings.")


def gp_log_prior(coeffs, L, logdet_K):
    """
    GP log prior for coefficients.

    coeffs: (M, T), dtype float32
    L: (T, T) Cholesky factor of K_jittered
    logdet_K: scalar log |K_jittered|

    Returns scalar log p(c) assuming independent modes, each ~ N(0, K).
    """
    M, T = coeffs.shape
    logp = torch.tensor(0.0, dtype=torch.float32, device=coeffs.device)

    for m in range(M):
        c_m = coeffs[m]  # (T,)
        # Solve K x = c using cholesky_solve with A = K = L L^T
        x = torch.cholesky_solve(c_m.unsqueeze(1), L)  # (T,1)
        x = x.squeeze(1)  # (T,)

        quad = torch.dot(c_m, x)
        logp += -0.5 * (quad + logdet_K + T * np.log(2.0 * np.pi))

    return logp


# ============================
# 8. Forward model: coeffs -> P(ice-free) in time
# ============================

def predict_ice_free_prob(coeffs, alpha=1.0):
    """
    coeffs: (M, T)

    Returns:
        p: (J, T), probability of being ice-free at each site & time.

    Uses logistic(SDF) with SDF < 0 => ice-covered, SDF > 0 => ice-free.
    """
    # SDF at sites & times: d_{j,k} = mean_j + sum_m c[m,k] * phi_{m,j}
    # mode_vals_sites_t: (J, M), coeffs: (M, T) -> (J, T)
    sdf = mean_sdf_sites_t[:, None] + mode_vals_sites_t @ coeffs  # (J,T)

    # logistic on SDF: P(ice-free)
    p = torch.sigmoid(alpha * sdf)
    return p


# ============================
# 9. Likelihood: time-series Bernoulli with soft labels
# ============================

def log_likelihood(coeffs, alpha=10.0):
    """
    coeffs: (M, T)

    Uses Bernoulli log-likelihood over time with soft labels y_star_t (J,T),
    where y_star_t(j,k) is the target probability that site j is ice-free at time t_k.
    """
    p = predict_ice_free_prob(coeffs, alpha=alpha)  # (J,T)
    eps = 1e-8
    #l = torch.log((p - y_star_t)**2)

    #print(p - y_star_t)

    #l = l.sum()

    ll = (y_star_t * torch.log(p + eps) + (1.0 - y_star_t) * torch.log(1.0 - p + eps)).sum()
    #print(ll[0])
    return ll


def log_posterior(coeffs, alpha=10.0):
    """
    coeffs: (M, T)
    """
    return log_likelihood(coeffs, alpha=alpha) + gp_log_prior(coeffs, L, logdet_K)


# ============================
# 10. MAP optimization
# ============================

# Initialize coefficients to zero (or small noise)
coeffs = torch.nn.Parameter(torch.zeros(M, T, device=device, dtype=torch.float32))

optimizer = torch.optim.Adam([coeffs], lr=1e-2)

n_iters = 50000
print_every = 500

print("Starting MAP optimization...")
for it in range(n_iters):
    optimizer.zero_grad()
    neg_log_post = -log_posterior(coeffs, alpha=10.0)
    neg_log_post.backward()
    optimizer.step()

    if it % print_every == 0 or it == n_iters - 1:
        print(f"iter {it:4d}, -logpost = {neg_log_post.item():.3f}")

coeffs_map = coeffs.detach().cpu().numpy().astype(np.float32)  # (M, T)
print("Optimization done.")


# ============================
# 11. Diagnostics
# ============================

# A. Plot some coefficient trajectories (MAP) vs time
modes_to_plot = min(3, M)

fig, axes = plt.subplots(modes_to_plot, 1, figsize=(10, 3.5 * modes_to_plot), sharex=True)
if modes_to_plot == 1:
    axes = [axes]

for mi in range(modes_to_plot):
    ax = axes[mi]
    ax.plot(time_grid, coeffs_map[mi], color="red", lw=2, label="MAP")
    ax.axhline(0.0, color="k", lw=1, alpha=0.5)
    ax.set_ylabel(f"Coeff mode {mi}")
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Time (ka)")
fig.suptitle("MAP coefficient trajectories (SDF PCA modes)")
plt.tight_layout()
plt.show()


# B. For a couple of sites, plot target vs model P(ice-free) over time
coeffs_map_t = torch.from_numpy(coeffs_map).to(device=device, dtype=torch.float32)
p_map = predict_ice_free_prob(coeffs_map_t, alpha=alpha)  # (J,T)
p_map_np = p_map.detach().cpu().numpy()
y_star_np = y_star_t.detach().cpu().numpy()

n_sites_plot = min(3, J)
site_indices = np.linspace(0, J - 1, n_sites_plot, dtype=int)

fig, axes = plt.subplots(n_sites_plot, 1, figsize=(10, 3.5 * n_sites_plot), sharex=True)
if n_sites_plot == 1:
    axes = [axes]

for idx, ax in zip(site_indices, axes):
    ax.plot(time_grid, y_star_np[idx], "k--", lw=2, label="Target P(ice-free)")
    ax.plot(time_grid, p_map_np[idx], "r-", lw=2, label="Model P(ice-free)")
    ax.set_ylabel(f"Site {idx}")
    ax.legend(loc="lower right")

axes[-1].set_xlabel("Time (ka)")
fig.suptitle("Target vs model P(ice-free) over time at selected sites")
plt.tight_layout()
plt.show()


# C. Example reconstructed SDF field at some time (e.g. median age)
median_age = float(np.median(age_obs))
time_idx = int(np.argmin(np.abs(time_grid - median_age)))
time_val = time_grid[time_idx]

print(f"Plotting reconstructed SDF at t = {time_val:.2f} ka (time index {time_idx}).")

sdf_grid = mean_sdf.copy()
for m in range(M):
    sdf_grid += coeffs_map[m, time_idx] * modes[m]

sdf_grid += reference_field  # add baseline back in case PCA was pre-centered

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    sdf_grid,
    origin="lower",
    extent=(x.min(), x.max(), y.min(), y.max()),
    cmap="RdBu_r"
)
# Overlay 0-contour (ice margin)
X, Y = np.meshgrid(x, y)
ax.contour(X, Y, sdf_grid, levels=[0.0], colors="k", linewidths=1.0)
ax.set_title(f"Reconstructed SDF at t = {time_val:.2f} ka")
ax.set_xlabel("x (EPSG:3413)")
ax.set_ylabel("y (EPSG:3413)")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Signed distance (SDF units)")
plt.tight_layout()
plt.show()
