#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingData:
    xy_m: "np.ndarray"  # (N,2) in meters (EPSG:3413)
    y: "np.ndarray"  # (N,) normalized 0..1
    yerr: "np.ndarray"  # (N,) normalized 0..1
    mean0: "np.ndarray"  # (N,) mean field at points (normalized)
    Phi: "np.ndarray"  # (N,M) PCA mode values at points (normalized)
    max_age_years: float


def _bilinear_sample(field: "np.ndarray", xg: "np.ndarray", yg: "np.ndarray", x: "np.ndarray", y: "np.ndarray") -> "np.ndarray":
    """
    field: (ny, nx), xg: (nx,), yg: (ny,) (monotonic increasing), x/y: (N,)
    Returns (N,) float32 with NaNs for out-of-bounds.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xg = np.asarray(xg, dtype=np.float64)
    yg = np.asarray(yg, dtype=np.float64)

    nx = xg.size
    ny = yg.size
    if nx < 2 or ny < 2:
        raise ValueError("Grid must be at least 2x2 for bilinear sampling.")

    ix = np.searchsorted(xg, x, side="right") - 1
    iy = np.searchsorted(yg, y, side="right") - 1

    oob = (ix < 0) | (ix >= nx - 1) | (iy < 0) | (iy >= ny - 1)
    ix = np.clip(ix, 0, nx - 2)
    iy = np.clip(iy, 0, ny - 2)

    x0 = xg[ix]
    x1 = xg[ix + 1]
    y0 = yg[iy]
    y1 = yg[iy + 1]

    tx = np.where(x1 != x0, (x - x0) / (x1 - x0), 0.0)
    ty = np.where(y1 != y0, (y - y0) / (y1 - y0), 0.0)

    f00 = field[iy, ix]
    f10 = field[iy, ix + 1]
    f01 = field[iy + 1, ix]
    f11 = field[iy + 1, ix + 1]

    out = (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11
    out = out.astype(np.float32, copy=False)
    out[oob] = np.nan
    return out


def load_training_data(
    *,
    pca_nc: Path,
    ages_csv: Path,
    n_modes: int,
    clip_ages: bool,
    drop_oob: bool,
) -> TrainingData:
    import numpy as np
    import pandas as pd
    import xarray as xr

    pca = xr.open_dataset(pca_nc, decode_times=False)
    try:
        if "deglaciation_age_mean_norm" not in pca or "pca_mode_norm" not in pca:
            raise KeyError(
                f"{pca_nc} must contain 'deglaciation_age_mean_norm' and 'pca_mode_norm'. "
                f"Available: {list(pca.data_vars)}"
            )

        xg = pca["x"].values.astype(np.float64)
        yg = pca["y"].values.astype(np.float64)
        mean_norm = pca["deglaciation_age_mean_norm"].values.astype(np.float32)
        modes = pca["pca_mode_norm"].values.astype(np.float32)  # (mode, y, x)

        if "deglaciation_age_scale" not in pca:
            raise KeyError(f"{pca_nc} missing 'deglaciation_age_scale' (needed to normalize observation ages).")
        max_age_years = float(np.nanmax(pca["deglaciation_age_scale"].values.astype(np.float64)))
        if not (max_age_years > 0):
            raise ValueError(f"Invalid max_age_years={max_age_years} from {pca_nc}")
    finally:
        pca.close()

    df = pd.read_csv(ages_csv)
    required = {"x_3413", "y_3413", "ages", "errors"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{ages_csv} missing columns: {sorted(missing)}. Available: {list(df.columns)}")

    x = df["x_3413"].to_numpy(dtype=np.float64)
    y = df["y_3413"].to_numpy(dtype=np.float64)
    ages = df["ages"].to_numpy(dtype=np.float64)
    errs = df["errors"].to_numpy(dtype=np.float64)

    if clip_ages:
        ages = np.clip(ages, 0.0, max_age_years)
        errs = np.clip(errs, 0.0, max_age_years)
    else:
        ok = (ages >= 0.0) & (ages <= max_age_years)
        x, y, ages, errs = x[ok], y[ok], ages[ok], errs[ok]

    y_norm = (ages / max_age_years).astype(np.float32)
    yerr_norm = (errs / max_age_years).astype(np.float32)

    mean0 = _bilinear_sample(mean_norm, xg, yg, x, y)

    M = int(min(n_modes, modes.shape[0]))
    Phi = np.empty((x.size, M), dtype=np.float32)
    for m in range(M):
        Phi[:, m] = _bilinear_sample(modes[m, :, :], xg, yg, x, y)

    finite = np.isfinite(mean0) & np.all(np.isfinite(Phi), axis=1) & np.isfinite(y_norm) & np.isfinite(yerr_norm)
    if drop_oob:
        x, y, y_norm, yerr_norm, mean0, Phi = x[finite], y[finite], y_norm[finite], yerr_norm[finite], mean0[finite], Phi[finite]

    xy_m = np.stack([x, y], axis=1).astype(np.float64)
    return TrainingData(xy_m=xy_m, y=y_norm, yerr=yerr_norm, mean0=mean0, Phi=Phi, max_age_years=max_age_years)


def build_noise_net(hidden: int = 16):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(2, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1),
    )


def nll_cholesky(y: "torch.Tensor", mean: "torch.Tensor", K: "torch.Tensor", *, jitter: float = 1e-6) -> "torch.Tensor":
    import torch

    n = y.shape[0]
    Kj = K + (float(jitter) * torch.eye(n, dtype=K.dtype, device=K.device))
    L = torch.linalg.cholesky(Kj)
    r = (y - mean).unsqueeze(1)  # (N,1)
    alpha = torch.cholesky_solve(r, L)
    quad = (r * alpha).sum()
    logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
    return 0.5 * (quad + logdet + n * torch.log(torch.tensor(2.0 * torch.pi, dtype=K.dtype, device=K.device)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GP for deglaciation age with PCA-parameterized mean (gpytorch kernel + torch).")
    parser.add_argument("--pca-nc", type=Path, default=Path("data/deglaciation_snapshot_pca.nc"), help="PCA output NetCDF")
    parser.add_argument("--ages-csv", type=Path, default=Path("data/age_data_epsg3413.csv"), help="Age observations CSV (EPSG:3413)")
    parser.add_argument("--modes", type=int, default=15, help="Number of PCA anomaly modes in mean (default: 15)")
    parser.add_argument("--steps", type=int, default=1500, help="Training steps (default: 1500)")
    parser.add_argument("--lr", type=float, default=2e-2, help="Adam learning rate (default: 2e-2)")
    parser.add_argument(
        "--clip-ages",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Clip ages/errors to [0,max_age] instead of dropping out-of-range (default: true)",
    )
    parser.add_argument("--use-obs-errors", action="store_true", help="Include CSV errors as fixed additional noise")
    parser.add_argument("--device", type=str, default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--out", type=Path, default=Path("data/gp_deglaciation_fit_gpytorch.pt"), help="Output torch checkpoint")
    args = parser.parse_args()

    try:
        import gpytorch  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "gpytorch is not installed in this environment. Add it with `uv sync` after updating dependencies "
            "(pyproject.toml now includes gpytorch), or `pip install gpytorch`."
        ) from e

    import numpy as np
    import torch
    import gpytorch

    data = load_training_data(
        pca_nc=args.pca_nc,
        ages_csv=args.ages_csv,
        n_modes=int(args.modes),
        clip_ages=bool(args.clip_ages),
        drop_oob=True,
    )

    print(data)
    quit()

    N, M = data.Phi.shape
    print(f"Training points: N={N}, modes in mean: M={M}, max_age_years={data.max_age_years:g}")

    device = torch.device(args.device)
    dtype = torch.float64

    xy = torch.from_numpy(data.xy_m).to(device=device, dtype=dtype)  # meters
    xy_s = xy / 1e6  # scaled to O(1)

    y = torch.from_numpy(data.y).to(device=device, dtype=dtype)
    yerr = torch.from_numpy(data.yerr).to(device=device, dtype=dtype)
    mean0 = torch.from_numpy(data.mean0).to(device=device, dtype=dtype)
    Phi = torch.from_numpy(data.Phi).to(device=device, dtype=dtype)

    beta = torch.nn.Parameter(torch.zeros((M,), dtype=dtype, device=device))
    noise_net = build_noise_net(hidden=16).to(device=device, dtype=dtype)
    log_sigma_n0 = torch.nn.Parameter(torch.tensor(-3.0, dtype=dtype, device=device))

    # GP kernel on space only (x,y), with ARD lengthscales.
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)).to(device=device, dtype=dtype)

    params = [beta, log_sigma_n0] + list(noise_net.parameters()) + list(kernel.parameters())
    opt = torch.optim.Adam(params, lr=float(args.lr))

    min_noise = 1e-4
    jitter = 1e-6

    for step in range(int(args.steps)):
        opt.zero_grad(set_to_none=True)

        mean = mean0 + Phi @ beta

        # Dense covariance (N=1322 is OK).
        K = kernel(xy_s).evaluate()

        logn = log_sigma_n0 + noise_net(xy_s).squeeze(-1)
        sigma_n = torch.nn.functional.softplus(logn) + min_noise
        diag = sigma_n * sigma_n
        if args.use_obs_errors:
            diag = diag + yerr * yerr
        K = K + torch.diag(diag)

        nll = nll_cholesky(y, mean, K, jitter=jitter)
        nll.backward()
        opt.step()

        if step % 100 == 0 or step == int(args.steps) - 1:
            with torch.no_grad():
                sf = float(kernel.outputscale.sqrt().cpu())
                ls = kernel.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
                sn0 = float(torch.nn.functional.softplus(log_sigma_n0).cpu())
                print(f"step {step:4d}  nll={float(nll):.4f}  sigma_f={sf:.3g}  ell=({ls[0]:.3g},{ls[1]:.3g})  sigma_n0~{sn0:.3g}")

    ckpt = {
        "backend": "gpytorch-kernel + torch-cholesky-nll",
        "pca_nc": str(args.pca_nc),
        "ages_csv": str(args.ages_csv),
        "max_age_years": float(data.max_age_years),
        "modes": int(M),
        "state": {
            "beta": beta.detach().cpu(),
            "log_sigma_n0": log_sigma_n0.detach().cpu(),
            "noise_net": noise_net.state_dict(),
            "kernel": kernel.state_dict(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()

