#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingData:
    xy_m: "np.ndarray"  # (N,2) in meters (EPSG:3413)
    y: "np.ndarray"  # (N,) normalized 0..1
    yerr: "np.ndarray"  # (N,) normalized 0..1 (optional; can be zeros)
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

    # Avoid divide-by-zero if grid is degenerate (shouldn't happen).
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
    else:
        # Keep but set to NaN so the caller can decide; default is to drop.
        pass

    xy_m = np.stack([x, y], axis=1).astype(np.float64)

    return TrainingData(
        xy_m=xy_m,
        y=y_norm,
        yerr=yerr_norm,
        mean0=mean0,
        Phi=Phi,
        max_age_years=max_age_years,
    )


def build_noise_net(hidden: int = 16):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(2, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1),
    )


def rbf_kernel_aniso(xy: "torch.Tensor", *, log_sigma: "torch.Tensor", log_ell: "torch.Tensor") -> "torch.Tensor":
    """
    xy: (N,2), float64
    log_sigma: scalar
    log_ell: (2,) for x/y lengthscales in same units as xy
    returns: (N,N)
    """
    import torch

    sigma2 = torch.exp(2.0 * log_sigma)
    ell = torch.exp(log_ell)  # (2,)

    dx = (xy[:, None, :] - xy[None, :, :]) / ell[None, None, :]
    d2 = (dx * dx).sum(dim=-1)
    return sigma2 * torch.exp(-0.5 * d2)


def nll_gp(
    *,
    y: "torch.Tensor",
    mean: "torch.Tensor",
    K: "torch.Tensor",
    jitter: float,
    max_tries: int = 6,
) -> tuple["torch.Tensor", float]:
    import torch

    n = y.shape[0]
    eye = torch.eye(n, dtype=K.dtype, device=K.device)

    nug = float(jitter)
    last_err = None
    for _ in range(max_tries):
        try:
            Kj = K + nug * eye
            L = torch.linalg.cholesky(Kj)
            r = (y - mean).unsqueeze(1)  # (N,1)
            alpha = torch.cholesky_solve(r, L)  # (N,1)
            quad = (r * alpha).sum()
            logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
            nll = 0.5 * (quad + logdet + n * torch.log(torch.tensor(2.0 * torch.pi, dtype=K.dtype, device=K.device)))
            return nll, nug
        except RuntimeError as e:
            last_err = e
            nug *= 10.0
    raise RuntimeError(f"Cholesky failed even with jitter={nug:g}. Last error: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GP for deglaciation age with PCA-parameterized mean (PyTorch).")
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
    parser.add_argument("--no-drop-oob", action="store_true", help="Do not drop out-of-grid points (not recommended)")
    parser.add_argument("--use-obs-errors", action="store_true", help="Include CSV errors as fixed additional noise")
    parser.add_argument("--device", type=str, default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--out", type=Path, default=Path("data/gp_deglaciation_fit.pt"), help="Output torch checkpoint")
    args = parser.parse_args()

    import numpy as np
    import torch

    data = load_training_data(
        pca_nc=args.pca_nc,
        ages_csv=args.ages_csv,
        n_modes=int(args.modes),
        clip_ages=bool(args.clip_ages),
        drop_oob=not bool(args.no_drop_oob),
    )

    N, M = data.Phi.shape
    print(f"Training points: N={N}, modes in mean: M={M}, max_age_years={data.max_age_years:g}")

    device = torch.device(args.device)
    dtype = torch.float64

    xy = torch.from_numpy(data.xy_m).to(device=device, dtype=dtype)  # meters
    # Scale coords for stability (hundreds of km).
    xy_s = xy / 1e6

    y = torch.from_numpy(data.y).to(device=device, dtype=dtype)
    yerr = torch.from_numpy(data.yerr).to(device=device, dtype=dtype)
    mean0 = torch.from_numpy(data.mean0).to(device=device, dtype=dtype)
    Phi = torch.from_numpy(data.Phi).to(device=device, dtype=dtype)

    beta = torch.nn.Parameter(torch.zeros((M,), dtype=dtype, device=device))
    log_sigma_f = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype, device=device))
    log_ell = torch.nn.Parameter(torch.tensor([0.0, 0.0], dtype=dtype, device=device))  # lengthscales in scaled units

    noise_net = build_noise_net(hidden=16).to(device=device, dtype=torch.float64)
    log_sigma_n0 = torch.nn.Parameter(torch.tensor(-3.0, dtype=dtype, device=device))

    params = [beta, log_sigma_f, log_ell, log_sigma_n0] + list(noise_net.parameters())
    opt = torch.optim.Adam(params, lr=float(args.lr))

    jitter = 1e-6
    min_noise = 1e-4

    for step in range(int(args.steps)):
        opt.zero_grad(set_to_none=True)

        mean = mean0 + Phi @ beta

        K = rbf_kernel_aniso(xy_s, log_sigma=log_sigma_f, log_ell=log_ell)

        # Heteroskedastic noise: sigma_space(x)^2 (+ obs errors^2)
        logn = log_sigma_n0 + noise_net(xy_s).squeeze(-1)
        sigma_n = torch.nn.functional.softplus(logn) + min_noise
        diag = sigma_n * sigma_n
        if args.use_obs_errors:
            diag = diag + yerr * yerr
        K = K + torch.diag(diag)

        nll, used_jitter = nll_gp(y=y, mean=mean, K=K, jitter=jitter)
        nll.backward()
        opt.step()

        if step % 100 == 0 or step == int(args.steps) - 1:
            with torch.no_grad():
                print(
                    f"step {step:4d}  nll={float(nll):.4f}  jitter={used_jitter:.1e}  "
                    f"sigma_f={float(torch.exp(log_sigma_f)):.3g}  "
                    f"ell=({float(torch.exp(log_ell[0])):.3g},{float(torch.exp(log_ell[1])):.3g})  "
                    f"sigma_n0~{float(torch.nn.functional.softplus(log_sigma_n0)):.3g}"
                )

    ckpt = {
        "pca_nc": str(args.pca_nc),
        "ages_csv": str(args.ages_csv),
        "max_age_years": float(data.max_age_years),
        "modes": int(M),
        "state": {
            "beta": beta.detach().cpu(),
            "log_sigma_f": log_sigma_f.detach().cpu(),
            "log_ell": log_ell.detach().cpu(),
            "log_sigma_n0": log_sigma_n0.detach().cpu(),
            "noise_net": noise_net.state_dict(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
