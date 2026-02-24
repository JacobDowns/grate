#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


_WGS84_TO_EPSG3413 = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)


def _pick_col(df: pd.DataFrame, candidates: list[str], *, label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find {label} column. Tried: {candidates}. Available: {list(df.columns)}")


def _add_epsg3413_xy(df: pd.DataFrame, *, lat_col: str = "lat", lon_col: str = "lon") -> pd.DataFrame:
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Expected columns {lat_col!r} and {lon_col!r} to compute EPSG:3413 coordinates.")

    out = df.copy()
    lat = pd.to_numeric(out[lat_col], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(out[lon_col], errors="coerce").to_numpy(dtype=float)

    x = np.full(lat.shape, np.nan, dtype=float)
    y = np.full(lat.shape, np.nan, dtype=float)
    valid = np.isfinite(lat) & np.isfinite(lon)
    if np.any(valid):
        xv, yv = _WGS84_TO_EPSG3413.transform(lon[valid], lat[valid])
        x[valid] = np.asarray(xv, dtype=float)
        y[valid] = np.asarray(yv, dtype=float)

    out["x_3413"] = x
    out["y_3413"] = y
    return out


def summarize_cosmo_sites(
    df: pd.DataFrame,
    *,
    site_col: str,
    age_col: str,
    error_col: str | None,
    lat_col: str | None,
    lon_col: str | None,
    default_quality: str,
) -> pd.DataFrame:
    if site_col not in df.columns:
        raise ValueError(f"Cosmo CSV is missing required column: {site_col!r}")
    if age_col not in df.columns:
        raise ValueError(f"Cosmo CSV is missing required column: {age_col!r}")
    if error_col is not None and error_col not in df.columns:
        raise ValueError(f"Cosmo CSV is missing required column: {error_col!r}")
    if lat_col is not None and lat_col not in df.columns:
        raise ValueError(f"Cosmo CSV is missing required column: {lat_col!r}")
    if lon_col is not None and lon_col not in df.columns:
        raise ValueError(f"Cosmo CSV is missing required column: {lon_col!r}")

    out = df.copy()
    out[age_col] = pd.to_numeric(out[age_col], errors="coerce")
    if error_col is not None:
        out[error_col] = pd.to_numeric(out[error_col], errors="coerce")
    if lat_col is not None:
        out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    if lon_col is not None:
        out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")

    rows: list[dict[str, object]] = []
    for site, g in out.groupby(site_col, dropna=False):
        ages = g[age_col].dropna()
        n = int(ages.shape[0])
        if n == 0:
            continue

        mean = float(ages.mean())
        lat = float(g[lat_col].mean()) if lat_col is not None else float("nan")
        lon = float(g[lon_col].mean()) if lon_col is not None else float("nan")

        if n > 1:
            sample_std = float(ages.std(ddof=1))

            # Preferred: propagate per-observation uncertainties to get std error.
            # Var(mean) = sum(sigma_i^2) / n^2.
            if error_col is not None:
                err = g.loc[ages.index, error_col]
                err = err.where(err > 0).dropna()
            else:
                err = pd.Series(dtype=float)

            if int(err.shape[0]) == n:
                std_error = float((err.pow(2).sum() ** 0.5) / n)
                sd_kind = "errors_propagated"
                n_errors_used = n
            else:
                std_error = float(sample_std / (n**0.5))
                sd_kind = "sample_standard_error"
                n_errors_used = int(err.shape[0])

            rows.append(
                {
                    "site": site,
                    "n_obs": n,
                    "lat": lat,
                    "lon": lon,
                    "age_mean": mean,
                    "age_sd": float(std_error),
                    "age_sd_kind": sd_kind,
                    "age_sample_std": sample_std,
                    "age_standard_error": float(std_error),
                    "n_errors_used": n_errors_used,
                    "obs_type": "cosmogenic",
                    "quality": default_quality,
                }
            )
        else:
            # With a single observation, use the reported per-observation uncertainty when available.
            if error_col is not None:
                sd = g[error_col].iloc[0]
            else:
                sd = float("nan")
            rows.append(
                {
                    "site": site,
                    "n_obs": n,
                    "lat": lat,
                    "lon": lon,
                    "age_mean": mean,
                    "age_sd": float(sd) if pd.notna(sd) else float("nan"),
                    "age_sd_kind": "reported_sd" if pd.notna(sd) else "missing_sd",
                    "age_sample_std": float("nan"),
                    "age_standard_error": float("nan"),
                    "n_errors_used": 1 if pd.notna(sd) else 0,
                    "obs_type": "cosmogenic",
                    "quality": default_quality,
                }
            )

    return pd.DataFrame(rows).sort_values("site", kind="mergesort").reset_index(drop=True)


def summarize_radiocarbon_rows(
    df: pd.DataFrame,
    *,
    site_col: str,
    lat_col: str,
    lon_col: str,
    age_col: str,
    sd_col: str,
    sd_sigma: float,
    quality_col: str | None,
    default_quality: str,
) -> pd.DataFrame:
    required = {site_col, lat_col, lon_col, age_col, sd_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Radiocarbon CSV is missing required columns: {sorted(missing)}")

    out = df.copy()
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out[age_col] = pd.to_numeric(out[age_col], errors="coerce")
    out[sd_col] = pd.to_numeric(out[sd_col], errors="coerce")

    if not np.isfinite(sd_sigma) or sd_sigma <= 0:
        raise ValueError(f"Expected radiocarbon sd_sigma to be a positive finite number; got {sd_sigma!r}")

    # Radiocarbon errors in our source CSV are reported as N-sigma (commonly 2σ). Convert to 1σ.
    sd_1sigma = out[sd_col] / float(sd_sigma)

    if quality_col is not None and quality_col in out.columns:
        quality = out[quality_col]
    else:
        quality = pd.Series([default_quality] * int(out.shape[0]))

    if float(sd_sigma) == 1.0:
        sd_kind = "reported_sd"
    else:
        sigma_label = f"{sd_sigma:g}"
        sd_kind = f"reported_{sigma_label}sigma_scaled_to_1sigma"

    obs = pd.DataFrame(
        {
            "site": out[site_col],
            "n_obs": 1,
            "lat": out[lat_col],
            "lon": out[lon_col],
            "age_mean": out[age_col],
            "age_sd": sd_1sigma,
            "age_sd_kind": sd_kind,
            "age_sample_std": float("nan"),
            "age_standard_error": float("nan"),
            "n_errors_used": sd_1sigma.notna().astype(int),
            "obs_type": "radiocarbon",
            "quality": quality,
        }
    )
    return obs.sort_values("site", kind="mergesort").reset_index(drop=True)


def overall_summary(site_df: pd.DataFrame) -> pd.DataFrame:
    if site_df.empty:
        raise ValueError("No rows available to summarize.")
    means = pd.to_numeric(site_df["age_mean"], errors="coerce")
    means = means.dropna()
    return pd.DataFrame(
        [
            {
                "mean": float(means.mean()),
                "std": float(means.std(ddof=1)) if means.shape[0] > 1 else float("nan"),
                "n_obs": int(means.shape[0]),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine cosmogenic ages (grouped by site) and radiocarbon ages (per-row) into a single "
            "observation CSV compatible with scripts expecting columns like age_mean/age_sd and x_3413/y_3413."
        )
    )
    parser.add_argument(
        "--cosmo",
        type=Path,
        default=Path("data/age_data/cosmo_ages.csv"),
        help="Input cosmogenic ages CSV path (default: data/age_data/cosmo_ages.csv).",
    )
    parser.add_argument(
        "--carbon",
        type=Path,
        default=Path("data/age_data/carbon_ages.csv"),
        help="Input radiocarbon ages CSV path (default: data/age_data/carbon_ages.csv).",
    )
    parser.add_argument("--no-carbon", action="store_true", help="Do not append radiocarbon observations.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/age_data/combined_ages.csv"),
        help="Output combined observation CSV path (default: data/age_data/combined_ages.csv).",
    )
    parser.add_argument("--cosmo-site-col", type=str, default="site", help="Cosmo site column name.")
    parser.add_argument("--cosmo-age-col", type=str, default="ages", help="Cosmo age column name.")
    parser.add_argument(
        "--cosmo-error-col",
        type=str,
        default="errors",
        help="Cosmo per-observation SD column name (default: errors). Use '' to disable.",
    )
    parser.add_argument("--cosmo-lat-col", type=str, default="Latitude", help="Cosmo latitude column name.")
    parser.add_argument("--cosmo-lon-col", type=str, default="Longitude", help="Cosmo longitude column name.")
    parser.add_argument("--cosmo-quality", type=str, default="High", help="Quality label to assign to cosmo sites.")
    parser.add_argument(
        "--carbon-site-col",
        type=str,
        default=None,
        help="Radiocarbon site column name (default: auto-detect sample_name/source).",
    )
    parser.add_argument("--carbon-lat-col", type=str, default="lat", help="Radiocarbon latitude column name.")
    parser.add_argument("--carbon-lon-col", type=str, default="lon", help="Radiocarbon longitude column name.")
    parser.add_argument("--carbon-age-col", type=str, default="age_mean", help="Radiocarbon age mean column name.")
    parser.add_argument("--carbon-sd-col", type=str, default="age_std", help="Radiocarbon SD column name.")
    parser.add_argument(
        "--carbon-sd-sigma",
        type=float,
        default=2.0,
        help=(
            "Sigma level of radiocarbon uncertainties in the input CSV (default: 2.0). "
            "Set to 1.0 if your radiocarbon errors are already 1σ."
        ),
    )
    parser.add_argument(
        "--carbon-quality-col",
        type=str,
        default="quality",
        help="Radiocarbon quality column name (default: quality). Use '' to disable.",
    )
    parser.add_argument("--carbon-default-quality", type=str, default="Mid", help="Fallback quality for carbon rows.")
    args = parser.parse_args()

    cosmo_df = pd.read_csv(args.cosmo)
    cosmo_error_col = str(args.cosmo_error_col).strip()
    cosmo_sites = summarize_cosmo_sites(
        cosmo_df,
        site_col=str(args.cosmo_site_col),
        age_col=str(args.cosmo_age_col),
        error_col=cosmo_error_col if cosmo_error_col else None,
        lat_col=str(args.cosmo_lat_col) if str(args.cosmo_lat_col).strip() else None,
        lon_col=str(args.cosmo_lon_col) if str(args.cosmo_lon_col).strip() else None,
        default_quality=str(args.cosmo_quality),
    )

    if args.no_carbon:
        combined = cosmo_sites
    else:
        carbon_df = pd.read_csv(args.carbon)
        carbon_site_col = args.carbon_site_col
        if carbon_site_col is None:
            carbon_site_col = _pick_col(carbon_df, ["sample_name", "source", "site", "sample_id"], label="radiocarbon site")

        carbon_quality_col = str(args.carbon_quality_col).strip()
        carbon_sites = summarize_radiocarbon_rows(
            carbon_df,
            site_col=str(carbon_site_col),
            lat_col=str(args.carbon_lat_col),
            lon_col=str(args.carbon_lon_col),
            age_col=str(args.carbon_age_col),
            sd_col=str(args.carbon_sd_col),
            sd_sigma=float(args.carbon_sd_sigma),
            quality_col=str(args.carbon_quality_col) if carbon_quality_col else None,
            default_quality=str(args.carbon_default_quality),
        )
        combined = pd.concat([cosmo_sites, carbon_sites], ignore_index=True)

    combined = _add_epsg3413_xy(combined, lat_col="lat", lon_col="lon")
    stats = overall_summary(combined)
    print(stats.to_string(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
