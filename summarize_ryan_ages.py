from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


_WGS84_TO_EPSG3413 = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)


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


def _site_summary(
    df: pd.DataFrame,
    *,
    site_col: str,
    age_col: str,
    error_col: str,
    lat_col: str | None = None,
    lon_col: str | None = None,
) -> pd.DataFrame:
    if site_col not in df.columns:
        raise ValueError(f"Input is missing required column: {site_col!r}")
    if age_col not in df.columns:
        raise ValueError(f"Input is missing required column: {age_col!r}")
    if error_col not in df.columns:
        raise ValueError(f"Input is missing required column: {error_col!r}")
    if lat_col is not None and lat_col not in df.columns:
        raise ValueError(f"Input is missing required column: {lat_col!r}")
    if lon_col is not None and lon_col not in df.columns:
        raise ValueError(f"Input is missing required column: {lon_col!r}")

    out = df.copy()
    out[age_col] = pd.to_numeric(out[age_col], errors="coerce")
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

            # Standard error for the (unweighted) mean, propagated from per-observation
            # uncertainties: Var(mean) = sum(sigma_i^2) / n^2.
            err = g.loc[ages.index, error_col]
            err = err.where(err > 0).dropna()
            if int(err.shape[0]) == n:
                std_error = float((err.pow(2).sum() ** 0.5) / n)
                sd_kind = "errors_propagated"
            else:
                # Fall back to estimating from the sample variability if we don't have
                # uncertainties for all observations.
                std_error = float(sample_std / (n**0.5))
                sd_kind = "sample_standard_error"

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
                    "n_errors_used": int(err.shape[0]),
                    "obs_type": "cosmogenic",
                    "quality": "High",
                }
            )
        else:
            # With a single observation, use the provided per-observation uncertainty as the SD.
            sd = g[error_col].iloc[0]
            rows.append(
                {
                    "site": site,
                    "n_obs": n,
                    "lat": lat,
                    "lon": lon,
                    "age_mean": mean,
                    "age_sd": float(sd) if pd.notna(sd) else float("nan"),
                    "age_sd_kind": "reported_sd",
                    "age_sample_std": float("nan"),
                    "age_standard_error": float("nan"),
                    "n_errors_used": 1 if pd.notna(sd) else 0,
                    "obs_type": "cosmogenic",
                    "quality": "High",
                }
            )

    return pd.DataFrame(rows).sort_values("site", kind="mergesort").reset_index(drop=True)


def _carbon_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = {"source", "lat", "lon", "age_mean", "age_std", "quality"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Radiocarbon CSV is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["age_mean"] = pd.to_numeric(out["age_mean"], errors="coerce")
    out["age_std"] = pd.to_numeric(out["age_std"], errors="coerce")

    obs = pd.DataFrame(
        {
            "site": out["source"],
            "n_obs": 1,
            "lat": out["lat"],
            "lon": out["lon"],
            "age_mean": out["age_mean"],
            "age_sd": out["age_std"],
            "age_sd_kind": "reported_sd",
            "age_sample_std": float("nan"),
            "age_standard_error": float("nan"),
            "n_errors_used": out["age_std"].notna().astype(int),
            "obs_type": "radiocarbon",
            "quality": out["quality"],
        }
    )
    return obs.sort_values("site", kind="mergesort").reset_index(drop=True)


def _overall_summary(site_df: pd.DataFrame) -> pd.DataFrame:
    if site_df.empty:
        raise ValueError("No site-level rows available to summarize.")
    means = site_df["age_mean"]
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
            "Group cosmogenic ages by site; when multiple observations, compute standard error "
            "by propagating per-observation uncertainties from the 'errors' column, "
            "otherwise use the single observation and its reported standard deviation. "
            "Optionally appends radiocarbon ages (age_mean/age_std) for a comparable dataset."
        )
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=Path("data/ryan_data/cosmo_ages.csv"),
        help="Input cosmo ages CSV path (default: data/ryan_data/cosmo_ages.csv).",
    )
    parser.add_argument(
        "--carbon",
        dest="carbon_path",
        type=Path,
        default=Path("data/ryan_data/carbon_ages.csv"),
        help="Input radiocarbon ages CSV path (default: data/ryan_data/carbon_ages.csv).",
    )
    parser.add_argument("--no-carbon", action="store_true", help="Do not append radiocarbon observations.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output CSV path for the overall mean/std (single row) across observations.",
    )
    parser.add_argument(
        "--out-sites",
        type=Path,
        default=None,
        help="Optional output CSV path for the combined observation summary (cosmogenic sites + radiocarbon rows).",
    )
    parser.add_argument(
        "--out-observations",
        type=Path,
        default=None,
        help="Alias for --out-sites (preferred).",
    )
    parser.add_argument("--site-col", type=str, default="site", help="Column name for site ID.")
    parser.add_argument("--age-col", type=str, default="ages", help="Column name for ages.")
    parser.add_argument("--error-col", type=str, default="errors", help="Column name for per-observation SD.")
    parser.add_argument("--lat-col", type=str, default="Latitude", help="Cosmo latitude column name.")
    parser.add_argument("--lon-col", type=str, default="Longitude", help="Cosmo longitude column name.")
    args = parser.parse_args()

    cosmo_df = pd.read_csv(args.in_path)
    cosmo_site_df = _site_summary(
        cosmo_df,
        site_col=str(args.site_col),
        age_col=str(args.age_col),
        error_col=str(args.error_col),
        lat_col=str(args.lat_col) if args.lat_col else None,
        lon_col=str(args.lon_col) if args.lon_col else None,
    )
    if args.no_carbon:
        obs_df = cosmo_site_df
    else:
        carbon_df = pd.read_csv(args.carbon_path)
        carbon_df = _carbon_summary(carbon_df)
        obs_df = pd.concat([cosmo_site_df, carbon_df], ignore_index=True)

    obs_df = _add_epsg3413_xy(obs_df, lat_col="lat", lon_col="lon")
    overall_df = _overall_summary(obs_df)
    print(overall_df.to_string(index=False))

    out_obs_path = args.out_observations if args.out_observations is not None else args.out_sites
    if out_obs_path is not None:
        out_obs_path.parent.mkdir(parents=True, exist_ok=True)
        obs_df.to_csv(out_obs_path, index=False)
        print(f"Wrote: {out_obs_path}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        overall_df.to_csv(args.out, index=False)
        print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
