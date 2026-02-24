#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _to_float(x: str) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity-check that radiocarbon uncertainties in combined_ages.csv are 1σ values derived "
            "from the 2σ uncertainties in carbon_ages.csv."
        )
    )
    parser.add_argument(
        "--carbon",
        type=Path,
        default=Path("data/age_data/carbon_ages.csv"),
        help="Input radiocarbon ages CSV path (default: data/age_data/carbon_ages.csv).",
    )
    parser.add_argument(
        "--combined",
        type=Path,
        default=Path("data/age_data/combined_ages.csv"),
        help="Input combined ages CSV path (default: data/age_data/combined_ages.csv).",
    )
    parser.add_argument(
        "--carbon-site-col",
        type=str,
        default="sample_name",
        help="Site/sample id column in carbon CSV (default: sample_name).",
    )
    parser.add_argument(
        "--carbon-2sigma-col",
        type=str,
        default="age_std",
        help="2σ uncertainty column in carbon CSV (default: age_std).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Absolute tolerance for float comparisons (default: 1e-9).",
    )
    args = parser.parse_args()

    carbon_2sigma: dict[str, list[float]] = {}
    with args.carbon.open(newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise SystemExit(f"Missing header: {args.carbon}")
        for row in r:
            site = (row.get(args.carbon_site_col) or "").strip()
            if not site:
                continue
            v = _to_float(row.get(args.carbon_2sigma_col, ""))
            if v is None:
                continue
            carbon_2sigma.setdefault(site, []).append(v)

    checked = 0
    missing = 0
    mismatched = 0
    bad_kind = 0

    with args.combined.open(newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise SystemExit(f"Missing header: {args.combined}")
        for row in r:
            if (row.get("obs_type") or "").strip().lower() != "radiocarbon":
                continue

            site = (row.get("site") or "").strip()
            if not site:
                continue

            v2_list = carbon_2sigma.get(site)
            if not v2_list:
                missing += 1
                continue

            v1 = _to_float(row.get("age_sd", ""))
            if v1 is None:
                mismatched += 1
                continue

            checked += 1
            if not any(abs(v1 - (v2 / 2.0)) <= float(args.tolerance) for v2 in v2_list):
                mismatched += 1

            kind = (row.get("age_sd_kind") or "").strip()
            if kind not in {"reported_2sigma_scaled_to_1sigma", "reported_2sigma_scaled_to_1σ"}:
                bad_kind += 1

    print(
        "Radiocarbon 1σ checks:",
        f"checked={checked}",
        f"missing_in_carbon={missing}",
        f"mismatched_sd={mismatched}",
        f"bad_sd_kind={bad_kind}",
    )

    if missing or mismatched or bad_kind:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
