#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def _import_or_explain(module: str, install_hint: str):
    try:
        return __import__(module)
    except ModuleNotFoundError as e:
        if e.name != module:
            raise
        print(
            f"Missing dependency: {module}\n\n"
            "Install dependencies, then re-run. For this repo, the simplest is:\n"
            f"{install_hint}\n",
            file=sys.stderr,
        )
        raise


def _detect_lat_lon_columns(columns: list[str]) -> tuple[str, str]:
    cols = {c.lower().strip(): c for c in columns}

    lat_candidates = ["lat", "latitude", "y"]
    lon_candidates = ["lon", "lng", "long", "longitude", "x"]

    lat_col = next((cols[c] for c in lat_candidates if c in cols), None)
    lon_col = next((cols[c] for c in lon_candidates if c in cols), None)
    if not lat_col or not lon_col:
        raise SystemExit(
            "Could not auto-detect latitude/longitude columns.\n"
            f"Columns: {', '.join(columns)}\n"
            "Pass --lat-col and --lon-col explicitly."
        )
    return lat_col, lon_col


def _sanitize_field_names(columns: list[str]) -> tuple[dict[str, str], list[str]]:
    used: set[str] = set()
    mapping: dict[str, str] = {}
    sanitized: list[str] = []

    for original in columns:
        name = original.strip()
        safe = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()
        if not safe:
            safe = "field"
        if safe[0].isdigit():
            safe = f"f_{safe}"

        # Shapefile field names are typically limited to 10 chars.
        base = safe[:10]
        candidate = base
        i = 1
        while candidate in used:
            suffix = f"_{i}"
            candidate = (base[: max(1, 10 - len(suffix))] + suffix)[:10]
            i += 1

        used.add(candidate)
        mapping[original] = candidate
        sanitized.append(candidate)

    return mapping, sanitized


def _resolve_output_path(output: str | None, input_csv: Path) -> Path:
    if output is None:
        out_dir = input_csv.parent / "age_data_shapefile"
        return out_dir / f"{input_csv.stem}.shp"

    out = Path(output)
    if str(output).endswith(os.sep) or (out.exists() and out.is_dir()) or out.suffix == "":
        out.mkdir(parents=True, exist_ok=True)
        return out / f"{input_csv.stem}.shp"
    return out


def _delete_existing_shapefile_set(shp_path: Path) -> None:
    base = shp_path.with_suffix("")
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj", ".sbn", ".sbx"]:
        p = base.with_suffix(ext)
        if p.exists():
            p.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a CSV with lat/lon columns into an ESRI Shapefile for QGIS."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="data/age_data.csv",
        help="Input CSV path (default: data/age_data.csv).",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help=(
            "Output .shp path OR output directory. "
            "Default: data/age_data_shapefile/<input_stem>.shp"
        ),
    )
    parser.add_argument("--lat-col", default=None, help="Latitude column name (default: auto-detect).")
    parser.add_argument("--lon-col", default=None, help="Longitude column name (default: auto-detect).")
    parser.add_argument("--crs", default="EPSG:4326", help="CRS for the coordinates (default: EPSG:4326).")
    parser.add_argument("--encoding", default="utf-8", help="DBF encoding (default: utf-8).")
    parser.add_argument(
        "--fieldmap-json",
        default=None,
        help="Write a JSON mapping of original->shapefile field names (default: alongside output).",
    )
    parser.add_argument(
        "--keep-original-fields",
        action="store_true",
        help="Do not sanitize/truncate column names (may break shapefile writing).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing shapefile components at the output location.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        print(f"Input CSV not found: {input_csv}", file=sys.stderr)
        return 2

    output_shp = _resolve_output_path(args.output, input_csv)
    output_shp.parent.mkdir(parents=True, exist_ok=True)

    # Import geopandas/pandas with a repo-appropriate hint.
    install_hint = (
        "  uv sync\n"
        "  uv run python csv_to_shapefile.py\n"
        "\n"
        "Or (conda):\n"
        "  conda env create -f environment.yml\n"
        "  conda activate grate\n"
        "  python csv_to_shapefile.py\n"
    )
    _import_or_explain("pandas", install_hint)
    _import_or_explain("geopandas", install_hint)

    import pandas as pd  # noqa: E402
    import geopandas as gpd  # noqa: E402

    df = pd.read_csv(input_csv)

    lat_col, lon_col = (
        (args.lat_col, args.lon_col)
        if args.lat_col and args.lon_col
        else _detect_lat_lon_columns(list(df.columns))
    )

    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[lat_col, lon_col])
    dropped = before - len(df)

    if not args.keep_original_fields:
        non_geom_cols = [c for c in df.columns]
        mapping, _ = _sanitize_field_names(non_geom_cols)
        df = df.rename(columns=mapping)
        lat_col = mapping[lat_col]
        lon_col = mapping[lon_col]

        fieldmap_path = (
            Path(args.fieldmap_json)
            if args.fieldmap_json is not None
            else output_shp.with_suffix(".fields.json")
        )
        fieldmap_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=args.crs,
    )

    if output_shp.exists() and not args.force:
        print(
            f"Output already exists: {output_shp}\n"
            "Re-run with --force to overwrite.",
            file=sys.stderr,
        )
        return 2
    if args.force:
        _delete_existing_shapefile_set(output_shp)

    gdf.to_file(output_shp, driver="ESRI Shapefile", encoding=args.encoding)

    print(f"Wrote shapefile: {output_shp}")
    print(f"Features: {len(gdf)} (dropped {dropped} rows missing lat/lon)")
    print(f"CRS: {gdf.crs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
