#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot all moraine / ice-marginal landform polyline segments from the PaleoGrIS 1.0 shapefile."
        )
    )
    parser.add_argument(
        "--shp",
        type=Path,
        default=Path(
            "data/leger_data/Geomorphological_database/PaleoGrIS_1.0_ice_marginal_landforms_polylines.shp"
        ),
        help="Input polyline shapefile path (default: PaleoGrIS 1.0 polylines).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output image path (e.g. moraines.png). If omitted, only shows an interactive window.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for saved output image (default: 200).",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=0.15,
        help="Line width for segments (default: 0.15).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Line alpha (default: 0.6).",
    )
    parser.add_argument(
        "--simplify-m",
        type=float,
        default=0.0,
        help="Optional geometry simplification tolerance in meters (default: 0 = no simplify).",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="Optional bounding box filter in the shapefile CRS (EPSG:3413 for PaleoGrIS).",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show interactive plot window (default: true). Use --no-show in batch runs.",
    )
    args = parser.parse_args()

    import geopandas as gpd
    import matplotlib.pyplot as plt

    shp_path = args.shp
    if not shp_path.exists():
        raise FileNotFoundError(shp_path)

    read_kwargs: dict[str, object] = {"engine": "pyogrio"}
    if args.bbox is not None:
        minx, miny, maxx, maxy = (float(v) for v in args.bbox)
        if not (minx < maxx and miny < maxy):
            raise ValueError("--bbox must satisfy MINX<MAXX and MINY<MAXY")
        read_kwargs["bbox"] = (minx, miny, maxx, maxy)

    gdf = gpd.read_file(shp_path, **read_kwargs)
    if gdf.empty:
        raise ValueError("No features loaded (empty GeoDataFrame). Check --bbox or input file.")

    if float(args.simplify_m) > 0:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.simplify(float(args.simplify_m), preserve_topology=True)

    print(gdf.head())

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.set_title(f"Moraine segments ({len(gdf):,} polylines)\n{shp_path.name}")

    gdf.plot(ax=ax, color="black", linewidth=float(args.linewidth), alpha=float(args.alpha))

    # Make large renders faster when saving as vector; matplotlib will rasterize collections.
    for coll in ax.collections:
        coll.set_rasterized(True)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=int(args.dpi))
        print(f"Wrote: {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
