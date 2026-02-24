#!/usr/bin/env python3
import argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString
import torch 
import torch.nn as nn

class SecondOrderPolyMean(nn.Module):
    def __init__(self, input_dim):
        super(SecondOrderPolyMean, self).__init__()
        # Number of terms: linear (D) + squared (D) + cross terms (D * (D - 1) / 2)
        self.poly_dim = input_dim * 2 + (input_dim * (input_dim - 1)) // 2
        self.linear = nn.Linear(self.poly_dim, 1)

    def forward(self, x):
        # x shape is (..., D). We support arbitrary batch shapes for GPyTorch.
        input_dim = x.shape[-1]
        
        terms = [x, x ** 2]
        
        # Calculate cross-interaction terms (e.g., x1*x2, x1*x3...)
        cross_terms = []
        for i in range(input_dim):
            for j in range(i + 1, input_dim):
                cross_terms.append((x[..., i] * x[..., j]).unsqueeze(-1))
                
        if cross_terms:
            terms.append(torch.cat(cross_terms, dim=-1))
            
        poly_x = torch.cat(terms, dim=-1)
        return self.linear(poly_x).squeeze(-1)

def extract_tangents(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    records = []
    
    for idx, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue
            
        if isinstance(geom, LineString):
            lines = [geom]
        elif isinstance(geom, MultiLineString):
            lines = list(geom.geoms)
        else:
            continue
            
        for line in lines:
            if line.length <= 0:
                continue
                
            # 1. Get the exact midpoint
            midpoint = line.interpolate(0.5, normalized=True)
            
            # 2. Sample to get the local derivative
            p_before = line.interpolate(0.49, normalized=True)
            p_after = line.interpolate(0.51, normalized=True)
            
            # 3. Calculate tangent (dx, dy)
            dx = p_after.x - p_before.x
            dy = p_after.y - p_before.y
            
            # 4. Normalize to a unit vector
            magnitude = np.hypot(dx, dy)
            if magnitude == 0:
                continue
                
            v_x = dx / magnitude
            v_y = dy / magnitude
            
            records.append({
                "id": gdf.iloc[idx].get("id", idx),
                "x_3413": midpoint.x,
                "y_3413": midpoint.y,
                "vx": v_x,
                "vy": v_y,
                "segment_length_m": line.length
            })
            
    return pd.DataFrame(records)

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and plot tangent vectors from moraines.")
    parser.add_argument(
        "--shp", type=Path, 
        default=Path("data/leger_data/Geomorphological_database/PaleoGrIS_1.0_ice_marginal_landforms_polylines.shp")
    )
    parser.add_argument(
        "--out_csv", type=Path, 
        default=Path("data/age_data/moraine_tangents.csv"),
    )
    parser.add_argument(
        "--show", action=argparse.BooleanOptionalAction, default=True,
        help="Show the interactive vector plot."
    )
    args = parser.parse_args()

    print(f"Reading shapefile: {args.shp}")
    gdf = gpd.read_file(args.shp, engine="pyogrio")
    
    print("Extracting vectors...")
    tangent_df = extract_tangents(gdf)
    
    min_length = 1000.0
    valid_df = tangent_df[tangent_df["segment_length_m"] > min_length]
    print(f"Extracted {len(valid_df):,} valid tangent vectors.")
    
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    valid_df.to_csv(args.out_csv, index=False)
    print(f"Saved CSV to: {args.out_csv}")

    # --- Plotting ---
    if args.show:
        print("Generating plot... (This may take a moment for many segments)")
        fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
        ax.set_title(f"Moraine Segments and Tangent Vectors\n({len(valid_df):,} extracted midpoints)")

        # Plot the original moraine lines underneath
        gdf.plot(ax=ax, color="black", linewidth=0.5, alpha=0.5)

        # Overlay the tangent vectors using a quiver plot
        # pivot='mid' centers the arrow exactly on the midpoint coordinate
        ax.quiver(
            valid_df["x_3413"], valid_df["y_3413"], 
            valid_df["vx"], valid_df["vy"], 
            color="red", 
            pivot="mid", 
            width=0.002,      # Make arrows thin enough to read
            headwidth=3,      # Arrowhead proportions
            scale=50          # Adjusts the visual length of the unit vectors on the map
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (EPSG:3413)")
        ax.set_ylabel("y (EPSG:3413)")
        
        print("Displaying plot. Zoom in to see individual tangent arrows!")
        plt.show()

if __name__ == "__main__":
    main()