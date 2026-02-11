# Grate

A Python project for Greenland deglaciation analysis and modeling.

## Installation

```bash
uv sync
```

## Usage

The project includes various scripts for glacier analysis:
- `compute_sdfs.py` - Compute signed distance functions
- `do_pca.py` - Principal component analysis
- `train.py` - Training models
- `write_netcdfs.py` - NetCDF file operations
- `csv_to_shapefile.py` - Convert `data/age_data.csv` to a QGIS-loadable shapefile
- `nc_deglaciation_age_to_geotiff.py` - Export per-run `deglaciation_age` GeoTIFFs + an across-run mean GeoTIFF
