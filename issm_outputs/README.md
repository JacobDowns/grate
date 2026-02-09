# ISSM Data Extraction Tool

This folder contains utilities to inspect and extract ISSM (Ice-sheet and Sea-level System Model) `.mat` outputs.

The current extractor is designed for the newer run outputs (`run_*.mat`) that are MATLAB v7.3 (HDF5) and expose
time-series fields under `/run/tot` (equivalent to `data.run.tot` in MATLAB).

## Installation

```bash
pip install numpy h5py
```

## Usage

### Inspect a run file (fast, no full load)

```bash
python3 ../inspect_run_01.py --max-depth 2 --max-children 20
```

### Extract all runs to per-run HDF5 files

```bash
python3 extract_issm_data.py --input-dir . --pattern 'run_*.mat' --output-dir ../data/issm_extracted
```

### Options

- `--pattern`: Which run files to process (default: `run_*.mat`)
- `--fields`: Comma-separated fields to extract from `/run/tot`
- `--max-steps`: Only extract the first N time steps (useful for testing)

## Output format

For each input `run_XX.mat`, the extractor writes `run_XX_tot.h5` containing:

- `time`: 1D float64 array of length `nt`
- one dataset per extracted field (e.g. `Thickness`, `Vx`, `IceVolume`, ...), each shaped `(nt, ...)`

## Loading the extracted data in Python

```python
import h5py

f = h5py.File("data/issm_extracted/run_01_tot.h5", "r")
print(list(f.keys()))

time = f["time"][:]
thickness = f["Thickness"][:]   # shape (nt, nvert) (or similar)
vx = f["Vx"][:]                 # shape (nt, nvert)
ice_volume = f["IceVolume"][:]  # shape (nt,)
```

