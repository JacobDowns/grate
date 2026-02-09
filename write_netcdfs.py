import numpy as np
from scipy.io import loadmat
import matplotlib.tri as mtri
from matplotlib.tri import CubicTriInterpolator, LinearTriInterpolator
import xarray as xr
from pathlib import Path
import mat73
import matplotlib.pyplot as plt 


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Mesh info (MATLAB v7 file created by export_issm_mesh.m)
mesh_file = Path("issm_outputs/mesh_info.mat")

# Patterns for run files (MATLAB v7.3)
# These will be formatted with run_id (1..8)
thickness_pattern = "issm_outputs/Run_{run_id}_thickness.mat"
bed_pattern       = "issm_outputs/Run_{run_id}_bed.mat"

# Output NetCDF name pattern
output_pattern = "data/issm_run_{run_id}.nc"

# Runs to process
run_ids = [1,2,3,4,5,7,8]  # 1,2,3,4,5,6,7,8

# Grid spacing (meters)
dx = 1000.0 
dy = 1000.0  

# Optional margin around mesh bounding box (meters)
margin = 0.0


# ---------------------------------------------------------------------
# LOAD MESH ONCE
# ---------------------------------------------------------------------

print(f"Loading mesh from {mesh_file} ...")
mesh = loadmat(mesh_file)

x = mesh["x"].ravel()          # (nvertices,)
y = mesh["y"].ravel()          # (nvertices,)
elements = mesh["elements"]    # (nelem, 3), 1-based
lat = mesh["lat"].ravel()       # (nvertices,)
lon = mesh["lon"].ravel()
base_bed = mesh["bed"].ravel()      # (nvertices,)
base_surface = mesh["surface"].ravel()
base_thickness = mesh["thickness"].ravel()
epsg = mesh["epsg"][0]

nvert = x.size
nelem = elements.shape[0]
print(f"Mesh: {nvert} vertices, {nelem} elements")

# Convert to 0-based indices for Python
triangles = elements - 1

triang = mtri.Triangulation(x, y, triangles)


# ---------------------------------------------------------------------
# BUILD REGULAR 2.5 KM GRID ONCE
# ---------------------------------------------------------------------

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

xmin_g, xmax_g = xmin - margin, xmax + margin
ymin_g, ymax_g = ymin - margin, ymax + margin

nx = int(np.floor((xmax_g - xmin_g) / dx)) + 1
ny = int(np.floor((ymax_g - ymin_g) / dy)) + 1

xg = np.linspace(xmin_g, xmax_g, nx)
yg = np.linspace(ymin_g, ymax_g, ny)
Xg, Yg = np.meshgrid(xg, yg)

print("\nGrid specification:")
print(f"  dx, dy   = {dx}, {dy} m")
print(f"  x range  = [{xmin_g:.1f}, {xmax_g:.1f}] -> nx = {nx}")
print(f"  y range  = [{ymin_g:.1f}, {ymax_g:.1f}] -> ny = {ny}")


# ---------------------------------------------------------------------
# HELPER: PROCESS A SINGLE RUN
# ---------------------------------------------------------------------

def process_run(run_id: int):
    print(f"\n=== Processing run {run_id} ===")

    thickness_file = Path(thickness_pattern.format(run_id=run_id))
    bed_file       = Path(bed_pattern.format(run_id=run_id))
    output_nc      = Path(output_pattern.format(run_id=run_id))

    if not thickness_file.exists():
        raise FileNotFoundError(f"{thickness_file} not found")
    if not bed_file.exists():
        raise FileNotFoundError(f"{bed_file} not found")

    print(f"Loading thickness from {thickness_file} ...")
    thk_data = mat73.loadmat(thickness_file)
    print(thk_data)
    quit()

    print(f"Loading bed from {bed_file} ...")
    bed_data = mat73.loadmat(bed_file)

    # Adjust key names here if your variables are nested differently
    if "icethickness" not in thk_data:
        raise KeyError(
            f"'thickness' variable not found in {thickness_file}. "
            f"Available keys: {list(thk_data.keys())}"
        )
    if "bed" not in bed_data:
        raise KeyError(
            f"'bed' variable not found in {bed_file}. "
            f"Available keys: {list(bed_data.keys())}"
        )

    thickness = np.asarray(thk_data["icethickness"])  # (ntime, nvertices)
    bed       = np.asarray(bed_data["bed"])        # (ntime, nvertices)

    ntime, nvert_thk = thickness.shape
    ntime_bed, nvert_bed = bed.shape

    print(f"  thickness shape: {thickness.shape}")
    print(f"  bed shape:       {bed.shape}")

    if nvert_thk != nvert or nvert_bed != nvert:
        raise ValueError(
            f"Vertex count mismatch for run {run_id}: "
            f"mesh has {nvert}, thickness has {nvert_thk}, bed has {nvert_bed}"
        )
    if ntime_bed != ntime:
        raise ValueError(
            f"Time dimension mismatch for run {run_id}: "
            f"thickness has {ntime}, bed has {ntime_bed}"
        )

    ntime -= 1
    # Allocate output arrays for this run
    thickness_grid = np.empty((ntime, ny, nx), dtype=np.float32)
    bed_grid       = np.empty((ntime, ny, nx), dtype=np.float32)

    f_thk = LinearTriInterpolator(triang, base_thickness)
    f_bed = LinearTriInterpolator(triang, base_bed)

    print(f"Interpolating to grid for run {run_id} ...")
    for it in range(ntime):
        if it % 10 == 0 or it == ntime - 1:
            print(f"  timestep {it+1}/{ntime}")

        thk_t = thickness[it, :]
        bed_t = bed[it, :]


        f_thk = LinearTriInterpolator(triang, thk_t)
        f_bed = LinearTriInterpolator(triang, bed_t)

        thk_grid_t = f_thk(Xg, Yg)  # masked array (ny, nx)
        bed_grid_t = f_bed(Xg, Yg)  # masked array (ny, nx)

        thickness_grid[it] = thk_grid_t.filled(np.nan)
        bed_grid[it]       = bed_grid_t.filled(np.nan)



    print(f"Saving NetCDF for run {run_id} to {output_nc} ...")

    time = np.arange(ntime)  # replace with real time values if you have them

    ds = xr.Dataset(
        data_vars=dict(
            thickness=(("time", "y", "x"), thickness_grid),
            bed      =(("time", "y", "x"), bed_grid),
        ),
        coords=dict(
            time=("time", time),
            x=("x", xg),
            y=("y", yg),
        ),
        attrs=dict(
            description=f"ISSM run {run_id}: thickness & bed interpolated to 2.5 km grid",
            dx_m=dx,
            dy_m=dy,
            epsg=int(epsg),
        ),
    )

    engine = None
    try:
        import netCDF4  # noqa: F401

        engine = "netcdf4"
    except ImportError:
        try:
            import h5netcdf  # noqa: F401

            engine = "h5netcdf"
        except ImportError:
            pass

    if engine is None:
        raise RuntimeError(
            "Saving requires either netCDF4 or h5netcdf to avoid classic NetCDF size limits."
        )

    ds.to_netcdf(output_nc, engine=engine)
    print(f"Run {run_id} done.")


# ---------------------------------------------------------------------
# MAIN LOOP OVER RUNS
# ---------------------------------------------------------------------

if __name__ == "__main__":
    for run_id in run_ids:
        process_run(run_id)

    print("\nAll runs processed.")
