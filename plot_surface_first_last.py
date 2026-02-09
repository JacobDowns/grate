#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Mesh2D:
    x: "np.ndarray"  # (nvert,)
    y: "np.ndarray"  # (nvert,)
    triangles: "np.ndarray"  # (ntri, 3) int, 0-based
    epsg: int | None = None


def _ravel_col(v) -> "np.ndarray":
    import numpy as np

    arr = np.asarray(v)
    return arr.reshape(-1)


def load_mesh(mesh_file: Path) -> Mesh2D:
    import numpy as np
    from scipy.io import loadmat

    mesh = loadmat(mesh_file, squeeze_me=False, struct_as_record=False)
    keys = {k for k in mesh.keys() if not k.startswith("__")}
    required = {"x", "y", "elements"}
    missing = sorted(required - keys)
    if missing:
        raise KeyError(f"{mesh_file} missing keys: {missing}. Available keys: {sorted(keys)}")

    x = _ravel_col(mesh["x"]).astype(float)
    y = _ravel_col(mesh["y"]).astype(float)

    elements = np.asarray(mesh["elements"])

    if elements.ndim != 2 or elements.shape[1] < 3:
        raise ValueError(f"Unexpected 'elements' shape {elements.shape}; expected (nelem, >=3).")

    tri_1based = elements[:, :3].astype(np.int64)
    triangles = tri_1based - 1

    tri_sorted = np.sort(triangles, axis=1)
    _, uniq_idx = np.unique(tri_sorted, axis=0, return_index=True)
    triangles = triangles[np.sort(uniq_idx)].astype(np.int32)

    epsg = None
    if "epsg" in mesh:
        try:
            epsg = int(np.asarray(mesh["epsg"]).reshape(-1)[0])
        except Exception:
            epsg = None

    return Mesh2D(x=x, y=y, triangles=triangles, epsg=epsg)


def maybe_sample_triangles(triangles: "np.ndarray", max_tris: int | None, seed: int = 0) -> "np.ndarray":
    import numpy as np

    if max_tris is None or triangles.shape[0] <= max_tris:
        return triangles
    rng = np.random.default_rng(seed)
    idx = rng.choice(triangles.shape[0], size=max_tris, replace=False)
    return triangles[np.sort(idx)]


def main() -> None:
    default_mesh = Path(__file__).resolve().parent / "issm_outputs" / "mesh_info.mat"
    default_h5 = Path(__file__).resolve().parent / "data" / "issm_extracted" / "run_01_tot.h5"

    parser = argparse.ArgumentParser(description="Plot first and last Surface side-by-side for one extracted run.")
    parser.add_argument("--mesh-file", type=Path, default=default_mesh, help="Path to mesh_info.mat")
    parser.add_argument("--run-h5", type=Path, default=default_h5, help="Path to extracted run_XX_tot.h5")
    parser.add_argument("--field", type=str, default="Surface", help="Dataset name to plot (default: Surface)")
    parser.add_argument("--max-tris", type=int, default=500_000, help="Randomly sample at most this many triangles")
    parser.add_argument("--no-edges", action="store_true", help="Disable drawing triangle edges (faster)")
    parser.add_argument(
        "--clim-pctl",
        type=float,
        default=None,
        help="If set (e.g. 99.5), color limits use +/- that percentile (robust to outliers).",
    )
    parser.add_argument("--out", type=Path, default=None, help="If set, save figure to this path instead of showing.")
    args = parser.parse_args()

    import numpy as np
    import h5py
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    mesh = load_mesh(args.mesh_file)
    triangles = maybe_sample_triangles(mesh.triangles, args.max_tris)
    triang = mtri.Triangulation(mesh.x, mesh.y, triangles)

    with h5py.File(args.run_h5, "r") as f:
        if args.field not in f:
            raise KeyError(f"{args.run_h5} missing dataset {args.field!r}. Available: {sorted(f.keys())}")
        data = f[args.field]
        if data.ndim < 2 or data.shape[1] != mesh.x.size:
            raise ValueError(
                f"Unexpected {args.field} shape {data.shape}; expected (nt, {mesh.x.size}) for this mesh."
            )

        a0 = data[0, :].astype(float)
        a1 = data[-1, :].astype(float)
        t0 = float(f["time"][0]) if "time" in f else 0.0
        t1 = float(f["time"][-1]) if "time" in f else float(data.shape[0] - 1)

    vals = np.concatenate([a0, a1])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No finite values to plot.")

    if args.clim_pctl is not None:
        p = float(args.clim_pctl)
        if not (0.0 < p <= 100.0):
            raise ValueError("--clim-pctl must be in (0, 100].")
        vmin, vmax = np.nanpercentile(vals, [100.0 - p, p])
    else:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    for ax, arr, title in (
        (axes[0], a0, f"{args.field} (first)  t={t0:g}"),
        (axes[1], a1, f"{args.field} (last)   t={t1:g}"),
    ):
        m = ax.tripcolor(triang, arr, shading="flat", cmap="viridis", vmin=vmin, vmax=vmax)
        if not args.no_edges:
            ax.triplot(triang, color="k", linewidth=0.08, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    cbar = fig.colorbar(m, ax=axes, shrink=0.9)
    cbar.set_label(args.field)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=200)
        print(f"Saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

