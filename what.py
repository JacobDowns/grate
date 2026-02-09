#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


@dataclass(frozen=True)
class Mesh2D:
    x: np.ndarray  # (nvert,)
    y: np.ndarray  # (nvert,)
    triangles: np.ndarray  # (ntri, 3) int, 0-based
    fields: dict[str, np.ndarray]
    epsg: int | None = None


def _load_mat(path: Path) -> dict:
    try:
        from scipy.io import loadmat

        return loadmat(path)
    except Exception:
        import mat73

        return mat73.loadmat(path)


def _ravel_col(v) -> np.ndarray:
    arr = np.asarray(v)
    return arr.reshape(-1)


def _infer_layering(x: np.ndarray, y: np.ndarray, max_layers: int = 12) -> tuple[int, int]:
    nvert = int(x.size)
    for nlayers in range(max_layers, 1, -1):
        if nvert % nlayers != 0:
            continue
        n2d = nvert // nlayers
        x0, y0 = x[:n2d], y[:n2d]
        ok = True
        for k in range(1, nlayers):
            if not (np.allclose(x0, x[k * n2d : (k + 1) * n2d]) and np.allclose(y0, y[k * n2d : (k + 1) * n2d])):
                ok = False
                break
        if ok:
            return n2d, nlayers
    return nvert, 1


def load_mesh_2d(mesh_file: Path, collapse_layers: bool | None = None) -> Mesh2D:
    mesh = _load_mat(mesh_file)

    keys = {k for k in mesh.keys() if not k.startswith("__")}
    required = {"x", "y", "elements"}
    missing = sorted(required - keys)
    if missing:
        raise KeyError(f"{mesh_file} missing keys: {missing}. Available keys: {sorted(keys)}")

    x_all = _ravel_col(mesh["x"]).astype(float)
    y_all = _ravel_col(mesh["y"]).astype(float)

    elements = np.asarray(mesh["elements"])
    if elements.ndim != 2 or elements.shape[1] < 3:
        raise ValueError(f"Unexpected 'elements' shape {elements.shape}; expected (nelem, >=3).")

    n2d, nlayers = _infer_layering(x_all, y_all)
    if collapse_layers is None:
        collapse_layers = nlayers > 1 or elements.shape[1] > 3

    if collapse_layers:
        x = x_all[:n2d]
        y = y_all[:n2d]
        # Collapse any layered/extruded mesh indices to the base 2D vertex id.
        tri_raw = elements[:, :3].astype(np.int64)
        tri_1based = ((tri_raw - 1) % n2d) + 1
    else:
        x, y = x_all, y_all
        tri_1based = elements[:, :3].astype(np.int64)

    triangles = tri_1based - 1  # 0-based

    # Deduplicate triangles after collapsing layers (or if elements contain repeats).
    tri_sorted = np.sort(triangles, axis=1)
    _, uniq_idx = np.unique(tri_sorted, axis=0, return_index=True)
    triangles = triangles[np.sort(uniq_idx)]

    fields: dict[str, np.ndarray] = {}
    for name in ("bed", "surface", "thickness", "base", "lat", "lon"):
        if name in mesh:
            v = _ravel_col(mesh[name]).astype(float)
            fields[name] = v[: x.size]

    epsg = None
    if "epsg" in mesh:
        try:
            epsg = int(np.asarray(mesh["epsg"]).reshape(-1)[0])
        except Exception:
            epsg = None

    return Mesh2D(x=x, y=y, triangles=triangles.astype(np.int32), fields=fields, epsg=epsg)


def _maybe_sample_triangles(triangles: np.ndarray, max_tris: int | None, seed: int = 0) -> np.ndarray:
    if max_tris is None or triangles.shape[0] <= max_tris:
        return triangles
    rng = np.random.default_rng(seed)
    idx = rng.choice(triangles.shape[0], size=max_tris, replace=False)
    return triangles[np.sort(idx)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Load ISSM mesh_info.mat and plot the 2D mesh.")
    parser.add_argument(
        "--mesh-file",
        type=Path,
        default=Path(__file__).resolve().parent / "issm_outputs" / "mesh_info.mat",
        help="Path to mesh_info.mat exported from ISSM.",
    )
    parser.add_argument(
        "--collapse-layers",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Collapse extruded/layered meshes to a base 2D triangulation (default: auto).",
    )
    parser.add_argument(
        "--color",
        choices=("none", "bed", "surface", "thickness", "base"),
        default="none",
        help="Optional vertex field to color the mesh by.",
    )
    parser.add_argument(
        "--max-tris",
        type=int,
        default=None,
        help="Randomly sample at most this many triangles for faster plotting.",
    )
    parser.add_argument(
        "--no-edges",
        action="store_true",
        help="Disable drawing triangle edges (faster).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="If set, save the figure to this path instead of opening a window.",
    )
    args = parser.parse_args()

    mesh = load_mesh_2d(args.mesh_file, collapse_layers=args.collapse_layers)
    triangles = _maybe_sample_triangles(mesh.triangles, args.max_tris)
    triang = mtri.Triangulation(mesh.x, mesh.y, triangles)

    if args.max_tris is None and not args.no_edges and triang.triangles.shape[0] > 200_000:
        print(
            f"Plotting {triang.triangles.shape[0]:,} triangles with edges can be slow. "
            "Consider --max-tris or --no-edges."
        )

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    if args.color != "none":
        if args.color not in mesh.fields:
            raise KeyError(
                f"Field '{args.color}' not found in {args.mesh_file}. "
                f"Available: {sorted(mesh.fields.keys())}"
            )
        tpc = ax.tripcolor(triang, mesh.fields[args.color], shading="flat", cmap="viridis")
        fig.colorbar(tpc, ax=ax, label=args.color)

    if not args.no_edges:
        ax.triplot(triang, color="k", linewidth=0.15, alpha=0.6)

    ax.set_aspect("equal", adjustable="box")
    title = f"Mesh: {mesh.x.size:,} vertices, {triang.triangles.shape[0]:,} triangles"
    if mesh.epsg is not None:
        title += f" (EPSG:{mesh.epsg})"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=200)
        print(f"Saved mesh plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
