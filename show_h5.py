#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _fmt_bytes(n: int) -> str:
    n_f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_f < 1024 or unit == "TB":
            return f"{n_f:.0f} {unit}" if unit == "B" else f"{n_f:.2f} {unit}"
        n_f /= 1024.0
    return f"{n_f:.2f} TB"


def main() -> None:
    default_file = Path("data/issm_extracted/run_02_tot.h5")

    parser = argparse.ArgumentParser(description="Minimal inspector for extracted ISSM .h5 files.")
    parser.add_argument("h5_file", nargs="?", type=Path, default=default_file, help=f"Path to .h5 (default: {default_file})")
    parser.add_argument("--head", type=int, default=5, help="Number of values to print for 1D datasets (default: 5)")
    args = parser.parse_args()

    path = args.h5_file
    if not path.exists():
        raise FileNotFoundError(path)

    import h5py
    import numpy as np

    #print(f"{path} ({_fmt_bytes(path.stat().st_size)})")
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys())
        print(f"Datasets ({len(keys)}): {keys}")

        for k in keys:
            d = f[k]
            if not isinstance(d, h5py.Dataset):
                continue
            nbytes = int(np.prod(d.shape)) * int(d.dtype.itemsize) if d.shape is not None else 0
            print(f"- {k}: shape={d.shape}, dtype={d.dtype}, ~{_fmt_bytes(nbytes)}")

        # A tiny peek at time and any 1D scalars
        if "time" in f:
            t = f["time"][:]/ 1000.
            print(t[-1] - t[0])
            #print(t.cumsum())
            import matplotlib.pyplot as plt
            plt.plot(np.arange(len(t)), t)
            plt.show()
            #print(f"\ntime: n={t.size} head={t[: args.head]}")

        one_d = [k for k in keys if isinstance(f[k], h5py.Dataset) and len(f[k].shape) == 1 and k != "time"]
        if one_d:
            print("\n1D datasets (head):")
            for k in one_d[:20]:
                v = f[k][:]
                print(f"  {k}: {v[: args.head]}")
            if len(one_d) > 20:
                print(f"  ... ({len(one_d) - 20} more)")


if __name__ == "__main__":
    main()
