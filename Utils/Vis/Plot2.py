"""
plot_npys.py

Takes a folder of .npy files (e.g. RLFoundry's raw dumped metrics) and
plots each one as a line graph, saving everything into a 'graph' folder.

Usage:
    python plot_npys.py /path/to/npy_folder
    python plot_npys.py /path/to/npy_folder --out my_graphs
    python plot_npys.py /path/to/npy_folder --xkey global_step_np
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#000000",
]


def load_x_axis(folder: Path, xkey: str, length: int):
    """Try to use a step/index array as the x-axis, else fall back to range()."""
    xfile = folder / f"{xkey}.npy"
    if xfile.exists():
        x = np.load(xfile, allow_pickle=True)
        if x.ndim == 1 and len(x) == length:
            return x
    return np.arange(length)


def plot_array(name: str, arr: np.ndarray, x, out_dir: Path):
    arr = np.asarray(arr)

    # Scalar / 0-d: nothing to plot, just skip


    fig, ax = plt.subplots(figsize=(9, 5))

    if arr.ndim == 1:
        ax.plot(x[: len(arr)], arr, color=COLORS[0], linewidth=1.4)

    elif arr.ndim == 2:
        n_cols = arr.shape[1]
        # too many dims to legend sensibly -> just overlay them faintly
        show_legend = n_cols <= 8
        for i in range(n_cols):
            ax.plot(
                x[: arr.shape[0]],
                arr[:, i],
                color=COLORS[i % len(COLORS)],
                linewidth=1.1,
                alpha=0.9 if show_legend else 0.5,
                label=f"dim {i}" if show_legend else None,
            )
        if show_legend:
            ax.legend(fontsize=8, frameon=False)

    else:
        # Higher-dim arrays: flatten trailing dims per timestep and plot mean ± just the mean line
        flat = arr.reshape(arr.shape[0], -1)
        mean_line = flat.mean(axis=1)
        ax.plot(x[: len(mean_line)], mean_line, color=COLORS[0], linewidth=1.4)
        ax.set_title(f"{name}  (mean over shape {arr.shape[1:]})")

    if not ax.get_title():
        ax.set_title(name)
    ax.set_xlabel("step")
    ax.set_ylabel(name)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()

    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot every .npy in a folder as a line graph.")
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default="/Users/zahin/Documents/GitHub/Specialist project/Bipedal/PPO/BipedalWalker/2026-07-04_18-36-27/numpy/step_1024000",
        help="Folder containing .npy files (default: Bipedal/PPO/BipedalWalker/2026-07-04_18-36-27/numpy/step_1024000)",
    )
    parser.add_argument("--out", type=str, default="graph", help="Output folder name (default: graph)")
    parser.add_argument("--xkey", type=str, default="global_step_np",
                         help="Filename (no .npy) to use as x-axis if present (default: global_step_np)")
    args = parser.parse_args()

    folder = Path(args.path)
    if not folder.is_dir():
        sys.exit(f"Not a folder: {folder}")

    out_dir = folder / args.out
    out_dir.mkdir(exist_ok=True)

    npy_files = sorted(folder.glob("*.npy"))
    if not npy_files:
        sys.exit(f"No .npy files found in {folder}")

    for f in npy_files:
        name = f.stem
        if name == args.xkey:
            continue  # this one IS the x-axis, don't plot it against itself
        try:
            arr = np.load(f, allow_pickle=True)
        except Exception as e:
            print(f"skip {name}: failed to load ({e})")
            continue

        length = len(arr) if getattr(arr, "ndim", 0) > 0 else 1
        x = load_x_axis(folder, args.xkey, length)
        plot_array(name, arr, x, out_dir)

    print(f"\nDone. Graphs saved to: {out_dir}")


if __name__ == "__main__":
    main()