"""
Plot every .npy file found in an RLFoundry checkpoint "step" folder.

Just set FOLDER below to the step_<n> directory and run:
    python plot_step_arrays.py

Auto-detects a sensible plot type per array based on shape:
    - scalar / 0-d       -> printed only (no subplot)
    - 1-d                -> line plot (good for actions, rewards, biases)
    - 2-d                -> heatmap (good for weight matrices)
    - 2-d but "wide/thin" and name suggests a time series -> line plot of columns
    - 3-d+                -> histogram of flattened values (fallback)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---- just edit this and run the script ----
FOLDER = "/home/zahin/Documents/AI MSc/Specialist-project/Building-v0/2026-07-01_00-41-24/numpy/step_7110656"
OUT = None       # None -> saves as step_plots.png inside FOLDER
COLS = 3         # number of subplot columns
SHOW = False     # True -> also pop up an interactive window
# --------------------------------------------


TIME_SERIES_HINTS = ("action", "reward", "obs", "state", "done", "value", "return", "advantage")
WEIGHT_HINTS = ("weight", "bias", "kernel", "param")


def classify(name: str, arr: np.ndarray) -> str:
    lname = name.lower()
    if arr.ndim == 0:
        return "scalar"
    if arr.ndim == 1:
        return "line"
    if arr.ndim == 2:
        # small 2D + weight-ish name -> heatmap; otherwise if it looks like
        # (timesteps, features) with a time-series-y name, plot as lines
        if any(h in lname for h in WEIGHT_HINTS) and not any(h in lname for h in TIME_SERIES_HINTS):
            return "heatmap"
        if any(h in lname for h in TIME_SERIES_HINTS):
            return "line_multi"
        return "heatmap"
    return "hist"


def plot_array(ax, name: str, arr: np.ndarray):
    kind = classify(name, arr)

    if kind == "scalar":
        ax.text(0.5, 0.5, f"{name}\n= {arr.item():.6g}", ha="center", va="center", fontsize=10)
        ax.set_axis_off()

    elif kind == "line":
        ax.plot(arr, linewidth=0.8)
        ax.set_title(f"{name}  shape={arr.shape}", fontsize=9)
        ax.set_xlabel("index")

    elif kind == "line_multi":
        # plot up to 8 columns/dims to keep it readable
        n_dims = arr.shape[1] if arr.ndim == 2 else 1
        for d in range(min(n_dims, 8)):
            ax.plot(arr[:, d], linewidth=0.7, label=f"dim{d}")
        ax.set_title(f"{name}  shape={arr.shape}", fontsize=9)
        ax.set_xlabel("timestep")
        if n_dims > 1:
            ax.legend(fontsize=6, loc="upper right")

    elif kind == "heatmap":
        im = ax.imshow(arr, aspect="auto", cmap="RdBu_r",
                        vmin=-np.max(np.abs(arr)) if arr.size else -1,
                        vmax=np.max(np.abs(arr)) if arr.size else 1)
        ax.set_title(f"{name}  shape={arr.shape}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    else:  # hist
        ax.hist(arr.flatten(), bins=50, color="steelblue")
        ax.set_title(f"{name}  shape={arr.shape} (flattened)", fontsize=9)


def main():
    folder = Path(FOLDER)
    npy_files = sorted(folder.glob("*.npy"))
    if not npy_files:
        raise SystemExit(f"No .npy files found in {folder}")

    print(f"Found {len(npy_files)} .npy files:")
    arrays = {}
    for f in npy_files:
        arr = np.load(f)
        arrays[f.stem] = arr
        print(f"  {f.name:30s} shape={arr.shape} dtype={arr.dtype}")

    n = len(arrays)
    cols = COLS
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (name, arr) in zip(axes, arrays.items()):
        plot_array(ax, name, arr)

    # hide any unused axes
    for ax in axes[n:]:
        ax.set_axis_off()

    fig.suptitle(f"Step folder: {folder.name}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = Path(OUT) if OUT else folder / "step_plots.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot grid to: {out_path}")

    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()