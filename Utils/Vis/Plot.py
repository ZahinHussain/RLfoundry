from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


COLOURS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]


def plot_line(x, y, title, xlabel, ylabel, save_path, colour):

    plt.figure(figsize=(10, 5))

    plt.plot(
        x,
        y,
        color=colour,
        linewidth=2.5,
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main(path):
    folder = Path(path)
    npy_files = sorted(folder.glob("*.npy"))

    for f in npy_files:
        print(f.name)

    folder = Path(path)
    npy_files = sorted(folder.glob("*.npy"))

    print(f"Folder: {folder}")
    print(f"Files found: {len(npy_files)}")

    output_dir = folder / "graphs"

    if not output_dir.exists():
        output_dir.mkdir()
        print("Created: graphs/")
    else:
        print("Using existing: graphs/")

    for i, f in enumerate(npy_files):

        print(f.name)

        data = np.load(f, allow_pickle=True)

        data = np.array(data)

        # handle scalars safely
        if data.ndim == 0:
            data = np.array([data.item()])

        x = np.arange(len(data))

        plot_line(
            x,
            data,
            f.stem,
            "Step",
            "Value",
            output_dir / f"{f.stem}.png",
            COLOURS[i % len(COLOURS)],
        )




if __name__ == "__main__":
    main("/Users/zahin/Documents/GitHub/Specialist project/Bipedal/PPO/BipedalWalker/2026-07-04_18-36-27/numpy/step_1024000")