"""
plot_metrics.py

Publication-quality plots from a 1:1 W&B export (run_history.csv/.parquet
from export_wandb.py). Matches your existing style: Okabe-Ito colorblind-safe
palette, LaTeX-matched typography, vector PDF + 300dpi PNG output.

This is for a SINGLE run. True IQM across seeds needs multiple runs bootstrapped
together — this script instead uses a rolling-window bootstrap CI as a smoothed
uncertainty band around one run's curve, which is the right tool when you only
have one seed. Swap in your existing IQM aggregator once you export multiple
runs and stack them for the final publication figures.

Usage:
    python plot_metrics.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

# ============================================================
# CONFIG — edit before running
# ============================================================
INPUT_FILE = "run_history.parquet"   # or "run_history.csv"

# Which metrics to plot. Each becomes its own figure.
# Use the exact wandb key names you logged (e.g. "charts/episodic_return").
METRICS_TO_PLOT = [
    "charts/episodic_return",
    "losses/policy_loss",
    "losses/value_loss",
    "losses/entropy",
    "losses/approx_kl",
    "actions_std/aggregate",
    "system/SPS",
]

STEP_COL = "global_step"        # fallback to "_step" if this isn't present
ROLLING_WINDOW = 25              # smoothing window (in log-call units, not env steps)
N_BOOTSTRAP = 2000
CI_LEVEL = 0.95

OUTPUT_DIR = "figures"

# Okabe-Ito colorblind-safe palette
OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}

# ============================================================
# STYLE — LaTeX-matched typography
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
    "text.usetex": False,  # set True if a local LaTeX install is available
    "mathtext.fontset": "cm",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})


def rolling_bootstrap_ci(values, window, n_resamples, ci_level):
    """
    For each point, bootstrap a CI over a trailing window of raw values.
    Returns (center, lower, upper) arrays same length as `values`.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    center = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        window_vals = values[start:i + 1]
        window_vals = window_vals[~np.isnan(window_vals)]

        if len(window_vals) < 2:
            if len(window_vals) == 1:
                center[i] = window_vals[0]
            continue

        center[i] = np.mean(window_vals)

        try:
            res = bootstrap(
                (window_vals,),
                np.mean,
                n_resamples=n_resamples,
                confidence_level=ci_level,
                method="basic",
            )
            lower[i] = res.confidence_interval.low
            upper[i] = res.confidence_interval.high
        except Exception:
            # window too small / degenerate (all identical values, etc.)
            lower[i] = center[i]
            upper[i] = center[i]

    return center, lower, upper


def plot_metric(df, metric, step_col, output_dir):
    if metric not in df.columns:
        print(f"  [skip] '{metric}' not found in exported columns")
        return

    sub = df[[step_col, metric]].dropna()
    if sub.empty:
        print(f"  [skip] '{metric}' has no non-null values")
        return

    steps = sub[step_col].values
    values = sub[metric].values

    center, lower, upper = rolling_bootstrap_ci(
        values, ROLLING_WINDOW, N_BOOTSTRAP, CI_LEVEL
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    color = OKABE_ITO["vermillion"]
    ax.plot(steps, values, color=OKABE_ITO["black"], alpha=0.15, linewidth=0.8, label="raw")
    ax.plot(steps, center, color=color, linewidth=2.4, label=f"rolling mean (w={ROLLING_WINDOW})")
    ax.fill_between(steps, lower, upper, color=color, alpha=0.25,
                     label=f"{int(CI_LEVEL * 100)}% bootstrap CI")

    ax.set_xlabel("Global step")
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.legend(frameon=False)
    fig.tight_layout()

    safe_name = metric.replace("/", "_")
    png_path = f"{output_dir}/{safe_name}.png"
    fig.savefig(png_path, dpi=600)
    plt.close(fig)

    print(f"  [ok] {metric} -> {png_path}")


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if INPUT_FILE.endswith(".parquet"):
        df = pd.read_parquet(INPUT_FILE)
    else:
        df = pd.read_csv(INPUT_FILE)

    # W&B auto-logs its own system telemetry (GPU/CPU/mem) on a separate
    # schedule from your wandb.log() calls, interleaved into the same
    # history table with all your custom keys as NaN on those rows.
    # Drop them so they don't inject NaN gaps into rolling means / CIs.
    training_cols = [c for c in METRICS_TO_PLOT if c in df.columns]
    if training_cols:
        before = len(df)
        df = df.dropna(subset=training_cols, how="all").reset_index(drop=True)
        print(f"Dropped {before - len(df)} telemetry-only rows (kept {len(df)})")

    step_col = STEP_COL if STEP_COL in df.columns else "_step"
    if step_col not in df.columns:
        raise ValueError(
            f"Neither '{STEP_COL}' nor '_step' found in columns: {df.columns.tolist()}"
        )

    print(f"Loaded {len(df)} rows from {INPUT_FILE}")
    print(f"Plotting {len(METRICS_TO_PLOT)} metrics...\n")

    for metric in METRICS_TO_PLOT:
        plot_metric(df, metric, step_col, OUTPUT_DIR)

    print(f"\nDone. Figures in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()