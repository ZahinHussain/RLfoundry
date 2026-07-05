"""
export_wandb.py

1:1 export of a single W&B run's full logged history to CSV/Parquet.
No downsampling -- every step, every key you logged with wandb.log().

Usage:
    python export_wandb.py

Edit the CONFIG block below before running.
"""

import sys
import wandb
import csvCheck as pd
from tqdm import tqdm

# ============================================================
# CONFIG -- edit these before running
# ============================================================
ENTITY = "zahinahussain-digiterrarium"
PROJECT = "SustainGym-EVCharging-v0-ppo"

# You can set EITHER of these:
#   RUN_ID   -- the short alphanumeric id from a URL like
#               wandb.ai/<entity>/<project>/runs/<run_id>
#   RUN_NAME -- the human-readable run name shown in the UI
#               (e.g. "2026-07-04_23-09-51")
# If RUN_ID is blank, the script looks up the run by RUN_NAME instead.
# If both are blank, it lists every run in the project so you can pick one.
RUN_ID = ""
RUN_NAME = "2026-07-04_23-09-51"

# Only pull these keys instead of every column ever logged.
# Leave as None to pull everything (slower -- transfers every per-dim
# observation/action column too, which you likely don't need for plots).
# Example: KEYS = ["global_step", "charts/episodic_return", "losses/policy_loss"]
KEYS = None

# Rows per API page. wandb's default is small (a few hundred), which means
# many round trips for a run with thousands of logged steps. Bumping this
# cuts the number of requests dramatically -- biggest single speedup available.
PAGE_SIZE = 10000

OUTPUT_CSV = "run_history.csv"
OUTPUT_PARQUET = "run_history.parquet"  # smaller + faster to reload, keeps dtypes

# ============================================================
# EXPORT
# ============================================================

def list_runs(api):
    print(f"No RUN_ID set. Listing runs in {ENTITY}/{PROJECT}:\n")
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    if not runs:
        print("  (no runs found -- check ENTITY/PROJECT are correct)")
        return
    for run in runs:
        print(f"  id={run.id}  name={run.name}  state={run.state}  created={run.created_at}")
    print("\nCopy the id= value you want into RUN_ID at the top of this script, then re-run.")


def find_run_by_name(api, name):
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": name})
    runs = list(runs)
    if not runs:
        print(f"No run found with name '{name}' in {ENTITY}/{PROJECT}.\n")
        list_runs(api)
        sys.exit(1)
    if len(runs) > 1:
        print(f"Multiple runs found with name '{name}':")
        for r in runs:
            print(f"  id={r.id}  state={r.state}  created={r.created_at}")
        print("\nSet RUN_ID explicitly to pick one, then re-run.")
        sys.exit(1)
    return runs[0]


def main():
    api = wandb.Api()

    if RUN_ID:
        run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
    elif RUN_NAME:
        run = find_run_by_name(api, RUN_NAME)
    else:
        list_runs(api)
        sys.exit(0)

    print(f"Run: {run.name} ({run.id})")
    print(f"State: {run.state}")
    print(f"Config: {run.config}")

    # scan_history() streams every row ever logged, no downsampling,
    # no 500-point cap like the default history() call has.
    # wandb doesn't expose a total count up front, so the bar has no
    # fixed length -- it just shows rows/sec and a live count.
    rows = []
    for row in tqdm(
        run.scan_history(keys=KEYS, page_size=PAGE_SIZE),
        desc="Downloading history",
        unit=" rows",
    ):
        rows.append(row)

    if not rows:
        print("\nNo history rows returned -- run may have no logged metrics yet.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # global_step is what you pass to wandb.log(..., step=global_step) --
    # use it as the index so rows are in training order, not log-call order.
    if "global_step" in df.columns:
        df = df.sort_values("global_step").reset_index(drop=True)
    elif "_step" in df.columns:
        df = df.sort_values("_step").reset_index(drop=True)

    # Attach run metadata as columns so this file is self-describing
    # once you're combining multiple exports later.
    df["run_id"] = run.id
    df["run_name"] = run.name
    for k, v in run.config.items():
        # avoid clobbering existing columns (e.g. if you log a metric
        # with the same name as a config key)
        col = f"config_{k}" if k in df.columns else k

        # pandas tries to broadcast lists/dicts/tuples element-wise instead
        # of treating them as one scalar value per row (this is what crashed
        # on 'defaults': [] -- an empty list has length 0, which doesn't
        # match the row count). Stringify anything non-scalar so it's stored
        # as one repeated value per row instead.
        if isinstance(v, (list, dict, tuple, set)):
            v = str(v)

        df[col] = v

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nExported {len(df)} rows, {len(df.columns)} columns")
    print(f"  -> {OUTPUT_CSV}")

    try:
        df.to_parquet(OUTPUT_PARQUET, index=False)
        print(f"  -> {OUTPUT_PARQUET}")
    except ImportError:
        print(
            f"  (skipped {OUTPUT_PARQUET} -- install pyarrow to enable: "
            f"pip install pyarrow)"
        )

    print(f"\nColumns logged: {sorted(df.columns.tolist())}")


if __name__ == "__main__":
    main()