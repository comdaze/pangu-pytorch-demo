# SPDX-License-Identifier: Apache-2.0
"""Compute per-variable mean/std for the WRF dataset -> stats.json (CorrDiff format)."""
import json
import sys
import numpy as np
import xarray as xr

WIND_VARS = ["U1000", "U850", "U10m", "U100m", "V1000", "V850", "V10m", "V100m"]


def group_stats(data_path, group):
    out = {}
    with xr.open_dataset(data_path, group=group) as ds:
        for v in WIND_VARS:
            arr = ds[v].values.astype(np.float64)
            out[v] = {"mean": float(arr.mean()), "std": float(arr.std() + 1e-8)}
    return out


def main(data_path, stats_path):
    stats = {"input": group_stats(data_path, "input"),
             "output": group_stats(data_path, "output")}
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("wrote", stats_path)
    for g in ("input", "output"):
        print(g, {k: round(v["mean"], 2) for k, v in stats[g].items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
