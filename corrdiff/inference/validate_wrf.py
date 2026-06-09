# SPDX-License-Identifier: Apache-2.0
"""
Validate CorrDiff downscaling on the WRF dataset.

Reads the generation output (groups: input / truth / prediction) and:
  1. reports per-variable RMSE of
       - bilinear baseline (input)         vs truth
       - CorrDiff ensemble-mean prediction vs truth
     and the improvement of CorrDiff over the baseline.
  2. saves a comparison figure (input | CorrDiff mean | truth) for the 10 m
     wind speed at the first time step.

Usage:
    LD_LIBRARY_PATH=/opt/conda/lib python inference/validate_wrf.py \
        --nc corrdiff_output.nc --out inference/wrf_corrdiff_validation.png
"""
import argparse

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARS = ["U1000", "U850", "U10m", "U100m", "V1000", "V850", "V10m", "V100m"]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="corrdiff_output.nc")
    ap.add_argument("--out", default="inference/wrf_corrdiff_validation.png")
    args = ap.parse_args()

    inp = xr.open_dataset(args.nc, group="input")
    tru = xr.open_dataset(args.nc, group="truth")
    pred = xr.open_dataset(args.nc, group="prediction")

    print(f"times={inp.sizes['time']} ensembles={pred.sizes['ensemble']} "
          f"grid={tru.sizes['y']}x{tru.sizes['x']}\n")

    print(f"{'var':>7} | {'baseline RMSE':>13} | {'corrdiff RMSE':>13} | {'improve %':>9}")
    print("-" * 54)
    tot_b = tot_c = 0.0
    for v in VARS:
        t = tru[v].values                       # (time, y, x)
        b = inp[v].values                        # (time, y, x) bilinear baseline
        c = pred[v].values.mean(axis=0)          # ensemble mean -> (time, y, x)
        rb, rc = rmse(b, t), rmse(c, t)
        imp = 100.0 * (rb - rc) / rb if rb > 0 else 0.0
        tot_b += rb
        tot_c += rc
        print(f"{v:>7} | {rb:13.4f} | {rc:13.4f} | {imp:8.2f}%")
    imp_all = 100.0 * (tot_b - tot_c) / tot_b
    print("-" * 54)
    print(f"{'MEAN':>7} | {tot_b/len(VARS):13.4f} | {tot_c/len(VARS):13.4f} | {imp_all:8.2f}%\n")

    # 10 m wind speed comparison figure at first time step
    def wspd(ds, ens=False):
        u = ds["U10m"].values
        v = ds["V10m"].values
        if ens:
            u, v = u.mean(0), v.mean(0)
        return np.sqrt(u ** 2 + v ** 2)

    ws_in = wspd(inp)[0]
    ws_pr = wspd(pred, ens=True)[0]
    ws_tr = wspd(tru)[0]
    vmax = max(ws_in.max(), ws_pr.max(), ws_tr.max())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, data, title in zip(
        axes,
        [ws_in, ws_pr, ws_tr],
        ["Input (25km bilinear)", "CorrDiff (5km, ens-mean)", "Truth (5km WRF)"],
    ):
        im = ax.imshow(data, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="10 m wind speed (m/s)")
    fig.suptitle("CorrDiff WRF downscaling — 10 m wind speed", fontsize=13)
    fig.savefig(args.out, dpi=130)
    print(f"saved figure -> {args.out}")


if __name__ == "__main__":
    main()
